"""End-to-end pipeline: data -> features -> detectors -> health -> schedule.

Split protocol (leak-free):
- Detectors train ONLY on feature rows from healthy machines (train subset).
- The threshold comes from held-out healthy validation machines at a stated
  FPR budget.
- Metrics are computed on rows the detectors never trained on: all faulty
  machines' days (positives after the true onset, negatives before) plus the
  healthy validation machines' days (negatives).

Recommendation policy: recommend the simpler detector (PCA) unless the
autoencoder clearly wins — PR-AUC margin > MARGIN_TO_SWITCH. The measured
margin is always reported either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pdm.data import DataConfig, PlantData, generate
from pdm.detect import (
    AEDetector,
    DetectorReport,
    PCADetector,
    detection_delays,
    pr_auc,
    pr_curve,
    precision_at_k,
    roc_auc,
    threshold_at_fpr,
    top_sensors,
    torch_available,
)
from pdm.features import FeatureConfig, FeatureSet, build_features
from pdm.health import HealthReport, machine_health
from pdm.schedule import Job, ScheduleComparison, compare_schedules

MARGIN_TO_SWITCH = 0.03  # AE must beat PCA by more than this PR-AUC margin


@dataclass
class Alert:
    machine: int
    day: int
    score: float
    sensors: list[str]


@dataclass
class PipelineResult:
    data: PlantData
    features: FeatureSet
    train_machines: list[int]
    val_machines: list[int]
    reports: dict[str, DetectorReport]
    recommended: str
    margin: float  # AE PR-AUC minus PCA PR-AUC (nan if AE unavailable)
    chosen_scores: np.ndarray  # (n_machines, n_feature_days)
    chosen_threshold: float
    health: HealthReport
    alerts: list[Alert]
    schedule: ScheduleComparison
    fpr_budget: float
    explanations: dict[int, list[str]] = field(default_factory=dict)


def _evaluate(
    detector,
    fs: FeatureSet,
    data: PlantData,
    eval_machines: list[int],
    val_scores: np.ndarray,
    fpr_budget: float,
    k: int,
) -> tuple[DetectorReport, np.ndarray]:
    n_machines = fs.X.shape[0]
    d = fs.n_features
    all_scores = detector.score(fs.X.reshape(-1, d)).reshape(n_machines, -1)
    threshold = threshold_at_fpr(val_scores, fpr_budget)
    val_fpr = float((val_scores > threshold).mean())

    day_labels = data.day_labels()
    y = day_labels[np.ix_(eval_machines, fs.days)].ravel()
    s = all_scores[eval_machines].ravel()

    precision, recall = pr_curve(y, s)
    delays = []
    n_detected = 0
    faulty = data.faulty_machines()
    onset = {lab.machine: lab.onset_day for lab in data.labels}
    for m in faulty:
        first = detection_delays(all_scores[m], fs.days, onset[m], threshold)
        if first is not None:
            n_detected += 1
            delays.append(first - onset[m])
    report = DetectorReport(
        name=detector.name,
        roc_auc=roc_auc(y, s),
        pr_auc=pr_auc(y, s),
        precision_at_k=precision_at_k(y, s, k),
        k=min(k, y.size),
        threshold=threshold,
        fpr_budget=fpr_budget,
        val_fpr=val_fpr,
        mean_delay_days=float(np.mean(delays)) if delays else float("nan"),
        n_detected=n_detected,
        n_faulty=len(faulty),
        precision_curve=precision,
        recall_curve=recall,
    )
    return report, all_scores


def run_pipeline(
    config: DataConfig | None = None,
    feature_config: FeatureConfig | None = None,
    include_ae: bool = True,
    ae_epochs: int = 400,
    fpr_budget: float = 0.05,
    top_k: int = 50,
    n_jobs: int = 8,
    n_crews: int = 2,
    horizon_days: int = 4,
) -> PipelineResult:
    cfg = config or DataConfig()
    data = generate(cfg)
    fs = build_features(data, feature_config)
    d = fs.n_features

    healthy = data.healthy_machines()
    if len(healthy) < 2:
        raise ValueError("need at least 2 healthy machines for train/val split")
    rng = np.random.default_rng(cfg.seed + 1)
    order = list(rng.permutation(healthy))
    n_val = max(1, len(healthy) // 3)
    val_machines = sorted(int(m) for m in order[:n_val])
    train_machines = sorted(int(m) for m in order[n_val:])
    eval_machines = sorted(data.faulty_machines() + val_machines)

    X_train = fs.X[train_machines].reshape(-1, d)
    X_val = fs.X[val_machines].reshape(-1, d)

    reports: dict[str, DetectorReport] = {}
    score_matrices: dict[str, np.ndarray] = {}
    detectors: dict[str, object] = {}

    pca = PCADetector().fit(X_train)
    reports["pca"], score_matrices["pca"] = _evaluate(
        pca, fs, data, eval_machines, pca.score(X_val), fpr_budget, top_k
    )
    detectors["pca"] = pca

    margin = float("nan")
    if include_ae and torch_available():
        ae = AEDetector(epochs=ae_epochs).fit(X_train)
        reports["autoencoder"], score_matrices["autoencoder"] = _evaluate(
            ae, fs, data, eval_machines, ae.score(X_val), fpr_budget, top_k
        )
        detectors["autoencoder"] = ae
        margin = reports["autoencoder"].pr_auc - reports["pca"].pr_auc

    if "autoencoder" in reports and margin > MARGIN_TO_SWITCH:
        recommended = "autoencoder"
    else:
        recommended = "pca"

    chosen = detectors[recommended]
    chosen_scores = score_matrices[recommended]
    threshold = reports[recommended].threshold

    health = machine_health(chosen_scores, threshold)

    per_feat = chosen.per_feature_error(fs.X.reshape(-1, d)).reshape(fs.X.shape[0], -1, d)
    alerts: list[Alert] = []
    explanations: dict[int, list[str]] = {}
    for m in range(fs.X.shape[0]):
        hot = np.where(chosen_scores[m] > threshold)[0]
        for i in hot:
            alerts.append(
                Alert(
                    machine=m,
                    day=int(fs.days[i]),
                    score=float(chosen_scores[m, i]),
                    sensors=top_sensors(per_feat[m, i], fs.names),
                )
            )
        if hot.size:
            explanations[m] = top_sensors(per_feat[m, hot].mean(axis=0), fs.names)

    urgent = health.ranking[:n_jobs]
    urg = health.urgency[urgent]
    lo, hi = float(urg.min()), float(urg.max())
    span = hi - lo

    def _weight(u: float) -> int:
        if span < 1e-9:
            return 5
        return 1 + int(round(9.0 * (u - lo) / span))  # scale to 1..10 across the jobs

    jobs = [
        Job(
            machine=int(m),
            duration_h=int(3 + min(3, int((100 - health.final[m]) // 30))),
            weight=_weight(float(health.urgency[m])),
        )
        for m in urgent
    ]
    jobs_fifo_order = sorted(jobs, key=lambda j: j.machine)  # request order = machine id
    schedule = compare_schedules(
        jobs_fifo_order, n_crews=n_crews, n_days=horizon_days, day_hours=8
    )

    return PipelineResult(
        data=data,
        features=fs,
        train_machines=train_machines,
        val_machines=val_machines,
        reports=reports,
        recommended=recommended,
        margin=margin,
        chosen_scores=chosen_scores,
        chosen_threshold=threshold,
        health=health,
        alerts=alerts,
        schedule=schedule,
        fpr_budget=fpr_budget,
        explanations=explanations,
    )
