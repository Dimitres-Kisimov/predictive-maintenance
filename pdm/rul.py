"""Remaining-useful-life (RUL) estimation, honestly evaluated.

A prognostics layer on TOP of the anomaly detector. The health index in
`pdm/health.py` is deliberately only a heuristic 0-100 urgency ranking and says
so everywhere; this module is the actual time-to-failure model, and it is
evaluated against a ground-truth failure day rather than asserted.

What it does
------------
For the two PROGRESSIVE-degradation fault kinds (slow_drift, accelerating_drift)
the injected fault magnitude grows monotonically, so a failure day is well
defined: `PlantData.failure_day` solves the exact injected growth law for the
first day the fault-induced vibration rise reaches an illustrative failure level.
RUL at day t is then `failure_day - t`.

The predictor reuses the RECOMMENDED detector's own risk score - no new model is
trained on the raw sensors. We EWMA-smooth the daily anomaly score (the same
smoothing `pdm/health.py` uses) and express it as a log-scale exceedance over the
alert threshold ("severity"). A one-feature log-linear degradation model then
maps severity -> RUL:  log1p(RUL) = a + b * severity.

Crucially the model uses ONLY observable signals - the anomaly score - and never
the true onset day, so it reflects what would be knowable in production.

Honest evaluation
-----------------
- LEAVE-ONE-MACHINE-OUT cross-validation: each machine's RUL is predicted by a
  model fit only on the OTHER machines, so no machine is scored by a fit that saw
  it. RUL is only defined - and only scored - on the degradation phase
  (onset <= day <= failure_day) of machines whose failure is observed inside the
  window; right-censored machines are excluded (you cannot measure error against
  a failure you never saw).
- Reported: MAE and RMSE in days; a naive mean-RUL baseline (the same asymmetry
  honesty as the PCA-vs-AE and CP-SAT-vs-FIFO comparisons elsewhere in the repo);
  MAE inside the actionable near-failure window; alpha-accuracy (the alpha-lambda
  prognostics cone); and a per-machine prognostic horizon.

Honest framing
--------------
- All data is SYNTHETIC (pdm/data.py); no real telemetry.
- The failure LEVEL (FAILURE_VIB_RISE) is an ILLUSTRATIVE synthetic stand-in so
  RUL has a defined target - it is NOT a measured engineering limit, and the day
  counts move if you change it. RUL numbers measure the mechanics of the pipeline
  on this generator, not field performance.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pdm.data import FAILURE_VIB_RISE, PROGRESSIVE_FAULTS
from pdm.health import ewma

_EPS = 1e-12


@dataclass(frozen=True)
class RULConfig:
    """Settings for the RUL layer. The failure level is ILLUSTRATIVE (see module doc)."""

    vib_rise: float = FAILURE_VIB_RISE  # illustrative failure level (mm/s of induced rise)
    ewma_alpha: float = 0.3  # smoothing for the severity signal (matches health.py default)
    horizon_days: int = 10  # actionable near-failure window for the focused MAE
    alpha: float = 0.5  # alpha-lambda relative-accuracy cone half-width
    alpha_floor_days: float = 1.0  # absolute cone floor so RUL ~ 0 stays achievable
    unit: str = "days"


@dataclass(frozen=True)
class MachineRUL:
    """Held-out RUL quality for one machine."""

    machine: int
    fault_type: str
    n_days: int  # degradation-phase days scored for this machine
    onset_day: int
    failure_day: int
    mae: float
    prognostic_horizon: float  # earliest lead time (days) predictions stay inside the cone


@dataclass(frozen=True)
class RULEvaluation:
    config: RULConfig
    n_samples: int  # scored machine-days (degradation phase, observed failures)
    n_machines: int  # progressive machines with an observed in-window failure
    mae: float
    rmse: float
    baseline_mae: float  # naive per-fold mean-RUL predictor, for comparison
    mae_within_horizon: float  # MAE restricted to true RUL <= horizon_days
    n_within_horizon: int
    alpha_accuracy: float  # fraction of predictions inside the alpha-lambda cone
    mean_prognostic_horizon: float
    coef: tuple[float, float]  # global log-linear (a, b) fit on all scored samples
    machines: tuple[MachineRUL, ...]
    # pooled, aligned held-out arrays (for artifacts / tests)
    sample_machine: np.ndarray
    sample_day: np.ndarray
    sample_severity: np.ndarray
    true_rul: np.ndarray
    pred_rul: np.ndarray


# ---------------------------------------------------------------- model


def severity_series(scores: np.ndarray, threshold: float, alpha: float = 0.3) -> np.ndarray:
    """Log-scale exceedance of the EWMA-smoothed anomaly score over the threshold.

    0 while the smoothed score sits at/below the alert threshold, rising
    monotonically as it climbs above it. This is exactly the degradation signal
    the heuristic health index is built from, reused here as the RUL predictor.
    """
    smooth = ewma(np.asarray(scores, dtype=float), alpha=alpha)
    return np.log1p(np.clip(smooth / max(threshold, _EPS) - 1.0, 0.0, None))


def fit_loglinear(severity: np.ndarray, rul: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of log1p(RUL) = a + b * severity. Deterministic."""
    sev = np.asarray(severity, dtype=float)
    y = np.log1p(np.asarray(rul, dtype=float))
    design = np.column_stack([np.ones_like(sev), sev])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1])


def predict_rul(coef: tuple[float, float], severity: np.ndarray) -> np.ndarray:
    """Predicted RUL (days, >= 0) from fitted coefficients."""
    a, b = coef
    z = np.clip(a + b * np.asarray(severity, dtype=float), -50.0, 50.0)
    return np.clip(np.expm1(z), 0.0, None)


# ---------------------------------------------------------------- sample building


def _collect_samples(result, cfg: RULConfig):
    """Degradation-phase (severity, RUL) samples for observed-failure machines.

    Returns aligned arrays (machine, day, severity, rul) and a machine->fault_type
    / onset / failure_day map. Only progressive-fault machines whose failure day
    falls inside the observation window contribute.
    """
    data = result.data
    fs = result.features
    threshold = result.chosen_threshold
    n_days = data.n_days
    day_index = {int(t): i for i, t in enumerate(fs.days)}

    machines, days, sev, rul = [], [], [], []
    meta: dict[int, tuple[str, int, int]] = {}
    for lab in data.labels:
        if lab.fault_type not in PROGRESSIVE_FAULTS:
            continue
        fday = data.failure_day(lab.machine, vib_rise=cfg.vib_rise)
        if fday is None or fday >= n_days:  # censored: failure not observed in-window
            continue
        severity = severity_series(
            result.chosen_scores[lab.machine], threshold, alpha=cfg.ewma_alpha
        )
        for t in range(lab.onset_day, fday + 1):
            if t not in day_index:  # day has no feature row (pre-window warm-up)
                continue
            machines.append(lab.machine)
            days.append(t)
            sev.append(float(severity[day_index[t]]))
            rul.append(float(fday - t))
        meta[lab.machine] = (lab.fault_type, lab.onset_day, fday)
    return (
        np.asarray(machines, dtype=int),
        np.asarray(days, dtype=int),
        np.asarray(sev, dtype=float),
        np.asarray(rul, dtype=float),
        meta,
    )


def _prognostic_horizon(days: np.ndarray, true: np.ndarray, in_cone: np.ndarray) -> float:
    """Earliest lead time (days before failure) from which predictions stay in-cone.

    Standard prognostics metric: walk backward from the failure day; the horizon
    is the true RUL at the start of the longest run of in-cone predictions that
    reaches failure. 0 if the final prediction is already outside the cone.
    """
    order = np.argsort(days)  # ascending day -> descending RUL toward failure
    cone = in_cone[order]
    rul = true[order]
    ph = 0.0
    for i in range(len(cone) - 1, -1, -1):
        if not cone[i]:
            break
        ph = float(rul[i])
    return ph


# ---------------------------------------------------------------- evaluation


def evaluate_rul(result, config: RULConfig | None = None) -> RULEvaluation:
    """Leave-one-machine-out RUL evaluation on a finished PipelineResult."""
    cfg = config or RULConfig()
    machine, day, sev, rul, meta = _collect_samples(result, cfg)
    uniq = np.unique(machine)
    if uniq.size < 2:
        raise ValueError(
            "need >= 2 progressive-fault machines with an observed in-window failure "
            "for leave-one-machine-out RUL cross-validation"
        )

    pred = np.full(sev.shape, np.nan)
    base = np.full(sev.shape, np.nan)
    for m in uniq:
        test = machine == m
        train = ~test
        coef = fit_loglinear(sev[train], rul[train])
        pred[test] = predict_rul(coef, sev[test])
        base[test] = float(rul[train].mean())

    err = pred - rul
    abs_err = np.abs(err)
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(np.mean(err**2)))
    baseline_mae = float(np.abs(base - rul).mean())

    hz = rul <= cfg.horizon_days
    mae_hz = float(abs_err[hz].mean()) if hz.any() else float("nan")

    in_cone = abs_err <= (cfg.alpha * rul + cfg.alpha_floor_days)
    alpha_acc = float(in_cone.mean())

    machines: list[MachineRUL] = []
    phs: list[float] = []
    for m in uniq:
        test = machine == m
        ft, onset, fday = meta[int(m)]
        ph = _prognostic_horizon(day[test], rul[test], in_cone[test])
        phs.append(ph)
        machines.append(
            MachineRUL(
                machine=int(m),
                fault_type=ft,
                n_days=int(test.sum()),
                onset_day=int(onset),
                failure_day=int(fday),
                mae=float(abs_err[test].mean()),
                prognostic_horizon=ph,
            )
        )

    return RULEvaluation(
        config=cfg,
        n_samples=int(sev.size),
        n_machines=int(uniq.size),
        mae=mae,
        rmse=rmse,
        baseline_mae=baseline_mae,
        mae_within_horizon=mae_hz,
        n_within_horizon=int(hz.sum()),
        alpha_accuracy=alpha_acc,
        mean_prognostic_horizon=float(np.mean(phs)) if phs else float("nan"),
        coef=fit_loglinear(sev, rul),
        machines=tuple(machines),
        sample_machine=machine,
        sample_day=day,
        sample_severity=sev,
        true_rul=rul,
        pred_rul=pred,
    )


def recommendation_text(ev: RULEvaluation) -> str:
    """Plain-language, honestly-framed summary of the RUL evaluation."""
    c = ev.config
    lift = ev.baseline_mae - ev.mae
    lines = [
        "Remaining-useful-life estimation (SYNTHETIC data; ILLUSTRATIVE failure level - "
        "not a measured engineering limit)",
        f"  Failure level: {c.vib_rise:g} mm/s of fault-induced vibration rise "
        f"(illustrative); progressive faults only (slow_drift, accelerating_drift).",
        f"  Scored: {ev.n_samples} degradation-phase machine-days across {ev.n_machines} "
        f"machines, leave-one-machine-out.",
        "",
        "  Held-out accuracy:",
        f"    MAE               {ev.mae:.2f} {c.unit}",
        f"    RMSE              {ev.rmse:.2f} {c.unit}",
        f"    naive-mean MAE    {ev.baseline_mae:.2f} {c.unit} "
        f"(model is {lift:+.2f} {c.unit} vs this baseline)",
        f"    MAE (RUL<={c.horizon_days}{c.unit[0]})    {ev.mae_within_horizon:.2f} {c.unit} "
        f"over {ev.n_within_horizon} near-failure days",
        f"    alpha-accuracy    {ev.alpha_accuracy:.2f} (within +/-{c.alpha:g}*RUL"
        f"+{c.alpha_floor_days:g}{c.unit[0]} cone)",
        f"    prognostic horiz. {ev.mean_prognostic_horizon:.1f} {c.unit} mean lead time in-cone",
        "",
        "  The predictor reuses the recommended detector's own risk score and never sees the "
        "true onset day. The failure level is illustrative: day counts scale with it, so treat "
        "them as the SHAPE of the degradation-to-failure relationship, not calibrated field RUL.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- CSV


def write_csv(ev: RULEvaluation, path: str | Path) -> Path:
    """Write the held-out per-machine-day predictions as a CSV the README references."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    c = ev.config
    order = np.lexsort((ev.sample_day, ev.sample_machine))
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["# SYNTHETIC data; failure level is an ILLUSTRATIVE assumption, not a measured limit"]
        )
        w.writerow(
            [
                f"# failure_vib_rise={c.vib_rise:g}mm/s",
                f"ewma_alpha={c.ewma_alpha:g}",
                f"model=log1p(RUL)=a+b*severity a={ev.coef[0]:.4f} b={ev.coef[1]:.4f}",
            ]
        )
        w.writerow(
            [
                f"# LOMO_MAE_days={ev.mae:.4f}",
                f"RMSE_days={ev.rmse:.4f}",
                f"naive_mean_MAE_days={ev.baseline_mae:.4f}",
                f"MAE_within_{c.horizon_days}d={ev.mae_within_horizon:.4f}",
                f"alpha{c.alpha:g}_accuracy={ev.alpha_accuracy:.4f}",
                f"n_samples={ev.n_samples}",
                f"n_machines={ev.n_machines}",
            ]
        )
        w.writerow(
            ["machine", "day", "fault_type", "onset_day", "failure_day", "severity",
             "true_rul_days", "pred_rul_days", "abs_error_days"]
        )
        meta = {mr.machine: mr for mr in ev.machines}
        for i in order:
            m = int(ev.sample_machine[i])
            mr = meta[m]
            w.writerow(
                [
                    m,
                    int(ev.sample_day[i]),
                    mr.fault_type,
                    mr.onset_day,
                    mr.failure_day,
                    f"{ev.sample_severity[i]:.5f}",
                    f"{ev.true_rul[i]:.1f}",
                    f"{ev.pred_rul[i]:.3f}",
                    f"{abs(ev.pred_rul[i] - ev.true_rul[i]):.3f}",
                ]
            )
    return path


# ---------------------------------------------------------------- hand-drawn SVG

_SVG_INK = "#0b0b0b"
_SVG_INK2 = "#52514e"
_SVG_MUTED = "#898781"
_SVG_GRID = "#e1e0d9"
_SVG_SURFACE = "#fcfcfb"
_SVG_IDEAL = "#898781"  # ideal y=x line
_SVG_DRIFT = "#2a78d6"  # slow_drift points (repo primary)
_SVG_ACCEL = "#008300"  # accelerating_drift points (repo secondary)


def _nice_top(v: float) -> float:
    if v <= 0:
        return 1.0
    mag = 10.0 ** np.floor(np.log10(v))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if m * mag >= v:
            return m * mag
    return 10.0 * mag


def write_svg(ev: RULEvaluation, path: str | Path) -> Path:
    """Hand-drawn, dependency-free, deterministic predicted-vs-true RUL scatter.

    Points sit against the ideal y=x line inside the alpha-lambda accuracy cone;
    a side panel carries the headline metrics. No matplotlib, no wall-clock, no
    RNG - identical bytes every run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    c = ev.config

    axis_top = _nice_top(max(float(ev.true_rul.max()), float(ev.pred_rul.max()), 1.0))

    W, H = 720, 440
    ml, mr, mt, mb = 66, 210, 58, 60
    pw, ph = W - ml - mr, H - mt - mb

    def px(x: float) -> float:
        return ml + (x / axis_top if axis_top else 0.0) * pw

    def py(y: float) -> float:
        return mt + (1.0 - (y / axis_top if axis_top else 0.0)) * ph

    def esc(t: str) -> str:
        return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="Segoe UI, Helvetica, Arial, sans-serif">'
    )
    parts.append(f'<rect width="{W}" height="{H}" fill="{_SVG_SURFACE}"/>')
    parts.append(
        f'<text x="{ml}" y="26" font-size="15" font-weight="700" fill="{_SVG_INK}">'
        f"Remaining-useful-life: predicted vs true (leave-one-machine-out)</text>"
    )
    parts.append(
        f'<text x="{ml}" y="44" font-size="10.5" fill="{_SVG_MUTED}">'
        f"SYNTHETIC data; ILLUSTRATIVE failure level ({c.vib_rise:g} mm/s rise), "
        f"not a measured limit</text>"
    )

    # grid + ticks (shared square scale)
    for i in range(6):
        v = axis_top * i / 5.0
        yy = py(v)
        xx = px(v)
        parts.append(
            f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml + pw}" y2="{yy:.1f}" '
            f'stroke="{_SVG_GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{ml - 8}" y="{yy + 3.5:.1f}" font-size="9.5" text-anchor="end" '
            f'fill="{_SVG_MUTED}">{v:.0f}</text>'
        )
        parts.append(
            f'<line x1="{xx:.1f}" y1="{mt + ph}" x2="{xx:.1f}" y2="{mt + ph + 5}" '
            f'stroke="{_SVG_MUTED}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{xx:.1f}" y="{mt + ph + 18}" font-size="9.5" text-anchor="middle" '
            f'fill="{_SVG_MUTED}">{v:.0f}</text>'
        )

    # alpha-lambda cone (pred within +/- (alpha*true + floor)) as a shaded band
    def band(sign: float) -> str:
        pts = []
        for tv in (0.0, axis_top):
            pv = np.clip(tv + sign * (c.alpha * tv + c.alpha_floor_days), 0.0, axis_top)
            pts.append(f"{px(tv):.1f},{py(pv):.1f}")
        return " ".join(pts)

    parts.append(
        f'<polyline points="{band(1.0)} {band(-1.0).split()[1]} {band(-1.0).split()[0]}" '
        f'fill="{_SVG_ACCEL}" fill-opacity="0.06" stroke="none"/>'
    )
    # ideal y=x
    parts.append(
        f'<line x1="{px(0):.1f}" y1="{py(0):.1f}" x2="{px(axis_top):.1f}" y2="{py(axis_top):.1f}" '
        f'stroke="{_SVG_IDEAL}" stroke-width="1.4" stroke-dasharray="5 4"/>'
    )

    # scatter points, coloured by fault kind
    ft_by_machine = {mr.machine: mr.fault_type for mr in ev.machines}
    for i in range(ev.true_rul.size):
        m = int(ev.sample_machine[i])
        color = _SVG_ACCEL if ft_by_machine[m] == "accelerating_drift" else _SVG_DRIFT
        parts.append(
            f'<circle cx="{px(float(ev.true_rul[i])):.1f}" '
            f'cy="{py(float(ev.pred_rul[i])):.1f}" r="3.1" fill="{color}" '
            f'fill-opacity="0.72" stroke="{_SVG_SURFACE}" stroke-width="0.6"/>'
        )

    parts.append(
        f'<text x="{ml + pw / 2:.1f}" y="{H - 20}" font-size="11" text-anchor="middle" '
        f'fill="{_SVG_INK2}">True RUL (days to failure)</text>'
    )
    parts.append(
        f'<text transform="translate(18,{mt + ph / 2:.1f}) rotate(-90)" font-size="11" '
        f'text-anchor="middle" fill="{_SVG_INK2}">Predicted RUL (days)</text>'
    )

    # side panel
    tx = ml + pw + 22
    ty = mt + 6
    panel = [
        ("RUL accuracy (LOMO)", _SVG_INK, "700"),
        (f"MAE        {ev.mae:.2f} d", _SVG_INK2, "400"),
        (f"RMSE       {ev.rmse:.2f} d", _SVG_INK2, "400"),
        (f"naive MAE  {ev.baseline_mae:.2f} d", _SVG_MUTED, "400"),
        (f"MAE<={c.horizon_days}d    {ev.mae_within_horizon:.2f} d", _SVG_INK2, "400"),
        (f"a-acc {c.alpha:g}   {ev.alpha_accuracy:.2f}", _SVG_ACCEL, "700"),
        (f"progn.horz {ev.mean_prognostic_horizon:.1f} d", _SVG_INK2, "400"),
        ("", _SVG_INK2, "400"),
        (f"{ev.n_samples} machine-days", _SVG_MUTED, "400"),
        (f"{ev.n_machines} machines", _SVG_MUTED, "400"),
        ("", _SVG_INK2, "400"),
        ("blue = slow_drift", _SVG_DRIFT, "700"),
        ("green = accel_drift", _SVG_ACCEL, "700"),
    ]
    for i, (txt, color, weight) in enumerate(panel):
        if not txt:
            continue
        parts.append(
            f'<text x="{tx}" y="{ty + i * 17:.1f}" font-size="10.5" '
            f'font-family="Consolas, monospace" font-weight="{weight}" '
            f'fill="{color}">{esc(txt)}</text>'
        )

    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return path


def build_reference_artifacts(
    result, outdir: str | Path = "docs", config: RULConfig | None = None
) -> dict[str, Path]:
    """(Re)write the committed reference CSV + SVG for a finished pipeline run.

    Deterministic for a given pipeline result: identical bytes every run.
    """
    outdir = Path(outdir)
    ev = evaluate_rul(result, config=config)
    return {
        "csv": write_csv(ev, outdir / "rul_eval.csv"),
        "svg": write_svg(ev, outdir / "rul_eval.svg"),
    }
