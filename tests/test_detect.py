from __future__ import annotations

import numpy as np

from pdm.data import SENSORS
from pdm.detect import (
    PCADetector,
    pr_auc,
    precision_at_k,
    roc_auc,
    threshold_at_fpr,
    top_sensors,
)


def test_metric_sanity_known_values():
    y = np.array([0, 0, 0, 1, 1])
    s_perfect = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert roc_auc(y, s_perfect) == 1.0
    assert pr_auc(y, s_perfect) == 1.0
    assert precision_at_k(y, s_perfect, 2) == 1.0
    s_worst = s_perfect[::-1]
    assert roc_auc(y, s_worst) == 0.0
    # thresholds: 10% budget on 100 clean scores -> ~10 exceed
    clean = np.linspace(0.0, 1.0, 100)
    thr = threshold_at_fpr(clean, 0.10)
    assert 0.85 <= thr <= 0.95


def test_pca_scores_finite_and_metrics_valid(pipeline_result):
    rep = pipeline_result.reports["pca"]
    assert np.isfinite(pipeline_result.chosen_scores).all()
    assert 0.0 <= rep.roc_auc <= 1.0
    assert 0.0 <= rep.pr_auc <= 1.0
    assert 0.0 <= rep.precision_at_k <= 1.0
    assert rep.threshold > 0.0


def test_threshold_respects_fpr_budget(pipeline_result):
    """Measured FPR on the clean validation split stays near the stated budget."""
    rep = pipeline_result.reports["pca"]
    assert rep.val_fpr <= rep.fpr_budget + 0.02


def test_injected_faults_detected_on_test_seed(pipeline_result):
    """At the stated FPR budget, PCA must alert on >= 80% of faulted machines."""
    rep = pipeline_result.reports["pca"]
    assert rep.n_faulty == 5
    assert rep.n_detected >= 4


def test_explanations_name_real_sensors(pipeline_result, small_features):
    assert pipeline_result.alerts, "expected at least one alert on the test seed"
    for alert in pipeline_result.alerts[:20]:
        assert len(alert.sensors) == 2
        assert set(alert.sensors) <= set(SENSORS)
    err = np.zeros(len(small_features.names))
    err[small_features.names.index("vibration_std")] = 5.0
    err[small_features.names.index("temperature_mean")] = 3.0
    assert top_sensors(err, small_features.names) == ["vibration", "temperature"]


def test_pca_reconstruction_identity_on_train_subspace():
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(200, 2))
    X = latent @ rng.normal(size=(2, 6)) + rng.normal(scale=1e-3, size=(200, 6))
    det = PCADetector(var_target=0.99).fit(X)
    scores = det.score(X)
    assert np.isfinite(scores).all()
    assert scores.mean() < 1e-3  # near-perfect reconstruction of its own subspace
