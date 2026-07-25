from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pdm.detect import AEDetector  # noqa: E402


def _toy_data():
    rng = np.random.default_rng(3)
    latent = rng.normal(size=(300, 3))
    return latent @ rng.normal(size=(3, 8)) + rng.normal(scale=0.05, size=(300, 8))


def test_ae_scores_finite_and_metrics_valid(pipeline_result):
    if "autoencoder" not in pipeline_result.reports:
        pytest.skip("torch unavailable in pipeline fixture")
    rep = pipeline_result.reports["autoencoder"]
    assert 0.0 <= rep.roc_auc <= 1.0
    assert 0.0 <= rep.pr_auc <= 1.0
    assert np.isfinite(rep.threshold)
    assert np.isfinite(pipeline_result.margin)


def test_ae_deterministic_for_fixed_seed():
    X = _toy_data()
    s1 = AEDetector(epochs=60, seed=11).fit(X).score(X)
    s2 = AEDetector(epochs=60, seed=11).fit(X).score(X)
    assert np.allclose(s1, s2)
    assert np.isfinite(s1).all()
