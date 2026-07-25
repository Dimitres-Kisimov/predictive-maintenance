"""Shared fixtures: a small, fast configuration reused across the suite."""

from __future__ import annotations

import pytest

from pdm.data import DataConfig, generate
from pdm.detect import torch_available
from pdm.features import build_features
from pdm.pipeline import run_pipeline

SMALL = DataConfig(
    n_machines=10,
    n_days=40,
    samples_per_day=12,
    seed=7,
    n_drift=2,
    n_spike=1,
    n_corr_break=1,
    n_accel=1,
)


@pytest.fixture(scope="session")
def small_config():
    return SMALL


@pytest.fixture(scope="session")
def small_data():
    return generate(SMALL)


@pytest.fixture(scope="session")
def small_features(small_data):
    return build_features(small_data)


@pytest.fixture(scope="session")
def pipeline_result():
    """One shared pipeline run on the small config (AE only if torch imports)."""
    return run_pipeline(
        config=SMALL,
        include_ae=torch_available(),
        ae_epochs=150,
        n_jobs=6,
        horizon_days=4,
    )
