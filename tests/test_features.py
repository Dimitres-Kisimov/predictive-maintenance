from __future__ import annotations

import dataclasses

import numpy as np

from pdm.data import PlantData
from pdm.features import build_features, feature_names


def test_feature_shapes_and_finite(small_data, small_features):
    cfg = small_data.config
    fs = small_features
    n_fdays = cfg.n_days - 5 + 1  # default window_days = 5
    assert fs.X.shape == (cfg.n_machines, n_fdays, len(feature_names()))
    assert np.isfinite(fs.X).all()
    assert fs.days[0] == 4
    assert fs.days[-1] == cfg.n_days - 1


def test_features_leak_free(small_data, small_features):
    """Features at day t must not change when all data after day t is removed."""
    cfg = small_data.config
    t_cut = 20  # keep days 0..19
    truncated = PlantData(
        readings=small_data.readings[:, : t_cut * cfg.samples_per_day, :],
        labels=small_data.labels,
        config=dataclasses.replace(cfg, n_days=t_cut),
    )
    fs_trunc = build_features(truncated)
    fs_full = small_features
    n_common = fs_trunc.X.shape[1]
    assert np.array_equal(fs_trunc.days, fs_full.days[:n_common])
    assert np.array_equal(fs_full.X[:, :n_common, :], fs_trunc.X)
