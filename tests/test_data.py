from __future__ import annotations

import dataclasses

import numpy as np

from pdm.data import FAULT_HEALTHY, FAULT_KINDS, SENSORS, generate


def test_generator_deterministic(small_config):
    a = generate(small_config)
    b = generate(small_config)
    assert np.array_equal(a.readings, b.readings)
    assert a.labels == b.labels


def test_generator_seed_changes_output(small_config):
    a = generate(small_config)
    b = generate(dataclasses.replace(small_config, seed=small_config.seed + 1))
    assert not np.array_equal(a.readings, b.readings)


def test_shapes_and_labels(small_data):
    cfg = small_data.config
    assert small_data.readings.shape == (
        cfg.n_machines,
        cfg.n_days * cfg.samples_per_day,
        len(SENSORS),
    )
    assert np.isfinite(small_data.readings).all()
    faulty = small_data.faulty_machines()
    assert len(faulty) == cfg.n_faulty
    lo = int(cfg.n_days * cfg.onset_min_frac)
    hi = int(cfg.n_days * cfg.onset_max_frac)
    for lab in small_data.labels:
        if lab.fault_type == FAULT_HEALTHY:
            assert lab.onset_day == -1
        else:
            assert lab.fault_type in FAULT_KINDS
            assert lo <= lab.onset_day <= hi
    y = small_data.day_labels()
    assert y.shape == (cfg.n_machines, cfg.n_days)
    for lab in small_data.labels:
        if lab.fault_type != FAULT_HEALTHY:
            assert y[lab.machine, lab.onset_day] == 1
            assert y[lab.machine, lab.onset_day - 1] == 0
        else:
            assert y[lab.machine].sum() == 0
