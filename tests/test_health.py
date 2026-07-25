from __future__ import annotations

import numpy as np

from pdm.health import health_series


def test_health_monotone_under_monotone_degradation():
    """A score series that only rises can never gain health."""
    scores = np.linspace(0.5, 8.0, 40)  # monotone increasing anomaly score
    h = health_series(scores, threshold=1.0)
    assert np.all(np.diff(h) <= 1e-12)
    assert h[0] > h[-1]


def test_health_range_and_ranking(pipeline_result):
    hr = pipeline_result.health
    assert np.all(hr.health >= 0.0)
    assert np.all(hr.health <= 100.0)
    n_machines = pipeline_result.data.config.n_machines
    assert sorted(hr.ranking) == list(range(n_machines))
    # on the test seed, the most urgent machine is truly faulted
    faulty = set(pipeline_result.data.faulty_machines())
    assert hr.ranking[0] in faulty
