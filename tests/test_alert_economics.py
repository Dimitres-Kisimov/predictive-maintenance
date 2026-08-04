from __future__ import annotations

import numpy as np

from pdm.alert_economics import (
    CostConfig,
    build_reference_artifacts,
    curve_rows,
    economics_from_result,
    recommendation_text,
    sweep_thresholds,
    write_csv,
    write_svg,
)


def test_sweep_known_values_and_cost_math():
    """Perfect-separation toy case: exact confusion counts and cost formula."""
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    econ = sweep_thresholds(y, s, cost=CostConfig(100.0, 10.0), n_periods=2)

    assert econ.n_samples == 4
    assert econ.n_positives == 2
    # a clean threshold between 0.2 and 0.8 separates perfectly -> zero cost
    assert econ.best.expected_cost == 0.0
    assert econ.best.precision == 1.0
    assert econ.best.recall == 1.0
    assert econ.best.fp == 0 and econ.best.fn == 0
    assert 0.2 <= econ.best.threshold < 0.8

    # extremes are present in the sweep
    assert any(p.alerts == 4 for p in econ.points), "alert-on-everything point missing"
    assert any(p.alerts == 0 for p in econ.points), "alert-on-nothing point missing"

    # cost formula holds at every operating point
    for p in econ.points:
        assert p.expected_cost == 100.0 * p.fn + 10.0 * p.fp
        assert p.alerts == p.tp + p.fp
        assert p.tp + p.fp + p.fn + p.tn == 4
        assert p.alerts_per_period == p.alerts / 2


def test_alert_on_nothing_is_all_missed():
    y = np.array([0, 1, 1])
    s = np.array([0.0, 1.0, 2.0])
    econ = sweep_thresholds(y, s, cost=CostConfig(1000.0, 10.0), n_periods=1)
    worst = max(econ.points, key=lambda p: p.missed_failures)
    assert worst.missed_failures == 2  # both true positives missed
    assert worst.alerts == 0
    assert worst.expected_cost == 2000.0


def test_best_never_worse_than_current(pipeline_result):
    """The swept minimum is by construction <= the detector's live threshold cost."""
    econ = economics_from_result(pipeline_result)
    assert econ.current is not None
    assert econ.n_positives > 0
    assert econ.best.expected_cost <= econ.current.expected_cost + 1e-9
    # rates stay in range and the workload axis is well defined
    for p in (econ.best, econ.current):
        assert 0.0 <= p.recall <= 1.0
        assert np.isnan(p.precision) or 0.0 <= p.precision <= 1.0
        assert p.alerts_per_period >= 0.0


def test_recommendation_text_is_honest(pipeline_result):
    econ = economics_from_result(pipeline_result)
    text = recommendation_text(econ)
    assert "SYNTHETIC" in text
    assert "ILLUSTRATIVE" in text
    assert "not a business guarantee" in text
    assert f"{econ.best.threshold:.4f}" in text


def test_cost_ratio_moves_the_optimum():
    """Higher missed-failure cost pushes the optimum to alert more (lower threshold)."""
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.3).astype(int)
    s = y * 0.6 + rng.random(400) * 0.8  # separable but noisy
    cheap_miss = sweep_thresholds(y, s, cost=CostConfig(100.0, 50.0)).best
    dear_miss = sweep_thresholds(y, s, cost=CostConfig(2000.0, 50.0)).best
    assert dear_miss.threshold <= cheap_miss.threshold
    assert dear_miss.recall >= cheap_miss.recall


def test_curve_rows_include_key_points(pipeline_result):
    econ = economics_from_result(pipeline_result)
    rows = curve_rows(econ, max_points=20)
    assert len(rows) <= len(econ.points)
    thrs = {p.threshold for p in rows}
    assert econ.best.threshold in thrs
    assert econ.current.threshold in thrs


def test_csv_and_svg_are_deterministic_and_labelled(pipeline_result, tmp_path):
    econ = economics_from_result(pipeline_result)

    csv1 = (tmp_path / "a.csv").with_suffix(".csv")
    write_csv(econ, csv1)
    body = csv1.read_text(encoding="utf-8")
    assert "SYNTHETIC" in body
    assert "ILLUSTRATIVE" in body
    assert "cost-optimal" in body

    svg1 = tmp_path / "a.svg"
    svg2 = tmp_path / "b.svg"
    write_svg(econ, svg1)
    write_svg(econ, svg2)
    b1 = svg1.read_bytes()
    b2 = svg2.read_bytes()
    assert b1 == b2, "SVG must be byte-identical across runs (no wall-clock/RNG)"
    text = b1.decode("utf-8")
    assert text.lstrip().startswith("<svg")
    assert text.rstrip().endswith("</svg>")
    assert "SYNTHETIC" in text


def test_reference_artifacts_written(pipeline_result, tmp_path):
    paths = build_reference_artifacts(pipeline_result, tmp_path)
    assert paths["csv"].exists() and paths["csv"].stat().st_size > 0
    assert paths["svg"].exists() and paths["svg"].stat().st_size > 0
