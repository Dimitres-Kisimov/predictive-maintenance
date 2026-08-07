from __future__ import annotations

import numpy as np
import pytest

from pdm.data import (
    ACCEL_VIB_PER_DAY2,
    DRIFT_VIB_PER_DAY,
    FAILURE_VIB_RISE,
    FAULT_ACCEL,
    FAULT_DRIFT,
    PROGRESSIVE_FAULTS,
    DataConfig,
    generate,
)
from pdm.rul import (
    RULConfig,
    build_reference_artifacts,
    evaluate_rul,
    fit_loglinear,
    predict_rul,
    recommendation_text,
    severity_series,
    write_csv,
    write_svg,
)


def _injected_vib_rise(fault_type: str, days_after_onset: int) -> float:
    """Fault-induced vibration rise the generator injects (mm/s), from data.py constants."""
    d = max(days_after_onset, 0)
    if fault_type == FAULT_DRIFT:
        return DRIFT_VIB_PER_DAY * d
    if fault_type == FAULT_ACCEL:
        return ACCEL_VIB_PER_DAY2 * d**2
    raise ValueError(fault_type)


def test_failure_day_is_first_crossing_of_the_injected_growth_law():
    """Ground truth ties to the exact signal: rise reaches the level at failure_day, not before."""
    data = generate()  # default seed-42 fleet
    checked = 0
    for lab in data.labels:
        if lab.fault_type not in PROGRESSIVE_FAULTS:
            continue
        fd = data.failure_day(lab.machine)
        assert fd is not None
        # the day before failure is still below the level; failure_day is at/above it
        assert _injected_vib_rise(lab.fault_type, fd - 1 - lab.onset_day) < FAILURE_VIB_RISE
        assert _injected_vib_rise(lab.fault_type, fd - lab.onset_day) >= FAILURE_VIB_RISE - 1e-9
        checked += 1
    assert checked >= 2


def test_failure_day_none_for_non_progressive_faults():
    data = generate()
    for lab in data.labels:
        if lab.fault_type not in PROGRESSIVE_FAULTS:
            assert data.failure_day(lab.machine) is None


def test_failure_day_scales_with_illustrative_level():
    """A higher illustrative failure level pushes the failure day later (monotone)."""
    data = generate()
    prog = [lab.machine for lab in data.labels if lab.fault_type in PROGRESSIVE_FAULTS]
    m = prog[0]
    assert data.failure_day(m, vib_rise=0.30) <= data.failure_day(m, vib_rise=0.60)


def test_fit_loglinear_recovers_a_log_linear_relationship():
    sev = np.linspace(0.0, 1.0, 40)  # kept where log1p(RUL) stays positive
    rul = np.expm1(2.2 - 1.8 * sev)  # exactly log-linear, RUL > 0 throughout
    a, b = fit_loglinear(sev, rul)
    assert a == pytest.approx(2.2, abs=1e-6)
    assert b == pytest.approx(-1.8, abs=1e-6)
    pred = predict_rul((a, b), sev)
    assert np.allclose(pred, rul, atol=1e-6)


def test_predict_rul_is_nonnegative_and_decreasing_in_severity():
    coef = (2.3, -2.4)  # RUL falls as severity rises
    sev = np.linspace(0.0, 3.0, 25)
    pred = predict_rul(coef, sev)
    assert np.all(pred >= 0.0)
    assert np.all(np.diff(pred) <= 1e-9)


def test_severity_is_zero_below_threshold_and_positive_above():
    scores = np.array([0.1, 0.1, 0.1, 0.5, 1.0, 2.0])
    sev = severity_series(scores, threshold=0.2, alpha=0.5)
    assert sev[0] == pytest.approx(0.0)  # smoothed score starts below threshold
    assert sev[-1] > 0.0
    assert np.all(sev >= 0.0)


def test_evaluate_basic_properties(pipeline_result):
    ev = evaluate_rul(pipeline_result)
    assert ev.n_machines >= 2
    assert ev.n_samples == ev.true_rul.size == ev.pred_rul.size > 0
    assert np.isfinite(ev.mae) and ev.mae >= 0.0
    assert ev.rmse >= ev.mae - 1e-9  # RMSE is never below MAE
    assert np.all(ev.true_rul >= 0.0)
    assert np.all(ev.pred_rul >= 0.0)
    assert 0.0 <= ev.alpha_accuracy <= 1.0
    # every scored machine is a progressive fault whose failure is observed in-window
    n_days = pipeline_result.data.n_days
    for mr in ev.machines:
        assert mr.fault_type in PROGRESSIVE_FAULTS
        assert mr.failure_day < n_days
        assert mr.onset_day <= mr.failure_day


def test_model_beats_the_naive_mean_baseline(pipeline_result):
    """The severity-driven model must add value over predicting the mean RUL."""
    ev = evaluate_rul(pipeline_result)
    assert ev.mae <= ev.baseline_mae


def test_evaluation_is_deterministic(pipeline_result):
    a = evaluate_rul(pipeline_result)
    b = evaluate_rul(pipeline_result)
    assert np.array_equal(a.pred_rul, b.pred_rul)
    assert np.array_equal(a.true_rul, b.true_rul)
    assert a.coef == b.coef


def test_all_failures_censored_raises(pipeline_result):
    """With an unreachable failure level every machine is right-censored -> no CV possible."""
    with pytest.raises(ValueError, match="observed in-window failure"):
        evaluate_rul(pipeline_result, RULConfig(vib_rise=100.0))


def test_recommendation_text_is_honest(pipeline_result):
    text = recommendation_text(evaluate_rul(pipeline_result))
    assert "SYNTHETIC" in text
    assert "ILLUSTRATIVE" in text
    assert "leave-one-machine-out" in text


def test_csv_and_svg_are_deterministic_and_labelled(pipeline_result, tmp_path):
    ev = evaluate_rul(pipeline_result)

    csv_path = tmp_path / "rul.csv"
    write_csv(ev, csv_path)
    body = csv_path.read_text(encoding="utf-8")
    assert "SYNTHETIC" in body
    assert "ILLUSTRATIVE" in body
    assert "true_rul_days" in body

    svg1, svg2 = tmp_path / "a.svg", tmp_path / "b.svg"
    write_svg(ev, svg1)
    write_svg(ev, svg2)
    b1 = svg1.read_bytes()
    assert b1 == svg2.read_bytes(), "SVG must be byte-identical across runs"
    text = b1.decode("utf-8")
    assert text.lstrip().startswith("<svg")
    assert text.rstrip().endswith("</svg>")
    assert "SYNTHETIC" in text


def test_reference_artifacts_written(pipeline_result, tmp_path):
    paths = build_reference_artifacts(pipeline_result, tmp_path)
    assert paths["csv"].exists() and paths["csv"].stat().st_size > 0
    assert paths["svg"].exists() and paths["svg"].stat().st_size > 0


def test_small_and_default_configs_both_have_observed_failures():
    """Both the test and the default fleets yield >= 2 scoreable progressive machines."""
    for cfg in (DataConfig(), DataConfig(n_machines=10, n_days=40, samples_per_day=12, seed=7,
                                         n_drift=2, n_spike=1, n_corr_break=1, n_accel=1)):
        data = generate(cfg)
        observed = [
            lab.machine
            for lab in data.labels
            if lab.fault_type in PROGRESSIVE_FAULTS
            and (fd := data.failure_day(lab.machine)) is not None
            and fd < data.n_days
        ]
        assert len(observed) >= 2
