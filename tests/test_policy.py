from __future__ import annotations

import math

import numpy as np
import pytest

from pdm.data import PROGRESSIVE_FAULTS
from pdm.policy import (
    POLICY_AGE,
    POLICY_CBM,
    POLICY_RTF,
    CBMPerformance,
    PolicyCostConfig,
    WeibullFit,
    age_replacement_cost_rate,
    build_reference_artifacts,
    cbm_cost_rate,
    compare_policies_from_result,
    fit_weibull,
    lifetimes_from_data,
    measure_cbm_performance,
    optimize_age_replacement,
    recommendation_text,
    truncated_mean_life,
    weibull_reliability,
    write_csv,
    write_svg,
)

# A small hand-set censored sample reused across the Weibull tests:
# 5 observed failures, 5 units suspended (still running) at day 60.
FAILURES = np.array([12.0, 25.0, 31.0, 44.0, 58.0])
SUSPENSIONS = np.array([60.0] * 5)


def _censored_loglik(beta: float, eta: float, fail: np.ndarray, susp: np.ndarray) -> float:
    """Independent right-censored Weibull log-likelihood, straight from the definition."""
    z_f = (fail / eta) ** beta
    z_s = (susp / eta) ** beta
    return float(
        np.sum(np.log(beta) - beta * np.log(eta) + (beta - 1.0) * np.log(fail) - z_f)
        - np.sum(z_s)
    )


# ---------------------------------------------------------------- Weibull MLE


def test_weibull_mle_satisfies_its_estimating_equations():
    """The fit must satisfy the exact MLE identities (hand-derivable from the likelihood)."""
    fit = fit_weibull(FAILURES, SUSPENSIONS)
    t_all = np.concatenate([FAILURES, SUSPENSIONS])
    r = FAILURES.size
    # scale identity: eta^beta = sum(t_i^beta) / r  (sum over ALL units)
    assert fit.scale**fit.shape == pytest.approx(np.sum(t_all**fit.shape) / r, rel=1e-9)
    # shape estimating equation: residual at beta-hat is ~ 0
    w = t_all**fit.shape
    resid = np.sum(w * np.log(t_all)) / np.sum(w) - 1.0 / fit.shape - np.mean(np.log(FAILURES))
    assert resid == pytest.approx(0.0, abs=1e-9)


def test_weibull_mle_maximizes_the_censored_likelihood():
    """Independent check: no (shape, scale) on a grid around the fit beats its likelihood."""
    fit = fit_weibull(FAILURES, SUSPENSIONS)
    ll_fit = _censored_loglik(fit.shape, fit.scale, FAILURES, SUSPENSIONS)
    for beta in fit.shape * np.linspace(0.7, 1.3, 61):
        for eta in fit.scale * np.linspace(0.85, 1.15, 61):
            assert ll_fit >= _censored_loglik(beta, eta, FAILURES, SUSPENSIONS) - 1e-9


def test_weibull_mle_recovers_known_parameters_on_a_complete_sample():
    rng = np.random.default_rng(123)
    true_shape, true_scale = 2.0, 30.0
    lives = true_scale * rng.weibull(true_shape, size=4000)
    fit = fit_weibull(lives, np.array([]))
    assert fit.shape == pytest.approx(true_shape, rel=0.05)
    assert fit.scale == pytest.approx(true_scale, rel=0.03)
    assert fit.n_failures == 4000
    assert fit.n_suspensions == 0


def test_weibull_fit_input_validation():
    with pytest.raises(ValueError, match="at least 2 observed failures"):
        fit_weibull(np.array([10.0]), SUSPENSIONS)
    with pytest.raises(ValueError, match="positive"):
        fit_weibull(np.array([10.0, 0.0]), SUSPENSIONS)
    with pytest.raises(ValueError, match="degenerate"):
        fit_weibull(np.array([30.0, 30.0, 30.0]), np.array([]))  # identical lives -> no MLE


def test_b_lives_and_mtbf_match_their_definitions():
    """Cross-check the closed forms against the reliability function and its integral."""
    fit = fit_weibull(FAILURES, SUSPENSIONS)
    assert weibull_reliability(fit, fit.b10) == pytest.approx(0.90, abs=1e-12)
    assert weibull_reliability(fit, fit.b50) == pytest.approx(0.50, abs=1e-12)
    assert fit.b10 < fit.b50 < fit.mtbf * 2.0
    # integral of R over (0, inf) IS the MTBF; the truncated form must converge to it
    assert truncated_mean_life(fit, 50.0 * fit.scale) == pytest.approx(fit.mtbf, rel=1e-12)


def test_truncated_mean_life_matches_the_erf_closed_form():
    """Hand-check for shape=2: integral_0^T exp(-(t/eta)^2) dt = (sqrt(pi)/2)*eta*erf(T/eta)."""
    fit = WeibullFit.from_params(2.0, 10.0)
    expected = (math.sqrt(math.pi) / 2.0) * 10.0 * math.erf(7.0 / 10.0)
    assert truncated_mean_life(fit, 7.0) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------- age replacement


def test_age_replacement_cost_rate_hand_computed():
    """g(T) for shape=2 checked against a fully hand-derived closed form."""
    fit = WeibullFit.from_params(2.0, 30.0)
    cost = PolicyCostConfig(cost_unplanned_failure=1000.0, cost_planned_replacement=250.0)
    rel = math.exp(-0.25)  # R(15) = exp(-(15/30)^2)
    denom = (math.sqrt(math.pi) / 2.0) * 30.0 * math.erf(0.5)
    expected = (250.0 * rel + 1000.0 * (1.0 - rel)) / denom
    assert age_replacement_cost_rate(fit, cost, 15.0) == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ValueError, match="positive"):
        age_replacement_cost_rate(fit, cost, 0.0)


def test_memoryless_lifetimes_collapse_to_run_to_failure():
    """Textbook base case: shape=1 (exponential) -> NO preventive age helps."""
    fit = WeibullFit.from_params(1.0, 30.0)  # MTBF = 30
    cost = PolicyCostConfig(cost_unplanned_failure=1000.0, cost_planned_replacement=250.0)
    pol = optimize_age_replacement(fit, cost)
    assert not pol.finite_optimum
    assert pol.optimal_age_days is None
    assert pol.cost_rate == pytest.approx(1000.0 / 30.0, rel=1e-12)  # == run-to-failure
    assert pol.cost_rate == pol.rtf_cost_rate


def test_equal_costs_collapse_to_run_to_failure():
    """If a planned replacement costs as much as a failure, replacing early cannot pay."""
    fit = WeibullFit.from_params(3.0, 30.0)  # strong wear-out, but no cost asymmetry
    cost = PolicyCostConfig(cost_unplanned_failure=500.0, cost_planned_replacement=500.0)
    pol = optimize_age_replacement(fit, cost)
    assert not pol.finite_optimum
    assert pol.cost_rate == pol.rtf_cost_rate


def test_wearout_lifetimes_have_a_finite_cheaper_optimum():
    fit = WeibullFit.from_params(3.0, 30.0)
    cost = PolicyCostConfig(cost_unplanned_failure=1000.0, cost_planned_replacement=250.0)
    pol = optimize_age_replacement(fit, cost)
    assert pol.finite_optimum
    assert 0.0 < pol.optimal_age_days < fit.mtbf  # replace before the mean life
    assert pol.cost_rate < pol.rtf_cost_rate
    # optimality: T* is at least as cheap as every point of a dense independent grid
    dense = np.linspace(0.5, 3.0 * fit.scale, 4001)
    assert pol.cost_rate <= float(np.min(age_replacement_cost_rate(fit, cost, dense))) + 1e-9


def test_optimal_age_moves_earlier_when_failures_get_costlier():
    fit = WeibullFit.from_params(3.0, 30.0)
    cheap = optimize_age_replacement(fit, PolicyCostConfig(cost_unplanned_failure=500.0))
    dear = optimize_age_replacement(fit, PolicyCostConfig(cost_unplanned_failure=2000.0))
    assert cheap.finite_optimum and dear.finite_optimum
    assert dear.optimal_age_days < cheap.optimal_age_days


# ---------------------------------------------------------------- condition-based


def _perf(p_num: int, p_den: int, fa: int, neg: int, lead: int = 3) -> CBMPerformance:
    return CBMPerformance(
        lead_days=lead,
        n_failures=p_den,
        n_detected_in_time=p_num,
        fraction_in_time=p_num / p_den,
        false_alarms=fa,
        n_negative_days=neg,
        false_alarms_per_machine_day=fa / neg,
    )


def test_cbm_cost_rate_hand_computed():
    """(2/3 * 250 + 1/3 * 1000) / 30 + 0.05 * 50 = 500/30 + 2.5."""
    cost = PolicyCostConfig()
    rate = cbm_cost_rate(_perf(2, 3, 5, 100), mtbf=30.0, cost=cost)
    assert rate == pytest.approx(500.0 / 30.0 + 2.5, rel=1e-12)


def test_cbm_edge_cases_bracket_the_policy():
    cost = PolicyCostConfig()
    # perfect detector, no false alarms: every failure becomes a planned repair
    assert cbm_cost_rate(_perf(4, 4, 0, 100), 30.0, cost) == pytest.approx(250.0 / 30.0)
    # useless detector: run-to-failure PLUS the false-alarm inspection load
    assert cbm_cost_rate(_perf(0, 4, 10, 100), 30.0, cost) == pytest.approx(
        1000.0 / 30.0 + 0.1 * 50.0
    )
    with pytest.raises(ValueError, match="mtbf"):
        cbm_cost_rate(_perf(1, 2, 0, 10), 0.0, cost)


def test_measure_cbm_performance_counts_are_consistent(pipeline_result):
    perf = measure_cbm_performance(pipeline_result, lead_days=3)
    assert 0 <= perf.n_detected_in_time <= perf.n_failures
    assert 0.0 <= perf.fraction_in_time <= 1.0
    assert 0 <= perf.false_alarms <= perf.n_negative_days
    # a stricter lead-time requirement can never detect MORE failures in time
    stricter = measure_cbm_performance(pipeline_result, lead_days=6)
    assert stricter.n_detected_in_time <= perf.n_detected_in_time
    with pytest.raises(ValueError, match="lead_days"):
        measure_cbm_performance(pipeline_result, lead_days=-1)


# ---------------------------------------------------------------- fleet integration


def test_lifetimes_from_data_match_the_ground_truth(small_data):
    failures, suspensions = lifetimes_from_data(small_data)
    expected = sorted(
        float(fd)
        for lab in small_data.labels
        if lab.fault_type in PROGRESSIVE_FAULTS
        and (fd := small_data.failure_day(lab.machine)) is not None
        and fd < small_data.n_days
    )
    assert failures.tolist() == expected
    assert failures.size + suspensions.size == small_data.config.n_machines
    assert np.all(suspensions == float(small_data.n_days))  # censored at the window end


def test_compare_policies_from_result_basic(pipeline_result):
    cmp = compare_policies_from_result(pipeline_result)
    assert cmp.recommended in (POLICY_RTF, POLICY_AGE, POLICY_CBM)
    rates = cmp.rates()
    assert set(rates) == {POLICY_RTF, POLICY_AGE, POLICY_CBM}
    assert all(np.isfinite(v) and v > 0.0 for v in rates.values())
    # the recommendation is exactly the cheapest policy
    assert rates[cmp.recommended] == min(rates.values())
    assert cmp.fit.n_failures >= 2
    assert cmp.fit.n_failures + cmp.fit.n_suspensions == pipeline_result.data.config.n_machines
    # the age policy can never beat its own asymptote's better side dishonestly:
    # its rate is at most the run-to-failure rate by construction
    assert cmp.age.cost_rate <= cmp.rtf_cost_rate + 1e-12


def test_comparison_is_deterministic(pipeline_result):
    a = compare_policies_from_result(pipeline_result)
    b = compare_policies_from_result(pipeline_result)
    assert a.recommended == b.recommended
    assert (a.fit.shape, a.fit.scale, a.fit.mtbf) == (b.fit.shape, b.fit.scale, b.fit.mtbf)
    assert a.rates() == b.rates()
    assert np.array_equal(a.age.curve_cost_rate, b.age.curve_cost_rate)


def test_recommendation_text_is_honest(pipeline_result):
    text = recommendation_text(compare_policies_from_result(pipeline_result))
    assert "SYNTHETIC" in text
    assert "ILLUSTRATIVE" in text
    assert "modelled, not measured" in text
    for name in (POLICY_RTF, POLICY_AGE, POLICY_CBM):
        assert name in text


def test_csv_and_svg_are_deterministic_and_labelled(pipeline_result, tmp_path):
    cmp = compare_policies_from_result(pipeline_result)

    csv_path = tmp_path / "policy.csv"
    write_csv(cmp, csv_path)
    body = csv_path.read_text(encoding="utf-8")
    assert "SYNTHETIC" in body
    assert "ILLUSTRATIVE" in body
    assert "cost_rate_per_machine_day" in body
    assert "recommended" in body

    svg1, svg2 = tmp_path / "a.svg", tmp_path / "b.svg"
    write_svg(cmp, svg1)
    write_svg(cmp, svg2)
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
