"""Maintenance-policy economics: Weibull lifetimes + the age-replacement optimization.

The classic reliability-engineering decision this repo had not yet answered:
given how these machines fail, WHICH maintenance policy is cheapest to run?

Three named policies are priced on the same fleet, in cost per machine-day:

- run-to-failure:   fix machines only when they break. Cost rate = Cf / MTBF.
- age-replacement:  preventively replace at a fixed age T (the Barlow-Proschan
                    age-replacement policy). Cost rate
                    g(T) = [Cp*R(T) + Cf*F(T)] / integral_0^T R(t) dt,
                    minimized over T. For wear-out lifetimes (Weibull shape > 1)
                    and Cp < Cf a finite optimal age T* exists; for memoryless
                    lifetimes (shape = 1) no preventive age beats run-to-failure
                    - the module reproduces that textbook result exactly.
- condition-based:  repair when the repo's OWN anomaly detector raises an alert
                    early enough to plan the work. Priced from the detector's
                    MEASURED performance on this fleet (fraction of failures
                    alerted with enough lead time, false-alarm rate), not from
                    an assumed detection probability.

Lifetime model
--------------
Machine lives come from the generator's ground-truth functional-failure day
(`PlantData.failure_day` - the same single source of truth the RUL layer uses).
Machines whose failure is observed in-window are failures; every other machine
is a right-censored SUSPENSION at the window end - healthy machines, progressive
machines whose failure lands beyond the window, and the spike / correlation-break
machines (their fault has no monotone magnitude, so they never reach the
vibration-rise functional-failure definition; for THIS failure mode they are
suspensions). A two-parameter Weibull is fit by maximum likelihood WITH the
censoring (the standard field-data "failures + suspensions" analysis), and the
fit reports the recognizable reliability numbers: shape beta, scale eta, MTBF,
and the B10 / B50 lives.

Honest framing
--------------
- All data is SYNTHETIC (pdm/data.py); the failure level behind `failure_day`
  is an ILLUSTRATIVE stand-in, not a measured engineering limit.
- The cost rates are ILLUSTRATIVE planning assumptions chosen to show the SHAPE
  of the policy trade-off - they are not currency and not a business guarantee.
  The ranking of policies moves with the cost ratios.
- The Weibull fit extrapolates from ONE heavily right-censored observation
  window (few failures, many suspensions), so MTBF and the B-lives are
  modelled, not measured.
- The condition-based cycle cost divides by the mean life (detection happens
  close to failure, so the shortened cycle is a second-order effect we ignore),
  and it assumes an alert with >= `lead_days` of lead time always converts an
  unplanned failure into a planned repair. Both are stated simplifications.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import gammainc  # regularized lower incomplete gamma P(s, x)

from pdm.data import PROGRESSIVE_FAULTS, PlantData
from pdm.detect import detection_delays

POLICY_RTF = "run-to-failure"
POLICY_AGE = "age-replacement"
POLICY_CBM = "condition-based"

# Weibull-shape search bracket for the MLE bisection. Shapes outside this range
# mean degenerate lifetime data (e.g. all failure times identical) - we raise
# rather than report a meaningless fit.
_BETA_LO = 0.05
_BETA_HI = 200.0


@dataclass(frozen=True)
class PolicyCostConfig:
    """ILLUSTRATIVE synthetic cost rates - NOT a business guarantee.

    Deliberately consistent with pdm/alert_economics.py: an unplanned failure
    books the same 1000 units a missed failure does there, and an inspection
    triggered by a false alarm books the same 50 units a false alarm does. The
    planned-replacement cost encodes the usual asymmetry that planned work
    (parts ordered, crew scheduled, line drained on purpose) is far cheaper
    than an emergency repair. Only the RATIOS drive the policy ranking.
    """

    cost_unplanned_failure: float = 1000.0  # emergency repair + idle line (per failure)
    cost_planned_replacement: float = 250.0  # planned preventive repair (per replacement)
    cost_inspection: float = 50.0  # chasing one false alarm (per inspection)
    unit: str = "illustrative cost units"


# ---------------------------------------------------------------- Weibull fit


@dataclass(frozen=True)
class WeibullFit:
    """Two-parameter Weibull lifetime fit (right-censored MLE) with B-lives."""

    shape: float  # beta: <1 infant mortality, =1 memoryless, >1 wear-out
    scale: float  # eta: characteristic life (63.2% failed), in days
    n_failures: int
    n_suspensions: int  # right-censored units (no failure observed in-window)
    mtbf: float  # eta * Gamma(1 + 1/beta), days - a MODEL extrapolation
    b10: float  # age by which 10% of units are expected to fail, days
    b50: float  # median life, days

    @classmethod
    def from_params(cls, shape: float, scale: float) -> WeibullFit:
        """A fit object from known parameters (for what-if analysis and tests)."""
        if shape <= 0.0 or scale <= 0.0:
            raise ValueError("Weibull shape and scale must be positive")
        return cls(
            shape=float(shape),
            scale=float(scale),
            n_failures=0,
            n_suspensions=0,
            mtbf=float(scale * math.gamma(1.0 + 1.0 / shape)),
            b10=float(scale * (-math.log(0.9)) ** (1.0 / shape)),
            b50=float(scale * math.log(2.0) ** (1.0 / shape)),
        )


def weibull_reliability(fit: WeibullFit, t: np.ndarray | float) -> np.ndarray | float:
    """Survival function R(t) = exp(-(t/eta)^beta)."""
    t = np.asarray(t, dtype=float)
    out = np.exp(-((np.clip(t, 0.0, None) / fit.scale) ** fit.shape))
    return float(out) if out.ndim == 0 else out


def _shape_equation(
    beta: float, t_all: np.ndarray, log_t_all: np.ndarray, mean_log_t_fail: float
) -> float:
    """The Weibull-MLE estimating equation in beta (monotone increasing, one root).

    Scale-invariant: `t_all` / `log_t_all` may be pre-scaled by any constant as
    long as `mean_log_t_fail` uses the same scaling.
    """
    w = t_all**beta
    return float(np.sum(w * log_t_all) / np.sum(w) - 1.0 / beta - mean_log_t_fail)


def fit_weibull(failure_times: np.ndarray, suspension_times: np.ndarray) -> WeibullFit:
    """Right-censored two-parameter Weibull MLE (profile likelihood + bisection).

    `failure_times` are observed lives; `suspension_times` are right-censoring
    ages of units that had NOT failed when observation stopped. Deterministic:
    fixed-iteration bisection, no RNG. Requires >= 2 failures (a lifetime
    distribution cannot be inferred from fewer) and strictly positive times.
    """
    fail = np.asarray(failure_times, dtype=float)
    susp = np.asarray(suspension_times, dtype=float)
    if fail.size < 2:
        raise ValueError("need at least 2 observed failures to fit a Weibull lifetime model")
    t_all = np.concatenate([fail, susp])
    if np.any(t_all <= 0.0):
        raise ValueError("all lifetimes and suspension ages must be positive")

    # Scale by the largest observation so t^beta never overflows (the equation
    # is scale-invariant; eta is un-scaled afterwards).
    t_max = float(t_all.max())
    ts = t_all / t_max
    log_ts = np.log(ts)
    mean_log_fail = float(np.mean(np.log(fail / t_max)))

    lo, hi = _BETA_LO, _BETA_HI
    f_lo = _shape_equation(lo, ts, log_ts, mean_log_fail)
    f_hi = _shape_equation(hi, ts, log_ts, mean_log_fail)
    if not (f_lo < 0.0 < f_hi):
        raise ValueError(
            "degenerate lifetime data: the Weibull shape MLE lies outside "
            f"[{_BETA_LO}, {_BETA_HI}] (e.g. all failure times identical)"
        )
    for _ in range(100):  # fixed iterations -> deterministic to float precision
        mid = 0.5 * (lo + hi)
        if _shape_equation(mid, ts, log_ts, mean_log_fail) < 0.0:
            lo = mid
        else:
            hi = mid
    beta = 0.5 * (lo + hi)

    r = fail.size
    eta = t_max * float(np.sum(ts**beta) / r) ** (1.0 / beta)
    base = WeibullFit.from_params(beta, eta)
    return WeibullFit(
        shape=base.shape,
        scale=base.scale,
        n_failures=int(r),
        n_suspensions=int(susp.size),
        mtbf=base.mtbf,
        b10=base.b10,
        b50=base.b50,
    )


def lifetimes_from_data(data: PlantData) -> tuple[np.ndarray, np.ndarray]:
    """(failures, suspensions) in days, from the generator's ground truth.

    A machine is a FAILURE if its functional-failure day (`PlantData.failure_day`,
    the illustrative vibration-rise definition) is observed inside the window;
    every other machine is a SUSPENSION right-censored at the window end -
    healthy machines, progressives failing beyond the window, and spike /
    correlation-break machines (whose fault never reaches this failure mode's
    definition). Sorted ascending for determinism.
    """
    failures: list[float] = []
    suspensions: list[float] = []
    for lab in data.labels:
        fd = data.failure_day(lab.machine)
        if fd is not None and fd < data.n_days:
            failures.append(float(fd))
        else:
            suspensions.append(float(data.n_days))
    return np.sort(np.asarray(failures)), np.sort(np.asarray(suspensions))


# ---------------------------------------------------------------- age replacement


def truncated_mean_life(fit: WeibullFit, t: np.ndarray | float) -> np.ndarray | float:
    """integral_0^t R(u) du in closed form: MTBF * P(1/beta, (t/eta)^beta).

    P is the regularized lower incomplete gamma function; as t -> inf this
    approaches the MTBF exactly.
    """
    t = np.asarray(t, dtype=float)
    x = (np.clip(t, 0.0, None) / fit.scale) ** fit.shape
    out = fit.mtbf * gammainc(1.0 / fit.shape, x)
    return float(out) if out.ndim == 0 else out


def age_replacement_cost_rate(
    fit: WeibullFit, cost: PolicyCostConfig, t: np.ndarray | float
) -> np.ndarray | float:
    """Long-run cost per machine-day of 'replace preventively at age t'.

    The classic renewal-reward rate g(t) = [Cp*R(t) + Cf*F(t)] / integral_0^t R.
    As t -> inf this approaches the run-to-failure rate Cf / MTBF.
    """
    t_arr = np.asarray(t, dtype=float)
    if np.any(t_arr <= 0.0):
        raise ValueError("replacement age must be positive")
    rel = weibull_reliability(fit, t_arr)
    denom = truncated_mean_life(fit, t_arr)
    out = (cost.cost_planned_replacement * rel + cost.cost_unplanned_failure * (1.0 - rel)) / denom
    return float(out) if np.ndim(out) == 0 else out


def _golden_section(f, lo: float, hi: float, iters: int = 90) -> tuple[float, float]:
    """Deterministic golden-section minimization of a unimodal f on [lo, hi]."""
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)
    x = 0.5 * (a + b)
    return x, f(x)


@dataclass(frozen=True)
class AgeReplacementPolicy:
    """The optimized age-replacement policy, plus its full cost-rate curve."""

    finite_optimum: bool  # False -> no preventive age beats run-to-failure
    optimal_age_days: float | None  # T*, None when finite_optimum is False
    cost_rate: float  # g(T*), or the run-to-failure rate when no finite T* exists
    rtf_cost_rate: float  # Cf / MTBF, the g(T -> inf) asymptote
    curve_age_days: np.ndarray
    curve_cost_rate: np.ndarray


def optimize_age_replacement(
    fit: WeibullFit, cost: PolicyCostConfig, n_grid: int = 2048
) -> AgeReplacementPolicy:
    """Minimize g(T) over T: grid scan + golden-section refine. Deterministic.

    The grid spans up to the 99.99th lifetime percentile, where g has reached
    its run-to-failure asymptote. If the minimum sits at the grid end (or no T
    is cheaper than run-to-failure), the honest answer is that NO preventive
    age helps - the textbook case for memoryless (shape <= 1) lifetimes or for
    Cp >= Cf - and the policy collapses to run-to-failure.
    """
    rtf = cost.cost_unplanned_failure / fit.mtbf
    t_hi = fit.scale * math.log(1e4) ** (1.0 / fit.shape)  # 99.99th percentile
    grid = np.linspace(t_hi / n_grid, t_hi, n_grid)
    g = age_replacement_cost_rate(fit, cost, grid)
    i = int(np.argmin(g))
    if i == n_grid - 1:
        return AgeReplacementPolicy(False, None, rtf, rtf, grid, g)
    lo = float(grid[i - 1]) if i > 0 else float(grid[i]) * 0.5
    hi = float(grid[i + 1])

    def _g(t: float) -> float:
        return float(age_replacement_cost_rate(fit, cost, t))

    t_star, g_star = _golden_section(_g, lo, hi)
    if g_star >= rtf * (1.0 - 1e-9):
        return AgeReplacementPolicy(False, None, rtf, rtf, grid, g)
    return AgeReplacementPolicy(True, float(t_star), float(g_star), rtf, grid, g)


# ---------------------------------------------------------------- condition-based


@dataclass(frozen=True)
class CBMPerformance:
    """The detector's MEASURED performance feeding the condition-based policy."""

    lead_days: int  # planning lead time an alert must give to convert the repair
    n_failures: int  # observed in-window functional failures assessed
    n_detected_in_time: int  # first at/after-onset alert >= lead_days before failure
    fraction_in_time: float
    false_alarms: int  # alert-days on truly pre-onset/healthy held-out machine-days
    n_negative_days: int
    false_alarms_per_machine_day: float


def measure_cbm_performance(result, lead_days: int = 3) -> CBMPerformance:
    """Measure the recommended detector's timely-detection and false-alarm rates.

    Timeliness: for each progressive machine whose functional failure is
    observed in-window, the first alert at/after the true onset must land at
    least `lead_days` before the failure day (enough time to plan a crew visit
    - the pipeline schedules over a matching multi-day horizon).
    False alarms reuse the alert-economics protocol: alerts on truly negative
    (pre-onset + healthy-validation) held-out machine-days.
    """
    if lead_days < 0:
        raise ValueError("lead_days must be >= 0")
    data = result.data
    fs = result.features
    threshold = result.chosen_threshold
    n_days = data.n_days

    n_fail = 0
    n_timely = 0
    for lab in data.labels:
        if lab.fault_type not in PROGRESSIVE_FAULTS:
            continue
        fd = data.failure_day(lab.machine)
        if fd is None or fd >= n_days:  # censored: failure not observed in-window
            continue
        n_fail += 1
        first = detection_delays(
            result.chosen_scores[lab.machine], fs.days, lab.onset_day, threshold
        )
        if first is not None and first <= fd - lead_days:
            n_timely += 1
    if n_fail == 0:
        raise ValueError("no observed in-window functional failures to assess CBM against")

    eval_machines = sorted(data.faulty_machines() + list(result.val_machines))
    y = data.day_labels()[np.ix_(eval_machines, fs.days)].ravel()
    s = result.chosen_scores[eval_machines].ravel()
    alert = s > threshold
    fp = int(np.count_nonzero(alert & (y == 0)))
    n_neg = int(np.count_nonzero(y == 0))
    return CBMPerformance(
        lead_days=int(lead_days),
        n_failures=n_fail,
        n_detected_in_time=n_timely,
        fraction_in_time=n_timely / n_fail,
        false_alarms=fp,
        n_negative_days=n_neg,
        false_alarms_per_machine_day=fp / n_neg if n_neg else float("nan"),
    )


def cbm_cost_rate(perf: CBMPerformance, mtbf: float, cost: PolicyCostConfig) -> float:
    """Cost per machine-day of the condition-based policy.

    Per failure cycle: a timely alert converts the event into a planned repair
    (Cp), a missed/late one stays an unplanned failure (Cf); divided by the
    mean life (cycle-shortening from acting slightly early is a stated
    second-order simplification). On top, every false alarm books one
    inspection at its measured per-machine-day rate.
    """
    if mtbf <= 0.0:
        raise ValueError("mtbf must be positive")
    p = perf.fraction_in_time
    per_cycle = p * cost.cost_planned_replacement + (1.0 - p) * cost.cost_unplanned_failure
    return per_cycle / mtbf + perf.false_alarms_per_machine_day * cost.cost_inspection


# ---------------------------------------------------------------- comparison


@dataclass(frozen=True)
class PolicyComparison:
    cost: PolicyCostConfig
    fit: WeibullFit
    rtf_cost_rate: float
    age: AgeReplacementPolicy
    cbm: CBMPerformance
    cbm_rate: float
    recommended: str  # POLICY_RTF / POLICY_AGE / POLICY_CBM (cheapest; ties -> simpler)

    def rates(self) -> dict[str, float]:
        """Policy -> cost rate per machine-day, in comparison order."""
        return {
            POLICY_RTF: self.rtf_cost_rate,
            POLICY_AGE: self.age.cost_rate,
            POLICY_CBM: self.cbm_rate,
        }


def compare_policies_from_result(
    result, cost: PolicyCostConfig | None = None, lead_days: int = 3
) -> PolicyComparison:
    """Price all three policies on a finished PipelineResult's fleet.

    The Weibull lifetime model comes from the generator's ground-truth failure
    days (failures + right-censored suspensions); the condition-based policy is
    priced from the recommended detector's own measured alerts. Deterministic.
    """
    cost = cost or PolicyCostConfig()
    failures, suspensions = lifetimes_from_data(result.data)
    fit = fit_weibull(failures, suspensions)
    age = optimize_age_replacement(fit, cost)
    perf = measure_cbm_performance(result, lead_days=lead_days)
    cbm = cbm_cost_rate(perf, fit.mtbf, cost)
    # cheapest wins; exact ties go to the SIMPLER policy (rtf < age < cbm),
    # matching the repo's prefer-the-simpler-model policy elsewhere.
    order = [(age.rtf_cost_rate, 0, POLICY_RTF), (age.cost_rate, 1, POLICY_AGE), (cbm, 2, POLICY_CBM)]
    recommended = min(order)[2]
    return PolicyComparison(
        cost=cost,
        fit=fit,
        rtf_cost_rate=age.rtf_cost_rate,
        age=age,
        cbm=perf,
        cbm_rate=cbm,
        recommended=recommended,
    )


def recommendation_text(cmp: PolicyComparison) -> str:
    """Plain-language, honestly-framed policy comparison."""
    c = cmp.cost
    fit = cmp.fit
    wear = "wear-out (shape > 1: preventive action CAN pay)" if fit.shape > 1.0 else (
        "memoryless/infant (shape <= 1: age-based replacement cannot beat run-to-failure)"
    )
    if cmp.age.finite_optimum:
        age_line = (
            f"    {POLICY_AGE:<17}{cmp.age.cost_rate:8.2f}   "
            f"(replace at T* = {cmp.age.optimal_age_days:.1f} d)"
        )
    else:
        age_line = (
            f"    {POLICY_AGE:<17}{cmp.age.cost_rate:8.2f}   "
            f"(no finite optimal age - collapses to run-to-failure)"
        )
    lines = [
        "Maintenance-policy comparison (SYNTHETIC data; ILLUSTRATIVE cost rates - "
        "not a business guarantee)",
        f"  Cost assumptions: unplanned failure {c.cost_unplanned_failure:g}, planned "
        f"replacement {c.cost_planned_replacement:g}, inspection {c.cost_inspection:g} "
        f"{c.unit}.",
        f"  Weibull lifetime fit ({fit.n_failures} failures + {fit.n_suspensions} "
        f"right-censored suspensions, MLE):",
        f"    shape beta = {fit.shape:.2f} - {wear}",
        f"    scale eta  = {fit.scale:.1f} d   MTBF = {fit.mtbf:.1f} d   "
        f"B10 = {fit.b10:.1f} d   B50 = {fit.b50:.1f} d  (modelled, not measured)",
        "",
        f"  Cost rate per machine-day ({c.unit}):",
        f"    {POLICY_RTF:<17}{cmp.rtf_cost_rate:8.2f}   (Cf / MTBF baseline)",
        age_line,
        f"    {POLICY_CBM:<17}{cmp.cbm_rate:8.2f}   "
        f"({cmp.cbm.n_detected_in_time}/{cmp.cbm.n_failures} failures alerted >= "
        f"{cmp.cbm.lead_days} d early; {cmp.cbm.false_alarms_per_machine_day:.3f} "
        f"false alarms/machine-day)",
        "",
        f"  Recommended under these assumptions: {cmp.recommended.upper()}",
        "",
        "  The lifetime model extrapolates from one heavily censored synthetic window and "
        "the cost rates are illustrative planning assumptions - read the ranking as the "
        "SHAPE of the policy trade-off, and substitute measured lifetimes and verified "
        "costs before acting on it.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- CSV

_CURVE_POINTS = 48


def write_csv(cmp: PolicyComparison, path: str | Path) -> Path:
    """Write the policy comparison + the age-replacement cost curve as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fit = cmp.fit
    c = cmp.cost
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["# SYNTHETIC data; cost rates are ILLUSTRATIVE assumptions, not real currency"])
        w.writerow(
            [
                f"# cost_unplanned_failure={c.cost_unplanned_failure:g}",
                f"cost_planned_replacement={c.cost_planned_replacement:g}",
                f"cost_inspection={c.cost_inspection:g}",
                f"unit={c.unit}",
            ]
        )
        w.writerow(
            [
                f"# weibull_shape={fit.shape:.4f}",
                f"scale_days={fit.scale:.2f}",
                f"mtbf_days={fit.mtbf:.2f}",
                f"b10_days={fit.b10:.2f}",
                f"b50_days={fit.b50:.2f}",
                f"n_failures={fit.n_failures}",
                f"n_suspensions={fit.n_suspensions}",
            ]
        )
        w.writerow(
            [
                f"# cbm_lead_days={cmp.cbm.lead_days}",
                f"detected_in_time={cmp.cbm.n_detected_in_time}/{cmp.cbm.n_failures}",
                f"false_alarms_per_machine_day={cmp.cbm.false_alarms_per_machine_day:.4f}",
            ]
        )
        w.writerow(["policy", "cost_rate_per_machine_day", "parameter", "note"])
        t_star = (
            f"T*={cmp.age.optimal_age_days:.2f}d" if cmp.age.finite_optimum else "no finite T*"
        )
        for name, rate, param in (
            (POLICY_RTF, cmp.rtf_cost_rate, ""),
            (POLICY_AGE, cmp.age.cost_rate, t_star),
            (POLICY_CBM, cmp.cbm_rate, f"lead>={cmp.cbm.lead_days}d"),
        ):
            w.writerow(
                [name, f"{rate:.4f}", param, "recommended" if name == cmp.recommended else ""]
            )
        w.writerow(["# age-replacement cost-rate curve g(T)"])
        w.writerow(["replacement_age_days", "cost_rate_per_machine_day", "note"])
        n = cmp.age.curve_age_days.size
        keep = set(np.linspace(0, n - 1, _CURVE_POINTS).round().astype(int).tolist())
        i_min = int(np.argmin(cmp.age.curve_cost_rate))
        keep.add(i_min)
        for i in sorted(keep):
            w.writerow(
                [
                    f"{cmp.age.curve_age_days[i]:.3f}",
                    f"{cmp.age.curve_cost_rate[i]:.4f}",
                    "grid-min" if i == i_min else "",
                ]
            )
    return path


# ---------------------------------------------------------------- hand-drawn SVG

_SVG_INK = "#0b0b0b"
_SVG_INK2 = "#52514e"
_SVG_MUTED = "#898781"
_SVG_GRID = "#e1e0d9"
_SVG_SURFACE = "#fcfcfb"
_SVG_AGE = "#2a78d6"  # age-replacement curve (repo's validated primary)
_SVG_CBM = "#008300"  # condition-based line (repo's validated secondary)
_SVG_RTF = _SVG_INK2  # run-to-failure baseline: a neutral dashed reference line


def _nice_top(v: float) -> float:
    if v <= 0:
        return 1.0
    mag = 10.0 ** np.floor(np.log10(v))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if m * mag >= v:
            return m * mag
    return 10.0 * mag


def write_svg(cmp: PolicyComparison, path: str | Path) -> Path:
    """Hand-drawn, dependency-free, deterministic SVG of the policy comparison.

    x-axis = preventive replacement age T (days); y-axis = cost rate per
    machine-day. The age-replacement curve g(T) runs against two horizontal
    references: the run-to-failure baseline (its own asymptote) and the
    condition-based policy priced from the detector's measured performance.
    No matplotlib, no wall-clock, no RNG - identical bytes every run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fit = cmp.fit

    # x spans to ~2 mean lives (where g has visibly settled onto its asymptote),
    # rounded up to a multiple of 25 days for clean ticks.
    x_span = min(2.0 * fit.mtbf, float(cmp.age.curve_age_days[-1]))
    x_max = 25.0 * math.ceil(x_span / 25.0) if x_span >= 25.0 else _nice_top(x_span)
    y_ref = max(cmp.rtf_cost_rate, cmp.cbm_rate, cmp.age.cost_rate)
    y_max = _nice_top(1.6 * y_ref)

    # dense display curve over the visible range (deterministic)
    ts = np.linspace(x_max / 240.0, x_max, 240)
    gs = age_replacement_cost_rate(fit, cmp.cost, ts)
    visible = gs <= y_max  # g(T) -> inf as T -> 0; clip the off-scale prefix

    W, H = 720, 440
    ml, mr, mt, mb = 66, 230, 58, 60
    pw, ph = W - ml - mr, H - mt - mb

    def px(x: float) -> float:
        return ml + (x / x_max if x_max else 0.0) * pw

    def py(y: float) -> float:
        return mt + (1.0 - (y / y_max if y_max else 0.0)) * ph

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
        f"Maintenance-policy economics: cost rate vs preventive replacement age</text>"
    )
    parts.append(
        f'<text x="{ml}" y="44" font-size="10.5" fill="{_SVG_MUTED}">'
        f"SYNTHETIC fleet; ILLUSTRATIVE cost rates - Weibull fit from {fit.n_failures} "
        f"failures + {fit.n_suspensions} suspensions (modelled, not measured)</text>"
    )

    # grid + ticks
    for i in range(6):
        yv = y_max * i / 5.0
        yy = py(yv)
        parts.append(
            f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml + pw}" y2="{yy:.1f}" '
            f'stroke="{_SVG_GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{ml - 8}" y="{yy + 3.5:.1f}" font-size="9.5" text-anchor="end" '
            f'fill="{_SVG_MUTED}">{yv:.0f}</text>'
        )
        xv = x_max * i / 5.0
        xx = px(xv)
        parts.append(
            f'<line x1="{xx:.1f}" y1="{mt + ph}" x2="{xx:.1f}" y2="{mt + ph + 5}" '
            f'stroke="{_SVG_MUTED}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{xx:.1f}" y="{mt + ph + 18}" font-size="9.5" text-anchor="middle" '
            f'fill="{_SVG_MUTED}">{xv:.0f}</text>'
        )
    parts.append(
        f'<text x="{ml + pw / 2:.1f}" y="{H - 20}" font-size="11" text-anchor="middle" '
        f'fill="{_SVG_INK2}">Preventive replacement age T (days)</text>'
    )
    parts.append(
        f'<text transform="translate(18,{mt + ph / 2:.1f}) rotate(-90)" font-size="11" '
        f'text-anchor="middle" fill="{_SVG_INK2}">Cost rate per machine-day '
        f"({esc(cmp.cost.unit)})</text>"
    )

    # run-to-failure baseline: neutral dashed reference (the g(T) asymptote)
    y_rtf = py(cmp.rtf_cost_rate)
    parts.append(
        f'<line x1="{ml}" y1="{y_rtf:.1f}" x2="{ml + pw}" y2="{y_rtf:.1f}" '
        f'stroke="{_SVG_RTF}" stroke-width="1.6" stroke-dasharray="6 4"/>'
    )
    parts.append(
        f'<text x="{ml + 6}" y="{y_rtf - 6:.1f}" font-size="10" font-weight="700" '
        f'fill="{_SVG_RTF}">run-to-failure  {cmp.rtf_cost_rate:.1f}</text>'
    )

    # condition-based line (measured detector performance)
    y_cbm = py(cmp.cbm_rate)
    parts.append(
        f'<line x1="{ml}" y1="{y_cbm:.1f}" x2="{ml + pw}" y2="{y_cbm:.1f}" '
        f'stroke="{_SVG_CBM}" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="{ml + 6}" y="{y_cbm - 6:.1f}" font-size="10" font-weight="700" '
        f'fill="{_SVG_CBM}">condition-based  {cmp.cbm_rate:.1f}</text>'
    )

    # age-replacement cost curve
    poly = " ".join(
        f"{px(float(t)):.1f},{py(float(g)):.1f}"
        for t, g, v in zip(ts, gs, visible, strict=True)
        if v
    )
    parts.append(
        f'<polyline points="{poly}" fill="none" stroke="{_SVG_AGE}" '
        f'stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>'
    )
    if cmp.age.finite_optimum and cmp.age.optimal_age_days <= x_max:
        cx, cy = px(cmp.age.optimal_age_days), py(cmp.age.cost_rate)
        parts.append(
            f'<line x1="{cx:.1f}" y1="{mt}" x2="{cx:.1f}" y2="{mt + ph}" '
            f'stroke="{_SVG_AGE}" stroke-width="1" stroke-dasharray="3 3" opacity="0.7"/>'
        )
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="{_SVG_AGE}" '
            f'stroke="{_SVG_SURFACE}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{cx + 9:.1f}" y="{cy + 16:.1f}" font-size="10" font-weight="700" '
            f'fill="{_SVG_AGE}">age T*={cmp.age.optimal_age_days:.0f}d  '
            f"{cmp.age.cost_rate:.1f}</text>"
        )

    # side panel
    tx = ml + pw + 22
    ty = mt + 6
    rec = cmp.recommended
    rate_by = cmp.rates()
    panel: list[tuple[str, str, str]] = [
        ("Weibull lifetime fit (MLE)", _SVG_INK, "700"),
        (f"shape beta {fit.shape:6.2f}", _SVG_INK2, "400"),
        (f"scale eta  {fit.scale:6.1f} d", _SVG_INK2, "400"),
        (f"MTBF       {fit.mtbf:6.1f} d", _SVG_INK2, "400"),
        (f"B10 life   {fit.b10:6.1f} d", _SVG_INK2, "400"),
        (f"{fit.n_failures} failures + {fit.n_suspensions} susp.", _SVG_MUTED, "400"),
        ("", _SVG_INK2, "400"),
        ("Cost rate / machine-day", _SVG_INK, "700"),
    ]
    for name, color in (
        (POLICY_RTF, _SVG_RTF),
        (POLICY_AGE, _SVG_AGE),
        (POLICY_CBM, _SVG_CBM),
    ):
        mark = " <-" if name == rec else ""
        panel.append((f"{name:<15}{rate_by[name]:7.2f}{mark}", color, "700" if name == rec else "400"))
    panel += [
        ("", _SVG_INK2, "400"),
        (f"recommended: {rec}", _SVG_INK, "700"),
        ("(illustrative costs; the", _SVG_MUTED, "400"),
        ("ranking moves with them)", _SVG_MUTED, "400"),
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
    result, outdir: str | Path = "docs", cost: PolicyCostConfig | None = None
) -> dict[str, Path]:
    """(Re)write the committed reference CSV + SVG for a finished pipeline run.

    Deterministic for a given pipeline result: identical bytes every run.
    """
    outdir = Path(outdir)
    cmp = compare_policies_from_result(result, cost=cost)
    return {
        "csv": write_csv(cmp, outdir / "policy_comparison.csv"),
        "svg": write_svg(cmp, outdir / "policy_comparison.svg"),
    }
