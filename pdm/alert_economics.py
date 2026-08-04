"""Alert-threshold economics / alert-fatigue analysis.

A decision layer on TOP of the anomaly detector. The detector emits a continuous
risk score per machine-day; turning that score into *alerts* needs a threshold,
and the threshold is a business trade-off, not a modelling detail:

- too low  -> analysts drown in false alarms (alert fatigue) and start ignoring
              the queue;
- too high -> real failures slip through with no alert at all.

This module sweeps the alert threshold over the detector's own risk scores on the
held-out (test) machine-days and, at each operating point, reports:

- precision / recall,
- alerts per period (the analyst workload / "alert-fatigue" axis),
- missed failures (false negatives),
- an expected cost = COST_MISSED_FAILURE * FN + COST_FALSE_ALARM * FP.

It then returns the cost-minimising operating point and a plain-language
recommendation.

IMPORTANT - honest framing:
- The scores come from a detector evaluated on SYNTHETIC, seeded data
  (pdm/data.py); no real telemetry is involved.
- The cost rates below are ILLUSTRATIVE synthetic assumptions, chosen only to
  show the SHAPE of the trade-off. They are NOT a business guarantee and the
  numbers are not real currency. The cost-minimising threshold moves with the
  cost RATIO, so plug in your own verified costs before using any of this
  operationally.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CostConfig:
    """ILLUSTRATIVE synthetic cost rates - NOT a business guarantee.

    Only the RATIO between the two drives the cost-minimising threshold. The
    defaults encode the usual asymmetry in maintenance: an unplanned failure
    that no alert caught (an emergency repair plus the idle line behind it) is
    far more expensive than a false alarm (an analyst spends time confirming a
    machine is fine).
    """

    cost_missed_failure: float = 1000.0  # cost booked per missed failure (a false negative)
    cost_false_alarm: float = 50.0  # cost booked per false alarm (a false positive)
    unit: str = "illustrative cost units"

    @property
    def ratio(self) -> float:
        """Missed-failure cost expressed in multiples of one false alarm."""
        return self.cost_missed_failure / max(self.cost_false_alarm, 1e-12)


@dataclass(frozen=True)
class OperatingPoint:
    """One alert threshold and everything it implies on the test set."""

    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    alerts: int  # tp + fp: machine-days that would raise an alert
    alerts_per_period: float  # alerts / n_periods -> analyst load per period
    missed_failures: int  # fn: true failure-days with no alert
    expected_cost: float


@dataclass(frozen=True)
class AlertEconomics:
    cost: CostConfig
    n_periods: int  # number of periods (days) the test window spans
    n_samples: int  # machine-days evaluated
    n_positives: int  # true failure machine-days in the test set
    points: tuple[OperatingPoint, ...]  # full exact sweep, threshold ascending
    best: OperatingPoint  # cost-minimising operating point over the sweep
    current: OperatingPoint | None  # the detector's live threshold, for reference


def _operating_point(
    y: np.ndarray,
    s: np.ndarray,
    threshold: float,
    cost: CostConfig,
    n_periods: int,
) -> OperatingPoint:
    """Confusion counts, rates and expected cost at one threshold (alert if s > t)."""
    alert = s > threshold
    tp = int(np.count_nonzero(alert & (y == 1)))
    fp = int(np.count_nonzero(alert & (y == 0)))
    fn = int(np.count_nonzero(~alert & (y == 1)))
    tn = int(np.count_nonzero(~alert & (y == 0)))
    alerts = tp + fp
    precision = tp / alerts if alerts else float("nan")
    n_pos = tp + fn
    recall = tp / n_pos if n_pos else float("nan")
    expected_cost = cost.cost_missed_failure * fn + cost.cost_false_alarm * fp
    return OperatingPoint(
        threshold=float(threshold),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        alerts=alerts,
        alerts_per_period=alerts / n_periods if n_periods else float("nan"),
        missed_failures=fn,
        expected_cost=float(expected_cost),
    )


def sweep_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    cost: CostConfig | None = None,
    n_periods: int = 1,
    current_threshold: float | None = None,
) -> AlertEconomics:
    """Exact threshold sweep over the detector's risk scores.

    The candidate thresholds are every distinct score value plus one just below
    the minimum, so the sweep spans the two extremes (alert on everything ->
    alert on nothing). `alert = score > threshold`, matching the rest of the
    package. The cost-minimising point is exact over this sweep; ties are broken
    toward the HIGHER threshold (fewer alerts = less analyst load at equal cost).
    """
    cost = cost or CostConfig()
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape:
        raise ValueError("y_true and scores must have the same shape")
    if y.size == 0:
        raise ValueError("need at least one sample")

    uniq = np.unique(s)
    below_min = np.nextafter(uniq[0], -np.inf)
    thresholds = np.concatenate(([below_min], uniq))

    points = tuple(_operating_point(y, s, float(t), cost, n_periods) for t in thresholds)
    # cost first, then prefer the higher threshold (less alert fatigue) on a tie
    best = min(points, key=lambda p: (p.expected_cost, -p.threshold))
    current = (
        _operating_point(y, s, float(current_threshold), cost, n_periods)
        if current_threshold is not None
        else None
    )
    return AlertEconomics(
        cost=cost,
        n_periods=int(n_periods),
        n_samples=int(y.size),
        n_positives=int(y.sum()),
        points=points,
        best=best,
        current=current,
    )


def economics_from_result(result, cost: CostConfig | None = None) -> AlertEconomics:
    """Build the alert economics from a finished PipelineResult.

    Uses the recommended detector's risk scores on exactly the held-out
    machine-days the detector was scored on (all faulty machines' days plus the
    healthy validation machines' days) and the generator's ground-truth
    onset labels. `current_threshold` is the detector's live FPR-budget
    threshold, so the sweep can be read against where the pipeline runs today.
    """
    data = result.data
    fs = result.features
    eval_machines = sorted(data.faulty_machines() + list(result.val_machines))
    y = data.day_labels()[np.ix_(eval_machines, fs.days)].ravel()
    s = result.chosen_scores[eval_machines].ravel()
    n_periods = int(fs.days.size)  # days in the test window = the analyst's periods
    return sweep_thresholds(
        y,
        s,
        cost=cost,
        n_periods=n_periods,
        current_threshold=result.chosen_threshold,
    )


def curve_rows(econ: AlertEconomics, max_points: int = 48) -> list[OperatingPoint]:
    """A compact, evenly-spaced subset of the sweep for tables/plots.

    Always includes the two extremes, the cost-minimising point and (if given)
    the detector's current operating point, so nothing important is dropped.
    """
    pts = econ.points
    n = len(pts)
    if n <= max_points:
        keep = set(range(n))
    else:
        keep = set(np.linspace(0, n - 1, max_points).round().astype(int).tolist())
    keep |= {0, n - 1}
    by_thr = {p.threshold: i for i, p in enumerate(pts)}
    keep.add(by_thr[econ.best.threshold])
    if econ.current is not None and econ.current.threshold in by_thr:
        keep.add(by_thr[econ.current.threshold])
    rows = [pts[i] for i in sorted(keep)]
    if econ.current is not None and econ.current.threshold not in by_thr:
        rows.append(econ.current)
        rows.sort(key=lambda p: p.threshold)
    return rows


def _tag(econ: AlertEconomics, p: OperatingPoint) -> str:
    tags = []
    if p.threshold == econ.best.threshold:
        tags.append("cost-optimal")
    if econ.current is not None and p.threshold == econ.current.threshold:
        tags.append("current")
    return "+".join(tags)


def write_csv(econ: AlertEconomics, path: str | Path, max_points: int = 48) -> Path:
    """Write the swept operating points as a CSV the README can reference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = curve_rows(econ, max_points=max_points)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["# SYNTHETIC data; cost rates are ILLUSTRATIVE assumptions, not real currency"])
        w.writerow(
            [
                f"# cost_missed_failure={econ.cost.cost_missed_failure:g}",
                f"cost_false_alarm={econ.cost.cost_false_alarm:g}",
                f"ratio={econ.cost.ratio:g}",
                f"unit={econ.cost.unit}",
            ]
        )
        w.writerow(
            [
                f"# n_samples={econ.n_samples}",
                f"n_positive_failure_days={econ.n_positives}",
                f"n_periods_days={econ.n_periods}",
            ]
        )
        w.writerow(
            [
                "threshold",
                "precision",
                "recall",
                "alerts",
                "alerts_per_day",
                "missed_failures",
                "false_alarms",
                "expected_cost",
                "note",
            ]
        )
        for p in rows:
            w.writerow(
                [
                    f"{p.threshold:.6f}",
                    "nan" if np.isnan(p.precision) else f"{p.precision:.4f}",
                    "nan" if np.isnan(p.recall) else f"{p.recall:.4f}",
                    p.alerts,
                    f"{p.alerts_per_period:.3f}",
                    p.missed_failures,
                    p.fp,
                    f"{p.expected_cost:.1f}",
                    _tag(econ, p),
                ]
            )
    return path


def recommendation_text(econ: AlertEconomics) -> str:
    """Plain-language recommendation for the cost-minimising operating point."""
    b = econ.best
    c = econ.cost
    lines = [
        "Alert-threshold economics (SYNTHETIC data; ILLUSTRATIVE cost rates - not a "
        "business guarantee)",
        f"  Cost assumptions: 1 missed failure = {c.cost_missed_failure:g} {c.unit}; "
        f"1 false alarm = {c.cost_false_alarm:g} {c.unit} (ratio {c.ratio:g}:1).",
        f"  Test set: {econ.n_samples} machine-days over {econ.n_periods} days, "
        f"{econ.n_positives} true failure-days.",
        "",
        "  Cost-minimising operating point:",
        f"    threshold        {b.threshold:.4f}",
        f"    precision/recall {b.precision:.3f} / {b.recall:.3f}",
        f"    analyst load     {b.alerts_per_period:.2f} alerts/day "
        f"({b.alerts} alerts over the window)",
        f"    missed failures  {b.missed_failures} of {econ.n_positives} failure-days",
        f"    expected cost    {b.expected_cost:.0f} {c.unit}",
    ]
    if econ.current is not None:
        cur = econ.current
        delta = cur.expected_cost - b.expected_cost
        if abs(delta) < 1e-9:
            verdict = (
                "The detector's current FPR-budget threshold already sits at the "
                "cost-minimising point under these assumptions."
            )
        else:
            direction = "lower" if cur.threshold > b.threshold else "raise"
            verdict = (
                f"Under these assumptions the current FPR-budget threshold "
                f"({cur.threshold:.4f}) is not cost-minimising: it books "
                f"{cur.expected_cost:.0f} vs {b.expected_cost:.0f} {c.unit} "
                f"({delta:+.0f}, {cur.missed_failures} vs {b.missed_failures} missed "
                f"failures at {cur.alerts_per_period:.2f} vs {b.alerts_per_period:.2f} "
                f"alerts/day). To reach it, {direction} the threshold toward "
                f"{b.threshold:.4f}."
            )
        lines += ["", "  " + verdict]
    lines += [
        "",
        "  Read this as the SHAPE of the fatigue/coverage trade-off, not a dollar figure: "
        "the cost-minimising threshold shifts with the cost ratio, so substitute your own "
        "verified costs before acting.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- hand-drawn SVG


_SVG_INK = "#0b0b0b"
_SVG_INK2 = "#52514e"
_SVG_MUTED = "#898781"
_SVG_GRID = "#e1e0d9"
_SVG_SURFACE = "#fcfcfb"
_SVG_PRIMARY = "#2a78d6"  # cost curve (repo's validated primary)
_SVG_BEST = "#008300"  # cost-optimal marker (repo's validated secondary)
_SVG_CURRENT = "#c25a00"  # current-threshold marker


def _nice_top(v: float) -> float:
    """A round-ish upper bound at or above v for the y-axis."""
    if v <= 0:
        return 1.0
    mag = 10.0 ** np.floor(np.log10(v))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if m * mag >= v:
            return m * mag
    return 10.0 * mag


def write_svg(econ: AlertEconomics, path: str | Path) -> Path:
    """Hand-drawn, dependency-free, deterministic SVG of the cost curve.

    x-axis = alerts per day (the analyst-workload / alert-fatigue axis);
    y-axis = expected cost. The cost-minimising and current operating points are
    marked. No matplotlib, no wall-clock, no RNG - identical bytes every run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pts = sorted(econ.points, key=lambda p: p.alerts_per_period)
    xs = [p.alerts_per_period for p in pts]
    ys = [p.expected_cost for p in pts]
    x_max = max(xs) if xs else 1.0
    y_max = _nice_top(max(ys) if ys else 1.0)

    W, H = 720, 440
    ml, mr, mt, mb = 78, 210, 58, 64
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
        f"Alert-threshold economics: expected cost vs analyst load</text>"
    )
    parts.append(
        f'<text x="{ml}" y="44" font-size="10.5" fill="{_SVG_MUTED}">'
        f"SYNTHETIC data; cost rates are ILLUSTRATIVE assumptions (ratio "
        f"{econ.cost.ratio:g}:1), not a business guarantee</text>"
    )

    # y grid + labels (5 ticks)
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
    # x ticks (0..x_max, 5 ticks)
    for i in range(6):
        xv = x_max * i / 5.0
        xx = px(xv)
        parts.append(
            f'<line x1="{xx:.1f}" y1="{mt + ph}" x2="{xx:.1f}" y2="{mt + ph + 5}" '
            f'stroke="{_SVG_MUTED}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{xx:.1f}" y="{mt + ph + 18}" font-size="9.5" text-anchor="middle" '
            f'fill="{_SVG_MUTED}">{xv:.1f}</text>'
        )
    # axis titles
    parts.append(
        f'<text x="{ml + pw / 2:.1f}" y="{H - 22}" font-size="11" text-anchor="middle" '
        f'fill="{_SVG_INK2}">Alerts per day (analyst workload -> alert fatigue)</text>'
    )
    parts.append(
        f'<text transform="translate(20,{mt + ph / 2:.1f}) rotate(-90)" font-size="11" '
        f'text-anchor="middle" fill="{_SVG_INK2}">Expected cost ({esc(econ.cost.unit)})</text>'
    )

    # cost curve polyline
    poly = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys, strict=True))
    parts.append(
        f'<polyline points="{poly}" fill="none" stroke="{_SVG_PRIMARY}" '
        f'stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>'
    )

    def marker(p: OperatingPoint, color: str, label: str) -> None:
        cx, cy = px(p.alerts_per_period), py(p.expected_cost)
        parts.append(
            f'<line x1="{cx:.1f}" y1="{mt}" x2="{cx:.1f}" y2="{mt + ph}" '
            f'stroke="{color}" stroke-width="1" stroke-dasharray="3 3" opacity="0.7"/>'
        )
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="{color}" '
            f'stroke="{_SVG_SURFACE}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{cx + 9:.1f}" y="{cy - 7:.1f}" font-size="10" font-weight="700" '
            f'fill="{color}">{esc(label)}</text>'
        )

    if econ.current is not None:
        marker(econ.current, _SVG_CURRENT, "current")
    marker(econ.best, _SVG_BEST, "cost-optimal")

    # side panel: the numbers for the cost-optimal point
    b = econ.best
    tx = ml + pw + 22
    ty = mt + 6
    panel = [
        ("Cost-optimal operating point", _SVG_INK, "700"),
        (f"threshold  {b.threshold:.3f}", _SVG_INK2, "400"),
        (f"precision  {b.precision:.3f}", _SVG_INK2, "400"),
        (f"recall     {b.recall:.3f}", _SVG_INK2, "400"),
        (f"alerts/day {b.alerts_per_period:.2f}", _SVG_INK2, "400"),
        (f"missed     {b.missed_failures}/{econ.n_positives} days", _SVG_INK2, "400"),
        (f"exp. cost  {b.expected_cost:.0f}", _SVG_BEST, "700"),
    ]
    if econ.current is not None:
        cur = econ.current
        panel += [
            ("", _SVG_INK2, "400"),
            ("Current threshold (5% FPR)", _SVG_INK, "700"),
            (f"threshold  {cur.threshold:.3f}", _SVG_INK2, "400"),
            (f"alerts/day {cur.alerts_per_period:.2f}", _SVG_INK2, "400"),
            (f"missed     {cur.missed_failures}/{econ.n_positives} days", _SVG_INK2, "400"),
            (f"exp. cost  {cur.expected_cost:.0f}", _SVG_CURRENT, "700"),
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
    result, outdir: str | Path = "docs", cost: CostConfig | None = None
) -> dict[str, Path]:
    """(Re)write the committed reference CSV + SVG for a finished pipeline run.

    Deterministic for a given pipeline result: identical bytes every run. These
    are the artifacts the README references.
    """
    outdir = Path(outdir)
    econ = economics_from_result(result, cost=cost)
    return {
        "csv": write_csv(econ, outdir / "alert_economics.csv"),
        "svg": write_svg(econ, outdir / "alert_economics.svg"),
    }
