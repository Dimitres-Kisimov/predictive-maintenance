"""Per-machine degradation/health index.

IMPORTANT: this is a HEURISTIC degradation score on a 0-100 scale derived from
the smoothed anomaly-score trend. It is NOT a certified remaining-useful-life
(RUL) prediction and must not be treated as one. It exists to rank machines by
urgency for maintenance planning.

Construction: EWMA-smooth the daily anomaly score, express it as a multiple of
the alert threshold, and map the ratio range [1x .. saturation x threshold] onto
health [100 .. 0] on a LOG scale (reconstruction errors span orders of
magnitude). Below threshold -> 100 (healthy). The mapping is monotone: a
machine whose smoothed score only rises can never gain health (tested).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


def ewma(x: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    out = np.empty_like(np.asarray(x, dtype=float))
    acc = float(x[0])
    for i, v in enumerate(np.asarray(x, dtype=float)):
        acc = alpha * v + (1.0 - alpha) * acc
        out[i] = acc
    return out


def health_series(
    scores: np.ndarray, threshold: float, alpha: float = 0.3, saturation: float = 1000.0
) -> np.ndarray:
    """Health 0-100 for one machine's daily anomaly scores (heuristic, not RUL).

    `saturation` is the score-to-threshold ratio at which health reaches 0.
    """
    smooth = ewma(scores, alpha=alpha)
    ratio = smooth / max(threshold, _EPS)
    exceed = np.log1p(np.clip(ratio - 1.0, 0.0, None)) / np.log1p(saturation - 1.0)
    return 100.0 * (1.0 - np.clip(exceed, 0.0, 1.0))


@dataclass
class HealthReport:
    health: np.ndarray  # (n_machines, n_days) health index over time
    final: np.ndarray  # (n_machines,) latest health
    slope: np.ndarray  # (n_machines,) health change per day over the last week
    severity: np.ndarray  # (n_machines,) log-scale exceedance of the smoothed score
    urgency: np.ndarray  # (n_machines,) higher = service sooner
    ranking: list[int]  # machine ids, most urgent first


def machine_health(
    score_matrix: np.ndarray,
    threshold: float,
    alpha: float = 0.3,
    saturation: float = 1000.0,
    slope_days: int = 7,
) -> HealthReport:
    n_machines, n_days = score_matrix.shape
    health = np.vstack(
        [
            health_series(score_matrix[m], threshold, alpha=alpha, saturation=saturation)
            for m in range(n_machines)
        ]
    )
    final = health[:, -1]
    w = min(slope_days, n_days - 1)
    slope = (health[:, -1] - health[:, -1 - w]) / max(w, 1)
    # Severity keeps ranking meaningful once the 0-100 index has saturated at 0:
    # log-scale how far the latest smoothed score sits above the alert threshold.
    smooth_final = np.array([ewma(score_matrix[m], alpha=alpha)[-1] for m in range(n_machines)])
    severity = np.log1p(np.clip(smooth_final / max(threshold, _EPS) - 1.0, 0.0, None))
    urgency = (100.0 - final) + 5.0 * np.clip(-slope, 0.0, None) + severity
    ranking = list(np.argsort(-urgency, kind="stable").astype(int))
    return HealthReport(
        health=health,
        final=final,
        slope=slope,
        severity=severity,
        urgency=urgency,
        ranking=ranking,
    )
