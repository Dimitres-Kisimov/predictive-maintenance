"""Rolling-window features per machine/day.

For each machine and day t, features are computed from a trailing window of
`window_days` days that ENDS at day t (inclusive). No sample after day t is
touched, so there is no look-ahead leakage: recomputing on data truncated at
day t yields identical features for all days <= t (guarded by a test).

Per sensor: mean, std, slope (least-squares over the window), and 3 FFT band
log-powers (numpy rfft, DC excluded, bins split into equal thirds).
Plus the 6 pairwise sensor correlations within the window (correlation-break
faults show up here).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pdm.data import SENSORS, PlantData

FFT_BANDS = 3


@dataclass(frozen=True)
class FeatureConfig:
    window_days: int = 5


@dataclass(frozen=True)
class FeatureSet:
    X: np.ndarray  # (n_machines, n_feature_days, n_features)
    days: np.ndarray  # day index of each feature row (window END day)
    names: tuple[str, ...]

    @property
    def n_features(self) -> int:
        return self.X.shape[2]


def feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for s in SENSORS:
        names.extend([f"{s}_mean", f"{s}_std", f"{s}_slope"])
        names.extend([f"{s}_fftband{b}" for b in range(FFT_BANDS)])
    for i in range(len(SENSORS)):
        for j in range(i + 1, len(SENSORS)):
            names.append(f"corr_{SENSORS[i]}_{SENSORS[j]}")
    return tuple(names)


def _window_features(win: np.ndarray) -> np.ndarray:
    """Features for one window of shape (n_samples, n_sensors)."""
    n = win.shape[0]
    tt = np.arange(n, dtype=float)
    tt = (tt - tt.mean()) / n
    denom = float(np.dot(tt, tt))
    feats: list[float] = []
    for s in range(win.shape[1]):
        x = win[:, s]
        mu = float(x.mean())
        sd = float(x.std())
        xd = x - mu
        slope = float(np.dot(tt, xd) / denom)
        power = np.abs(np.fft.rfft(xd)) ** 2 / n
        bands = np.array_split(power[1:], FFT_BANDS)
        feats.extend([mu, sd, slope] + [float(np.log1p(b.sum())) for b in bands])
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(win.T)
    iu = np.triu_indices(win.shape[1], k=1)
    feats.extend(np.nan_to_num(corr[iu], nan=0.0).tolist())
    return np.asarray(feats)


def build_features(data: PlantData, fcfg: FeatureConfig | None = None) -> FeatureSet:
    cfg = fcfg or FeatureConfig()
    spd = data.config.samples_per_day
    n_days = data.n_days
    w = cfg.window_days
    if n_days < w:
        raise ValueError("not enough days for one feature window")
    days = np.arange(w - 1, n_days)
    names = feature_names()
    n_machines = data.readings.shape[0]
    X = np.zeros((n_machines, len(days), len(names)))
    for m in range(n_machines):
        for k, t in enumerate(days):
            win = data.readings[m, (t - w + 1) * spd : (t + 1) * spd, :]
            X[m, k] = _window_features(win)
    return FeatureSet(X=X, days=days, names=names)
