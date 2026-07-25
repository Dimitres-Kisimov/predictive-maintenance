"""Seeded synthetic sensor data generator for a small machine fleet.

Everything produced here is SYNTHETIC. A deterministic generator builds
multivariate sensor readings (temperature, vibration, current, pressure) for
~20 machines over ~60 days. The four sensors are correlated through a shared
latent "load" process with a daily cycle. A subset of machines degrades:

- slow_drift:          slow linear rise in vibration/temperature
- sudden_spike:        short high-amplitude spikes after onset
- correlation_break:   vibration decouples from the shared load
- accelerating_drift:  quadratic (accelerating) rise after onset

Ground-truth fault types and onset days are exposed for honest evaluation.
Same config (incl. seed) -> bit-identical output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

SENSORS = ("temperature", "vibration", "current", "pressure")

FAULT_HEALTHY = "healthy"
FAULT_DRIFT = "slow_drift"
FAULT_SPIKE = "sudden_spike"
FAULT_CORR = "correlation_break"
FAULT_ACCEL = "accelerating_drift"

FAULT_KINDS = (FAULT_DRIFT, FAULT_SPIKE, FAULT_CORR, FAULT_ACCEL)

_BASES = np.array([70.0, 2.0, 30.0, 5.0])  # degC, mm/s, A, bar
_GAINS = np.array([8.0, 1.2, 12.0, 1.5])  # coupling to the latent load
_NOISE_SD = np.array([0.5, 0.08, 0.6, 0.05])


@dataclass(frozen=True)
class DataConfig:
    n_machines: int = 20
    n_days: int = 60
    samples_per_day: int = 24
    seed: int = 42
    n_drift: int = 4
    n_spike: int = 2
    n_corr_break: int = 2
    n_accel: int = 2
    onset_min_frac: float = 0.40
    onset_max_frac: float = 0.75

    @property
    def n_faulty(self) -> int:
        return self.n_drift + self.n_spike + self.n_corr_break + self.n_accel


@dataclass(frozen=True)
class MachineLabel:
    machine: int
    fault_type: str
    onset_day: int  # -1 for healthy machines


@dataclass(frozen=True)
class PlantData:
    readings: np.ndarray  # (n_machines, n_days * samples_per_day, len(SENSORS))
    labels: tuple[MachineLabel, ...]
    config: DataConfig

    @property
    def n_days(self) -> int:
        return self.readings.shape[1] // self.config.samples_per_day

    def faulty_machines(self) -> list[int]:
        return [lab.machine for lab in self.labels if lab.fault_type != FAULT_HEALTHY]

    def healthy_machines(self) -> list[int]:
        return [lab.machine for lab in self.labels if lab.fault_type == FAULT_HEALTHY]

    def day_labels(self) -> np.ndarray:
        """0/1 matrix (n_machines, n_days): 1 on and after a machine's true onset day."""
        y = np.zeros((self.config.n_machines, self.n_days), dtype=int)
        for lab in self.labels:
            if lab.fault_type != FAULT_HEALTHY and lab.onset_day < self.n_days:
                y[lab.machine, lab.onset_day :] = 1
        return y


def _ar1(rng: np.random.Generator, n: int, phi: float = 0.9) -> np.ndarray:
    return lfilter([1.0], [1.0, -phi], rng.normal(size=n))


def generate(config: DataConfig | None = None) -> PlantData:
    """Generate the synthetic fleet. Deterministic for a given config."""
    cfg = config or DataConfig()
    if cfg.n_faulty > cfg.n_machines:
        raise ValueError("more faulty machines than machines")
    rng = np.random.default_rng(cfg.seed)
    spd = cfg.samples_per_day
    n_t = cfg.n_days * spd
    t = np.arange(n_t)
    hour = (t % spd) * (24.0 / spd)
    day = t // spd

    faulty = rng.choice(cfg.n_machines, size=cfg.n_faulty, replace=False)
    kinds = (
        [FAULT_DRIFT] * cfg.n_drift
        + [FAULT_SPIKE] * cfg.n_spike
        + [FAULT_CORR] * cfg.n_corr_break
        + [FAULT_ACCEL] * cfg.n_accel
    )
    lo = int(cfg.n_days * cfg.onset_min_frac)
    hi = max(lo + 1, int(cfg.n_days * cfg.onset_max_frac))
    onsets = rng.integers(lo, hi, size=cfg.n_faulty)
    fault_type = {int(m): k for m, k in zip(faulty, kinds, strict=True)}
    onset_day = {int(m): int(o) for m, o in zip(faulty, onsets, strict=True)}

    readings = np.zeros((cfg.n_machines, n_t, len(SENSORS)))
    for m in range(cfg.n_machines):
        base = _BASES + rng.normal(0.0, [1.5, 0.10, 1.0, 0.15])
        phase = rng.uniform(-1.5, 1.5)
        load = 1.0 + 0.2 * np.sin(2.0 * np.pi * (hour - 9.0 - phase) / 24.0) + 0.05 * _ar1(rng, n_t)
        x = base + np.outer(load, _GAINS) + rng.normal(size=(n_t, len(SENSORS))) * _NOISE_SD

        kind = fault_type.get(m, FAULT_HEALTHY)
        if kind != FAULT_HEALTHY:
            onset = onset_day[m]
            dsince = np.clip(day - onset, 0, None).astype(float)
            if kind == FAULT_DRIFT:
                x[:, 1] += 0.030 * dsince
                x[:, 0] += 0.120 * dsince
            elif kind == FAULT_ACCEL:
                x[:, 1] += 0.0045 * dsince**2
                x[:, 0] += 0.020 * dsince**2
            elif kind == FAULT_SPIKE:
                for d in range(onset, cfg.n_days):
                    idx = d * spd + rng.choice(spd, size=3, replace=False)
                    x[idx, 1] += rng.uniform(1.5, 3.0, size=3)
                    x[idx, 2] += rng.uniform(6.0, 10.0, size=3)
                    x[idx, 0] += rng.uniform(2.0, 5.0, size=3)
            elif kind == FAULT_CORR:
                post = day >= onset
                load2 = (
                    1.0
                    + 0.2 * np.sin(2.0 * np.pi * (hour + 8.0 - phase) / 24.0)
                    + 0.05 * _ar1(rng, n_t)
                )
                x[post, 1] = (
                    base[1]
                    + _GAINS[1] * load2[post]
                    + rng.normal(size=int(post.sum())) * _NOISE_SD[1]
                )
        readings[m] = x

    labels = tuple(
        MachineLabel(m, fault_type.get(m, FAULT_HEALTHY), onset_day.get(m, -1))
        for m in range(cfg.n_machines)
    )
    return PlantData(readings=readings, labels=labels, config=cfg)
