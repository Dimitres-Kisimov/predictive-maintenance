"""Two anomaly detectors, honestly compared.

1. PCADetector — numpy SVD principal-component reconstruction error. Simple,
   fast, no training loop. This is the baseline.
2. AEDetector — small PyTorch autoencoder trained on healthy-only windows
   (full-batch, seeded, deterministic). torch is imported lazily so the rest
   of the package works without it (CI installs torch best-effort).

Both score a sample by reconstruction error and expose per-feature error so
an alert can be explained ("flagged on vibration + temperature"). Thresholds
come from a clean validation split at a stated false-positive-rate budget.

Metric helpers (ROC-AUC, PR-AUC / average precision, precision@k) are
implemented here with numpy/scipy since scikit-learn is not part of the stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import rankdata

from pdm.data import SENSORS

_EPS = 1e-12


def _fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


class PCADetector:
    """PCA reconstruction-error detector via numpy SVD."""

    name = "pca"

    def __init__(self, var_target: float = 0.90):
        self.var_target = var_target
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.components: np.ndarray | None = None
        self.n_components: int = 0

    def fit(self, X: np.ndarray) -> PCADetector:
        self.mu, self.sd = _fit_standardizer(X)
        Z = (X - self.mu) / self.sd
        _, S, Vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
        var = S**2
        cum = np.cumsum(var) / max(var.sum(), _EPS)
        k = int(np.searchsorted(cum, self.var_target) + 1)
        self.n_components = max(2, min(k, Vt.shape[0]))
        self.components = Vt[: self.n_components]
        return self

    def per_feature_error(self, X: np.ndarray) -> np.ndarray:
        if self.components is None:
            raise RuntimeError("fit first")
        Z = (X - self.mu) / self.sd
        recon = Z @ self.components.T @ self.components
        return (Z - recon) ** 2

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.per_feature_error(X).mean(axis=1)


class AEDetector:
    """Small torch autoencoder; reconstruction-error scores.

    Trained full-batch with a fixed torch seed so repeated fits on the same
    data give identical scores on CPU.
    """

    name = "autoencoder"

    def __init__(
        self,
        hidden: int = 16,
        code: int = 6,
        epochs: int = 400,
        lr: float = 1e-3,
        seed: int = 0,
    ):
        self.hidden = hidden
        self.code = code
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self._model = None

    @staticmethod
    def _torch():
        import torch

        return torch

    def fit(self, X: np.ndarray) -> AEDetector:
        torch = self._torch()
        torch.manual_seed(self.seed)
        self.mu, self.sd = _fit_standardizer(X)
        Z = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        d = Z.shape[1]
        model = torch.nn.Sequential(
            torch.nn.Linear(d, self.hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, self.code),
            torch.nn.ReLU(),
            torch.nn.Linear(self.code, self.hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, d),
        )
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(model(Z), Z)
            loss.backward()
            opt.step()
        model.eval()
        self._model = model
        return self

    def per_feature_error(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit first")
        torch = self._torch()
        Z = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        with torch.no_grad():
            recon = self._model(Z)
        return ((Z - recon) ** 2).numpy()

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.per_feature_error(X).mean(axis=1)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


# ---------------------------------------------------------------- metrics


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based ROC-AUC (Mann-Whitney), ties handled by average ranks."""
    y = np.asarray(y_true).astype(bool)
    n1 = int(y.sum())
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def pr_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Precision and recall at every score cut, sorted by descending score."""
    y = np.asarray(y_true).astype(int)
    order = np.argsort(-np.asarray(scores), kind="stable")
    tp = np.cumsum(y[order])
    n1 = max(int(y.sum()), 1)
    precision = tp / np.arange(1, y.size + 1)
    recall = tp / n1
    return precision, recall


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision (step-wise integral of the PR curve)."""
    if int(np.asarray(y_true).sum()) == 0:
        return float("nan")
    precision, recall = pr_curve(y_true, scores)
    drecall = np.diff(np.concatenate(([0.0], recall)))
    return float(np.sum(precision * drecall))


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    y = np.asarray(y_true).astype(int)
    k = min(k, y.size)
    order = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return float(y[order].mean())


def threshold_at_fpr(clean_scores: np.ndarray, fpr_budget: float) -> float:
    """Score threshold so that ~fpr_budget of clean validation rows alert."""
    return float(np.quantile(np.asarray(clean_scores), 1.0 - fpr_budget))


@dataclass
class DetectorReport:
    name: str
    roc_auc: float
    pr_auc: float
    precision_at_k: float
    k: int
    threshold: float
    fpr_budget: float
    val_fpr: float
    mean_delay_days: float  # mean days from true onset to first alert (detected machines)
    n_detected: int
    n_faulty: int
    precision_curve: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    recall_curve: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


def detection_delays(
    score_rows: np.ndarray, days: np.ndarray, onset: int, threshold: float
) -> int | None:
    """First alert day at/after onset for one machine; None if never alerted."""
    mask = (score_rows > threshold) & (days >= onset)
    if not mask.any():
        return None
    return int(days[mask][0])


def top_sensors(per_feature_err_row: np.ndarray, names: tuple[str, ...], top: int = 2) -> list[str]:
    """Aggregate per-feature error to per-sensor error; return top sensor names.

    Correlation features split their error evenly between the two sensors.
    """
    agg = dict.fromkeys(SENSORS, 0.0)
    for err, name in zip(per_feature_err_row, names, strict=True):
        if name.startswith("corr_"):
            _, a, b = name.split("_", 2)
            agg[a] += float(err) / 2.0
            agg[b] += float(err) / 2.0
        else:
            sensor = name.split("_", 1)[0]
            agg[sensor] += float(err)
    ranked = sorted(agg, key=agg.get, reverse=True)
    return ranked[:top]
