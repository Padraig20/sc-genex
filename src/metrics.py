"""Evaluation metrics computed at the donor (pseudobulk) level.

Two things are checked, mirroring Variformer's cross-individual R2/Pearson
evaluation but extended with the uncertainty-quality check that is the whole
point of this POC:

1. `pseudobulk_correlation`: does predicted `mu` (from an unseen donor's
   personalized sequence) track that donor's true pseudobulk mean expression?
2. `sigma_calibration_correlation`: does predicted `sigma` track that donor's
   true empirical cell-cell std? If it does, the model has learned something
   about expression *specificity/variability* from genotype alone, which is
   the hypothesis this POC is meant to test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr


@dataclass
class CorrelationResult:
    pearson_r: float
    pearson_p: float
    r2: float
    n: int


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _correlation(y_true: np.ndarray, y_pred: np.ndarray) -> CorrelationResult:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return CorrelationResult(pearson_r=float("nan"), pearson_p=float("nan"), r2=float("nan"), n=len(y_true))
    r, p = pearsonr(y_true, y_pred)
    return CorrelationResult(pearson_r=float(r), pearson_p=float(p), r2=_r2_score(y_true, y_pred), n=len(y_true))


def pseudobulk_correlation(pred_mu: np.ndarray, true_pseudobulk_mean: np.ndarray) -> CorrelationResult:
    """Pearson r / R2 between predicted mu and true pseudobulk mean, across donors."""
    return _correlation(np.asarray(true_pseudobulk_mean, dtype=np.float64), np.asarray(pred_mu, dtype=np.float64))


def sigma_calibration_correlation(pred_sigma: np.ndarray, empirical_std: np.ndarray) -> CorrelationResult:
    """Pearson r / R2 between predicted sigma and true empirical cell-cell std, across donors."""
    return _correlation(np.asarray(empirical_std, dtype=np.float64), np.asarray(pred_sigma, dtype=np.float64))
