"""Generic SNP-dosage regression helpers, shared by the heritability/PrediXcan
pipeline (`src/heritability.py`) and the final model-vs-PrediXcan comparison
(`src/evaluate.py`).

Ported from the original single-gene POC's `src/diagnose.py` (elastic-net
baseline + permutation testing), with no dependency on gene- or
cell-type-specific data loading, so it can be reused across many genes.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNet, ElasticNetCV


def is_constant(values: np.ndarray, atol: float = 1e-12) -> bool:
    """True if `values` has (numerically) zero variance.

    A real scenario, not just an edge case: an undetected gene, or a
    cell-type/donor combination too small to have any expression spread.
    Correlation is undefined in that case, and several downstream
    computations (elastic net's alpha-path/CV selection, permutation nulls)
    become ill-posed rather than erroring, so this is checked explicitly.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return True
    return bool(np.std(values) <= atol)


def safe_pearsonr(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """`pearsonr`, but returns (nan, nan) for constant/too-short input instead of warning."""
    if len(a) < 3 or is_constant(a) or is_constant(b):
        return float("nan"), float("nan")
    r, p = pearsonr(a, b)
    return float(r), float(p)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination: 1 - SS_res / SS_tot."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


class ElasticNetBaseline:
    """Elastic net regression (standardized features) on raw SNP dosages.

    Elastic net (L1 + L2 penalty) is the standard choice for exactly this
    kind of "predict a trait from many, often correlated (LD-linked) SNPs"
    problem -- it's the model behind PrediXcan/TWAS-style genetic prediction
    of expression. The L1 term encourages sparsity (a handful of tagging
    SNPs, not all of them with tiny weights) and the L2 term keeps it stable
    when SNPs are highly correlated, which is the norm within a ~40kb window.

    `alpha`/`l1_ratio` are selected once via `ElasticNetCV`'s internal
    k-fold CV (see `fit_cv`), then reused as *fixed* hyperparameters for
    plain (non-CV) refits, e.g. inside a permutation-test loop, since
    re-running a full CV grid search per permutation would be far too slow.
    """

    def __init__(self, alpha: float, l1_ratio: float, max_iter: int = 10_000):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNetBaseline":
        # float64 throughout: sklearn's ElasticNetCV/ElasticNet internal Gram-matrix
        # precompute path has a known float32 precision bug (a Gram entry recomputed
        # in float64 fails strict equality against the float32-computed original,
        # raising "Gram matrix ... did not pass validation").
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0)
        self.x_std[self.x_std == 0] = 1.0
        x_scaled = (X - self.x_mean) / self.x_std

        self._model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=self.max_iter)
        self._model.fit(x_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        x_scaled = (X - self.x_mean) / self.x_std
        return self._model.predict(x_scaled)

    @staticmethod
    def fit_cv(
        X: np.ndarray, y: np.ndarray, l1_ratios: Sequence[float], cv: int, seed: int, max_iter: int = 10_000
    ) -> "ElasticNetBaseline":
        """Selects `alpha` (via ElasticNetCV's automatic path) and `l1_ratio` via internal k-fold CV."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x_mean, x_std = X.mean(axis=0), X.std(axis=0)
        x_std[x_std == 0] = 1.0
        x_scaled = (X - x_mean) / x_std

        cv_folds = min(cv, len(y))  # ElasticNetCV requires cv <= n_samples
        cv_model = ElasticNetCV(l1_ratio=list(l1_ratios), cv=max(cv_folds, 2), random_state=seed, max_iter=max_iter)
        cv_model.fit(x_scaled, y)

        model = ElasticNetBaseline(alpha=float(cv_model.alpha_), l1_ratio=float(cv_model.l1_ratio_), max_iter=max_iter)
        model.x_mean, model.x_std, model._model = x_mean, x_std, cv_model
        return model


def impute_and_filter_dosage(
    dosage: np.ndarray, positions: List[int], ref_alt: List[Tuple[str, str]], max_missing_frac: float = 0.2
) -> Tuple[np.ndarray, List[int], List[Tuple[str, str]], np.ndarray]:
    """Drops variants with too much missingness or no variance, mean-imputes the rest.

    Returns the filtered/imputed dosage matrix, filtered positions/ref_alt,
    and each surviving variant's alt-allele frequency (for MAF filtering).
    """
    if dosage.shape[1] == 0:
        return dosage, positions, ref_alt, np.zeros(0)

    missing_frac = np.isnan(dosage).mean(axis=0)
    col_std = np.nanstd(dosage, axis=0)
    keep = (missing_frac <= max_missing_frac) & (col_std > 0)

    dosage = dosage[:, keep]
    positions = [p for p, k in zip(positions, keep) if k]
    ref_alt = [ra for ra, k in zip(ref_alt, keep) if k]

    col_means = np.nanmean(dosage, axis=0)
    nan_mask = np.isnan(dosage)
    dosage = np.where(nan_mask, col_means, dosage)
    allele_freq = col_means / 2.0

    return dosage, positions, ref_alt, allele_freq


def filter_by_maf(
    dosage: np.ndarray, positions: List[int], ref_alt: List[Tuple[str, str]], allele_freq: np.ndarray, maf_min: float
) -> Tuple[np.ndarray, List[int], List[Tuple[str, str]], np.ndarray]:
    """Drops variants whose minor-allele frequency is below `maf_min` (PrediXcan-style QC)."""
    if dosage.shape[1] == 0:
        return dosage, positions, ref_alt, allele_freq
    maf = np.minimum(allele_freq, 1.0 - allele_freq)
    keep = maf >= maf_min
    return dosage[:, keep], [p for p, k in zip(positions, keep) if k], [ra for ra, k in zip(ref_alt, keep) if k], allele_freq[keep]


def permutation_test_baseline(
    dosage: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
    l1_ratio: float,
    n_permutations: int,
    rng: np.random.RandomState,
) -> Tuple[float, np.ndarray, float]:
    """Null: shuffle fit-set labels only, keep the real test labels fixed.

    Answers: could a model fit on this many donors/variants achieve this
    test-set performance by chance alone? Uses a *fixed* (alpha, l1_ratio)
    rather than re-running CV per permutation, which would be far too slow.
    """
    if is_constant(y[fit_idx]) or is_constant(y[test_idx]):
        return float("nan"), np.array([]), float("nan")

    observed_model = ElasticNetBaseline(alpha, l1_ratio).fit(dosage[fit_idx], y[fit_idx])
    observed_r2 = r2_score(y[test_idx], observed_model.predict(dosage[test_idx]))

    null_r2 = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled_y_fit = rng.permutation(y[fit_idx])
        model = ElasticNetBaseline(alpha, l1_ratio).fit(dosage[fit_idx], shuffled_y_fit)
        null_r2[i] = r2_score(y[test_idx], model.predict(dosage[test_idx]))

    p_value = float(np.mean(null_r2 >= observed_r2))
    return observed_r2, null_r2, p_value
