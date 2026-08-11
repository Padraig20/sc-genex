"""Diagnostics for a (gene, cell_type) pair: how "easy" is this prediction task?

Runs quickly with **no GPU and no Enformer** -- useful for sanity-checking
whether a `train.py` run's "amazing" test-set performance reflects genuinely
learned regulatory grammar, or is mostly explained by (a) a large, well-known
local genetic effect that a trivial linear model could also capture, (b) a
small test set that inflates apparent correlation, or (c) confounds in how the
targets were constructed.

Checks performed:
  1. Expression-distribution diagnostics: zero-inflation/dropout, skew, and
     whether `n_cells` per donor confounds the pseudobulk mean/std (fewer
     cells -> noisier empirical std estimate, unrelated to true biology).
  2. A flag for genes inside the classical MHC/HLA region (chr6:28.48-33.45Mb,
     GRCh38). HLA genes are notorious for unusually large, simple genetic
     effects on expression (e.g. structural gene presence/absence tied to a
     nearby paralogue's haplotype, long high-LD blocks), which can make a gene
     look "trivially predictable" for reasons that have little to do with
     whether the sequence model learned anything subtle.
  3. Top individual variants (in the same TSS window used for training) by
     raw correlation with pseudobulk expression -- a strong single hit is a
     red flag that this is a simple eQTL, not complex regulatory grammar.
  4. A "cheap" linear baseline: elastic net regression (like a PrediXcan/TWAS
     model) on raw SNP dosages in that same window, evaluated on the *same*
     donor train/(val+)test split `train.py` would use. If this baseline's
     test R2/Pearson r is close to Enformer's (pass `--predictions-csv` to
     compare directly), the deep sequence model likely isn't adding much. Run
     for *both* the pseudobulk mean (a standard eQTL-style question) *and*
     the empirical cell-cell std (a vQTL-style question: is cell-to-cell
     variability itself locally genetically determined, or is it something
     the sequence model would have to learn beyond what nearby SNPs alone
     capture?).
  5. Permutation tests (label-shuffling) giving assumption-free empirical
     p-values for both baselines (mean and std) and, if provided, the actual
     Enformer test predictions -- flags results that look strong mostly
     because the test set is small.
  6. If `--predictions-csv` is given: prediction-quality stats (Pearson r,
     Pearson p, R2, plus a permutation p-value) *and* a predicted-vs-true
     scatter plot for *both* heads -- `mu` (mean expression) and `sigma`
     (cell-cell std) -- on the test set, since "amazing results" usually
     refers to `mu` but the whole point of this POC is also `sigma`.

Example:
    python src/diagnose.py \\
        --gene ENSG00000198502 --cell-type "memory B cell" \\
        --h5ad-path data/onek1k_cellxgene_standardized.h5ad \\
        --vcf-dir /path/to/genotypes --gtf /path/to/gencode.gtf.gz \\
        --predictions-csv results/ACTB_memoryBcell/predictions_test_final.csv \\
        --out-dir results/ACTB_memoryBcell/diagnostics
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, skew
from sklearn.linear_model import ElasticNet, ElasticNetCV

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.preprocess import DonorTable, build_donor_table, load_gene_celltype_cells, split_donors
from src.genome import (
    GeneAnnotation,
    GeneRecord,
    VCFGenotypeReader,
    chrom_for_gene,
    get_tss_window,
    onek1k_sample_id_from_donor_id,
)
from src.metrics import CorrelationResult, pseudobulk_correlation, r2_score, sigma_calibration_correlation

# GRCh38 coordinates of the classical MHC region (chr6), 0-based-inclusive-ish;
# exact boundaries vary slightly by source, this is the commonly used span.
MHC_REGION_GRCH38 = {"chrom": "6", "start": 28_477_797, "end": 33_448_354}


def is_constant(values: np.ndarray, atol: float = 1e-12) -> bool:
    """True if `values` has (numerically) zero variance.

    This is a real scenario here, not just an edge case: e.g. a gene with no
    detected expression at all in a cell type, or a cell type where almost
    every donor has exactly 1 cell (so empirical std is 0 for all of them).
    Correlation is undefined in that case, and several downstream
    computations (elastic net's alpha-path/CV selection, permutation nulls)
    become ill-posed or silently misleading rather than erroring, so this is
    checked explicitly everywhere it matters instead of relying on NaN
    propagation.
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


# --------------------------------------------------------------------------- #
# Expression-distribution diagnostics
# --------------------------------------------------------------------------- #


@dataclass
class DistributionSummary:
    n: int
    mean: float
    median: float
    std: float
    cv: float
    skewness: float
    min: float
    max: float
    p10: float
    p90: float
    frac_le_threshold: float


def summarize_distribution(values: np.ndarray, near_zero_threshold: float = 0.05) -> DistributionSummary:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std())
    return DistributionSummary(
        n=len(values),
        mean=mean,
        median=float(np.median(values)),
        std=std,
        cv=(std / mean) if mean != 0 else float("nan"),
        skewness=float(skew(values)) if len(values) > 2 and std > 0 else float("nan"),
        min=float(values.min()),
        max=float(values.max()),
        p10=float(np.percentile(values, 10)),
        p90=float(np.percentile(values, 90)),
        frac_le_threshold=float(np.mean(values <= near_zero_threshold)),
    )


def n_cells_confound_check(table: DonorTable) -> Dict[str, float]:
    """Checks whether donor cell counts confound the pseudobulk targets.

    A donor's empirical cell-cell std is estimated from only `n_cells`
    samples, so donors with few cells will have systematically noisier (often
    inflated) std *estimates* even with identical true biological
    variability. If `sigma` predictions end up correlating with `n_cells`
    rather than with true variability, that would show up here too.
    """
    donor_ids = table.donor_ids
    n_cells = np.array([table.n_cells[d] for d in donor_ids], dtype=np.float64)
    pb_std = np.array([table.pseudobulk_std[d] for d in donor_ids], dtype=np.float64)
    pb_mean = np.array([table.pseudobulk_mean[d] for d in donor_ids], dtype=np.float64)

    r_std, p_std = safe_pearsonr(n_cells, pb_std)
    r_mean, p_mean = safe_pearsonr(n_cells, pb_mean)
    return {
        "n_donors": len(donor_ids),
        "n_cells_median": float(np.median(n_cells)),
        "n_cells_min": float(n_cells.min()),
        "n_cells_max": float(n_cells.max()),
        "corr_ncells_vs_pseudobulk_std": r_std,
        "corr_ncells_vs_pseudobulk_std_p": p_std,
        "corr_ncells_vs_pseudobulk_mean": r_mean,
        "corr_ncells_vs_pseudobulk_mean_p": p_mean,
    }


def is_in_mhc_region(gene: GeneRecord) -> bool:
    chrom = gene.chrom.lstrip("chr")
    return chrom == MHC_REGION_GRCH38["chrom"] and MHC_REGION_GRCH38["start"] <= gene.tss <= MHC_REGION_GRCH38["end"]


# --------------------------------------------------------------------------- #
# Cheap linear baseline (elastic net on raw SNP dosages)
# --------------------------------------------------------------------------- #


class ElasticNetBaseline:
    """Elastic net regression (standardized features) on raw SNP dosages.

    Elastic net (L1 + L2 penalty) is the standard choice for exactly this
    kind of "predict a trait from many, often correlated (LD-linked) SNPs"
    problem -- it's the model behind PrediXcan/TWAS-style genetic prediction
    of expression. The L1 term encourages sparsity (a handful of tagging
    SNPs, not all of them with tiny weights) and the L2 term keeps it stable
    when SNPs are highly correlated, which is the norm within a ~50kb window.

    `alpha`/`l1_ratio` are selected once via `ElasticNetCV`'s internal
    k-fold CV (see `fit_cv`), then reused as *fixed* hyperparameters for
    plain (non-CV) refits inside the permutation-test loop, since re-running
    a full CV grid search per permutation would be far too slow.
    """

    def __init__(self, alpha: float, l1_ratio: float, max_iter: int = 10_000):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNetBaseline":
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0)
        self.x_std[self.x_std == 0] = 1.0
        x_scaled = (X - self.x_mean) / self.x_std

        self._model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=self.max_iter)
        self._model.fit(x_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_scaled = (X - self.x_mean) / self.x_std
        return self._model.predict(x_scaled)

    @staticmethod
    def fit_cv(
        X: np.ndarray, y: np.ndarray, l1_ratios: Sequence[float], cv: int, seed: int, max_iter: int = 10_000
    ) -> "ElasticNetBaseline":
        """Selects `alpha` (via ElasticNetCV's automatic path) and `l1_ratio` via internal k-fold CV."""
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
    and each surviving variant's alt-allele frequency (for reporting).
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


def top_variant_correlations(
    dosage: np.ndarray, positions: List[int], ref_alt: List[Tuple[str, str]], allele_freq: np.ndarray, y: np.ndarray, top_k: int = 10
) -> pd.DataFrame:
    if is_constant(y):
        # Nothing for any variant to explain -- avoid a ConstantInputWarning per variant.
        return pd.DataFrame()
    rows = []
    for j in range(dosage.shape[1]):
        col = dosage[:, j]
        if is_constant(col):
            continue
        r, p = pearsonr(col, y)
        rows.append(
            {
                "position_0based": positions[j],
                "ref": ref_alt[j][0],
                "alt": ref_alt[j][1],
                "alt_allele_freq": float(allele_freq[j]),
                "pearson_r": float(r),
                "pearson_p": float(p),
            }
        )
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.reindex(df["pearson_r"].abs().sort_values(ascending=False).index).head(top_k).reset_index(drop=True)


def fit_linear_baseline(
    dosage: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    test_idx: np.ndarray,
    l1_ratios: Sequence[float],
    cv: int,
    seed: int,
) -> Dict:
    """Fits the elastic net baseline (alpha/l1_ratio selected via internal CV on `fit_idx`) and evaluates on `test_idx`.

    `fit_idx` is expected to be train+val donors combined: `ElasticNetCV`
    does its own internal k-fold cross-validation to pick hyperparameters, so
    a separate held-out val split (as the hand-rolled ridge baseline used)
    isn't needed here -- using train+val together just gives the CV more
    data to work with. `test_idx` remains untouched until final evaluation.
    """
    if is_constant(y[fit_idx]):
        return {
            "skipped": True,
            "reason": "training-set targets are constant across donors -- nothing for a linear model to predict",
        }

    model = ElasticNetBaseline.fit_cv(dosage[fit_idx], y[fit_idx], l1_ratios, cv, seed)
    test_pred = model.predict(dosage[test_idx])
    r, p = safe_pearsonr(y[test_idx], test_pred)
    return {
        "best_alpha": float(model.alpha),
        "best_l1_ratio": float(model.l1_ratio),
        "test_pearson_r": r,
        "test_pearson_p": p,
        "test_r2": r2_score(y[test_idx], test_pred),
        "test_pred": test_pred,
        "model": model,
    }


# --------------------------------------------------------------------------- #
# Permutation tests
# --------------------------------------------------------------------------- #


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
    """Null: shuffle *fit-set* (train+val) labels only (breaks genotype-phenotype pairing
    for fitting), keep the real test labels fixed. Answers: could a model fit on this many
    donors and this many variants achieve this test-set performance by chance alone, with no
    real genotype-phenotype relationship? Important because elastic net with many variants and
    few donors can overfit and "explain" noise.

    Uses a *fixed* (alpha, l1_ratio) -- selected once outside this function via
    `ElasticNetBaseline.fit_cv` on the real data -- rather than re-running a full CV grid
    search per permutation, which would be far too slow at `n_permutations` in the thousands.

    Returns (nan, empty array, nan) if either the fit-set or test targets are
    constant -- there is nothing for the model to fit, or nothing to
    evaluate against, so R2 is undefined and a p-value would be meaningless
    (comparisons against NaN silently evaluate to False, which would
    otherwise produce a spuriously "significant" p-value of 0.0).
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


def permutation_test_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, n_permutations: int, rng: np.random.RandomState
) -> Tuple[float, np.ndarray, float]:
    """Null: random pairing between a fixed set of predictions and true values.

    Model-agnostic significance test for a reported correlation given only
    `len(y_true)` samples -- doesn't need retraining, works directly off a
    `predictions_test_final.csv` from `train.py`. Complements the parametric
    p-value already reported by `train.py`/`src/metrics.py` (which assumes
    bivariate normality) with an assumption-free version.

    Returns (nan, empty array, nan) if either side is constant -- e.g. every
    test donor happens to share the same true pseudobulk mean, or the model
    predicted a constant `mu` -- since correlation (and re-pairing it
    `n_permutations` times) is undefined/meaningless in that case.
    """
    if is_constant(y_true) or is_constant(y_pred):
        return float("nan"), np.array([]), float("nan")

    observed_r, _ = pearsonr(y_true, y_pred)
    null_r = np.empty(n_permutations)
    for i in range(n_permutations):
        null_r[i] = pearsonr(y_true, rng.permutation(y_pred))[0]
    p_value = float(np.mean(np.abs(null_r) >= abs(observed_r)))
    return float(observed_r), null_r, p_value


# --------------------------------------------------------------------------- #
# Per-target (mu or sigma) baseline orchestration
# --------------------------------------------------------------------------- #


@dataclass
class BaselineDiagnostics:
    target_label: str
    top_variants: pd.DataFrame
    baseline: Dict
    observed_r2: float
    null_r2: np.ndarray
    permutation_p: float
    targets_are_constant: bool


def run_target_baseline_diagnostics(
    dosage: np.ndarray,
    positions: List[int],
    ref_alt: List[Tuple[str, str]],
    allele_freq: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    test_idx: np.ndarray,
    target_label: str,
    l1_ratios: Sequence[float],
    cv: int,
    seed: int,
    n_permutations: int,
    rng: np.random.RandomState,
    top_k: int,
) -> BaselineDiagnostics:
    """Runs top-variant correlations + the elastic-net baseline + its permutation test for one
    target (either the pseudobulk mean/`mu`, or the empirical std/`sigma`), with identical
    logic/printing for both -- so the exact same rigor applied to "can genotype predict mean
    expression" (a standard eQTL question) is also applied to "can genotype predict cell-cell
    variability" (a vQTL-style question), which is the whole extra hypothesis this POC exists to
    test.
    """
    print(f"### Target: {target_label} ###\n")

    if is_constant(y):
        print(
            f"[!] {target_label} is constant across all donors -- there is nothing for any variant "
            "or linear model to explain. Skipping top-variant correlations, the elastic-net "
            "baseline, and its permutation test for this target (all undefined for a constant target)."
        )
        print()
        skipped_baseline = {"skipped": True, "reason": f"{target_label} is constant across all donors"}
        return BaselineDiagnostics(target_label, pd.DataFrame(), skipped_baseline, float("nan"), np.array([]), float("nan"), True)

    top_variants = top_variant_correlations(dosage, positions, ref_alt, allele_freq, y, top_k=top_k)
    print(f"--- Top {len(top_variants)} variants by |correlation| with {target_label} ---")
    print(top_variants.to_string(index=False))
    if len(top_variants) > 0 and top_variants.iloc[0]["pearson_p"] < 1e-4 and abs(top_variants.iloc[0]["pearson_r"]) > 0.5:
        print(
            f"[!] A single variant already strongly correlates with {target_label} -- this looks like "
            "a simple, large-effect QTL rather than something requiring learned sequence context."
        )
    print()

    baseline = fit_linear_baseline(dosage, y, fit_idx, test_idx, l1_ratios, cv, seed)
    if baseline.get("skipped"):
        print(f"--- Elastic-net baseline for {target_label}: skipped ({baseline['reason']}) ---\n")
        return BaselineDiagnostics(target_label, top_variants, baseline, float("nan"), np.array([]), float("nan"), False)

    print(f"--- Elastic-net baseline for {target_label} (raw SNP dosages, same donor split) ---")
    print(
        f"best_alpha={baseline['best_alpha']:.4g} best_l1_ratio={baseline['best_l1_ratio']:.2f} "
        f"test_pearson_r={baseline['test_pearson_r']:.4f} test_pearson_p={baseline['test_pearson_p']:.4g} "
        f"test_r2={baseline['test_r2']:.4f} (n_test={len(test_idx)})"
    )
    print()

    observed_r2, null_r2, perm_p = permutation_test_baseline(
        dosage, y, fit_idx, test_idx, baseline["best_alpha"], baseline["best_l1_ratio"], n_permutations, rng
    )
    if np.isnan(perm_p):
        print(f"--- Permutation test ({target_label}): skipped (constant target in fit-set or test set) ---\n")
    else:
        print(f"--- Permutation test: elastic-net baseline for {target_label} vs. label-shuffled null ---")
        print(
            f"observed test R2={observed_r2:.4f}, null mean={null_r2.mean():.4f} +/- {null_r2.std():.4f}, "
            f"empirical p-value={perm_p:.4g} (n_permutations={n_permutations})"
        )
        print()

    return BaselineDiagnostics(target_label, top_variants, baseline, observed_r2, null_r2, perm_p, False)


# --------------------------------------------------------------------------- #
# Model prediction quality (mu and sigma, from a completed train.py run)
# --------------------------------------------------------------------------- #

STATS_GLOSSARY = (
    "  Pearson r:      linear correlation between predicted and true values, range -1..1 "
    "(1 = perfect positive correlation, 0 = none, -1 = perfect inverse).\n"
    "  Pearson p:      parametric significance of that r, assuming bivariate normality -- "
    "probability of seeing |r| this large by chance if truly uncorrelated; smaller = more significant.\n"
    "  R2:             fraction of the true values' variance explained by the predictions "
    "(1 = perfect, 0 = no better than always predicting the mean, <0 = worse than that).\n"
    "  permutation p:  assumption-free version of Pearson p -- fraction of random re-pairings of "
    "the same predictions/true values that correlate at least as strongly; more robust than the "
    "parametric p-value when n is small or non-normal."
)


@dataclass
class ModelEvalResult:
    corr: CorrelationResult
    permutation_p: float


def evaluate_model_predictions(
    pred_df: pd.DataFrame, n_permutations: int, rng: np.random.RandomState
) -> Tuple[ModelEvalResult, ModelEvalResult]:
    """Prediction-quality stats for both model heads on a `predictions_test_final.csv`-style df.

    `mu` is evaluated against the true pseudobulk mean, `sigma` against the true empirical
    cell-cell std -- exactly the two correlations `train.py` optimizes for and reports each
    epoch, recomputed here so they can be examined/plotted independently of a training run.
    """
    y_true_mean = pred_df["y_true_pseudobulk_mean"].to_numpy()
    y_pred_mu = pred_df["y_pred_mu"].to_numpy()
    y_true_std = pred_df["y_true_empirical_std"].to_numpy()
    y_pred_sigma = pred_df["y_pred_sigma"].to_numpy()

    mu_corr = pseudobulk_correlation(y_pred_mu, y_true_mean)
    sigma_corr = sigma_calibration_correlation(y_pred_sigma, y_true_std)
    _, _, mu_perm_p = permutation_test_predictions(y_true_mean, y_pred_mu, n_permutations, rng)
    _, _, sigma_perm_p = permutation_test_predictions(y_true_std, y_pred_sigma, n_permutations, rng)

    return ModelEvalResult(mu_corr, mu_perm_p), ModelEvalResult(sigma_corr, sigma_perm_p)


def make_model_evaluation_plots(out_dir: str, pred_df: pd.DataFrame, mu_eval: ModelEvalResult, sigma_eval: ModelEvalResult) -> None:
    """Predicted-vs-true scatter plots (test set) for both `mu` and `sigma`, each annotated
    with Pearson r/p, R2, and n -- the direct visual answer to "how well does the model
    actually predict mean expression, and separately, cell-cell variability?"
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plots] matplotlib not installed, skipping model-evaluation plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    def _scatter_with_stats(ax, y_true: np.ndarray, y_pred: np.ndarray, title: str, xlabel: str, ylabel: str, result: ModelEvalResult) -> None:
        if len(y_true) == 0:
            ax.axis("off")
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            return
        ax.scatter(y_true, y_pred, s=16, alpha=0.65, edgecolors="none")
        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        corr = result.corr
        stats_text = (
            f"pearson r = {corr.pearson_r:.3f}\npearson p = {corr.pearson_p:.3g}\n"
            f"R2 = {corr.r2:.3f}\npermutation p = {result.permutation_p:.3g}\nn = {corr.n}"
        )
        ax.text(
            0.03, 0.97, stats_text, transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )
        ax.legend(loc="lower right", fontsize=8)

    _scatter_with_stats(
        axes[0],
        pred_df["y_true_pseudobulk_mean"].to_numpy(),
        pred_df["y_pred_mu"].to_numpy(),
        "mu: predicted vs. true pseudobulk mean (test set)",
        "true pseudobulk mean",
        "predicted mu",
        mu_eval,
    )
    _scatter_with_stats(
        axes[1],
        pred_df["y_true_empirical_std"].to_numpy(),
        pred_df["y_pred_sigma"].to_numpy(),
        "sigma: predicted vs. true empirical std (test set)",
        "true empirical std",
        "predicted sigma",
        sigma_eval,
    )

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model_predictions_vs_truth.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plots] saved to {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--gene", required=True)
    data_group.add_argument("--cell-type", required=True)
    data_group.add_argument("--h5ad-path", required=True)
    data_group.add_argument("--min-cells-per-donor", type=int, default=5)
    data_group.add_argument("--val-frac", type=float, default=0.15)
    data_group.add_argument("--test-frac", type=float, default=0.15)
    data_group.add_argument("--seed", type=int, default=0, help="Must match the --seed used for the train.py run being diagnosed")

    genome_group = parser.add_argument_group("genome")
    genome_group.add_argument("--vcf-dir", required=True)
    genome_group.add_argument("--vcf-filename-template", default="{chrom}.vcf.gz")
    genome_group.add_argument("--vcf-chrom-style", choices=["chr", "no_chr"], default=None)
    genome_group.add_argument("--gtf", required=True)
    genome_group.add_argument("--seq-len", type=int, default=49152, help="Window size; should match the train.py run being diagnosed")
    genome_group.add_argument("--vcf-sample-id-scheme", choices=["onek1k", "identity"], default="onek1k")
    genome_group.add_argument("--vcf-sample-id-prefix", default="OneK1K_")
    genome_group.add_argument("--donor-id-map", default=None)
    genome_group.add_argument("--max-missing-frac", type=float, default=0.2, help="Drop variants missing in more than this fraction of donors")

    baseline_group = parser.add_argument_group("baseline")
    baseline_group.add_argument(
        "--l1-ratios",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        help="ElasticNetCV l1_ratio grid (0=pure ridge, 1=pure lasso); alpha is selected automatically along its regularization path",
    )
    baseline_group.add_argument("--cv", type=int, default=5, help="Number of CV folds for ElasticNetCV's internal hyperparameter search")
    baseline_group.add_argument("--top-k-variants", type=int, default=10)
    baseline_group.add_argument("--n-permutations", type=int, default=2000)

    parser.add_argument("--predictions-csv", default=None, help="predictions_test_final.csv from a completed train.py run, for direct comparison")
    parser.add_argument("--out-dir", default=None, help="If set, saves a summary CSV/JSON and diagnostic plots here")

    return parser.parse_args()


def load_donor_id_map(path: Optional[str]) -> Dict[str, str]:
    if path is None:
        return {}
    df = pd.read_csv(path, header=None, names=["h5ad_donor_id", "vcf_sample_id"])
    return dict(zip(df["h5ad_donor_id"].astype(str), df["vcf_sample_id"].astype(str)))


def make_plots(
    out_dir: str,
    table: DonorTable,
    dosage: np.ndarray,
    positions: List[int],
    y_mu: np.ndarray,
    y_sigma: np.ndarray,
    mu_result: "BaselineDiagnostics",
    sigma_result: "BaselineDiagnostics",
) -> None:
    """2x3 grid: distribution/confound diagnostics on top, mu-baseline and sigma-baseline
    diagnostics side by side below -- so the "can a simple linear model predict this?" check
    is visually comparable between mean expression and cell-cell variability.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plots] matplotlib not installed, skipping plots")
        return

    def _scatter_top_variant(ax, top_variants: pd.DataFrame, y: np.ndarray, ylabel: str) -> None:
        if len(top_variants) == 0 or dosage.shape[1] == 0:
            ax.axis("off")
            ax.text(0.5, 0.5, "no usable variants\nor constant target", ha="center", va="center")
            return
        top_pos = top_variants.iloc[0]["position_0based"]
        j = positions.index(top_pos)
        ax.scatter(dosage[:, j], y, s=10, alpha=0.6)
        ax.set_title(f"Top variant (pos {top_pos}) dosage vs. {ylabel}")
        ax.set_xlabel("alt-allele dosage")
        ax.set_ylabel(ylabel)

    def _permutation_hist(ax, null_r2: np.ndarray, observed_r2: float, title: str) -> None:
        if len(null_r2) == 0 or np.isnan(observed_r2):
            ax.axis("off")
            ax.text(0.5, 0.5, "permutation test skipped\n(constant target)", ha="center", va="center")
            return
        ax.hist(null_r2, bins=40, alpha=0.7, label="permutation null")
        ax.axvline(observed_r2, color="red", label="observed")
        ax.set_title(title)
        ax.legend()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0, 0].hist(list(table.pseudobulk_mean.values()), bins=30)
    axes[0, 0].set_title("Pseudobulk mean expression across donors")
    axes[0, 0].set_xlabel("normalized expression")

    axes[0, 1].scatter([table.n_cells[d] for d in table.donor_ids], [table.pseudobulk_std[d] for d in table.donor_ids], s=10, alpha=0.6)
    axes[0, 1].set_title("n_cells vs. empirical cell-cell std (confound check)")
    axes[0, 1].set_xlabel("n_cells")
    axes[0, 1].set_ylabel("pseudobulk std")

    _scatter_top_variant(axes[0, 2], mu_result.top_variants, y_mu, "pseudobulk mean expression")
    _permutation_hist(axes[1, 0], mu_result.null_r2, mu_result.observed_r2, "mu elastic-net baseline: permutation test (test R2)")
    _scatter_top_variant(axes[1, 1], sigma_result.top_variants, y_sigma, "empirical cell-cell std")
    _permutation_hist(axes[1, 2], sigma_result.null_r2, sigma_result.observed_r2, "sigma elastic-net baseline: permutation test (test R2)")

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "diagnostics.png"), dpi=150)
    plt.close(fig)
    print(f"[plots] saved to {os.path.join(out_dir, 'diagnostics.png')}")


def main() -> None:
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    print(f"=== Diagnostics for gene={args.gene} cell_type='{args.cell_type}' ===\n")

    # 1. Expression distribution
    cell_data = load_gene_celltype_cells(args.h5ad_path, args.gene, args.cell_type)
    dropout_rate = float(np.mean(cell_data.raw_count == 0))
    table = build_donor_table(cell_data, min_cells_per_donor=args.min_cells_per_donor)

    pb_mean_summary = summarize_distribution(np.array(list(table.pseudobulk_mean.values())))
    pb_std_summary = summarize_distribution(np.array(list(table.pseudobulk_std.values())))
    confound = n_cells_confound_check(table)

    print("--- Expression distribution ---")
    print(f"raw dropout rate (fraction of cells with zero count): {dropout_rate:.3f}")
    print(f"pseudobulk mean across donors: {pb_mean_summary}")
    print(f"pseudobulk (within-donor) std across donors: {pb_std_summary}")
    print(f"n_cells / confound checks: {confound}")
    if abs(confound["corr_ncells_vs_pseudobulk_std"]) > 0.3:
        print(
            "  [!] pseudobulk std correlates with n_cells (|r|>0.3) -- part of the sigma "
            "signal may reflect sampling noise from variable cell counts, not pure biology."
        )
    print()

    # 2. MHC/HLA region flag
    annotation = GeneAnnotation(args.gtf)
    gene_record = annotation.get(args.gene)
    if is_in_mhc_region(gene_record):
        print("--- MHC/HLA region flag ---")
        print(
            f"[!] {args.gene} (chr{gene_record.chrom.lstrip('chr')}:{gene_record.tss}) lies within the classical "
            f"MHC region (chr6:{MHC_REGION_GRCH38['start']}-{MHC_REGION_GRCH38['end']}, GRCh38). Genes here "
            "(especially HLA class I/II genes) are known for unusually large, simple genetic effects on "
            "expression -- including structural gene presence/absence tied to a nearby paralogue's haplotype, "
            "and long, high-LD haplotype blocks. Strong performance here may reflect the model picking up one "
            "of these large, well-tagged effects rather than learning subtle regulatory grammar. Consider "
            "re-running this diagnostic (and training) on a non-MHC gene as a comparison point."
        )
        print()

    # 3-5. Genotype-based diagnostics: top variants + elastic-net baseline, for mu AND sigma
    vcf_reader = VCFGenotypeReader(args.vcf_dir, filename_template=args.vcf_filename_template)
    donor_id_map = load_donor_id_map(args.donor_id_map)
    sample_id_fn = (
        (lambda d: onek1k_sample_id_from_donor_id(d, prefix=args.vcf_sample_id_prefix))
        if args.vcf_sample_id_scheme == "onek1k"
        else (lambda d: d)
    )
    sample_ids = [donor_id_map.get(d, sample_id_fn(d)) for d in table.donor_ids]

    chrom = chrom_for_gene(gene_record, args.vcf_chrom_style)
    start, end = get_tss_window(gene_record, args.seq_len)
    positions, ref_alt, raw_dosage = vcf_reader.get_dosage_matrix_in_region(chrom, start, end, sample_ids)
    dosage, positions, ref_alt, allele_freq = impute_and_filter_dosage(raw_dosage, positions, ref_alt, args.max_missing_frac)

    y_mu = np.array([table.pseudobulk_mean[d] for d in table.donor_ids], dtype=np.float64)
    y_sigma = np.array([table.pseudobulk_std[d] for d in table.donor_ids], dtype=np.float64)

    print("--- Genotype window ---")
    print(f"window: {chrom}:{start}-{end} ({args.seq_len} bp, TSS-centered)")
    print(f"variants found: {len(raw_dosage[0]) if raw_dosage.ndim == 2 else 0} raw -> {dosage.shape[1]} after missingness/monomorphic filtering")
    print()

    if dosage.shape[1] == 0:
        print("[!] No usable variants in this window -- cannot run the elastic-net baseline or permutation tests.")
        return

    split = split_donors(table.donor_ids, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)
    donor_to_row = {d: i for i, d in enumerate(table.donor_ids)}
    train_idx = np.array([donor_to_row[d] for d in split.train])
    val_idx = np.array([donor_to_row[d] for d in split.val])
    test_idx = np.array([donor_to_row[d] for d in split.test])
    # ElasticNetCV does its own internal k-fold CV for hyperparameter selection,
    # so train+val are combined into a single fit-set (see `fit_linear_baseline`).
    fit_idx = np.concatenate([train_idx, val_idx])

    mu_result = run_target_baseline_diagnostics(
        dosage, positions, ref_alt, allele_freq, y_mu, fit_idx, test_idx,
        "pseudobulk mean expression (mu)", args.l1_ratios, args.cv, args.seed, args.n_permutations, rng, args.top_k_variants,
    )
    sigma_result = run_target_baseline_diagnostics(
        dosage, positions, ref_alt, allele_freq, y_sigma, fit_idx, test_idx,
        "empirical cell-cell std (sigma)", args.l1_ratios, args.cv, args.seed, args.n_permutations, rng, args.top_k_variants,
    )

    if not mu_result.targets_are_constant and not sigma_result.targets_are_constant:
        mu_skipped, sigma_skipped = mu_result.baseline.get("skipped"), sigma_result.baseline.get("skipped")
        print("--- Can a simple linear model (elastic net) predict sigma as well as it predicts mu? ---")
        if mu_skipped or sigma_skipped:
            print("skipped (one or both baselines could not be fit -- see above)")
        else:
            mu_r2, mu_p = mu_result.baseline["test_r2"], mu_result.permutation_p
            sigma_r2, sigma_p = sigma_result.baseline["test_r2"], sigma_result.permutation_p
            print(f"mu (mean expression):    elastic-net test R2={mu_r2:.4f} (permutation p={mu_p:.4g})")
            print(f"sigma (cell-cell std):   elastic-net test R2={sigma_r2:.4f} (permutation p={sigma_p:.4g})")
            if not np.isnan(sigma_p) and sigma_p < 0.1 and sigma_r2 >= mu_r2 - 0.05:
                print(
                    "[!] Local genotype predicts cell-cell variability about as well as it predicts mean "
                    "expression -- evidence of a genetic (vQTL-like) component to expression variability at "
                    "this locus, not just measurement/sampling noise. This would support sigma being a "
                    "biologically learnable target here, not just noise for Enformer to fit."
                )
            elif np.isnan(sigma_p) or sigma_p >= 0.1:
                print(
                    "Local genotype does not significantly predict cell-cell variability here -- sigma may be "
                    "driven more by non-genetic/stochastic factors (or by the n_cells sampling-noise confound "
                    "flagged above) than by nearby SNPs, at least at the level a simple linear model can detect."
                )
            else:
                print(
                    "Local genotype predicts mu much better than sigma -- consistent with mean expression being "
                    "under stronger simple local genetic control than cell-cell variability at this locus."
                )
        print()

    predictions_summary = None
    mu_eval = sigma_eval = None
    pred_df = None
    if args.predictions_csv is not None:
        pred_df = pd.read_csv(args.predictions_csv)
        y_true_mean = pred_df["y_true_pseudobulk_mean"].to_numpy()
        y_true_std = pred_df["y_true_empirical_std"].to_numpy()

        print("--- Model prediction quality: mu and sigma (test set, from --predictions-csv) ---")
        print(STATS_GLOSSARY)
        if is_constant(y_true_mean):
            print("[!] note: true pseudobulk means in --predictions-csv are constant -- mu stats are undefined (nan)")
        if is_constant(y_true_std):
            print("[!] note: true empirical stds in --predictions-csv are constant -- sigma stats are undefined (nan)")

        mu_eval, sigma_eval = evaluate_model_predictions(pred_df, args.n_permutations, rng)
        print(
            f"mu (mean expression):  pearson_r={mu_eval.corr.pearson_r:.4f} pearson_p={mu_eval.corr.pearson_p:.4g} "
            f"r2={mu_eval.corr.r2:.4f} permutation_p={mu_eval.permutation_p:.4g} (n={mu_eval.corr.n})"
        )
        print(
            f"sigma (cell-cell std):  pearson_r={sigma_eval.corr.pearson_r:.4f} pearson_p={sigma_eval.corr.pearson_p:.4g} "
            f"r2={sigma_eval.corr.r2:.4f} permutation_p={sigma_eval.permutation_p:.4g} (n={sigma_eval.corr.n})"
        )
        print()

        model_r, model_r2 = mu_eval.corr.pearson_r, mu_eval.corr.r2
        mu_baseline = mu_result.baseline
        if np.isnan(model_r):
            print("--- Enformer mu vs. elastic-net baseline: skipped (constant true value or prediction) ---\n")
        else:
            print("--- Enformer mu vs. elastic-net baseline ---")
            print(f"Enformer:  test_pearson_r={model_r:.4f} test_r2={model_r2:.4f} (n={mu_eval.corr.n}), permutation p-value={mu_eval.permutation_p:.4g}")
            if mu_baseline.get("skipped"):
                print("Baseline:  skipped (constant pseudobulk target)")
            else:
                print(f"Baseline:  test_pearson_r={mu_baseline['test_pearson_r']:.4f} test_r2={mu_baseline['test_r2']:.4f} (n={len(test_idx)})")
                if abs(model_r) - abs(mu_baseline["test_pearson_r"]) < 0.1:
                    print(
                        "[!] The elastic-net SNP baseline is within ~0.1 Pearson r of the Enformer model. This "
                        "suggests the task may reduce to standard eQTL detection (a few strongly-tagging variants), "
                        "and the deep sequence model may not be learning much beyond that."
                    )
                else:
                    print(
                        "The Enformer model notably outperforms the elastic-net baseline -- more consistent with "
                        "the model using sequence context beyond a simple additive SNP effect."
                    )
            print()

        predictions_summary = {
            "mu_pearson_r": mu_eval.corr.pearson_r,
            "mu_pearson_p": mu_eval.corr.pearson_p,
            "mu_r2": mu_eval.corr.r2,
            "mu_permutation_p": mu_eval.permutation_p,
            "sigma_pearson_r": sigma_eval.corr.pearson_r,
            "sigma_pearson_p": sigma_eval.corr.pearson_p,
            "sigma_r2": sigma_eval.corr.r2,
            "sigma_permutation_p": sigma_eval.permutation_p,
            "n_test": int(mu_eval.corr.n),
        }

    def _summarize_target(result: "BaselineDiagnostics") -> None:
        if result.targets_are_constant:
            print(f"- {result.target_label} is constant across donors -- baseline diagnostics skipped entirely")
            return
        top_r = result.top_variants["pearson_r"].abs().max() if len(result.top_variants) > 0 else None
        print(f"- {result.target_label}: top single-variant |r|=" + (f"{top_r:.4f}" if top_r is not None else "n/a (no usable variants)"))
        if result.baseline.get("skipped"):
            print(f"  elastic-net baseline: skipped ({result.baseline['reason']})")
        else:
            print(f"  elastic-net baseline test R2: {result.baseline['test_r2']:.4f} (permutation p={result.permutation_p:.4g})")

    print("=== Summary / suggested interpretation ===")
    print(f"- n_donors used: {len(table.donor_ids)} (train={len(train_idx)} val={len(val_idx)} test={len(test_idx)})")
    print(f"- in MHC/HLA region: {is_in_mhc_region(gene_record)}")
    _summarize_target(mu_result)
    _summarize_target(sigma_result)
    if predictions_summary is not None:
        print(
            f"- Enformer mu test R2 (from --predictions-csv): {predictions_summary['mu_r2']:.4f} "
            f"(permutation p={predictions_summary['mu_permutation_p']:.4g})"
        )
        print(
            f"- Enformer sigma test R2 (from --predictions-csv): {predictions_summary['sigma_r2']:.4f} "
            f"(permutation p={predictions_summary['sigma_permutation_p']:.4g})"
        )
    print(
        "If the linear baseline alone explains most of the reported performance, if a single variant "
        "dominates, and/or if the test set is small (~20-30 donors), treat 'amazing' results with caution -- "
        "try a gene outside the MHC region and/or one with a weaker known eQTL for a harder comparison point."
    )

    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)
        mu_result.top_variants.to_csv(os.path.join(args.out_dir, "top_variants_mu.csv"), index=False)
        sigma_result.top_variants.to_csv(os.path.join(args.out_dir, "top_variants_sigma.csv"), index=False)

        def _baseline_summary(result: "BaselineDiagnostics") -> Dict:
            return {
                "targets_are_constant": result.targets_are_constant,
                "baseline": {k: v for k, v in result.baseline.items() if k not in ("test_pred", "model")},
                "baseline_permutation_p_value": result.permutation_p,
            }

        summary = {
            "gene": args.gene,
            "cell_type": args.cell_type,
            "dropout_rate": dropout_rate,
            "pseudobulk_mean_distribution": asdict(pb_mean_summary),
            "pseudobulk_std_distribution": asdict(pb_std_summary),
            "n_cells_confound": confound,
            "in_mhc_region": is_in_mhc_region(gene_record),
            "n_variants_in_window": int(dosage.shape[1]),
            "mu_baseline": _baseline_summary(mu_result),
            "sigma_baseline": _baseline_summary(sigma_result),
            "predictions_comparison": predictions_summary,
        }
        pd.Series(summary).to_json(os.path.join(args.out_dir, "diagnostics_summary.json"), indent=2)
        make_plots(args.out_dir, table, dosage, positions, y_mu, y_sigma, mu_result, sigma_result)
        if pred_df is not None:
            make_model_evaluation_plots(args.out_dir, pred_df, mu_eval, sigma_eval)


if __name__ == "__main__":
    main()
