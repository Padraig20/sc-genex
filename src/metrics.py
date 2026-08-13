"""Evaluation metrics computed at the donor (pseudobulk) level.

Two kinds of checks live here:

1. `pseudobulk_correlation`/`sigma_calibration_correlation`: single-group
   Pearson r/R2 (does predicted `mu` track true pseudobulk mean expression,
   across whatever set of donors is passed in? does predicted `sigma` track
   true empirical cell-cell std?).
2. `grouped_correlation`: pSAGE-net's actual model-selection/reporting
   metric -- **per-gene** Pearson r (correlating predictions to truth across
   *donors*, separately for each gene) and **per-donor** Pearson r
   (correlating across *genes*, separately for each donor), each summarized
   by their median across genes/donors. This is what `src/train.py` computes
   for every cell of the 4-way seen/unseen-gene x seen/unseen-individual
   evaluation matrix, and what `src/evaluate.py` uses for the final
   model-vs-PrediXcan comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


@dataclass
class CorrelationResult:
    pearson_r: float
    pearson_p: float
    r2: float
    n: int


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination: 1 - SS_res / SS_tot."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _correlation(y_true: np.ndarray, y_pred: np.ndarray) -> CorrelationResult:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    n = len(y_true)
    if n < 2:
        return CorrelationResult(pearson_r=float("nan"), pearson_p=float("nan"), r2=float("nan"), n=n)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        # Pearson correlation is undefined when either side is constant --
        # e.g. every donor has the same pseudobulk target (small/degenerate
        # eval set), or -- for sigma calibration -- every donor has exactly 1
        # cell so empirical std is 0 for all of them. Skip rather than let
        # scipy raise a ConstantInputWarning and return nan; R2 is still
        # well-defined unless `y_true` itself is constant (handled inside
        # `r2_score`), so it's still computed.
        return CorrelationResult(pearson_r=float("nan"), pearson_p=float("nan"), r2=r2_score(y_true, y_pred), n=n)
    r, p = pearsonr(y_true, y_pred)
    return CorrelationResult(pearson_r=float(r), pearson_p=float(p), r2=r2_score(y_true, y_pred), n=n)


def pseudobulk_correlation(pred_mu: np.ndarray, true_pseudobulk_mean: np.ndarray) -> CorrelationResult:
    """Pearson r / R2 between predicted mu and true pseudobulk mean, across donors."""
    return _correlation(np.asarray(true_pseudobulk_mean, dtype=np.float64), np.asarray(pred_mu, dtype=np.float64))


def sigma_calibration_correlation(pred_sigma: np.ndarray, empirical_std: np.ndarray) -> CorrelationResult:
    """Pearson r / R2 between predicted sigma and true empirical cell-cell std, across donors."""
    return _correlation(np.asarray(empirical_std, dtype=np.float64), np.asarray(pred_sigma, dtype=np.float64))


def _grouped_correlation_table(df: pd.DataFrame, group_col: str, true_col: str, pred_col: str) -> pd.DataFrame:
    """One `_correlation` per distinct value of `group_col` (e.g. one Pearson r per gene, across donors)."""
    rows = []
    for group_value, group_df in df.groupby(group_col):
        result = _correlation(group_df[true_col].to_numpy(dtype=np.float64), group_df[pred_col].to_numpy(dtype=np.float64))
        rows.append(
            {
                group_col: group_value,
                "pearson_r": result.pearson_r,
                "pearson_p": result.pearson_p,
                "r2": result.r2,
                "n": result.n,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class GroupedCorrelationResult:
    """Per-gene and per-donor Pearson r tables, plus their medians -- pSAGE-net's
    primary reporting metric (Fig 1c/1d: distribution of per-gene Pearson r across
    a set of held-out individuals).
    """

    per_gene: pd.DataFrame  # columns: gene_id, pearson_r, pearson_p, r2, n
    per_donor: pd.DataFrame  # columns: donor_id, pearson_r, pearson_p, r2, n
    median_per_gene_r: float
    median_per_donor_r: float
    n_genes: int
    n_donors: int


def grouped_correlation(
    df: pd.DataFrame,
    gene_col: str = "gene_id",
    donor_col: str = "donor_id",
    true_col: str = "y_true",
    pred_col: str = "y_pred",
) -> GroupedCorrelationResult:
    """Computes per-gene (across donors) and per-donor (across genes) Pearson r.

    `df` must have one row per `(gene, donor)` example with `true_col`/
    `pred_col` columns. Used for every cell of the 4-way seen/unseen-gene x
    seen/unseen-individual evaluation matrix in `src/train.py`: e.g. for the
    "unseen gene / unseen individual" cell, `per_gene` answers "for this
    never-trained-on gene, how well does the model rank never-seen
    individuals by predicted expression?", matching the paper's headline
    evaluation.
    """
    per_gene = _grouped_correlation_table(df, gene_col, true_col, pred_col)
    per_donor = _grouped_correlation_table(df, donor_col, true_col, pred_col)
    median_gene_r = float(np.nanmedian(per_gene["pearson_r"])) if len(per_gene) else float("nan")
    median_donor_r = float(np.nanmedian(per_donor["pearson_r"])) if len(per_donor) else float("nan")
    return GroupedCorrelationResult(
        per_gene=per_gene,
        per_donor=per_donor,
        median_per_gene_r=median_gene_r,
        median_per_donor_r=median_donor_r,
        n_genes=len(per_gene),
        n_donors=len(per_donor),
    )
