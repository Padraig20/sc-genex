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
  4. A "cheap" linear-eQTL baseline: ridge regression on raw SNP dosages in
     that same window, evaluated on the *same* donor train/val/test split
     `train.py` would use. If this baseline's test R2/Pearson r is close to
     Enformer's (pass `--predictions-csv` to compare directly), the deep
     sequence model likely isn't adding much.
  5. Permutation tests (label-shuffling) giving assumption-free empirical
     p-values for both the cheap baseline and, if provided, the actual
     Enformer test predictions -- flags results that look strong mostly
     because the test set is small.

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
from src.metrics import r2_score

# GRCh38 coordinates of the classical MHC region (chr6), 0-based-inclusive-ish;
# exact boundaries vary slightly by source, this is the commonly used span.
MHC_REGION_GRCH38 = {"chrom": "6", "start": 28_477_797, "end": 33_448_354}


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
        skewness=float(skew(values)) if len(values) > 2 else float("nan"),
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

    def safe_corr(a, b):
        if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0:
            r, p = pearsonr(a, b)
            return float(r), float(p)
        return float("nan"), float("nan")

    r_std, p_std = safe_corr(n_cells, pb_std)
    r_mean, p_mean = safe_corr(n_cells, pb_mean)
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
# Cheap linear-eQTL baseline
# --------------------------------------------------------------------------- #


class RidgeBaseline:
    """Minimal closed-form ridge regression (standardized features, centered target).

    Deliberately not using scikit-learn to keep this POC's dependency surface
    small; the problem size here (dozens to a few hundred donors x variants)
    makes a hand-rolled closed-form solve entirely adequate.
    """

    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeBaseline":
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0)
        self.x_std[self.x_std == 0] = 1.0
        x_scaled = (X - self.x_mean) / self.x_std
        self.y_mean = float(y.mean())

        n_features = x_scaled.shape[1]
        gram = x_scaled.T @ x_scaled + self.alpha * np.eye(n_features)
        self.beta = np.linalg.solve(gram, x_scaled.T @ (y - self.y_mean))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_scaled = (X - self.x_mean) / self.x_std
        return x_scaled @ self.beta + self.y_mean


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
    rows = []
    for j in range(dosage.shape[1]):
        col = dosage[:, j]
        if np.std(col) == 0:
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


def fit_eqtl_baseline(
    dosage: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    alphas: Sequence[float],
) -> Dict:
    """Fits the ridge baseline (selecting alpha on val, or train if val is tiny) and evaluates on test."""
    best_alpha, best_val_r2, best_model = alphas[0], -np.inf, None
    for alpha in alphas:
        model = RidgeBaseline(alpha).fit(dosage[train_idx], y[train_idx])
        eval_idx = val_idx if len(val_idx) >= 3 else train_idx
        val_r2 = r2_score(y[eval_idx], model.predict(dosage[eval_idx]))
        if not np.isnan(val_r2) and val_r2 > best_val_r2:
            best_alpha, best_val_r2, best_model = alpha, val_r2, model

    test_pred = best_model.predict(dosage[test_idx])
    r, p = (pearsonr(y[test_idx], test_pred) if len(test_idx) > 2 else (float("nan"), float("nan")))
    return {
        "best_alpha": best_alpha,
        "selection_val_r2": best_val_r2,
        "test_pearson_r": float(r),
        "test_pearson_p": float(p),
        "test_r2": r2_score(y[test_idx], test_pred),
        "test_pred": test_pred,
        "model": best_model,
    }


# --------------------------------------------------------------------------- #
# Permutation tests
# --------------------------------------------------------------------------- #


def permutation_test_baseline(
    dosage: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
    n_permutations: int,
    rng: np.random.RandomState,
) -> Tuple[float, np.ndarray, float]:
    """Null: shuffle *training* labels only (breaks genotype-phenotype pairing for fitting),
    keep the real test labels fixed. Answers: could a model fit on this many training
    donors and this many variants achieve this test-set performance by chance alone,
    with no real genotype-phenotype relationship? Important because ridge with many
    variants and few donors can overfit and "explain" noise.
    """
    observed_model = RidgeBaseline(alpha).fit(dosage[train_idx], y[train_idx])
    observed_r2 = r2_score(y[test_idx], observed_model.predict(dosage[test_idx]))

    null_r2 = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled_y_train = rng.permutation(y[train_idx])
        model = RidgeBaseline(alpha).fit(dosage[train_idx], shuffled_y_train)
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
    """
    observed_r, _ = pearsonr(y_true, y_pred)
    null_r = np.empty(n_permutations)
    for i in range(n_permutations):
        null_r[i] = pearsonr(y_true, rng.permutation(y_pred))[0]
    p_value = float(np.mean(np.abs(null_r) >= abs(observed_r)))
    return float(observed_r), null_r, p_value


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
    baseline_group.add_argument("--alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0, 1000.0])
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


def make_plots(out_dir: str, table: DonorTable, top_variants: pd.DataFrame, dosage: np.ndarray, positions: List[int], y: np.ndarray, baseline_null: np.ndarray, baseline_observed: float) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plots] matplotlib not installed, skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    axes[0, 0].hist(list(table.pseudobulk_mean.values()), bins=30)
    axes[0, 0].set_title("Pseudobulk mean expression across donors")
    axes[0, 0].set_xlabel("normalized expression")

    axes[0, 1].scatter([table.n_cells[d] for d in table.donor_ids], [table.pseudobulk_std[d] for d in table.donor_ids], s=10, alpha=0.6)
    axes[0, 1].set_title("n_cells vs. empirical cell-cell std (confound check)")
    axes[0, 1].set_xlabel("n_cells")
    axes[0, 1].set_ylabel("pseudobulk std")

    if len(top_variants) > 0 and dosage.shape[1] > 0:
        top_pos = top_variants.iloc[0]["position_0based"]
        j = positions.index(top_pos)
        axes[1, 0].scatter(dosage[:, j], y, s=10, alpha=0.6)
        axes[1, 0].set_title(f"Top variant (pos {top_pos}) dosage vs. pseudobulk mean")
        axes[1, 0].set_xlabel("alt-allele dosage")
        axes[1, 0].set_ylabel("pseudobulk mean expression")
    else:
        axes[1, 0].axis("off")

    axes[1, 1].hist(baseline_null, bins=40, alpha=0.7, label="permutation null")
    axes[1, 1].axvline(baseline_observed, color="red", label="observed")
    axes[1, 1].set_title("Linear-baseline permutation test (test R2)")
    axes[1, 1].legend()

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

    # 3-4. Genotype-based diagnostics: top variants + cheap linear-eQTL baseline
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

    y = np.array([table.pseudobulk_mean[d] for d in table.donor_ids], dtype=np.float64)

    print("--- Genotype window ---")
    print(f"window: {chrom}:{start}-{end} ({args.seq_len} bp, TSS-centered)")
    print(f"variants found: {len(raw_dosage[0]) if raw_dosage.ndim == 2 else 0} raw -> {dosage.shape[1]} after missingness/monomorphic filtering")
    print()

    if dosage.shape[1] == 0:
        print("[!] No usable variants in this window -- cannot run the linear-eQTL baseline or permutation tests.")
        return

    top_variants = top_variant_correlations(dosage, positions, ref_alt, allele_freq, y, top_k=args.top_k_variants)
    print(f"--- Top {len(top_variants)} variants by |correlation| with pseudobulk expression ---")
    print(top_variants.to_string(index=False))
    if len(top_variants) > 0 and top_variants.iloc[0]["pearson_p"] < 1e-4 and abs(top_variants.iloc[0]["pearson_r"]) > 0.5:
        print(
            "[!] A single variant already strongly correlates with expression -- this looks like a "
            "simple, large-effect eQTL rather than something requiring learned sequence context."
        )
    print()

    split = split_donors(table.donor_ids, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)
    donor_to_row = {d: i for i, d in enumerate(table.donor_ids)}
    train_idx = np.array([donor_to_row[d] for d in split.train])
    val_idx = np.array([donor_to_row[d] for d in split.val])
    test_idx = np.array([donor_to_row[d] for d in split.test])

    baseline = fit_eqtl_baseline(dosage, y, train_idx, val_idx, test_idx, args.alphas)
    print("--- Cheap linear-eQTL baseline (ridge on raw SNP dosages, same donor split) ---")
    print(
        f"best_alpha={baseline['best_alpha']} test_pearson_r={baseline['test_pearson_r']:.4f} "
        f"test_pearson_p={baseline['test_pearson_p']:.4g} test_r2={baseline['test_r2']:.4f} (n_test={len(test_idx)})"
    )
    print()

    observed_r2, null_r2, perm_p = permutation_test_baseline(
        dosage, y, train_idx, test_idx, baseline["best_alpha"], args.n_permutations, rng
    )
    print("--- Permutation test: linear baseline vs. train-label-shuffled null ---")
    print(
        f"observed test R2={observed_r2:.4f}, null mean={null_r2.mean():.4f} +/- {null_r2.std():.4f}, "
        f"empirical p-value={perm_p:.4g} (n_permutations={args.n_permutations})"
    )
    print()

    predictions_summary = None
    if args.predictions_csv is not None:
        pred_df = pd.read_csv(args.predictions_csv)
        y_true = pred_df["y_true_pseudobulk_mean"].to_numpy()
        y_pred = pred_df["y_pred_mu"].to_numpy()
        model_r, _ = pearsonr(y_true, y_pred)
        model_r2 = r2_score(y_true, y_pred)
        obs_r, null_r, model_perm_p = permutation_test_predictions(y_true, y_pred, args.n_permutations, rng)

        print("--- Enformer model vs. cheap linear baseline ---")
        print(f"Enformer:  test_pearson_r={model_r:.4f} test_r2={model_r2:.4f} (n={len(y_true)}), permutation p-value={model_perm_p:.4g}")
        print(f"Baseline:  test_pearson_r={baseline['test_pearson_r']:.4f} test_r2={baseline['test_r2']:.4f} (n={len(test_idx)})")
        if abs(model_r) - abs(baseline["test_pearson_r"]) < 0.1:
            print(
                "[!] The cheap linear SNP baseline is within ~0.1 Pearson r of the Enformer model. This "
                "suggests the task may reduce to standard eQTL detection (a few strongly-tagging variants), "
                "and the deep sequence model may not be learning much beyond that."
            )
        else:
            print(
                "The Enformer model notably outperforms the cheap linear baseline -- more consistent with the "
                "model using sequence context beyond a simple additive SNP effect."
            )
        predictions_summary = {
            "model_test_pearson_r": float(model_r),
            "model_test_r2": float(model_r2),
            "model_permutation_p_value": model_perm_p,
        }
        print()

    print("=== Summary / suggested interpretation ===")
    print(f"- n_donors used: {len(table.donor_ids)} (train={len(train_idx)} val={len(val_idx)} test={len(test_idx)})")
    print(f"- in MHC/HLA region: {is_in_mhc_region(gene_record)}")
    print(f"- top single-variant |r|: {top_variants['pearson_r'].abs().max():.4f}" if len(top_variants) > 0 else "- no usable variants")
    print(f"- linear-eQTL baseline test R2: {baseline['test_r2']:.4f} (permutation p={perm_p:.4g})")
    if predictions_summary is not None:
        print(f"- Enformer model test R2 (from --predictions-csv): {predictions_summary['model_test_r2']:.4f} (permutation p={predictions_summary['model_permutation_p_value']:.4g})")
    print(
        "If the linear baseline alone explains most of the reported performance, if a single variant "
        "dominates, and/or if the test set is small (~20-30 donors), treat 'amazing' results with caution -- "
        "try a gene outside the MHC region and/or one with a weaker known eQTL for a harder comparison point."
    )

    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)
        top_variants.to_csv(os.path.join(args.out_dir, "top_variants.csv"), index=False)
        summary = {
            "gene": args.gene,
            "cell_type": args.cell_type,
            "dropout_rate": dropout_rate,
            "pseudobulk_mean_distribution": asdict(pb_mean_summary),
            "pseudobulk_std_distribution": asdict(pb_std_summary),
            "n_cells_confound": confound,
            "in_mhc_region": is_in_mhc_region(gene_record),
            "n_variants_in_window": int(dosage.shape[1]),
            "baseline": {k: v for k, v in baseline.items() if k not in ("test_pred", "model")},
            "baseline_permutation_p_value": perm_p,
            "predictions_comparison": predictions_summary,
        }
        pd.Series(summary).to_json(os.path.join(args.out_dir, "diagnostics_summary.json"), indent=2)
        make_plots(args.out_dir, table, top_variants, dosage, positions, y, null_r2, observed_r2)


if __name__ == "__main__":
    main()
