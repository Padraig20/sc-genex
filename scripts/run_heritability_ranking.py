"""CLI: rank protein-coding autosomal genes by PrediXcan-style heritability
for one or all cell types, and select the top-N genes to train on.

This is a prerequisite step for `src/train.py`: the top-N gene list it writes
(`--top-genes-csv`) is the training/eval gene universe, and the underlying
per-gene rankings (`--out-csv`) are reused by `src/evaluate.py` for the final
model-vs-PrediXcan comparison.

Example:
    python scripts/run_heritability_ranking.py \\
        --h5ad-path data/onek1k_cellxgene_standardized.h5ad \\
        --cell-type "naive B cell" \\
        --vcf-dir /path/to/genotypes --gtf /path/to/gencode.gtf.gz \\
        --out-dir results/heritability/naive_B_cell \\
        --top-n 1000 --n-workers 8
"""
from __future__ import annotations

import argparse
import os
import re
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from data.preprocess import get_celltype_donor_ids
from src.heritability import rank_genes_by_heritability, select_top_genes
from src.splits import split_donors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--h5ad-path", required=True)
    cell_type_group = data_group.add_mutually_exclusive_group(required=True)
    cell_type_group.add_argument("--cell-type", help="Exact `obs['cell_type']` value, e.g. 'naive B cell'")
    cell_type_group.add_argument(
        "--all-cell-types",
        action="store_true",
        help="Run every non-null cell type in `obs['cell_type']`; each type gets a subdirectory under --out-dir",
    )
    data_group.add_argument("--min-cells-per-donor", type=int, default=5)
    data_group.add_argument(
        "--val-frac", type=float, default=0.10, help="Paper default: ROSMAP individuals split 80%%/10%%/10%% train/val/test"
    )
    data_group.add_argument("--test-frac", type=float, default=0.10)
    data_group.add_argument("--seed", type=int, default=0, help="Must match the --seed used for src/train.py so donor splits line up")

    genome_group = parser.add_argument_group("genome")
    genome_group.add_argument("--vcf-dir", required=True)
    genome_group.add_argument("--vcf-filename-template", default="{chrom}.vcf.gz")
    genome_group.add_argument("--vcf-chrom-style", choices=["chr", "no_chr"], default=None)
    genome_group.add_argument("--gtf", required=True)
    genome_group.add_argument("--window", type=int, default=40_000, help="TSS-centered window size for the PrediXcan regression (paper default: 40kb)")
    genome_group.add_argument("--vcf-sample-id-scheme", choices=["onek1k", "identity"], default="onek1k")
    genome_group.add_argument("--vcf-sample-id-prefix", default="OneK1K_")
    genome_group.add_argument("--donor-id-map", default=None)
    genome_group.add_argument("--max-missing-frac", type=float, default=0.2)

    baseline_group = parser.add_argument_group("baseline")
    baseline_group.add_argument("--maf-min", type=float, default=0.01, help="Minor-allele-frequency threshold (paper default: 0.01)")
    baseline_group.add_argument("--l1-ratio", type=float, default=0.5, help="ElasticNet l1_ratio, fixed (paper default: 0.5)")
    baseline_group.add_argument("--cv", type=int, default=5, help="ElasticNetCV internal fold count for alpha selection")
    baseline_group.add_argument("--n-workers", type=int, default=1, help="Parallel worker processes across genes")

    parser.add_argument("--top-n", type=int, default=1000)
    parser.add_argument("--out-dir", required=True)

    return parser.parse_args()


def load_donor_id_map(path):
    if path is None:
        return {}
    df = pd.read_csv(path, header=None, names=["h5ad_donor_id", "vcf_sample_id"])
    return dict(zip(df["h5ad_donor_id"].astype(str), df["vcf_sample_id"].astype(str)))


def get_all_cell_types(h5ad_path: str) -> list[str]:
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        return sorted(adata.obs["cell_type"].dropna().astype(str).unique().tolist())
    finally:
        adata.file.close()


def cell_type_dirname(cell_type: str) -> str:
    """Return a filesystem-safe, readable directory name."""
    dirname = re.sub(r"[^A-Za-z0-9._-]+", "_", cell_type).strip("._-")
    return dirname or "unnamed_cell_type"


def run_cell_type(args: argparse.Namespace, cell_type: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    donor_ids = get_celltype_donor_ids(args.h5ad_path, cell_type)
    donor_split = split_donors(donor_ids, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)
    print(
        f"[donors] cell_type='{cell_type}': {len(donor_ids)} total -> "
        f"train={len(donor_split.train)} val={len(donor_split.val)} test={len(donor_split.test)}"
    )

    ranking_csv = os.path.join(out_dir, "heritability_ranking.csv")
    ranking_df = rank_genes_by_heritability(
        h5ad_path=args.h5ad_path,
        cell_type=cell_type,
        gtf_path=args.gtf,
        vcf_dir=args.vcf_dir,
        donor_split=donor_split,
        vcf_filename_template=args.vcf_filename_template,
        vcf_chrom_style=args.vcf_chrom_style,
        vcf_sample_id_scheme=args.vcf_sample_id_scheme,
        vcf_sample_id_prefix=args.vcf_sample_id_prefix,
        donor_id_map=load_donor_id_map(args.donor_id_map),
        window=args.window,
        maf_min=args.maf_min,
        l1_ratio=args.l1_ratio,
        cv=args.cv,
        seed=args.seed,
        max_missing_frac=args.max_missing_frac,
        min_cells_per_donor=args.min_cells_per_donor,
        n_workers=args.n_workers,
        out_csv=ranking_csv,
    )

    n_ranked = ranking_df["val_pearson_r"].notna().sum()
    print(f"[heritability] ranked {n_ranked}/{len(ranking_df)} genes successfully (rest skipped -- see 'skipped_reason' column)")

    top_genes_df = select_top_genes(ranking_df, top_n=args.top_n)
    top_genes_csv = os.path.join(out_dir, f"top_{args.top_n}_genes.csv")
    top_genes_df.to_csv(top_genes_csv, index=False)
    print(f"[heritability] wrote top {len(top_genes_df)} genes (by val Pearson r) to {top_genes_csv}")
    print(top_genes_df.head(10).to_string(index=False))


def main() -> None:
    args = parse_args()

    if not args.all_cell_types:
        run_cell_type(args, args.cell_type, args.out_dir)
        return

    cell_types = get_all_cell_types(args.h5ad_path)
    print(f"[heritability] found {len(cell_types)} cell types")
    failures = []
    for index, cell_type in enumerate(cell_types, start=1):
        print(f"\n[heritability] cell type {index}/{len(cell_types)}: {cell_type!r}")
        out_dir = os.path.join(args.out_dir, cell_type_dirname(cell_type))
        try:
            run_cell_type(args, cell_type, out_dir)
        except Exception as exc:  # noqa: BLE001 - one unsuitable cell type should not stop the batch
            failures.append((cell_type, str(exc)))
            print(f"[heritability] FAILED for {cell_type!r}: {exc}")

    print(f"\n[heritability] completed {len(cell_types) - len(failures)}/{len(cell_types)} cell types")
    if failures:
        print("[heritability] failed cell types:")
        for cell_type, reason in failures:
            print(f"  - {cell_type}: {reason}")


if __name__ == "__main__":
    main()
