"""Final comparison: trained `PSAGEnetSC` vs. PrediXcan (ElasticNet), on the
same held-out test donors -- reproducing the paper's Fig 1c/1d "how much does
a sequence model beat/match PrediXcan" comparison.

For every TRAIN gene ("seen" -- the sequence model actually trained on it),
PrediXcan is *refit* on train+val donors (using more data than the original
heritability-ranking fit in `scripts/run_heritability_ranking.py`, which only
used train donors for a fair train/val split) and evaluated on the same
held-out test donors the sequence model is evaluated on, for an
apples-to-apples comparison. TEST genes ("unseen") have no PrediXcan
counterpart here: PrediXcan doesn't have a "genes it was/wasn't trained on"
concept the way the sequence model does (every gene gets an independent
per-gene fit), so there's no fair "unseen gene" PrediXcan baseline to compare
against -- the sequence model's own per-gene Pearson r is still reported for
these genes, just with `predixcan_pearson_r` left NaN.

Example:
    python src/evaluate.py \\
        --results-dir results/naive_B_cell_psagenet_sc \\
        --h5ad-path data/onek1k_cellxgene_standardized.h5ad --cell-type "naive B cell" \\
        --vcf-dir /path/to/genotypes --genome-fasta /path/to/hg38.fa --gtf /path/to/gencode.gtf.gz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.preprocess import build_multigene_donor_table, compute_population_means, load_celltype_multigene_cells
from src.dataset import PersonalGenomeEvalDataset, PersonalGenomeSequenceBuilder
from src.genome import GeneAnnotation, ReferenceGenome, VCFGenotypeReader, chrom_for_gene, get_tss_window, onek1k_sample_id_from_donor_id
from src.heritability import fit_gene_elasticnet
from src.metrics import grouped_correlation
from src.model import PSAGEnetSC
from src.train import load_donor_id_map, run_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--results-dir", required=True, help="Output dir from src/train.py (has best_model.pt, *_split.csv, model_config.json)")

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--h5ad-path", required=True)
    data_group.add_argument("--cell-type", required=True)
    data_group.add_argument("--min-cells-per-donor", type=int, default=5)

    genome_group = parser.add_argument_group("genome")
    genome_group.add_argument("--vcf-dir", required=True)
    genome_group.add_argument("--vcf-filename-template", default="{chrom}.vcf.gz")
    genome_group.add_argument("--vcf-chrom-style", choices=["chr", "no_chr"], default=None)
    genome_group.add_argument("--genome-fasta", required=True)
    genome_group.add_argument("--gtf", required=True)
    genome_group.add_argument("--vcf-sample-id-scheme", choices=["onek1k", "identity"], default="onek1k")
    genome_group.add_argument("--vcf-sample-id-prefix", default="OneK1K_")
    genome_group.add_argument("--donor-id-map", default=None)

    predixcan_group = parser.add_argument_group("predixcan")
    predixcan_group.add_argument("--window", type=int, default=40_000, help="Must match scripts/run_heritability_ranking.py's --window")
    predixcan_group.add_argument("--maf-min", type=float, default=0.01)
    predixcan_group.add_argument("--l1-ratio", type=float, default=0.5)
    predixcan_group.add_argument("--cv", type=int, default=5)
    predixcan_group.add_argument("--max-missing-frac", type=float, default=0.2)
    predixcan_group.add_argument("--min-fit-donors", type=int, default=10, help="Skip PrediXcan refit for a gene with fewer than this many fit-set donors")
    predixcan_group.add_argument("--min-test-donors", type=int, default=3)

    eval_group = parser.add_argument_group("evaluation")
    eval_group.add_argument("--max-pairs", type=int, default=None, help="Optional cap on (gene, donor) pairs per quadrant (default: no cap)")
    eval_group.add_argument("--eval-batch-size", type=int, default=64)
    eval_group.add_argument("--seed", type=int, default=0)
    eval_group.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def load_split(path: str, id_col: str) -> Dict[str, List[str]]:
    df = pd.read_csv(path)
    return {split: df.loc[df["split"] == split, id_col].astype(str).tolist() for split in df["split"].unique()}


def main() -> None:
    args = parse_args()

    donor_splits = load_split(os.path.join(args.results_dir, "donor_split.csv"), "donor_id")
    gene_splits = load_split(os.path.join(args.results_dir, "gene_split.csv"), "gene_id")
    train_donors = donor_splits.get("train", [])
    val_donors = donor_splits.get("val", [])
    test_donors = donor_splits.get("test", [])
    train_genes = gene_splits.get("train", [])
    test_genes = gene_splits.get("test", [])
    all_genes = train_genes + gene_splits.get("val", []) + test_genes
    print(f"[splits] donors: train={len(train_donors)} val={len(val_donors)} test={len(test_donors)}")
    print(f"[splits] genes: train(seen)={len(train_genes)} test(unseen)={len(test_genes)}")

    with open(os.path.join(args.results_dir, "model_config.json")) as fh:
        model_config = json.load(fh)

    annotation = GeneAnnotation(args.gtf)
    gene_records = {g: annotation.get(g) for g in all_genes}

    cell_data = load_celltype_multigene_cells(args.h5ad_path, args.cell_type, all_genes)
    table = build_multigene_donor_table(cell_data, min_cells_per_donor=args.min_cells_per_donor)
    compute_population_means(table, train_donors)

    reference = ReferenceGenome(args.genome_fasta)
    vcf_reader = VCFGenotypeReader(args.vcf_dir, filename_template=args.vcf_filename_template)
    donor_id_map = load_donor_id_map(args.donor_id_map)
    if args.vcf_sample_id_scheme == "onek1k":
        sample_id_fn = lambda d: onek1k_sample_id_from_donor_id(d, prefix=args.vcf_sample_id_prefix)  # noqa: E731
    else:
        sample_id_fn = lambda d: d  # noqa: E731
    resolve_sample_id = lambda d: donor_id_map.get(d, sample_id_fn(d))  # noqa: E731

    sequence_builder = PersonalGenomeSequenceBuilder(
        reference=reference,
        vcf_reader=vcf_reader,
        genes=gene_records,
        seq_len=model_config["input_length"],
        vcf_chrom_style=args.vcf_chrom_style,
        donor_id_to_sample_id=donor_id_map,
        sample_id_fn=sample_id_fn,
    )

    model = PSAGEnetSC(**model_config)
    model.load_state_dict(torch.load(os.path.join(args.results_dir, "best_model.pt"), map_location=args.device))
    model = model.to(args.device)

    print("[model] evaluating seen genes x test donors, unseen genes x test donors")
    seen_ds = PersonalGenomeEvalDataset(train_genes, test_donors, table, sequence_builder, max_pairs=args.max_pairs, seed=args.seed)
    unseen_ds = PersonalGenomeEvalDataset(test_genes, test_donors, table, sequence_builder, max_pairs=args.max_pairs, seed=args.seed)

    seen_df = run_eval(model, seen_ds, args.device, args.eval_batch_size)
    unseen_df = run_eval(model, unseen_ds, args.device, args.eval_batch_size)

    seen_model_r = grouped_correlation(seen_df, true_col="y_true_mean", pred_col="y_pred_mean").per_gene
    unseen_model_r = grouped_correlation(unseen_df, true_col="y_true_mean", pred_col="y_pred_mean").per_gene
    seen_model_r["gene_split"] = "seen"
    unseen_model_r["gene_split"] = "unseen"
    model_r = pd.concat([seen_model_r, unseen_model_r], ignore_index=True).rename(
        columns={"pearson_r": "model_pearson_r", "pearson_p": "model_pearson_p", "n": "model_n_test_donors"}
    )[["gene_id", "gene_split", "model_pearson_r", "model_pearson_p", "model_n_test_donors"]]

    print(f"[predixcan] refitting {len(train_genes)} seen-gene ElasticNet models on train+val donors, evaluating on test donors")
    fit_donors = train_donors + val_donors
    predixcan_rows = []
    for gene_id in train_genes:
        gene_record = gene_records[gene_id]
        chrom = chrom_for_gene(gene_record, args.vcf_chrom_style)
        start, end = get_tss_window(gene_record, args.window)

        y_fit = np.array([table.pseudobulk_mean.get((gene_id, d), np.nan) for d in fit_donors])
        y_test = np.array([table.pseudobulk_mean.get((gene_id, d), np.nan) for d in test_donors])
        fit_mask, test_mask = ~np.isnan(y_fit), ~np.isnan(y_test)
        if fit_mask.sum() < args.min_fit_donors or test_mask.sum() < args.min_test_donors:
            predixcan_rows.append({"gene_id": gene_id, "predixcan_pearson_r": float("nan"), "predixcan_n_variants": 0})
            continue

        sample_ids_fit = [resolve_sample_id(d) for d, keep in zip(fit_donors, fit_mask) if keep]
        sample_ids_test = [resolve_sample_id(d) for d, keep in zip(test_donors, test_mask) if keep]
        result = fit_gene_elasticnet(
            gene_id,
            chrom,
            start,
            end,
            args.vcf_dir,
            args.vcf_filename_template,
            sample_ids_fit,
            sample_ids_test,
            y_fit[fit_mask],
            y_test[test_mask],
            args.maf_min,
            args.l1_ratio,
            args.cv,
            args.seed,
            args.max_missing_frac,
        )
        predixcan_rows.append(
            {"gene_id": gene_id, "predixcan_pearson_r": result.val_pearson_r, "predixcan_n_variants": result.n_variants}
        )

    predixcan_df = pd.DataFrame(predixcan_rows)
    comparison = model_r.merge(predixcan_df, on="gene_id", how="left")

    out_path = os.path.join(args.results_dir, "model_vs_predixcan.csv")
    comparison.to_csv(out_path, index=False)
    print(f"[evaluate] wrote {out_path}")

    seen_rows = comparison[(comparison["gene_split"] == "seen") & comparison["predixcan_pearson_r"].notna()]
    if len(seen_rows) > 0:
        win_frac = float((seen_rows["model_pearson_r"] > seen_rows["predixcan_pearson_r"]).mean())
        print(
            f"[evaluate] seen genes (n={len(seen_rows)}): median model r={seen_rows['model_pearson_r'].median():.4f}, "
            f"median PrediXcan r={seen_rows['predixcan_pearson_r'].median():.4f}, model wins on {win_frac:.1%} of genes"
        )
    else:
        print("[evaluate] no seen genes had a valid PrediXcan refit -- check --min-fit-donors/--min-test-donors and genotype coverage")

    unseen_rows = comparison[comparison["gene_split"] == "unseen"]
    if len(unseen_rows) > 0:
        print(f"[evaluate] unseen genes (n={len(unseen_rows)}): median model r={unseen_rows['model_pearson_r'].median():.4f} (no PrediXcan baseline)")


if __name__ == "__main__":
    main()
