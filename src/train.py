"""Trains a Variformer-style, single-cell, Gaussian-NLL model for one (gene,
cell_type) pair on OneK1K.

Faithful to Variformer (https://github.com/shirondru/enformer_fine_tuning)
in spirit: personalized, TSS-centered DNA input per donor fine-tunes a
pretrained Enformer. Two POC-specific changes:
  1. Training targets are per-cell (not pseudobulked) expression values --
     see `src/loss.py` for how this is made computationally efficient while
     remaining mathematically equivalent to per-cell training.
  2. The loss is a Gaussian NLL over a learned (mu, sigma) pair per donor,
     instead of MSE over a single value, so the model also learns to predict
     cell-cell variability (a proxy for expression specificity).

Validation/test are always pseudobulk (per-donor mean), per the plan.

Example:
    python src/train.py \\
        --gene ENSG00000075624 --cell-type "naive B cell" \\
        --h5ad-path data/onek1k_cellxgene_standardized.h5ad \\
        --vcf-dir /path/to/genotypes --genome-fasta /path/to/hg38.fa \\
        --gtf /path/to/annotation.gtf.gz --out-dir results/ACTB_naiveB
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.preprocess import prepare_gene_celltype_data
from src.dataset import (
    DonorSequenceBuilder,
    PseudobulkEvalDataset,
    SingleCellDonorDataset,
    collate_donor_batch,
)
from src.genome import GeneAnnotation, ReferenceGenome, VCFGenotypeReader, onek1k_sample_id_from_donor_id
from src.loss import per_cell_gaussian_nll
from src.metrics import pseudobulk_correlation, sigma_calibration_correlation
from src.model import VariformerGNLL
from src.wandb_logger import get_logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--gene", required=True, help="Ensembl gene ID (preferred) or gene symbol")
    data_group.add_argument("--cell-type", required=True, help="Exact `obs['cell_type']` value, e.g. 'naive B cell'")
    data_group.add_argument("--h5ad-path", required=True)
    data_group.add_argument("--min-cells-per-donor", type=int, default=5)
    data_group.add_argument("--val-frac", type=float, default=0.15)
    data_group.add_argument("--test-frac", type=float, default=0.15)

    genome_group = parser.add_argument_group("genome")
    genome_group.add_argument("--vcf-dir", required=True, help="Directory containing one VCF per chromosome")
    genome_group.add_argument("--vcf-filename-template", default="{chrom}.vcf.gz")
    genome_group.add_argument(
        "--vcf-chrom-style",
        choices=["chr", "no_chr"],
        default=None,
        help="Normalize chromosome naming to VCF convention if it differs from the GTF's",
    )
    genome_group.add_argument("--genome-fasta", required=True, help="Indexed reference genome FASTA (e.g. hg38)")
    genome_group.add_argument("--gtf", required=True, help="GTF/GFF gene annotation, used for TSS lookup")
    genome_group.add_argument("--seq-len", type=int, default=49152)
    genome_group.add_argument(
        "--vcf-sample-id-scheme",
        choices=["onek1k", "identity"],
        default="onek1k",
        help=(
            "How to map h5ad donor_id -> genotype-file sample ID. 'onek1k' (default) "
            "implements the confirmed OneK1K convention: donor_id '{X}_{Y}' -> sample "
            "ID '{prefix}{Y}' (e.g. '10_10' -> 'OneK1K_10'). 'identity' assumes they "
            "are the same string. Use --donor-id-map for exceptions to either."
        ),
    )
    genome_group.add_argument(
        "--vcf-sample-id-prefix",
        default="OneK1K_",
        help="Prefix used by --vcf-sample-id-scheme=onek1k (ignored otherwise)",
    )
    genome_group.add_argument(
        "--donor-id-map",
        default=None,
        help=(
            "Optional 2-column CSV (h5ad_donor_id,vcf_sample_id) for donors that don't "
            "follow --vcf-sample-id-scheme; entries here always take precedence"
        ),
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--random-weights", action="store_true", help="Smoke-test only: skip the pretrained Enformer download")
    model_group.add_argument("--freeze-enformer", action="store_true", help="Only train the (mu, sigma) head")
    model_group.add_argument("--finetune-last-n-layers", type=int, default=None)

    train_group = parser.add_argument_group("training")
    train_group.add_argument("--batch-size", type=int, default=8, help="Donors per training step")
    train_group.add_argument("--epochs", type=int, default=20)
    train_group.add_argument("--lr", type=float, default=5e-6)
    train_group.add_argument("--weight-decay", type=float, default=0.0)
    train_group.add_argument("--grad-clip", type=float, default=0.05)
    train_group.add_argument("--eval-every", type=int, default=1)
    train_group.add_argument("--patience", type=int, default=10)
    train_group.add_argument("--seed", type=int, default=0)
    train_group.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train_group.add_argument("--out-dir", default=None)
    train_group.add_argument("--wandb", action="store_true")

    return parser.parse_args()


def load_donor_id_map(path: Optional[str]) -> Dict[str, str]:
    if path is None:
        return {}
    df = pd.read_csv(path, header=None, names=["h5ad_donor_id", "vcf_sample_id"])
    return dict(zip(df["h5ad_donor_id"].astype(str), df["vcf_sample_id"].astype(str)))


def build_datasets(args: argparse.Namespace):
    table, split = prepare_gene_celltype_data(
        h5ad_path=args.h5ad_path,
        gene=args.gene,
        cell_type=args.cell_type,
        min_cells_per_donor=args.min_cells_per_donor,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    annotation = GeneAnnotation(args.gtf)
    gene_record = annotation.get(args.gene)
    reference = ReferenceGenome(args.genome_fasta)
    vcf_reader = VCFGenotypeReader(args.vcf_dir, filename_template=args.vcf_filename_template)
    donor_id_map = load_donor_id_map(args.donor_id_map)

    if args.vcf_sample_id_scheme == "onek1k":
        sample_id_fn = lambda donor_id: onek1k_sample_id_from_donor_id(donor_id, prefix=args.vcf_sample_id_prefix)
    else:
        sample_id_fn = lambda donor_id: donor_id

    sequence_builder = DonorSequenceBuilder(
        reference=reference,
        vcf_reader=vcf_reader,
        gene=gene_record,
        seq_len=args.seq_len,
        vcf_chrom_style=args.vcf_chrom_style,
        donor_id_to_sample_id=donor_id_map,
        sample_id_fn=sample_id_fn,
    )

    train_ds = SingleCellDonorDataset(split.train, table, sequence_builder)
    val_ds = PseudobulkEvalDataset(split.val, table, sequence_builder)
    test_ds = PseudobulkEvalDataset(split.test, table, sequence_builder)

    print(
        f"[data] donors -> train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"(min_cells_per_donor={args.min_cells_per_donor})"
    )
    return train_ds, val_ds, test_ds, gene_record


@torch.no_grad()
def run_pseudobulk_eval(model: VariformerGNLL, dataset: PseudobulkEvalDataset, device: str) -> pd.DataFrame:
    model.eval()
    loader = DataLoader(dataset, batch_size=max(1, len(dataset)), shuffle=False)
    rows: List[dict] = []
    for seqs, pseudobulk_mean, empirical_std, n_cells, donor_ids in loader:
        seqs = seqs.to(device)
        mu, sigma = model(seqs)
        mu = mu.cpu().numpy()
        sigma = sigma.cpu().numpy()
        pseudobulk_mean = pseudobulk_mean.numpy()
        empirical_std = empirical_std.numpy()
        n_cells = n_cells.numpy()
        for i, donor_id in enumerate(donor_ids):
            rows.append(
                {
                    "donor_id": donor_id,
                    "y_true_pseudobulk_mean": float(pseudobulk_mean[i]),
                    "y_true_empirical_std": float(empirical_std[i]),
                    "y_pred_mu": float(mu[i]),
                    "y_pred_sigma": float(sigma[i]),
                    "n_cells": int(n_cells[i]),
                }
            )
    return pd.DataFrame(rows)


def summarize_eval(df: pd.DataFrame, prefix: str) -> Dict[str, float]:
    mean_corr = pseudobulk_correlation(df["y_pred_mu"].to_numpy(), df["y_true_pseudobulk_mean"].to_numpy())
    sigma_corr = sigma_calibration_correlation(df["y_pred_sigma"].to_numpy(), df["y_true_empirical_std"].to_numpy())
    return {
        f"{prefix}/pseudobulk_pearson_r": mean_corr.pearson_r,
        # Parametric p-value (assumes bivariate normality) for the pseudobulk
        # correlation given only `n_donors` samples -- a quick, free sanity
        # check on how surprising the reported correlation really is. See
        # `src/diagnose.py` for a more robust, assumption-free permutation
        # version of this same question, plus other "is this too easy?" checks.
        f"{prefix}/pseudobulk_pearson_p": mean_corr.pearson_p,
        f"{prefix}/pseudobulk_r2": mean_corr.r2,
        f"{prefix}/sigma_calibration_pearson_r": sigma_corr.pearson_r,
        f"{prefix}/sigma_calibration_pearson_p": sigma_corr.pearson_p,
        f"{prefix}/sigma_calibration_r2": sigma_corr.r2,
        f"{prefix}/n_donors": mean_corr.n,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = args.out_dir or os.path.join("results", f"{args.gene}_{args.cell_type}".replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)

    train_ds, val_ds, test_ds, gene_record = build_datasets(args)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_donor_batch
    )

    model = VariformerGNLL(
        random_weights=args.random_weights,
        freeze_enformer=args.freeze_enformer,
        finetune_last_n_layers_only=args.finetune_last_n_layers,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    logger = get_logger(args.wandb, config=vars(args))

    best_metric = -float("inf")
    best_state: Optional[dict] = None
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for seqs, cell_targets, donor_ids in train_loader:
            seqs = seqs.to(args.device)
            cell_targets = [t.to(args.device) for t in cell_targets]

            mu, sigma = model(seqs)
            loss = per_cell_gaussian_nll(mu, sigma, cell_targets)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        log_payload = {"train/loss": train_loss, "epoch": epoch}
        print(f"[epoch {epoch}] train_loss={train_loss:.4f}")

        if epoch % args.eval_every == 0:
            val_df = run_pseudobulk_eval(model, val_ds, args.device)
            val_metrics = summarize_eval(val_df, "val")
            log_payload.update(val_metrics)
            val_df.to_csv(os.path.join(out_dir, f"predictions_val_epoch{epoch}.csv"), index=False)
            print(f"[epoch {epoch}] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in val_metrics.items()))

            monitor = val_metrics["val/pseudobulk_pearson_r"]
            if not np.isnan(monitor) and monitor > best_metric:
                best_metric = monitor
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        logger.log(log_payload, step=epoch)

        if epochs_without_improvement >= args.patience:
            print(f"[early stop] no val/pseudobulk_pearson_r improvement for {args.patience} evals")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_df = run_pseudobulk_eval(model, test_ds, args.device)
    test_metrics = summarize_eval(test_df, "test")
    test_df.to_csv(os.path.join(out_dir, "predictions_test_final.csv"), index=False)
    logger.log(test_metrics)
    print("[test] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in test_metrics.items()))

    torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
    logger.finish()


if __name__ == "__main__":
    main()
