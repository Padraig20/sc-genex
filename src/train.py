"""Trains `PSAGEnetSC` -- a from-scratch, pSAGE-net-inspired compact CNN --
on single-cell OneK1K expression for one cell type, across the top-N
heritability-ranked genes (see `scripts/run_heritability_ranking.py`).

Training is per-cell (no pseudobulking): every `(gene, donor)` example is
scored against every one of that donor's actual per-cell expression values
via a per-cell-weighted Gaussian NLL on the difference head (`src/loss.py`).
Evaluation is always pseudobulk (per-donor mean/std), across all 4 cells of
the seen/unseen-gene x seen/unseen-individual matrix (`src/metrics.py`).

Example:
    python src/train.py \\
        --h5ad-path data/onek1k_cellxgene_standardized.h5ad \\
        --cell-type "naive B cell" \\
        --top-genes-csv results/heritability/naive_B_cell/top_1000_genes.csv \\
        --vcf-dir /path/to/genotypes --genome-fasta /path/to/hg38.fa --gtf /path/to/gencode.gtf.gz \\
        --out-dir results/naive_B_cell_psagenet_sc
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.preprocess import build_multigene_donor_table, compute_population_means, get_celltype_donor_ids, load_celltype_multigene_cells
from src.dataset import (
    PersonalGenomeDatasetSC,
    PersonalGenomeEvalDataset,
    PersonalGenomeSequenceBuilder,
    collate_eval_batch,
    collate_personal_genome_batch,
)
from src.genome import GeneAnnotation, ReferenceGenome, VCFGenotypeReader, onek1k_sample_id_from_donor_id
from src.loss import psagenet_sc_loss
from src.metrics import grouped_correlation
from src.model import PSAGEnetSC, load_trunk_from_reference_model, validate_trunk_hyperparams
from src.splits import DonorSplit, GENE_SPLIT_SCHEMES, GeneSplit, select_gene_split, split_donors
from src.wandb_logger import get_logger

# The 4 held-out cells of the seen/unseen-gene x seen/unseen-individual matrix
# (see plan); "seen_gene_val_donor" is a 5th, training-time-only cell used for
# early stopping (mirrors the paper's model-selection criterion) and is not
# part of the final reported matrix.
EVAL_MATRIX_CELLS = ("seen_gene_seen_donor", "seen_gene_unseen_donor", "unseen_gene_seen_donor", "unseen_gene_unseen_donor")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--h5ad-path", required=True)
    data_group.add_argument("--cell-type", required=True, help="Exact `obs['cell_type']` value, e.g. 'naive B cell'")
    data_group.add_argument(
        "--top-genes-csv",
        required=True,
        help="CSV with a 'gene_id' column (output of scripts/run_heritability_ranking.py's top_N_genes.csv)",
    )
    data_group.add_argument("--min-cells-per-donor", type=int, default=5)
    data_group.add_argument(
        "--donor-val-frac", type=float, default=0.10, help="Paper default: ROSMAP individuals split 80%%/10%%/10%% train/val/test"
    )
    data_group.add_argument("--donor-test-frac", type=float, default=0.10)
    data_group.add_argument(
        "--gene-split-scheme",
        choices=GENE_SPLIT_SCHEMES,
        default="paper",
        help=(
            "'paper' (default): the paper's fixed chromosome ranges (train=chr1-16, val=chr17/18/21/22, "
            "test=chr19/20), directly comparable to the paper's own results. 'greedy': a dynamic, "
            "gene-count-balanced fallback (--gene-val-frac/--gene-test-frac) for when 'paper' leaves a "
            "split empty -- likely for a small/biased --top-genes-csv gene subset."
        ),
    )
    data_group.add_argument("--gene-val-frac", type=float, default=0.15, help="Only used when --gene-split-scheme greedy")
    data_group.add_argument("--gene-test-frac", type=float, default=0.15, help="Only used when --gene-split-scheme greedy")
    data_group.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Also used for train/eval shuffling; must match --seed used for scripts/run_heritability_ranking.py so donor splits agree",
    )

    genome_group = parser.add_argument_group("genome")
    genome_group.add_argument("--vcf-dir", required=True, help="Directory containing one (phased) VCF per chromosome")
    genome_group.add_argument("--vcf-filename-template", default="{chrom}.vcf.gz")
    genome_group.add_argument("--vcf-chrom-style", choices=["chr", "no_chr"], default=None)
    genome_group.add_argument("--genome-fasta", required=True, help="Indexed reference genome FASTA (e.g. hg38)")
    genome_group.add_argument("--gtf", required=True, help="GTF/GFF gene annotation, used for TSS lookup")
    genome_group.add_argument("--seq-len", type=int, default=40_000, help="TSS-centered window size (paper default: 40kb)")
    genome_group.add_argument("--vcf-sample-id-scheme", choices=["onek1k", "identity"], default="onek1k")
    genome_group.add_argument("--vcf-sample-id-prefix", default="OneK1K_")
    genome_group.add_argument("--donor-id-map", default=None, help="Optional 2-col CSV (h5ad_donor_id,vcf_sample_id) override")
    genome_group.add_argument(
        "--strict-phasing",
        action="store_true",
        help="Raise instead of skipping heterozygous sites that aren't marked phased in the VCF",
    )

    model_group = parser.add_argument_group("model (defaults are the paper's bolded hyperparameters)")
    model_group.add_argument("--first-layer-kernel-number", type=int, default=900)
    model_group.add_argument("--first-layer-kernel-size", type=int, default=10)
    model_group.add_argument("--int-layers-kernel-number", type=int, default=256)
    model_group.add_argument("--int-layers-kernel-size", type=int, default=5)
    model_group.add_argument("--hidden-size", type=int, default=256)
    model_group.add_argument("--n-conv-blocks", type=int, default=5)
    model_group.add_argument("--n-dilated-conv-blocks", type=int, default=0)
    model_group.add_argument("--h-layers", type=int, default=1)
    model_group.add_argument("--pooling-size", type=int, default=10)
    model_group.add_argument("--pooling-type", choices=["avg", "max"], default="avg")
    model_group.add_argument("--no-batch-norm", action="store_true")
    model_group.add_argument(
        "--increasing-dilation",
        action="store_true",
        help="Exponentially increase dilation across --n-dilated-conv-blocks (default: keep it fixed at 2)",
    )
    model_group.add_argument("--dropout", type=float, default=0.0)
    model_group.add_argument("--subtract-or-concat", choices=["subtract", "concat"], default="subtract")

    train_group = parser.add_argument_group("training")
    train_group.add_argument("--batch-size", type=int, default=32, help="(gene, donor) pairs per training step")
    train_group.add_argument("--eval-batch-size", type=int, default=64)
    train_group.add_argument("--epochs", type=int, default=50)
    train_group.add_argument("--lr", type=float, default=1e-3)
    train_group.add_argument("--weight-decay", type=float, default=0.0)
    train_group.add_argument("--grad-clip", type=float, default=1.0)
    train_group.add_argument("--eval-every", type=int, default=1)
    train_group.add_argument("--patience", type=int, default=10)
    train_group.add_argument("--lam-ref", type=float, default=1.0, help="Mean-head MSE loss weight (paper default: 1)")
    train_group.add_argument("--lam-diff", type=float, default=10.0, help="Diff-head GNLL loss weight (paper default: 10)")
    train_group.add_argument(
        "--lr-scheduler",
        choices=["cyclic", "none"],
        default="none",
        help=(
            "'cyclic' wraps AdamW in CyclicLR with base_lr=lr/2, max_lr=lr*2, cycle_momentum=False, stepped every "
            "training batch (matches scripts/run_pretrain_reference_model.py's default r-SAGE-net-style recipe, which "
            "the paper also uses for p-SAGE-net). Default 'none' keeps a constant --lr, preserving prior behavior."
        ),
    )
    train_group.add_argument(
        "--lr-scheduler-step-size-up",
        type=int,
        default=2000,
        help="Training iterations for CyclicLR to ramp base_lr -> max_lr (and the same count back down); ignored if --lr-scheduler none",
    )
    train_group.add_argument(
        "--max-eval-pairs",
        type=int,
        default=5000,
        help="Cap each eval-matrix cell's (gene, donor) pairs via random subsampling, for tractable per-epoch/final eval cost",
    )
    train_group.add_argument("--num-workers", type=int, default=0)
    train_group.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train_group.add_argument("--out-dir", default=None)
    train_group.add_argument("--wandb", action="store_true")
    train_group.add_argument("--no-save-checkpoint", action="store_true")
    train_group.add_argument(
        "--init-from-reference-model",
        default=None,
        help=(
            "Path to a scripts/run_pretrain_reference_model.py --out-dir (r-SAGE-net-style pretraining). "
            "If set, warm-starts PSAGEnetSC's shared conv/pooling trunk from that run's reference_model.pt "
            "-- only the trunk transfers (mean/diff/diff_sigma heads are always randomly initialized), "
            "matching the paper's r-SAGE-net -> p-SAGE-net recipe. Trunk hyperparameters must match exactly "
            "(see src/model.py::TRUNK_HYPERPARAM_KEYS)."
        ),
    )

    return parser.parse_args()


def load_donor_id_map(path: Optional[str]) -> Dict[str, str]:
    if path is None:
        return {}
    df = pd.read_csv(path, header=None, names=["h5ad_donor_id", "vcf_sample_id"])
    return dict(zip(df["h5ad_donor_id"].astype(str), df["vcf_sample_id"].astype(str)))


def load_top_genes(path: str) -> List[str]:
    df = pd.read_csv(path)
    if "gene_id" not in df.columns:
        raise ValueError(f"{path} must have a 'gene_id' column (e.g. output of scripts/run_heritability_ranking.py)")
    return df["gene_id"].astype(str).tolist()


def build_pipeline(args: argparse.Namespace):
    gene_ids = load_top_genes(args.top_genes_csv)
    print(f"[genes] loaded {len(gene_ids)} candidate genes from {args.top_genes_csv}")

    annotation = GeneAnnotation(args.gtf)
    gene_records = {g: annotation.get(g) for g in gene_ids}

    cell_data = load_celltype_multigene_cells(args.h5ad_path, args.cell_type, gene_ids)
    table = build_multigene_donor_table(cell_data, min_cells_per_donor=args.min_cells_per_donor)
    print(f"[data] {len(table.donor_ids)} donors have >= {args.min_cells_per_donor} cells for at least one gene")

    donor_universe = get_celltype_donor_ids(args.h5ad_path, args.cell_type)
    raw_donor_split = split_donors(donor_universe, val_frac=args.donor_val_frac, test_frac=args.donor_test_frac, seed=args.seed)
    table_donor_set = set(table.donor_ids)
    donor_split = DonorSplit(
        train=[d for d in raw_donor_split.train if d in table_donor_set],
        val=[d for d in raw_donor_split.val if d in table_donor_set],
        test=[d for d in raw_donor_split.test if d in table_donor_set],
    )
    print(
        f"[donors] train={len(donor_split.train)} val={len(donor_split.val)} test={len(donor_split.test)} "
        f"(out of {len(donor_universe)} total for this cell type)"
    )

    gene_to_chrom = {g: gene_records[g].chrom for g in gene_ids}
    gene_split = select_gene_split(
        gene_to_chrom, scheme=args.gene_split_scheme, val_frac=args.gene_val_frac, test_frac=args.gene_test_frac, seed=args.seed
    )
    print(f"[genes] scheme={args.gene_split_scheme!r}: {gene_split.summary()}")

    compute_population_means(table, donor_split.train)

    donor_id_map = load_donor_id_map(args.donor_id_map)
    if args.vcf_sample_id_scheme == "onek1k":
        sample_id_fn = lambda d: onek1k_sample_id_from_donor_id(d, prefix=args.vcf_sample_id_prefix)  # noqa: E731
    else:
        sample_id_fn = lambda d: d  # noqa: E731

    def _make_sequence_builder() -> PersonalGenomeSequenceBuilder:
        # A fresh ReferenceGenome/VCFGenotypeReader per builder -- NOT shared between the
        # training builder (below) and the eval builder. `pysam` file handles must never be
        # shared across a `fork()`: the training DataLoader workers are (re)forked every epoch
        # (no `persistent_workers`), and if the *main* process had already opened VCF handles
        # on the same reader (e.g. from an eval pass), those handles would get duplicated into
        # every worker, sharing the same underlying kernel file offset -- concurrent tabix
        # reads across those forked siblings then race and corrupt each other's bgzf stream
        # (surfaces as "CRC32 checksum mismatch" / "truncated file" from pysam/htslib).
        # Keeping the training-path reader untouched by the (single-process) eval path means
        # its `_open_files` is always empty at fork time, so every worker always opens its own
        # independent, unshared file descriptors.
        return PersonalGenomeSequenceBuilder(
            reference=ReferenceGenome(args.genome_fasta),
            vcf_reader=VCFGenotypeReader(args.vcf_dir, filename_template=args.vcf_filename_template),
            genes=gene_records,
            seq_len=args.seq_len,
            vcf_chrom_style=args.vcf_chrom_style,
            donor_id_to_sample_id=donor_id_map,
            sample_id_fn=sample_id_fn,
            strict_phasing=args.strict_phasing,
        )

    train_sequence_builder = _make_sequence_builder()
    eval_sequence_builder = _make_sequence_builder()

    train_ds = PersonalGenomeDatasetSC(gene_split.train, donor_split.train, table, train_sequence_builder)
    print(f"[data] train pairs (train genes x train donors): {len(train_ds)}")

    eval_gene_donor_lists = {
        "seen_gene_seen_donor": (gene_split.train, donor_split.train),
        "seen_gene_val_donor": (gene_split.train, donor_split.val),
        "seen_gene_unseen_donor": (gene_split.train, donor_split.test),
        "unseen_gene_seen_donor": (gene_split.test, donor_split.train),
        "unseen_gene_unseen_donor": (gene_split.test, donor_split.test),
    }
    eval_datasets = {}
    for name, (genes, donors) in eval_gene_donor_lists.items():
        eval_datasets[name] = PersonalGenomeEvalDataset(
            genes, donors, table, eval_sequence_builder, max_pairs=args.max_eval_pairs, seed=args.seed
        )
        print(f"[data] eval set '{name}': {len(eval_datasets[name])} pairs")

    return train_ds, eval_datasets, donor_split, gene_split


@torch.no_grad()
def run_eval(
    model: PSAGEnetSC, dataset: PersonalGenomeEvalDataset, device: str, batch_size: int, desc: str = "eval"
) -> pd.DataFrame:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_eval_batch)
    rows: List[dict] = []
    for ref, mat, pat, pseudobulk_mean, pseudobulk_std, population_mean, n_cells, gene_ids, donor_ids in tqdm(
        loader, desc=desc, leave=False
    ):
        ref, mat, pat = ref.to(device), mat.to(device), pat.to(device)
        m_hat, d_hat, sigma_d_hat = model(ref, mat, pat)
        mu_hat = (m_hat + d_hat).cpu().numpy()
        m_hat_np = m_hat.cpu().numpy()
        d_hat_np = d_hat.cpu().numpy()
        sigma_np = sigma_d_hat.cpu().numpy()
        pseudobulk_mean_np = pseudobulk_mean.numpy()
        pseudobulk_std_np = pseudobulk_std.numpy()
        population_mean_np = population_mean.numpy()
        n_cells_np = n_cells.numpy()
        for i in range(len(gene_ids)):
            rows.append(
                {
                    "gene_id": gene_ids[i],
                    "donor_id": donor_ids[i],
                    "y_true_mean": float(pseudobulk_mean_np[i]),
                    "y_true_std": float(pseudobulk_std_np[i]),
                    "y_true_population_mean": float(population_mean_np[i]),
                    "y_pred_mean": float(mu_hat[i]),
                    "y_pred_m_hat": float(m_hat_np[i]),
                    "y_pred_d_hat": float(d_hat_np[i]),
                    "y_pred_sigma": float(sigma_np[i]),
                    "n_cells": int(n_cells_np[i]),
                }
            )
    return pd.DataFrame(rows)


def summarize_eval(df: pd.DataFrame, prefix: str) -> Dict[str, float]:
    mean_grouped = grouped_correlation(df, true_col="y_true_mean", pred_col="y_pred_mean")
    sigma_grouped = grouped_correlation(df, true_col="y_true_std", pred_col="y_pred_sigma")
    return {
        f"{prefix}/mean_median_per_gene_pearson_r": mean_grouped.median_per_gene_r,
        f"{prefix}/mean_median_per_donor_pearson_r": mean_grouped.median_per_donor_r,
        f"{prefix}/sigma_median_per_gene_pearson_r": sigma_grouped.median_per_gene_r,
        f"{prefix}/sigma_median_per_donor_pearson_r": sigma_grouped.median_per_donor_r,
        f"{prefix}/n_genes": mean_grouped.n_genes,
        f"{prefix}/n_donors": mean_grouped.n_donors,
        f"{prefix}/n_pairs": len(df),
    }


def save_splits(out_dir: str, donor_split: DonorSplit, gene_split: GeneSplit) -> None:
    donor_rows = (
        [{"donor_id": d, "split": "train"} for d in donor_split.train]
        + [{"donor_id": d, "split": "val"} for d in donor_split.val]
        + [{"donor_id": d, "split": "test"} for d in donor_split.test]
    )
    pd.DataFrame(donor_rows).to_csv(os.path.join(out_dir, "donor_split.csv"), index=False)

    gene_rows = (
        [{"gene_id": g, "split": "train"} for g in gene_split.train]
        + [{"gene_id": g, "split": "val"} for g in gene_split.val]
        + [{"gene_id": g, "split": "test"} for g in gene_split.test]
    )
    pd.DataFrame(gene_rows).to_csv(os.path.join(out_dir, "gene_split.csv"), index=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = args.out_dir or os.path.join("results", args.cell_type.replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)

    train_ds, eval_datasets, donor_split, gene_split = build_pipeline(args)
    save_splits(out_dir, donor_split, gene_split)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_personal_genome_batch,
        num_workers=args.num_workers,
    )

    model_config = {
        "input_length": args.seq_len,
        "first_layer_kernel_number": args.first_layer_kernel_number,
        "int_layers_kernel_number": args.int_layers_kernel_number,
        "first_layer_kernel_size": args.first_layer_kernel_size,
        "int_layers_kernel_size": args.int_layers_kernel_size,
        "hidden_size": args.hidden_size,
        "n_conv_blocks": args.n_conv_blocks,
        "n_dilated_conv_blocks": args.n_dilated_conv_blocks,
        "h_layers": args.h_layers,
        "pooling_size": args.pooling_size,
        "pooling_type": args.pooling_type,
        "batch_norm": not args.no_batch_norm,
        "increasing_dilation": args.increasing_dilation,
        "dropout": args.dropout,
        "subtract_or_concat": args.subtract_or_concat,
    }
    with open(os.path.join(out_dir, "model_config.json"), "w") as fh:
        json.dump(model_config, fh, indent=2)

    model = PSAGEnetSC(**model_config).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] PSAGEnetSC with {n_params:,} parameters, trunk output dim={model.trunk.output_dim}")

    if args.init_from_reference_model:
        reference_config_path = os.path.join(args.init_from_reference_model, "reference_model_config.json")
        reference_ckpt_path = os.path.join(args.init_from_reference_model, "reference_model.pt")
        with open(reference_config_path) as fh:
            reference_config = json.load(fh)
        validate_trunk_hyperparams(model_config, reference_config)
        reference_state_dict = torch.load(reference_ckpt_path, map_location=args.device)
        load_trunk_from_reference_model(model, reference_state_dict)
        print(f"[model] warm-started trunk from {reference_ckpt_path} (r-SAGE-net-style pretraining)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_scheduler == "cyclic":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=args.lr / 2, max_lr=args.lr * 2, step_size_up=args.lr_scheduler_step_size_up, cycle_momentum=False
        )
    elif args.lr_scheduler != "none":
        raise ValueError(f"Unknown --lr-scheduler={args.lr_scheduler!r}, expected 'none' or 'cyclic'")
    logger = get_logger(args.wandb, project="sc-genex-psagenet-sc", config=vars(args))

    best_metric = -float("inf")
    best_state: Optional[dict] = None
    epochs_without_improvement = 0

    epoch_bar = tqdm(range(args.epochs), desc="epochs")
    for epoch in epoch_bar:
        model.train()
        losses, mean_losses, diff_losses = [], [], []
        batch_bar = tqdm(train_loader, desc=f"epoch {epoch} [train]", leave=False)
        for ref, mat, pat, population_mean, cell_diff, gene_ids, donor_ids in batch_bar:
            ref, mat, pat = ref.to(args.device), mat.to(args.device), pat.to(args.device)
            population_mean = population_mean.to(args.device)
            cell_diff = [t.to(args.device) for t in cell_diff]

            m_hat, d_hat, sigma_d_hat = model(ref, mat, pat)
            loss_components = psagenet_sc_loss(
                m_hat, d_hat, sigma_d_hat, population_mean, cell_diff, lam_ref=args.lam_ref, lam_diff=args.lam_diff
            )

            optimizer.zero_grad()
            loss_components.total.backward()
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            losses.append(loss_components.total.item())
            mean_losses.append(loss_components.mean_loss.item())
            diff_losses.append(loss_components.diff_loss.item())
            batch_bar.set_postfix(loss=f"{losses[-1]:.4f}", mean=f"{mean_losses[-1]:.4f}", diff=f"{diff_losses[-1]:.4f}")

        log_payload = {
            "train/loss": float(np.mean(losses)) if losses else float("nan"),
            "train/mean_loss": float(np.mean(mean_losses)) if mean_losses else float("nan"),
            "train/diff_loss": float(np.mean(diff_losses)) if diff_losses else float("nan"),
            "epoch": epoch,
        }
        tqdm.write(f"[epoch {epoch}] train_loss={log_payload['train/loss']:.4f}")

        if epoch % args.eval_every == 0:
            monitor_df = run_eval(
                model, eval_datasets["seen_gene_val_donor"], args.device, args.eval_batch_size, desc=f"epoch {epoch} [val]"
            )
            monitor_metrics = summarize_eval(monitor_df, "val")
            log_payload.update(monitor_metrics)
            monitor = monitor_metrics["val/mean_median_per_gene_pearson_r"]
            tqdm.write(
                f"[epoch {epoch}] val/mean_median_per_gene_pearson_r="
                f"{monitor:.4f}" if not np.isnan(monitor) else f"[epoch {epoch}] val metric undefined (nan)"
            )

            if not np.isnan(monitor) and monitor > best_metric:
                best_metric = monitor
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                if not args.no_save_checkpoint:
                    # Persisted immediately (not just at the end) so a crash/OOM/preemption
                    # later in a long run doesn't lose an already-improved checkpoint.
                    torch.save(best_state, os.path.join(out_dir, "best_model.pt"))
            else:
                epochs_without_improvement += 1
            epoch_bar.set_postfix(val_r=f"{monitor:.4f}", best=f"{best_metric:.4f}", no_improve=epochs_without_improvement)

        logger.log(log_payload, step=epoch)

        if epochs_without_improvement >= args.patience:
            tqdm.write(f"[early stop] no val/mean_median_per_gene_pearson_r improvement for {args.patience} evals")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("[warning] validation metric was always nan -- reporting final-epoch weights instead of a 'best' checkpoint")

    print("[final] evaluating full 4-way seen/unseen-gene x seen/unseen-individual matrix")
    final_summary: Dict[str, float] = {}
    for name in tqdm(EVAL_MATRIX_CELLS, desc="final 4-way matrix"):
        df = run_eval(model, eval_datasets[name], args.device, args.eval_batch_size, desc=f"final [{name}]")
        df.to_csv(os.path.join(out_dir, f"predictions_{name}.csv"), index=False)
        metrics = summarize_eval(df, name)
        final_summary.update(metrics)
        mean_r = metrics[f"{name}/mean_median_per_gene_pearson_r"]
        sigma_r = metrics[f"{name}/sigma_median_per_gene_pearson_r"]
        tqdm.write(
            f"[final:{name}] median per-gene Pearson r: mean={mean_r:.4f} sigma={sigma_r:.4f} (n_pairs={metrics[f'{name}/n_pairs']})"
        )

    logger.log(final_summary)
    pd.DataFrame([final_summary]).to_csv(os.path.join(out_dir, "final_eval_matrix_summary.csv"), index=False)

    if args.no_save_checkpoint:
        print("[checkpoint] --no-save-checkpoint set, skipping best_model.pt")
    else:
        torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
    logger.finish()


if __name__ == "__main__":
    main()
