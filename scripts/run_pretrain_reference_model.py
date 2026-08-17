"""CLI: r-SAGE-net-style reference-only pretraining -- predicts a gene's
population-mean expression from reference sequence alone, across the
full protein-coding autosomal gene universe present in the h5ad.

The population-mean target defaults to **bulk** (all cell types pooled per
donor). Pass `--cell-type` to instead use that cell type's own pseudobulk,
matching `src/train.py` / `scripts/run_heritability_ranking.py`.

This is an optional prerequisite step for `src/train.py`: pass the resulting
`--out-dir` to `src/train.py --init-from-reference-model` to warm-start
`PSAGEnetSC`'s shared convolutional trunk from this run's `reference_model.pt`
before personal-genome fine-tuning on a specific `--cell-type`. Model
hyperparameter flags below intentionally share names with `src/train.py`'s
(a mismatch on any of `src/model.py::TRUNK_HYPERPARAM_KEYS` is caught, with a
clear error, at `--init-from-reference-model` load time).

Example (bulk, the default -- one reusable trunk for every fine-tuning cell type):
    python scripts/run_pretrain_reference_model.py \\
        --h5ad-path data/onek1k_cellxgene_standardized.h5ad \\
        --genome-fasta /path/to/hg38.fa --gtf /path/to/gencode.gtf.gz \\
        --out-dir results/reference_pretrain

Example (one cell type only):
    python scripts/run_pretrain_reference_model.py \\
        --h5ad-path data/onek1k_cellxgene_standardized.h5ad \\
        --cell-type "naive B cell" \\
        --genome-fasta /path/to/hg38.fa --gtf /path/to/gencode.gtf.gz \\
        --out-dir results/reference_pretrain_naive_B_cell
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from src.model import ReferenceExpressionModel
from src.pretrain import build_pretrain_pipeline, save_splits, set_seed, train_reference_model
from src.wandb_logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--h5ad-path", required=True)
    data_group.add_argument(
        "--cell-type",
        default=None,
        help=(
            "Exact `obs['cell_type']` value, e.g. 'naive B cell'. If omitted (default), the "
            "population-mean target is bulk pseudobulk (all cell types pooled per donor). If set, "
            "only cells of this type are used -- matching src/train.py / scripts/run_heritability_ranking.py."
        ),
    )
    data_group.add_argument(
        "--min-cells-per-donor",
        type=int,
        default=1,
        help=(
            "Min cells for a donor to count towards the population-mean target. Counted across any "
            "cell type when --cell-type is omitted (bulk); counted only within --cell-type when set"
        ),
    )
    data_group.add_argument(
        "--donor-val-frac", type=float, default=0.10, help="Paper default: ROSMAP individuals split 80%%/10%%/10%% train/val/test"
    )
    data_group.add_argument("--donor-test-frac", type=float, default=0.10)
    data_group.add_argument(
        "--gene-split-scheme",
        choices=["paper", "greedy"],
        default="paper",
        help=(
            "'paper' (default): the paper's fixed chromosome ranges (train=chr1-16, val=chr17/18/21/22, "
            "test=chr19/20) -- 78%%/11%%/11%%, matching r-SAGE-net's actual training setup ('n=14,786 "
            "training genes'). 'greedy': a dynamic, gene-count-balanced fallback (--gene-val-frac/"
            "--gene-test-frac)."
        ),
    )
    data_group.add_argument("--gene-val-frac", type=float, default=0.15, help="Only used when --gene-split-scheme greedy")
    data_group.add_argument("--gene-test-frac", type=float, default=0.15, help="Only used when --gene-split-scheme greedy")
    data_group.add_argument("--seed", type=int, default=0)

    genome_group = parser.add_argument_group("genome")
    genome_group.add_argument("--genome-fasta", required=True, help="Indexed reference genome FASTA (e.g. hg38)")
    genome_group.add_argument("--gtf", required=True, help="GTF/GFF gene annotation, used for TSS lookup")
    genome_group.add_argument("--seq-len", type=int, default=40_000, help="TSS-centered window size (paper default: 40kb)")
    genome_group.add_argument(
        "--chrom-style",
        choices=["chr", "no_chr"],
        default=None,
        help="Normalizes GTF <-> FASTA chromosome naming (e.g. 'chr1' vs '1'); no VCF is used in this pretraining stage",
    )

    model_group = parser.add_argument_group("model (defaults are the paper's bolded hyperparameters; same names as src/train.py)")
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

    train_group = parser.add_argument_group("training")
    train_group.add_argument("--batch-size", type=int, default=64, help="Genes per training step (no donor dimension)")
    train_group.add_argument("--eval-batch-size", type=int, default=256)
    train_group.add_argument("--epochs", type=int, default=100)
    train_group.add_argument("--lr", type=float, default=1e-3)
    train_group.add_argument("--weight-decay", type=float, default=0.0)
    train_group.add_argument("--grad-clip", type=float, default=1.0)
    train_group.add_argument("--patience", type=int, default=10)
    train_group.add_argument(
        "--lr-scheduler",
        choices=["cyclic", "none"],
        default="cyclic",
        help=(
            "'cyclic' (default, paper's recipe) wraps AdamW in CyclicLR with base_lr=lr/2, max_lr=lr*2, "
            "cycle_momentum=False, stepped every training batch -- helps the batch-norm-free first conv "
            "layer avoid dying-ReLU collapse. 'none' keeps a constant --lr."
        ),
    )
    train_group.add_argument(
        "--lr-scheduler-step-size-up",
        type=int,
        default=2000,
        help="Training iterations for CyclicLR to ramp base_lr -> max_lr (and the same count back down); ignored if --lr-scheduler none",
    )
    train_group.add_argument("--num-workers", type=int, default=0)
    train_group.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train_group.add_argument("--out-dir", required=True)
    train_group.add_argument("--wandb", action="store_true")
    train_group.add_argument("--no-save-checkpoint", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds, val_ds, test_ds, donor_split, gene_split = build_pretrain_pipeline(
        h5ad_path=args.h5ad_path,
        gtf_path=args.gtf,
        genome_fasta=args.genome_fasta,
        seq_len=args.seq_len,
        chrom_style=args.chrom_style,
        min_cells_per_donor=args.min_cells_per_donor,
        donor_val_frac=args.donor_val_frac,
        donor_test_frac=args.donor_test_frac,
        gene_split_scheme=args.gene_split_scheme,
        gene_val_frac=args.gene_val_frac,
        gene_test_frac=args.gene_test_frac,
        seed=args.seed,
        cell_type=args.cell_type,
    )
    save_splits(args.out_dir, donor_split, gene_split)

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
    }
    with open(os.path.join(args.out_dir, "reference_model_config.json"), "w") as fh:
        json.dump({**model_config, "cell_type": args.cell_type}, fh, indent=2)

    model = ReferenceExpressionModel(**model_config).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] ReferenceExpressionModel with {n_params:,} parameters, trunk output dim={model.trunk.output_dim}")

    logger = get_logger(args.wandb, project="sc-genex-reference-pretrain", config=vars(args))

    train_reference_model(
        model,
        train_ds,
        val_ds,
        test_ds,
        device=args.device,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        patience=args.patience,
        num_workers=args.num_workers,
        save_checkpoint=not args.no_save_checkpoint,
        lr_scheduler=args.lr_scheduler,
        lr_scheduler_step_size_up=args.lr_scheduler_step_size_up,
        logger=logger,
    )


if __name__ == "__main__":
    main()
