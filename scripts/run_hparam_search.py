"""CLI: reproduces the paper's r-SAGE-net hyperparameter tuning (Methods,
"R-SAGE-net training": "For the hyperparameter tuning process, we grid
search over: first_layer_kernel_number = (256, 900); int_layers_kernel_number
= (256, 512, 900); first_layer_kernel_size = (25, 10); n_conv_blocks = (5, 8);
pooling_size = (25, 10, 5); pooling_type = ('max', 'avg'); n_dilated_conv_blocks
= (0, 3, 5); dropout = (0, 0.2); h_layers = (2, 1); increasing_dilation =
(False, True); batch_norm = (True, False); hidden_size = (256, 512, 900);
learning_rate = (5e-4, 1e-3). Our selection of hyperparameters is bolded
above.").

Each trial is a full `scripts/run_pretrain_reference_model.py` subprocess run
(the paper does its hyperparameter tuning on r-SAGE-net specifically, then
reuses the winning config for p-SAGE-net verbatim: "All model hyperparameters
are the same for p-SAGE-net as r-SAGE-net"), scored by validation-gene
Pearson r -- exactly `train_reference_model`'s own early-stopping metric,
persisted incrementally to each trial's `best_val_summary.csv`. Once this
script reports a winner, pass its printed flags straight to both
`scripts/run_pretrain_reference_model.py` (for a final, longer pretraining
run) and `src/train.py` (whose `TRUNK_HYPERPARAM_KEYS` flags must match it
exactly for `--init-from-reference-model` to work).

Search modes:
  --mode one-at-a-time (default): vary ONE hyperparameter at a time away
    from the paper's own bolded/selected config (1 baseline + 17
    single-hyperparameter variations = 18 trials total). This is the only
    *tractable* reading of "grid search over: ... our selection is bolded"
    -- the literal full factorial grid below has 41,472 combinations
    (2x3x2x2x3x2x3x2x2x2x2x3x2), far more than any paper plausibly reports
    training, and is provided (`--mode grid`) mainly for completeness.
  --mode random: uniformly samples --n-trials distinct combinations from
    the full grid -- a practical way to explore more of the space than
    one-at-a-time without the intractable full factorial.
  --mode grid: the literal full factorial grid (41,472 combinations) --
    requires --max-trials >= the grid size (or --force), as a safety check
    against accidentally launching tens of thousands of training runs.

Resumable: a trial whose `best_val_summary.csv` already exists is skipped
(pass --no-resume to force a rerun of every trial). Failures in one trial
don't abort the search -- they're recorded in `leaderboard.csv` and the
search continues.

Example (paper's actual scale -- 18 trials):
    python scripts/run_hparam_search.py \\
        --h5ad-path data/onek1k_cellxgene_standardized.h5ad \\
        --genome-fasta /path/to/hg38.fa --gtf /path/to/gencode.gtf.gz \\
        --out-dir results/hparam_search --epochs 100 --patience 10

Example (random search over the full grid, 50 trials):
    python scripts/run_hparam_search.py \\
        --h5ad-path data/onek1k_cellxgene_standardized.h5ad \\
        --genome-fasta /path/to/hg38.fa --gtf /path/to/gencode.gtf.gz \\
        --out-dir results/hparam_search_random --mode random --n-trials 50
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
from typing import Any, Dict, List

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

# The paper's grid search space (Methods, "R-SAGE-net training"). Keys match
# scripts/run_pretrain_reference_model.py's argparse dest names (`lr` for its
# `--lr` flag, i.e. the paper's `learning_rate`); `int_layers_kernel_size` is
# excluded since the paper only ever gives it a single value, "(5)" -- i.e.
# it's fixed, not actually searched (matches every script/model default).
PAPER_GRID: Dict[str, List[Any]] = {
    "first_layer_kernel_number": [256, 900],
    "int_layers_kernel_number": [256, 512, 900],
    "first_layer_kernel_size": [25, 10],
    "n_conv_blocks": [5, 8],
    "pooling_size": [25, 10, 5],
    "pooling_type": ["max", "avg"],
    "n_dilated_conv_blocks": [0, 3, 5],
    "dropout": [0.0, 0.2],
    "h_layers": [2, 1],
    "increasing_dilation": [False, True],
    "batch_norm": [True, False],
    "hidden_size": [256, 512, 900],
    "lr": [5e-4, 1e-3],
}

# The paper's own bolded selection -- also every script/model default in this repo.
PAPER_DEFAULTS: Dict[str, Any] = {
    "first_layer_kernel_number": 900,
    "int_layers_kernel_number": 256,
    "first_layer_kernel_size": 10,
    "n_conv_blocks": 5,
    "pooling_size": 10,
    "pooling_type": "avg",
    "n_dilated_conv_blocks": 0,
    "dropout": 0.0,
    "h_layers": 1,
    "increasing_dilation": False,
    "batch_norm": True,
    "hidden_size": 256,
    "lr": 1e-3,
}

# Hyperparameters whose CLI representation is a boolean store_true flag
# rather than a plain `--key value` pair (one of them, batch_norm, is even
# *inverted*: True is the default/no-flag state).
_BOOL_HPARAMS = {"increasing_dilation", "batch_norm"}


def hparams_to_cli_args(hparams: Dict[str, Any]) -> List[str]:
    """Renders a hyperparameter dict as `scripts/run_pretrain_reference_model.py` CLI flags."""
    args: List[str] = []
    for key, value in hparams.items():
        if key == "increasing_dilation":
            if value:
                args.append("--increasing-dilation")
        elif key == "batch_norm":
            if not value:
                args.append("--no-batch-norm")
        else:
            args += [f"--{key.replace('_', '-')}", str(value)]
    return args


def one_at_a_time_trials() -> List[Dict[str, Any]]:
    """1 baseline (all paper defaults) + one trial per non-default grid value (17) = 18 total."""
    trials = [dict(PAPER_DEFAULTS)]
    for key, values in PAPER_GRID.items():
        for value in values:
            if value == PAPER_DEFAULTS[key]:
                continue
            trial = dict(PAPER_DEFAULTS)
            trial[key] = value
            trials.append(trial)
    return trials


def full_grid_trials() -> List[Dict[str, Any]]:
    """The literal full factorial grid (41,472 combinations)."""
    keys = list(PAPER_GRID.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*(PAPER_GRID[k] for k in keys))]


def random_trials(n_trials: int, seed: int) -> List[Dict[str, Any]]:
    """`n_trials` distinct combinations sampled uniformly from the full grid."""
    rng = random.Random(seed)
    keys = list(PAPER_GRID.keys())
    seen = set()
    trials: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = max(n_trials * 50, 1000)
    while len(trials) < n_trials and attempts < max_attempts:
        attempts += 1
        combo = tuple(rng.choice(PAPER_GRID[k]) for k in keys)
        if combo in seen:
            continue
        seen.add(combo)
        trials.append(dict(zip(keys, combo)))
    return trials


def trial_name(idx: int, hparams: Dict[str, Any]) -> str:
    diffs = {k: v for k, v in hparams.items() if PAPER_DEFAULTS.get(k) != v}
    if not diffs:
        return f"trial_{idx:04d}_baseline"
    if len(diffs) == 1:
        ((key, value),) = diffs.items()
        return f"trial_{idx:04d}_{key}-{str(value).replace('.', 'p')}"
    return f"trial_{idx:04d}"


def write_leaderboard(out_dir: str, results: List[dict]) -> str:
    df = pd.DataFrame(results)
    if "val/pearson_r" in df.columns:
        df = df.sort_values("val/pearson_r", ascending=False, na_position="last")
    path = os.path.join(out_dir, "leaderboard.csv")
    df.to_csv(path, index=False)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    data_group = parser.add_argument_group("data (forwarded to every trial, unchanged)")
    data_group.add_argument("--h5ad-path", required=True)
    data_group.add_argument(
        "--cell-type",
        default=None,
        help=(
            "Forwarded to every trial. If omitted (default), bulk pretraining. If set, only that "
            "cell type's pseudobulk is used -- see scripts/run_pretrain_reference_model.py --cell-type"
        ),
    )
    data_group.add_argument("--min-cells-per-donor", type=int, default=1)
    data_group.add_argument("--donor-val-frac", type=float, default=0.10)
    data_group.add_argument("--donor-test-frac", type=float, default=0.10)
    data_group.add_argument("--gene-split-scheme", choices=["paper", "greedy"], default="paper")
    data_group.add_argument("--gene-val-frac", type=float, default=0.15, help="Only used when --gene-split-scheme greedy")
    data_group.add_argument("--gene-test-frac", type=float, default=0.15, help="Only used when --gene-split-scheme greedy")
    data_group.add_argument("--seed", type=int, default=0, help="Also seeds --mode random's trial sampling")

    genome_group = parser.add_argument_group("genome (forwarded to every trial, unchanged)")
    genome_group.add_argument("--genome-fasta", required=True)
    genome_group.add_argument("--gtf", required=True)
    genome_group.add_argument("--seq-len", type=int, default=40_000)
    genome_group.add_argument("--chrom-style", choices=["chr", "no_chr"], default=None)

    train_group = parser.add_argument_group("training (paper's fixed r-SAGE-net recipe -- Methods, 'R-SAGE-net training')")
    train_group.add_argument("--batch-size", type=int, default=16, help="Paper default for r-SAGE-net")
    train_group.add_argument("--eval-batch-size", type=int, default=256)
    train_group.add_argument("--epochs", type=int, default=100, help="Paper: 'maximum of 100 epochs'")
    train_group.add_argument("--patience", type=int, default=10, help="Paper: 'early stopping (patience = 10)'")
    train_group.add_argument("--weight-decay", type=float, default=1e-5, help="Paper: Adam weight_decay=1e-5")
    train_group.add_argument("--grad-clip", type=float, default=1.0, help="Paper: gradient_clip_val=1")
    train_group.add_argument("--lr-scheduler", choices=["cyclic", "none"], default="cyclic", help="Paper: LR scheduler 'cycle'")
    train_group.add_argument("--lr-scheduler-step-size-up", type=int, default=2000)
    train_group.add_argument("--num-workers", type=int, default=0)
    train_group.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train_group.add_argument("--wandb", action="store_true", help="Log every trial to W&B (project sc-genex-reference-pretrain)")

    search_group = parser.add_argument_group("search")
    search_group.add_argument("--mode", choices=["one-at-a-time", "grid", "random"], default="one-at-a-time")
    search_group.add_argument("--n-trials", type=int, default=50, help="Only used by --mode random")
    search_group.add_argument(
        "--max-trials",
        type=int,
        default=200,
        help="Safety cap: refuse to launch more than this many trials unless --force is also passed",
    )
    search_group.add_argument("--force", action="store_true", help="Bypass --max-trials")
    search_group.add_argument("--no-resume", action="store_true", help="Rerun every trial, even ones with an existing best_val_summary.csv")
    search_group.add_argument("--dry-run", action="store_true", help="Print the planned trials/commands without running anything")
    search_group.add_argument("--out-dir", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "one-at-a-time":
        trials = one_at_a_time_trials()
    elif args.mode == "grid":
        trials = full_grid_trials()
    else:
        trials = random_trials(args.n_trials, args.seed)

    if len(trials) > args.max_trials and not args.force:
        raise SystemExit(
            f"--mode={args.mode!r} would launch {len(trials)} trials, exceeding --max-trials={args.max_trials}. "
            f"Pass --max-trials {len(trials)} (or --force) if you really want to run all of them, or use "
            f"--mode one-at-a-time (18 trials) / --mode random --n-trials N for a more tractable search."
        )
    print(f"[hparam-search] mode={args.mode!r}: {len(trials)} trial(s) planned, out_dir={args.out_dir}")

    fixed_args = [
        "--h5ad-path", args.h5ad_path,
        "--genome-fasta", args.genome_fasta,
        "--gtf", args.gtf,
        "--min-cells-per-donor", str(args.min_cells_per_donor),
        "--donor-val-frac", str(args.donor_val_frac),
        "--donor-test-frac", str(args.donor_test_frac),
        "--gene-split-scheme", args.gene_split_scheme,
        "--gene-val-frac", str(args.gene_val_frac),
        "--gene-test-frac", str(args.gene_test_frac),
        "--seed", str(args.seed),
        "--seq-len", str(args.seq_len),
        "--batch-size", str(args.batch_size),
        "--eval-batch-size", str(args.eval_batch_size),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--weight-decay", str(args.weight_decay),
        "--grad-clip", str(args.grad_clip),
        "--lr-scheduler", args.lr_scheduler,
        "--lr-scheduler-step-size-up", str(args.lr_scheduler_step_size_up),
        "--num-workers", str(args.num_workers),
        "--device", args.device,
    ]  # fmt: skip
    if args.chrom_style:
        fixed_args += ["--chrom-style", args.chrom_style]
    if args.cell_type:
        fixed_args += ["--cell-type", args.cell_type]
    if args.wandb:
        fixed_args.append("--wandb")

    script_path = os.path.join(_REPO_ROOT, "scripts", "run_pretrain_reference_model.py")
    resume = not args.no_resume
    results: List[dict] = []

    for idx, hparams in enumerate(trials):
        name = trial_name(idx, hparams)
        trial_out_dir = os.path.join(args.out_dir, name)
        summary_path = os.path.join(trial_out_dir, "best_val_summary.csv")
        prefix = f"[hparam-search] [{idx + 1}/{len(trials)}] {name}"

        if resume and os.path.exists(summary_path):
            print(f"{prefix}: already completed, skipping (--no-resume to force a rerun)")
        else:
            os.makedirs(trial_out_dir, exist_ok=True)
            with open(os.path.join(trial_out_dir, "trial_hparams.json"), "w") as fh:
                json.dump(hparams, fh, indent=2)

            cmd = [sys.executable, script_path, *fixed_args, *hparams_to_cli_args(hparams), "--out-dir", trial_out_dir]
            print(f"{prefix}: {' '.join(cmd)}")

            if args.dry_run:
                results.append({"trial": name, **hparams, "val/pearson_r": float("nan"), "status": "dry-run"})
                continue

            log_path = os.path.join(trial_out_dir, "train.log")
            with open(log_path, "w") as log_fh:
                proc = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                print(f"{prefix}: FAILED (exit {proc.returncode}), see {log_path}")
                results.append({"trial": name, **hparams, "val/pearson_r": float("nan"), "status": f"failed(exit={proc.returncode})"})
                write_leaderboard(args.out_dir, results)
                continue

        if os.path.exists(summary_path):
            row = pd.read_csv(summary_path).iloc[0].to_dict()
            row.update({"trial": name, **hparams, "status": "ok"})
            results.append(row)
        else:
            print(f"{prefix}: no best_val_summary.csv found (val Pearson r was always nan?)")
            results.append({"trial": name, **hparams, "val/pearson_r": float("nan"), "status": "no-val-summary"})
        write_leaderboard(args.out_dir, results)

    leaderboard_path = write_leaderboard(args.out_dir, results)
    print(f"[hparam-search] wrote leaderboard to {leaderboard_path}")

    ok_results = [r for r in results if r.get("status") == "ok" and pd.notna(r.get("val/pearson_r"))]
    if not ok_results:
        print("[hparam-search] no successful trial produced a defined val/pearson_r")
        return

    best = max(ok_results, key=lambda r: r["val/pearson_r"])
    best_hparams = {key: best[key] for key in PAPER_GRID}
    print(f"[hparam-search] best trial: {best['trial']} (val/pearson_r={best['val/pearson_r']:.4f})")
    print(
        "[hparam-search] equivalent flags for scripts/run_pretrain_reference_model.py / src/train.py:\n  "
        + " ".join(hparams_to_cli_args(best_hparams))
    )


if __name__ == "__main__":
    main()
