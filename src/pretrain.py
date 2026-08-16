"""r-SAGE-net-style reference-only pretraining: predicts a gene's population-mean
(bulk) expression from reference sequence alone, across a large, genome-wide
protein-coding-autosomal gene universe -- mirroring pSAGE-net's paper
("R-SAGE-net is trained on reference sequence and mean expression ... for
n = 14,786 training genes"). No personal genome/donor input exists anywhere in
this stage: unlike `src/dataset.py`/`src/train.py`, there is no per-donor
haplotype dimension at all, so `ReferenceExpressionDataset` (below) has one
item per *gene*, not per `(gene, donor)` pair.

The resulting trunk weights (`reference_model.pt`'s `trunk.*` keys) are meant
to warm-start `PSAGEnetSC`'s shared `CNNTrunk` before personal-genome
fine-tuning -- see `src/train.py --init-from-reference-model`. This module
holds the dataset + core train/eval loop; `scripts/run_pretrain_reference_model.py`
is the CLI entry point (mirrors the `src/heritability.py` / `scripts/run_heritability_ranking.py`
split).

Pretraining target: **bulk**, pooling a donor's cells across *all* cell types
(not one cell type) -- the closest OneK1K analogue of the paper's actual
bulk-tissue RNA-seq pretraining data (see `data.preprocess.load_bulk_pseudobulk_matrix`),
and it yields a single reusable pretrained trunk shared across every
downstream `--cell-type` fine-tuning run.
"""
from __future__ import annotations

import copy
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.preprocess import get_all_donor_ids, load_bulk_pseudobulk_matrix
from src.genome import GeneAnnotation, GeneRecord, ReferenceGenome, chrom_for_gene, get_tss_window, one_hot_encode_sequence
from src.heritability import autosomal_protein_coding_genes
from src.model import ReferenceExpressionModel
from src.regression_utils import safe_pearsonr
from src.splits import DEFAULT_SEED, DonorSplit, GeneSplit, split_donors, split_genes_by_chromosome
from src.wandb_logger import get_logger

__all__ = [
    "ReferenceExpressionDataset",
    "collate_reference_batch",
    "build_pretrain_pipeline",
    "run_eval",
    "summarize_eval",
    "save_splits",
    "train_reference_model",
    "set_seed",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ReferenceExpressionDataset(Dataset):
    """One item per *gene* (no donor dimension): TSS-window one-hot reference
    sequence + that gene's population-mean bulk-pseudobulk target.

    Reference sequences for every gene in `gene_ids` are fetched and one-hot
    encoded eagerly in `__init__` (not lazily on first access), since -- unlike
    `src/dataset.py`'s per-`(gene, donor)` haplotypes -- there is no donor
    multiplier and no VCF/pysam handles at all here (r-SAGE-net never touches
    genotypes), so none of the DataLoader-worker fork-safety concerns that
    motivate `PersonalGenomeSequenceBuilder`'s two-independent-readers /
    bounded-haplotype-cache design apply. This does mean the full gene
    universe's one-hot sequences are held in memory at once: for the paper-
    scale universe (~15-20K genes x 40kb x 4 bytes) that's ~9-12 GB, a real
    cost worth calling out but a fixed, predictable one (no per-donor
    multiplication, no cache eviction/growth over the course of training).
    """

    def __init__(
        self,
        gene_ids: Sequence[str],
        gene_records: Dict[str, GeneRecord],
        reference: ReferenceGenome,
        seq_len: int,
        population_mean: Dict[str, float],
        chrom_style: Optional[str] = None,
        desc: str = "caching reference sequences",
    ):
        kept_gene_ids = [g for g in gene_ids if g in population_mean and not np.isnan(population_mean[g])]
        if len(kept_gene_ids) == 0:
            raise ValueError("No genes with a defined population-mean target -- check min_cells_per_donor / donor split overlap")
        self.gene_ids = kept_gene_ids
        self.population_mean = population_mean

        self._sequences: Dict[str, np.ndarray] = {}
        for gene_id in tqdm(self.gene_ids, desc=desc, leave=False):
            gene = gene_records[gene_id]
            chrom = chrom_for_gene(gene, chrom_style)
            start, end = get_tss_window(gene, seq_len)
            self._sequences[gene_id] = one_hot_encode_sequence(reference.fetch(chrom, start, end))

    def __len__(self) -> int:
        return len(self.gene_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        gene_id = self.gene_ids[idx]
        ref = self._sequences[gene_id]
        target = self.population_mean[gene_id]
        return torch.from_numpy(ref), torch.tensor(target, dtype=torch.float32), gene_id


def collate_reference_batch(batch: List[Tuple[torch.Tensor, torch.Tensor, str]]):
    ref, target, gene_ids = zip(*batch)
    return torch.stack(ref, dim=0), torch.stack(target, dim=0), list(gene_ids)


def build_pretrain_pipeline(
    h5ad_path: str,
    gtf_path: str,
    genome_fasta: str,
    seq_len: int = 40_000,
    chrom_style: Optional[str] = None,
    min_cells_per_donor: int = 1,
    donor_val_frac: float = 0.15,
    donor_test_frac: float = 0.15,
    gene_val_frac: float = 0.15,
    gene_test_frac: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> Tuple[ReferenceExpressionDataset, ReferenceExpressionDataset, ReferenceExpressionDataset, DonorSplit, GeneSplit]:
    """End-to-end: gene universe -> bulk pseudobulk -> donor/gene splits -> population means -> datasets.

    Reuses the same building blocks as the rest of the pipeline: candidate
    genes via `autosomal_protein_coding_genes` (`src/heritability.py`, the
    same protein-coding/autosomal/present-in-h5ad filter used for the
    per-cell-type heritability ranking), donor/gene splitting via
    `src/splits.py`, and the population-mean convention (mean of *train*
    donors' pseudobulk means) from `data.preprocess.compute_population_means` --
    reimplemented here directly on the `[donors x genes]` bulk pseudobulk
    matrix rather than the per-cell `MultiGeneDonorTable`, since bulk
    pretraining never needs per-cell targets.
    """
    import anndata as ad

    annotation = GeneAnnotation(gtf_path)
    var_gene_ids = list(ad.read_h5ad(h5ad_path, backed="r").var.index.astype(str))
    candidate_genes = autosomal_protein_coding_genes(annotation, var_gene_ids)
    print(f"[pretrain] {len(candidate_genes)} protein-coding autosomal candidate genes found in h5ad var")
    if len(candidate_genes) == 0:
        raise ValueError("No candidate genes found -- check --gtf biotype field and h5ad var gene IDs match")

    gene_records = {g.gene_id: g for g in candidate_genes}
    gene_ids = list(gene_records.keys())

    pb = load_bulk_pseudobulk_matrix(h5ad_path, gene_ids, min_cells_per_donor=min_cells_per_donor)
    print(f"[pretrain] bulk pseudobulk matrix: {len(pb.donor_ids)} donors x {len(pb.gene_ids)} genes")

    donor_universe = get_all_donor_ids(h5ad_path)
    raw_donor_split = split_donors(donor_universe, val_frac=donor_val_frac, test_frac=donor_test_frac, seed=seed)
    pb_donor_set = set(pb.donor_ids)
    donor_split = DonorSplit(
        train=[d for d in raw_donor_split.train if d in pb_donor_set],
        val=[d for d in raw_donor_split.val if d in pb_donor_set],
        test=[d for d in raw_donor_split.test if d in pb_donor_set],
    )
    print(
        f"[pretrain] donors: train={len(donor_split.train)} val={len(donor_split.val)} test={len(donor_split.test)} "
        f"(out of {len(donor_universe)} total)"
    )
    if len(donor_split.train) == 0:
        raise ValueError(f"No train donors with >= {min_cells_per_donor} cells (any type) -- cannot compute population means")

    gene_to_chrom = {g.gene_id: g.chrom for g in candidate_genes}
    gene_split = split_genes_by_chromosome(gene_to_chrom, val_frac=gene_val_frac, test_frac=gene_test_frac, seed=seed)
    print(f"[pretrain] {gene_split.summary()}")

    # Population mean = mean of TRAIN donors' bulk pseudobulk means, same
    # convention as `data.preprocess.compute_population_means` (never uses
    # val/test donors, so val/test genes' targets don't leak val/test-donor
    # expression into a value they're also scored against).
    population_mean = pb.mean.loc[donor_split.train].mean(axis=0).to_dict()

    reference = ReferenceGenome(genome_fasta)
    train_ds = ReferenceExpressionDataset(
        gene_split.train, gene_records, reference, seq_len, population_mean, chrom_style, desc="caching train reference sequences"
    )
    val_ds = ReferenceExpressionDataset(
        gene_split.val, gene_records, reference, seq_len, population_mean, chrom_style, desc="caching val reference sequences"
    )
    test_ds = ReferenceExpressionDataset(
        gene_split.test, gene_records, reference, seq_len, population_mean, chrom_style, desc="caching test reference sequences"
    )
    print(f"[pretrain] genes with defined targets: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    return train_ds, val_ds, test_ds, donor_split, gene_split


@torch.no_grad()
def run_eval(
    model: ReferenceExpressionModel, dataset: ReferenceExpressionDataset, device: str, batch_size: int, desc: str = "eval"
) -> pd.DataFrame:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_reference_batch)
    rows: List[dict] = []
    for ref, target, gene_ids in tqdm(loader, desc=desc, leave=False):
        ref = ref.to(device)
        m_hat = model(ref).cpu().numpy()
        target_np = target.numpy()
        for i, gene_id in enumerate(gene_ids):
            rows.append({"gene_id": gene_id, "y_true_population_mean": float(target_np[i]), "y_pred_m_hat": float(m_hat[i])})
    return pd.DataFrame(rows)


def summarize_eval(df: pd.DataFrame, prefix: str) -> Dict[str, float]:
    """Single held-out-gene Pearson r (paper's own r-SAGE-net monitoring approach --
    no per-donor axis exists here to build a per-gene/per-donor grouped metric from,
    unlike `src/metrics.py::grouped_correlation`)."""
    y_true = df["y_true_population_mean"].to_numpy(dtype=np.float64)
    y_pred = df["y_pred_m_hat"].to_numpy(dtype=np.float64)
    r, p = safe_pearsonr(y_true, y_pred)
    mse = float(np.mean((y_true - y_pred) ** 2)) if len(df) else float("nan")
    return {
        f"{prefix}/pearson_r": r,
        f"{prefix}/pearson_p": p,
        f"{prefix}/mse": mse,
        f"{prefix}/n_genes": len(df),
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


def train_reference_model(
    model: ReferenceExpressionModel,
    train_ds: ReferenceExpressionDataset,
    val_ds: ReferenceExpressionDataset,
    test_ds: ReferenceExpressionDataset,
    device: str,
    out_dir: str,
    batch_size: int = 64,
    eval_batch_size: int = 256,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip: Optional[float] = 1.0,
    patience: int = 10,
    num_workers: int = 0,
    save_checkpoint: bool = True,
    logger=None,
) -> ReferenceExpressionModel:
    """Plain-MSE training loop with early stopping + incremental best-checkpoint
    saving on held-out (val) gene Pearson r -- structurally mirrors
    `src/train.py`'s loop (tqdm epoch/batch progress, early stopping,
    checkpointing) but much simpler: no donor dimension, no 4-way eval matrix,
    no GNLL -- just `F.mse_loss` against each gene's population-mean target.
    """
    logger = logger if logger is not None else get_logger(False)
    os.makedirs(out_dir, exist_ok=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_reference_batch, num_workers=num_workers
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_metric = -float("inf")
    best_state: Optional[dict] = None
    epochs_without_improvement = 0

    epoch_bar = tqdm(range(epochs), desc="epochs")
    for epoch in epoch_bar:
        model.train()
        losses: List[float] = []
        batch_bar = tqdm(train_loader, desc=f"epoch {epoch} [train]", leave=False)
        for ref, target, _gene_ids in batch_bar:
            ref, target = ref.to(device), target.to(device)
            m_hat = model(ref)
            loss = F.mse_loss(m_hat, target)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            losses.append(loss.item())
            batch_bar.set_postfix(loss=f"{losses[-1]:.4f}")

        log_payload = {"train/loss": float(np.mean(losses)) if losses else float("nan"), "epoch": epoch}
        tqdm.write(f"[epoch {epoch}] train_loss={log_payload['train/loss']:.4f}")

        val_df = run_eval(model, val_ds, device, eval_batch_size, desc=f"epoch {epoch} [val]")
        val_metrics = summarize_eval(val_df, "val")
        log_payload.update(val_metrics)
        monitor = val_metrics["val/pearson_r"]
        tqdm.write(
            f"[epoch {epoch}] val/pearson_r={monitor:.4f}" if not np.isnan(monitor) else f"[epoch {epoch}] val metric undefined (nan)"
        )

        if not np.isnan(monitor) and monitor > best_metric:
            best_metric = monitor
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            if save_checkpoint:
                # Persisted immediately (not just at the end) so a crash/OOM/preemption
                # later in a long run doesn't lose an already-improved checkpoint.
                torch.save(best_state, os.path.join(out_dir, "reference_model.pt"))
        else:
            epochs_without_improvement += 1
        epoch_bar.set_postfix(val_r=f"{monitor:.4f}", best=f"{best_metric:.4f}", no_improve=epochs_without_improvement)

        logger.log(log_payload, step=epoch)

        if epochs_without_improvement >= patience:
            tqdm.write(f"[early stop] no val/pearson_r improvement for {patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("[warning] validation metric was always nan -- reporting final-epoch weights instead of a 'best' checkpoint")

    print("[final] evaluating held-out test genes")
    test_df = run_eval(model, test_ds, device, eval_batch_size, desc="final [test]")
    test_df.to_csv(os.path.join(out_dir, "predictions_test_genes.csv"), index=False)
    test_metrics = summarize_eval(test_df, "test")
    tqdm.write(f"[final:test] Pearson r={test_metrics['test/pearson_r']:.4f} (n_genes={test_metrics['test/n_genes']})")
    logger.log(test_metrics)
    pd.DataFrame([test_metrics]).to_csv(os.path.join(out_dir, "final_test_summary.csv"), index=False)

    if save_checkpoint:
        torch.save(model.state_dict(), os.path.join(out_dir, "reference_model.pt"))
    else:
        print("[checkpoint] save_checkpoint=False, skipping reference_model.pt")
    logger.finish()

    return model
