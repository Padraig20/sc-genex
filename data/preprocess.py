"""Extracts per-cell expression for one (gene, cell_type) pair from the OneK1K
h5ad, normalizes it, groups it by donor, and produces donor train/val/test
splits.

Kept deliberately simple ("low-level POC"): uses AnnData's backed mode so the
~4GB h5ad is never fully loaded into RAM -- only the rows matching the
requested cell type and the single requested gene column are ever materialized.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import anndata as ad
except ImportError:  # pragma: no cover - optional at import time for tooling
    ad = None

DEFAULT_SEED = 0


def _resolve_gene_index(var: "pd.DataFrame", gene: str) -> int:
    """Resolves a gene identifier (Ensembl ID or symbol) to a `var` row position."""
    if gene in var.index:
        return var.index.get_loc(gene)
    if "feature_name" in var.columns:
        matches = np.where(var["feature_name"].astype(str).to_numpy() == gene)[0]
        if len(matches) > 0:
            return int(matches[0])
    raise KeyError(f"Gene '{gene}' not found by Ensembl ID or feature_name in var")


@dataclass
class CellLevelData:
    gene: str
    cell_type: str
    donor_id: np.ndarray  # [n_cells] str
    raw_count: np.ndarray  # [n_cells] float32
    total_count: np.ndarray  # [n_cells] float32, library size (e.g. nCount_RNA)
    normalized_expr: np.ndarray  # [n_cells] float32, log1p(CP-median)


def normalize_library_size(raw_count: np.ndarray, total_count: np.ndarray) -> np.ndarray:
    """CP-median + log1p normalization (standard scanpy-style per-cell normalization).

    This removes technical (sequencing-depth) variation between cells so that
    the remaining cell-cell spread reflects biological variability, which is
    exactly what the sigma head is meant to capture.
    """
    median_total = float(np.median(total_count))
    cp_median = raw_count / np.clip(total_count, 1.0, None) * median_total
    return np.log1p(cp_median).astype(np.float32)


def load_gene_celltype_cells(
    h5ad_path: str,
    gene: str,
    cell_type: str,
    donor_col: str = "donor_id",
    cell_type_col: str = "cell_type",
    library_size_col: str = "nCount_RNA",
) -> CellLevelData:
    """Extracts per-cell raw + normalized expression for one gene x cell-type."""
    if ad is None:
        raise ImportError("anndata is required to read the OneK1K h5ad file")

    adata = ad.read_h5ad(h5ad_path, backed="r")
    cell_mask = (adata.obs[cell_type_col] == cell_type).to_numpy()
    if cell_mask.sum() == 0:
        raise ValueError(f"No cells found for cell_type='{cell_type}'")

    gene_idx = _resolve_gene_index(adata.var, gene)

    sub = adata[cell_mask, [gene_idx]].to_memory()
    x = sub.X
    raw_count = np.asarray(x.todense() if hasattr(x, "todense") else x).reshape(-1).astype(np.float32)

    donor_id = sub.obs[donor_col].astype(str).to_numpy()
    total_count = sub.obs[library_size_col].to_numpy().astype(np.float32)
    normalized_expr = normalize_library_size(raw_count, total_count)

    return CellLevelData(
        gene=gene,
        cell_type=cell_type,
        donor_id=donor_id,
        raw_count=raw_count,
        total_count=total_count,
        normalized_expr=normalized_expr,
    )


@dataclass
class DonorSplit:
    train: List[str]
    val: List[str]
    test: List[str]


def split_donors(
    donor_ids: List[str],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> DonorSplit:
    """Seeded, non-overlapping donor split (mirrors Variformer's donor-level splits)."""
    rng = np.random.RandomState(seed)
    donors = sorted(set(donor_ids))
    rng.shuffle(donors)
    n = len(donors)
    n_val = int(round(n * val_frac))
    n_test = int(round(n * test_frac))
    val = donors[:n_val]
    test = donors[n_val : n_val + n_test]
    train = donors[n_val + n_test :]
    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))
    return DonorSplit(train=train, val=val, test=test)


@dataclass
class DonorTable:
    """Per-donor grouping of cell-level targets, plus pseudobulk summary stats."""

    donor_ids: List[str]
    cell_targets: Dict[str, np.ndarray]  # donor_id -> [n_cells_i] normalized expr
    pseudobulk_mean: Dict[str, float]
    pseudobulk_std: Dict[str, float]
    n_cells: Dict[str, int]


def build_donor_table(data: CellLevelData, min_cells_per_donor: int = 5) -> DonorTable:
    """Groups per-cell targets by donor and computes pseudobulk mean/std.

    Donors with fewer than `min_cells_per_donor` cells of this type are
    dropped: with too few cells, both the pseudobulk mean and (especially)
    the empirical cell-cell std are too noisy to be a meaningful training
    or evaluation target.
    """
    df = pd.DataFrame({"donor_id": data.donor_id, "normalized_expr": data.normalized_expr})
    grouped = df.groupby("donor_id")["normalized_expr"]

    cell_targets: Dict[str, np.ndarray] = {}
    pseudobulk_mean: Dict[str, float] = {}
    pseudobulk_std: Dict[str, float] = {}
    n_cells: Dict[str, int] = {}

    for donor_id, values in grouped:
        values = values.to_numpy(dtype=np.float32, copy=True)
        if len(values) < min_cells_per_donor:
            continue
        cell_targets[donor_id] = values
        pseudobulk_mean[donor_id] = float(values.mean())
        pseudobulk_std[donor_id] = float(values.std(ddof=0))
        n_cells[donor_id] = len(values)

    donor_ids = sorted(cell_targets.keys())
    return DonorTable(
        donor_ids=donor_ids,
        cell_targets=cell_targets,
        pseudobulk_mean=pseudobulk_mean,
        pseudobulk_std=pseudobulk_std,
        n_cells=n_cells,
    )


def prepare_gene_celltype_data(
    h5ad_path: str,
    gene: str,
    cell_type: str,
    min_cells_per_donor: int = 5,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> tuple[DonorTable, DonorSplit]:
    """End-to-end: load, normalize, group by donor, and split donors."""
    cell_data = load_gene_celltype_cells(h5ad_path, gene, cell_type)
    table = build_donor_table(cell_data, min_cells_per_donor=min_cells_per_donor)
    split = split_donors(table.donor_ids, val_frac=val_frac, test_frac=test_frac, seed=seed)
    return table, split


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preview per-cell/pseudobulk stats for a gene x cell-type")
    parser.add_argument("--h5ad-path", required=True)
    parser.add_argument("--gene", required=True)
    parser.add_argument("--cell-type", required=True)
    parser.add_argument("--min-cells-per-donor", type=int, default=5)
    args = parser.parse_args()

    cell_data = load_gene_celltype_cells(args.h5ad_path, args.gene, args.cell_type)
    table = build_donor_table(cell_data, min_cells_per_donor=args.min_cells_per_donor)
    print(f"Total cells: {len(cell_data.donor_id)}")
    print(f"Donors passing min-cells filter: {len(table.donor_ids)}")
    means = np.array(list(table.pseudobulk_mean.values()))
    stds = np.array(list(table.pseudobulk_std.values()))
    if len(means) > 0:
        print(f"Pseudobulk mean expr across donors: mean={means.mean():.3f} std={means.std():.3f}")
        print(f"Within-donor cell-cell std: mean={stds.mean():.3f} std={stds.std():.3f}")
