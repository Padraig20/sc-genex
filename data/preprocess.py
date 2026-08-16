"""Multi-gene, single-cell-type extraction from the OneK1K h5ad.

Two loaders exist, at different granularities, both built on a single backed
h5ad read (never materializing the full `[cells x genes]` matrix -- only the
requested cell type's rows and requested genes' columns are ever read):

1. `load_celltype_pseudobulk_matrix`: collapses straight to a `[donors x
   genes]` pseudobulk-mean matrix via sparse donor-indicator multiplication,
   so it stays cheap even across the full protein-coding-autosomal candidate
   gene universe (tens of thousands of genes) used by the heritability
   ranking in `src/heritability.py`.
2. `load_celltype_multigene_cells` + `build_multigene_donor_table`: keeps
   every individual cell's normalized expression value, for the (much
   smaller, ~1000-gene) final training gene set -- these per-cell values are
   the model's actual training targets (see `src/dataset.py`, `src/loss.py`).

Both use the same `log1p(CP-median)` per-cell normalization as the original
single-gene POC.

`load_celltype_pseudobulk_matrix`/`_read_celltype_gene_submatrix` also accept
`cell_type=None` ("bulk": pool every cell of every donor, any type), used by
the reference-only ("r-SAGE-net") pretraining stage in `src/pretrain.py`; see
`load_bulk_pseudobulk_matrix` for an explicit wrapper and `get_all_donor_ids`
for the corresponding donor universe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    import anndata as ad
except ImportError:  # pragma: no cover - optional at import time for tooling
    ad = None

# Re-exported for backwards compatibility with code that imported these from
# `data.preprocess` before donor/gene splitting moved to `src/splits.py`.
from src.splits import DEFAULT_SEED, DonorSplit, split_donors  # noqa: F401


def get_celltype_donor_ids(
    h5ad_path: str, cell_type: str, donor_col: str = "donor_id", cell_type_col: str = "cell_type"
) -> List[str]:
    """All distinct donors with at least one cell of `cell_type` (gene-independent).

    Used to compute a single canonical donor split (see `src/splits.py`) that
    is reused for heritability ranking, model training, and final evaluation
    -- so "unseen individual" means the same set of individuals everywhere in
    the pipeline, and heritability ranking never trains/ranks on donors held
    out as the final test set.
    """
    if ad is None:
        raise ImportError("anndata is required to read the OneK1K h5ad file")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    cell_mask = (adata.obs[cell_type_col] == cell_type).to_numpy()
    if cell_mask.sum() == 0:
        raise ValueError(f"No cells found for cell_type='{cell_type}'")
    donors = adata.obs.loc[cell_mask, donor_col].astype(str).unique().tolist()
    return sorted(donors)


def get_all_donor_ids(h5ad_path: str, donor_col: str = "donor_id") -> List[str]:
    """All distinct donors with any cells at all, regardless of cell type.

    Bulk equivalent of `get_celltype_donor_ids` -- used for the cell-type-
    agnostic reference-only ("r-SAGE-net") pretraining stage (`src/pretrain.py`),
    which pools all of a donor's cells (any type) into a single bulk-pseudobulk
    value per gene, mirroring the paper's actual bulk-tissue RNA-seq pretraining
    data (never any FACS-sorted subset).
    """
    if ad is None:
        raise ImportError("anndata is required to read the OneK1K h5ad file")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    donors = adata.obs[donor_col].astype(str).unique().tolist()
    return sorted(donors)


def _resolve_gene_index(var: "pd.DataFrame", gene: str) -> int:
    """Resolves a gene identifier (Ensembl ID or symbol) to a `var` row position."""
    if gene in var.index:
        return var.index.get_loc(gene)
    if "feature_name" in var.columns:
        matches = np.where(var["feature_name"].astype(str).to_numpy() == gene)[0]
        if len(matches) > 0:
            return int(matches[0])
    raise KeyError(f"Gene '{gene}' not found by Ensembl ID or feature_name in var")


def _normalize_sparse_rows(X: "sp.spmatrix", total_count: np.ndarray) -> "sp.csr_matrix":
    """CP-median + log1p per-cell (row) normalization, preserving sparsity.

    `log1p(0) == 0`, so scaling each row by a positive per-cell factor and
    then applying `log1p` to only the stored (nonzero) entries is exact --
    there's no need to ever densify the matrix.
    """
    X = X.tocsr()
    median_total = float(np.median(total_count))
    scaling = (median_total / np.clip(total_count, 1.0, None)).astype(np.float32)
    norm = sp.diags(scaling) @ X
    norm = norm.tocsr()
    norm.data = np.log1p(norm.data).astype(np.float32)
    return norm


def _read_celltype_gene_submatrix(
    h5ad_path: str,
    cell_type: Optional[str],
    gene_ids: Sequence[str],
    donor_col: str,
    cell_type_col: str,
    library_size_col: str,
):
    """Shared backed-mode h5ad read: cell-type-filtered rows, requested gene columns.

    `cell_type=None` means "bulk": no cell-type filter at all, every cell of
    every donor is included (mirroring a real bulk RNA-seq assay over the same
    sample, used by the reference-only pretraining stage in `src/pretrain.py`).
    """
    if ad is None:
        raise ImportError("anndata is required to read the OneK1K h5ad file")

    adata = ad.read_h5ad(h5ad_path, backed="r")
    if cell_type is None:
        cell_mask = np.ones(adata.n_obs, dtype=bool)
    else:
        cell_mask = (adata.obs[cell_type_col] == cell_type).to_numpy()
    if cell_mask.sum() == 0:
        raise ValueError(f"No cells found for cell_type='{cell_type}'")

    gene_positions = [_resolve_gene_index(adata.var, g) for g in gene_ids]
    sub = adata[cell_mask, gene_positions].to_memory()
    X = sub.X
    if not sp.issparse(X):
        X = sp.csr_matrix(np.asarray(X))
    total_count = sub.obs[library_size_col].to_numpy().astype(np.float32)
    donor_id = sub.obs[donor_col].astype(str).to_numpy()
    return X, donor_id, total_count


@dataclass
class PseudobulkMatrix:
    """Per-donor mean `log1p(CP-median)` expression for many genes at once."""

    donor_ids: List[str]
    gene_ids: List[str]
    mean: "pd.DataFrame"  # [donors x genes]
    n_cells: Dict[str, int]


def load_celltype_pseudobulk_matrix(
    h5ad_path: str,
    cell_type: Optional[str],
    gene_ids: Sequence[str],
    min_cells_per_donor: int = 1,
    donor_col: str = "donor_id",
    cell_type_col: str = "cell_type",
    library_size_col: str = "nCount_RNA",
) -> PseudobulkMatrix:
    """One backed read for arbitrarily many genes -> `[donors x genes]` pseudobulk means.

    Never densifies the full `[cells x genes]` matrix: normalization is done
    on the sparse matrix's nonzero entries directly (see
    `_normalize_sparse_rows`), and the donor-level mean is computed via a
    single sparse `[n_donors, n_cells] @ [n_cells, n_genes]` multiplication
    (a sparse "indicator/averaging" matrix), so memory stays bounded by
    `n_donors x n_genes` regardless of how many candidate genes are passed in
    -- this is what makes it cheap enough to run over the full
    protein-coding-autosomal gene universe for heritability ranking.

    `cell_type=None` computes a "bulk" pseudobulk instead: every one of a
    donor's cells (any type) is pooled together, mirroring a real bulk
    RNA-seq assay over the same sample -- used by the reference-only
    pretraining stage (`src/pretrain.py`, `load_bulk_pseudobulk_matrix` below).
    """
    X, donor_id, total_count = _read_celltype_gene_submatrix(
        h5ad_path, cell_type, gene_ids, donor_col, cell_type_col, library_size_col
    )
    norm = _normalize_sparse_rows(X, total_count)

    donors_all = sorted(set(donor_id))
    donor_to_idx = {d: i for i, d in enumerate(donors_all)}
    row_idx = np.array([donor_to_idx[d] for d in donor_id])
    n_cells_arr = np.bincount(row_idx, minlength=len(donors_all))

    indicator = sp.csr_matrix(
        (1.0 / n_cells_arr[row_idx], (row_idx, np.arange(len(donor_id)))),
        shape=(len(donors_all), len(donor_id)),
    )
    pseudobulk = indicator @ norm  # [n_donors, n_genes]
    pseudobulk = np.asarray(pseudobulk.todense()) if sp.issparse(pseudobulk) else np.asarray(pseudobulk)

    keep_mask = n_cells_arr >= min_cells_per_donor
    donors_kept = [d for d, keep in zip(donors_all, keep_mask) if keep]
    pseudobulk_kept = pseudobulk[keep_mask]

    mean_df = pd.DataFrame(pseudobulk_kept, index=donors_kept, columns=list(gene_ids))
    n_cells = {d: int(n_cells_arr[donor_to_idx[d]]) for d in donors_kept}
    return PseudobulkMatrix(donor_ids=donors_kept, gene_ids=list(gene_ids), mean=mean_df, n_cells=n_cells)


def load_bulk_pseudobulk_matrix(
    h5ad_path: str,
    gene_ids: Sequence[str],
    min_cells_per_donor: int = 1,
    donor_col: str = "donor_id",
    library_size_col: str = "nCount_RNA",
) -> PseudobulkMatrix:
    """Bulk (all-cell-types) pseudobulk matrix -- thin, explicit wrapper around
    `load_celltype_pseudobulk_matrix(..., cell_type=None, ...)`.

    Used for the r-SAGE-net-style reference-only pretraining stage
    (`src/pretrain.py`): pools every cell of every donor (regardless of
    `cell_type`) into one `[donors x genes]` pseudobulk-mean matrix, the
    closest OneK1K analogue of the paper's actual bulk-tissue RNA-seq
    pretraining data.
    """
    return load_celltype_pseudobulk_matrix(
        h5ad_path,
        cell_type=None,
        gene_ids=gene_ids,
        min_cells_per_donor=min_cells_per_donor,
        donor_col=donor_col,
        library_size_col=library_size_col,
    )


@dataclass
class MultiGeneCellData:
    """Per-cell normalized expression for a (typically small, ~1000-gene) gene set."""

    donor_id: np.ndarray  # [n_cells] str
    gene_ids: List[str]
    normalized_expr: np.ndarray  # [n_cells, n_genes] float32, log1p(CP-median)


def load_celltype_multigene_cells(
    h5ad_path: str,
    cell_type: str,
    gene_ids: Sequence[str],
    donor_col: str = "donor_id",
    cell_type_col: str = "cell_type",
    library_size_col: str = "nCount_RNA",
) -> MultiGeneCellData:
    """Per-cell (not pseudobulked) normalized expression for `gene_ids`.

    Unlike `load_celltype_pseudobulk_matrix`, this densifies the normalized
    matrix (`[n_cells, n_genes]`) since per-cell values -- not just donor
    means -- are the actual model training targets. Only intended for the
    final, much smaller (~1000-gene) training gene set, not the full
    candidate universe.
    """
    X, donor_id, total_count = _read_celltype_gene_submatrix(
        h5ad_path, cell_type, gene_ids, donor_col, cell_type_col, library_size_col
    )
    norm = _normalize_sparse_rows(X, total_count)
    normalized_expr = np.asarray(norm.todense(), dtype=np.float32)
    return MultiGeneCellData(donor_id=donor_id, gene_ids=list(gene_ids), normalized_expr=normalized_expr)


@dataclass
class MultiGeneDonorTable:
    """Per-(gene, donor) cell-level targets + pseudobulk stats, for many genes at once.

    `population_mean` (per gene, over TRAIN donors only) is filled in
    separately by `compute_population_means` once the donor split is known --
    it is the model's "mean expression" target (pSAGE-net's `m_g`, predicted
    from reference sequence alone) and must never be computed using val/test
    donors to avoid leaking their expression values into a target that val/
    test examples are also scored against.
    """

    gene_ids: List[str]
    donor_ids: List[str]
    cell_targets: Dict[Tuple[str, str], np.ndarray]  # (gene_id, donor_id) -> [n_cells_i]
    pseudobulk_mean: Dict[Tuple[str, str], float]
    pseudobulk_std: Dict[Tuple[str, str], float]
    n_cells: Dict[Tuple[str, str], int]
    population_mean: Dict[str, float] = field(default_factory=dict)

    def has(self, gene_id: str, donor_id: str) -> bool:
        return (gene_id, donor_id) in self.cell_targets


def build_multigene_donor_table(data: MultiGeneCellData, min_cells_per_donor: int = 5) -> MultiGeneDonorTable:
    """Groups per-cell values by (gene, donor), dropping (gene, donor) pairs below the min-cells filter.

    Vectorized over genes: cells are sorted by donor once, then each donor's
    contiguous block of rows is reduced (mean/std) across all gene columns in
    a single operation, rather than looping `groupby` per gene.
    """
    donor_id = data.donor_id
    order = np.argsort(donor_id, kind="stable")
    sorted_donors = donor_id[order]
    sorted_expr = data.normalized_expr[order]  # [n_cells, n_genes]

    unique_donors, start_idx, counts = np.unique(sorted_donors, return_index=True, return_counts=True)

    cell_targets: Dict[Tuple[str, str], np.ndarray] = {}
    pseudobulk_mean: Dict[Tuple[str, str], float] = {}
    pseudobulk_std: Dict[Tuple[str, str], float] = {}
    n_cells: Dict[Tuple[str, str], int] = {}
    donor_ids_kept: List[str] = []

    for donor, start, count in zip(unique_donors, start_idx, counts):
        if count < min_cells_per_donor:
            continue
        block = sorted_expr[start : start + count]  # [count, n_genes]
        means = block.mean(axis=0)
        stds = block.std(axis=0, ddof=0)
        donor_ids_kept.append(str(donor))
        for gi, gene in enumerate(data.gene_ids):
            key = (gene, str(donor))
            cell_targets[key] = block[:, gi].copy()
            pseudobulk_mean[key] = float(means[gi])
            pseudobulk_std[key] = float(stds[gi])
            n_cells[key] = int(count)

    return MultiGeneDonorTable(
        gene_ids=list(data.gene_ids),
        donor_ids=sorted(donor_ids_kept),
        cell_targets=cell_targets,
        pseudobulk_mean=pseudobulk_mean,
        pseudobulk_std=pseudobulk_std,
        n_cells=n_cells,
    )


def compute_population_means(table: MultiGeneDonorTable, train_donor_ids: Sequence[str]) -> Dict[str, float]:
    """Per-gene population mean = mean of TRAIN donors' pseudobulk means (pSAGE-net's `m_g`).

    Mutates and returns `table.population_mean`. Must be called with the
    training donor split *before* building any dataset, since this value is
    a fixed regression target shared by every (gene, donor) example
    regardless of split.
    """
    train_set = set(train_donor_ids)
    population_mean: Dict[str, float] = {}
    for gene in table.gene_ids:
        values = [table.pseudobulk_mean[(gene, d)] for d in table.donor_ids if d in train_set and table.has(gene, d)]
        population_mean[gene] = float(np.mean(values)) if values else float("nan")
    table.population_mean = population_mean
    return population_mean


def prepare_multigene_celltype_data(
    h5ad_path: str,
    cell_type: str,
    gene_ids: Sequence[str],
    min_cells_per_donor: int = 5,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> Tuple[MultiGeneDonorTable, DonorSplit]:
    """End-to-end: load per-cell values for `gene_ids`, group by donor, split donors, fill population means."""
    cell_data = load_celltype_multigene_cells(h5ad_path, cell_type, gene_ids)
    table = build_multigene_donor_table(cell_data, min_cells_per_donor=min_cells_per_donor)
    donor_split = split_donors(table.donor_ids, val_frac=val_frac, test_frac=test_frac, seed=seed)
    compute_population_means(table, donor_split.train)
    return table, donor_split


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preview per-cell/pseudobulk stats for a gene set x cell-type")
    parser.add_argument("--h5ad-path", required=True)
    parser.add_argument("--genes", nargs="+", required=True, help="One or more Ensembl gene IDs")
    parser.add_argument("--cell-type", required=True)
    parser.add_argument("--min-cells-per-donor", type=int, default=5)
    args = parser.parse_args()

    table, split = prepare_multigene_celltype_data(args.h5ad_path, args.cell_type, args.genes, args.min_cells_per_donor)
    print(f"Donors passing min-cells filter (any gene): {len(table.donor_ids)}")
    print(f"train={len(split.train)} val={len(split.val)} test={len(split.test)}")
    for gene in args.genes:
        print(f"  {gene}: population_mean(train)={table.population_mean.get(gene, float('nan')):.4f}")
