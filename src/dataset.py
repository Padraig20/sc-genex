"""PyTorch datasets for the multi-gene, multi-donor, single-cell pSAGE-net rebuild.

Two datasets, both built on `PersonalGenomeSequenceBuilder` (below), which
turns a `(gene, donor)` pair into `(ref_seq, mat_seq, pat_seq)` one-hot
tensors:

- `PersonalGenomeDatasetSC`: **training** dataset. One item per `(gene,
  donor)` pair drawn from `train_genes x train_donors`, with that pair's
  full, variable-length per-cell targets (used by `src/loss.py`'s per-cell
  GNLL -- no pseudobulking).
- `PersonalGenomeEvalDataset`: **evaluation** dataset. One item per `(gene,
  donor)` pair from an *arbitrary* genes x donors combination (used for all
  4 cells of the seen/unseen-gene x seen/unseen-individual matrix in
  `src/train.py`), with pseudobulk mean/std targets.
"""
from __future__ import annotations

import os
import sys
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.preprocess import MultiGeneDonorTable
from src.genome import (
    GeneRecord,
    ReferenceGenome,
    VCFGenotypeReader,
    build_haplotype_onehots,
    chrom_for_gene,
    get_tss_window,
    one_hot_encode_sequence,
    onek1k_sample_id_from_donor_id,
)


class _BoundedCache(OrderedDict):
    """Simple LRU cache: evicts the least-recently-used entry past `max_size`."""

    def __init__(self, max_size: int):
        super().__init__()
        self.max_size = max_size

    def get_or_none(self, key):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None

    def put(self, key, value) -> None:
        self[key] = value
        self.move_to_end(key)
        if len(self) > self.max_size:
            self.popitem(last=False)


class PersonalGenomeSequenceBuilder:
    """Builds reference + phased maternal/paternal one-hot sequences for `(gene, donor)` pairs.

    The reference sequence only depends on the gene (TSS window), so it's
    cached without bound (naturally capped at the number of distinct genes,
    e.g. ~1000). Maternal/paternal haplotypes depend on both gene and donor,
    so caching every combination would be unbounded (potentially hundreds of
    thousands of `[seq_len, 4]` float32 arrays) -- these are cached in a
    small bounded LRU (`max_haplotype_cache`) instead, which fully
    eliminates re-fetching for evaluation sets (queried identically every
    epoch, and capped in size by `PersonalGenomeEvalDataset`) while staying
    memory-safe for the much larger training set (where cache misses simply
    fall back to a fresh, cheap VCF region query).

    Donor ID resolution mirrors the original single-gene POC: the h5ad
    `donor_id` (e.g. `"10_10"`) is not generally the same string as the
    corresponding sample ID in the genotype files (e.g. `"OneK1K_10"`).
    `sample_id_fn` implements the default OneK1K convention; `donor_id_to_sample_id`
    is an explicit override dict, checked first.
    """

    def __init__(
        self,
        reference: ReferenceGenome,
        vcf_reader: VCFGenotypeReader,
        genes: Dict[str, GeneRecord],
        seq_len: int,
        vcf_chrom_style: Optional[str] = None,
        donor_id_to_sample_id: Optional[Dict[str, str]] = None,
        sample_id_fn: Callable[[str], str] = onek1k_sample_id_from_donor_id,
        strict_phasing: bool = False,
        max_haplotype_cache: int = 20_000,
    ):
        self.reference = reference
        self.vcf_reader = vcf_reader
        self.genes = genes
        self.seq_len = seq_len
        self.vcf_chrom_style = vcf_chrom_style
        self.donor_id_to_sample_id = donor_id_to_sample_id or {}
        self.sample_id_fn = sample_id_fn
        self.strict_phasing = strict_phasing

        self._window_cache: Dict[str, Tuple[str, int, int]] = {}
        self._ref_cache: Dict[str, np.ndarray] = {}
        self._hap_cache = _BoundedCache(max_haplotype_cache)

    def resolve_sample_id(self, donor_id: str) -> str:
        if donor_id in self.donor_id_to_sample_id:
            return self.donor_id_to_sample_id[donor_id]
        return self.sample_id_fn(donor_id)

    def _window(self, gene_id: str) -> Tuple[str, int, int]:
        if gene_id not in self._window_cache:
            gene = self.genes[gene_id]
            chrom = chrom_for_gene(gene, self.vcf_chrom_style)
            start, end = get_tss_window(gene, self.seq_len)
            self._window_cache[gene_id] = (chrom, start, end)
        return self._window_cache[gene_id]

    def build_reference(self, gene_id: str) -> np.ndarray:
        if gene_id not in self._ref_cache:
            chrom, start, end = self._window(gene_id)
            self._ref_cache[gene_id] = one_hot_encode_sequence(self.reference.fetch(chrom, start, end))
        return self._ref_cache[gene_id]

    def build_haplotypes(self, gene_id: str, donor_id: str) -> Tuple[np.ndarray, np.ndarray]:
        key = (gene_id, donor_id)
        cached = self._hap_cache.get_or_none(key)
        if cached is not None:
            return cached
        chrom, start, end = self._window(gene_id)
        sample_id = self.resolve_sample_id(donor_id)
        mat, pat = build_haplotype_onehots(
            self.reference, self.vcf_reader, sample_id, chrom, start, end, strict_phasing=self.strict_phasing
        )
        self._hap_cache.put(key, (mat, pat))
        return mat, pat

    def build(self, gene_id: str, donor_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ref = self.build_reference(gene_id)
        mat, pat = self.build_haplotypes(gene_id, donor_id)
        return ref, mat, pat


def _enumerate_available_pairs(
    gene_ids: Sequence[str], donor_ids: Sequence[str], table: MultiGeneDonorTable
) -> List[Tuple[str, str]]:
    return [(g, d) for g in gene_ids for d in donor_ids if table.has(g, d)]


class PersonalGenomeDatasetSC(Dataset):
    """Training dataset: one item per `(gene, donor)` pair, with per-cell targets.

    Each item is scored against every one of that donor's actual per-cell
    values for that gene (see `src/loss.py::per_cell_gaussian_nll`) -- there
    is no pseudobulking anywhere in the training path.
    """

    def __init__(
        self,
        gene_ids: Sequence[str],
        donor_ids: Sequence[str],
        table: MultiGeneDonorTable,
        sequence_builder: PersonalGenomeSequenceBuilder,
    ):
        self.pairs = _enumerate_available_pairs(gene_ids, donor_ids, table)
        if len(self.pairs) == 0:
            raise ValueError(
                "No (gene, donor) pairs available for training -- check min_cells_per_donor "
                "filtering and that the gene/donor lists overlap the donor table"
            )
        self.table = table
        self.sequence_builder = sequence_builder

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
        gene_id, donor_id = self.pairs[idx]
        ref, mat, pat = self.sequence_builder.build(gene_id, donor_id)
        cell_values = self.table.cell_targets[(gene_id, donor_id)]
        population_mean = self.table.population_mean[gene_id]
        cell_diff = (cell_values - population_mean).astype(np.float32)
        return (
            torch.from_numpy(ref),
            torch.from_numpy(mat),
            torch.from_numpy(pat),
            torch.tensor(population_mean, dtype=torch.float32),
            torch.from_numpy(cell_diff),
            gene_id,
            donor_id,
        )


def collate_personal_genome_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, str]]
):
    """Stacks fixed-length sequences/scalars; keeps variable-length per-cell target lists as-is."""
    ref, mat, pat, population_mean, cell_diff, gene_ids, donor_ids = zip(*batch)
    return (
        torch.stack(ref, dim=0),
        torch.stack(mat, dim=0),
        torch.stack(pat, dim=0),
        torch.stack(population_mean, dim=0),
        list(cell_diff),
        list(gene_ids),
        list(donor_ids),
    )


class PersonalGenomeEvalDataset(Dataset):
    """Evaluation dataset: one item per `(gene, donor)` pair, with pseudobulk targets.

    Unlike `PersonalGenomeDatasetSC`, `gene_ids`/`donor_ids` may be *any*
    combination (train or test genes x train or test donors) -- this is what
    lets `src/train.py` build all 4 cells of the seen/unseen-gene x
    seen/unseen-individual evaluation matrix from the same class.

    `max_pairs` caps the (potentially huge, e.g. hundreds of genes x hundreds
    of donors) full cross-product via seeded random subsampling, keeping
    per-epoch evaluation tractable; pass `None` to use every available pair.
    """

    def __init__(
        self,
        gene_ids: Sequence[str],
        donor_ids: Sequence[str],
        table: MultiGeneDonorTable,
        sequence_builder: PersonalGenomeSequenceBuilder,
        max_pairs: Optional[int] = None,
        seed: int = 0,
    ):
        pairs = _enumerate_available_pairs(gene_ids, donor_ids, table)
        if len(pairs) == 0:
            raise ValueError(
                "No (gene, donor) pairs available for evaluation -- check min_cells_per_donor "
                "filtering and that the gene/donor lists overlap the donor table"
            )
        if max_pairs is not None and len(pairs) > max_pairs:
            rng = np.random.RandomState(seed)
            keep_idx = sorted(rng.choice(len(pairs), size=max_pairs, replace=False).tolist())
            pairs = [pairs[i] for i in keep_idx]
        self.pairs = pairs
        self.table = table
        self.sequence_builder = sequence_builder

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, str, str]:
        gene_id, donor_id = self.pairs[idx]
        ref, mat, pat = self.sequence_builder.build(gene_id, donor_id)
        key = (gene_id, donor_id)
        pseudobulk_mean = self.table.pseudobulk_mean[key]
        pseudobulk_std = self.table.pseudobulk_std[key]
        n_cells = self.table.n_cells[key]
        population_mean = self.table.population_mean.get(gene_id, float("nan"))
        return (
            torch.from_numpy(ref),
            torch.from_numpy(mat),
            torch.from_numpy(pat),
            torch.tensor(pseudobulk_mean, dtype=torch.float32),
            torch.tensor(pseudobulk_std, dtype=torch.float32),
            torch.tensor(population_mean, dtype=torch.float32),
            n_cells,
            gene_id,
            donor_id,
        )


def collate_eval_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, str, str]]
):
    ref, mat, pat, pseudobulk_mean, pseudobulk_std, population_mean, n_cells, gene_ids, donor_ids = zip(*batch)
    return (
        torch.stack(ref, dim=0),
        torch.stack(mat, dim=0),
        torch.stack(pat, dim=0),
        torch.stack(pseudobulk_mean, dim=0),
        torch.stack(pseudobulk_std, dim=0),
        torch.stack(population_mean, dim=0),
        torch.tensor(n_cells, dtype=torch.long),
        list(gene_ids),
        list(donor_ids),
    )
