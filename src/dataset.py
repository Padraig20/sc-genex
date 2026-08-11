"""PyTorch datasets for training/evaluating the Variformer GNLL POC.

Because this POC only ever trains a single (gene, cell_type) pair, each
dataset item is simply "one donor": their personalized, TSS-centered one-hot
DNA sequence (constant per donor, cached after first computation) plus their
single-cell expression targets for that gene/cell-type.

`SingleCellDonorDataset` keeps the full, variable-length list of per-cell
targets per donor (used for training with `loss.per_cell_gaussian_nll`).
`PseudobulkEvalDataset` collapses that to the donor-level pseudobulk mean/std
(used for validation/test, matching Variformer's cross-individual evaluation).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.preprocess import DonorTable
from src.genome import (
    GeneRecord,
    ReferenceGenome,
    VCFGenotypeReader,
    build_personalized_onehot,
    chrom_for_gene,
    get_tss_window,
)


class DonorSequenceBuilder:
    """Builds (and caches) the personalized one-hot sequence for a gene, per donor."""

    def __init__(
        self,
        reference: ReferenceGenome,
        vcf_reader: VCFGenotypeReader,
        gene: GeneRecord,
        seq_len: int,
        vcf_chrom_style: Optional[str] = None,
        donor_id_to_sample_id: Optional[Dict[str, str]] = None,
    ):
        self.reference = reference
        self.vcf_reader = vcf_reader
        self.gene = gene
        self.seq_len = seq_len
        self.chrom = chrom_for_gene(gene, vcf_chrom_style)
        self.start, self.end = get_tss_window(gene, seq_len)
        self.donor_id_to_sample_id = donor_id_to_sample_id or {}
        self._cache: Dict[str, np.ndarray] = {}

    def build(self, donor_id: str) -> np.ndarray:
        if donor_id not in self._cache:
            sample_id = self.donor_id_to_sample_id.get(donor_id, donor_id)
            self._cache[donor_id] = build_personalized_onehot(
                self.reference, self.vcf_reader, sample_id, self.chrom, self.start, self.end
            )
        return self._cache[donor_id]


class SingleCellDonorDataset(Dataset):
    """One item per donor: personalized sequence + all of that donor's per-cell targets."""

    def __init__(self, donor_ids: Sequence[str], table: DonorTable, sequence_builder: DonorSequenceBuilder):
        self.donor_ids = [d for d in donor_ids if d in table.cell_targets]
        missing = set(donor_ids) - set(self.donor_ids)
        if missing:
            raise ValueError(f"Donors missing from donor table (below min-cells filter?): {sorted(missing)}")
        self.table = table
        self.sequence_builder = sequence_builder

    def __len__(self) -> int:
        return len(self.donor_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        donor_id = self.donor_ids[idx]
        seq = self.sequence_builder.build(donor_id)
        targets = self.table.cell_targets[donor_id]
        return torch.from_numpy(seq), torch.from_numpy(targets), donor_id


def collate_donor_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, str]]
) -> Tuple[torch.Tensor, List[torch.Tensor], List[str]]:
    """Stacks fixed-length sequences; keeps variable-length cell-target lists as-is."""
    seqs, targets, donor_ids = zip(*batch)
    return torch.stack(seqs, dim=0), list(targets), list(donor_ids)


class PseudobulkEvalDataset(Dataset):
    """One item per donor: personalized sequence + donor-level pseudobulk mean/std."""

    def __init__(self, donor_ids: Sequence[str], table: DonorTable, sequence_builder: DonorSequenceBuilder):
        self.donor_ids = [d for d in donor_ids if d in table.cell_targets]
        missing = set(donor_ids) - set(self.donor_ids)
        if missing:
            raise ValueError(f"Donors missing from donor table (below min-cells filter?): {sorted(missing)}")
        self.table = table
        self.sequence_builder = sequence_builder

    def __len__(self) -> int:
        return len(self.donor_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, str]:
        donor_id = self.donor_ids[idx]
        seq = self.sequence_builder.build(donor_id)
        pseudobulk_mean = self.table.pseudobulk_mean[donor_id]
        empirical_std = self.table.pseudobulk_std[donor_id]
        n_cells = self.table.n_cells[donor_id]
        return (
            torch.from_numpy(seq),
            torch.tensor(pseudobulk_mean, dtype=torch.float32),
            torch.tensor(empirical_std, dtype=torch.float32),
            n_cells,
            donor_id,
        )
