"""Train/val/test splitting: donors (random) and genes (chromosome-bucket).

Two independent splits are combined to build the 4-way seen/unseen
evaluation matrix (see `src/train.py`):
- `split_donors`: random, seeded individual-level split (same philosophy as
  the original single-gene POC and pSAGE-net's ROSMAP/GTEx individual
  splits).
- `split_genes_by_chromosome`: gene-level split, keeping whole chromosomes
  together per split (like pSAGE-net's/Enformer's fixed chromosome-range
  gene splits), to avoid nearby-locus leakage between train/val/test genes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

DEFAULT_SEED = 0


@dataclass
class DonorSplit:
    train: List[str]
    val: List[str]
    test: List[str]


def split_donors(
    donor_ids: Sequence[str],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> DonorSplit:
    """Seeded, non-overlapping donor split (random; mirrors pSAGE-net's individual-level splits)."""
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
class GeneSplit:
    train: List[str]
    val: List[str]
    test: List[str]
    chrom_assignment: Dict[str, str]  # chrom -> "train" / "val" / "test"

    def summary(self) -> str:
        n_total = len(self.train) + len(self.val) + len(self.test)
        return (
            f"genes: train={len(self.train)} ({len(self.train) / n_total:.1%}) "
            f"val={len(self.val)} ({len(self.val) / n_total:.1%}) "
            f"test={len(self.test)} ({len(self.test) / n_total:.1%}) "
            f"across {len(self.chrom_assignment)} chromosomes"
        )


def split_genes_by_chromosome(
    gene_to_chrom: Dict[str, str],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> GeneSplit:
    """Splits genes into train/val/test by assigning *whole chromosomes* to each split.

    Mirrors the rationale behind pSAGE-net's (and Enformer's) gene splits --
    train/val/test genes come from disjoint chromosomes, so nearby loci (and
    any shared local LD structure) never straddle a split boundary. Unlike
    the paper's *fixed* chromosome-range assignment (tuned for a ~20k-gene
    universe, e.g. "chromosomes 1-16 = train"), chromosomes are assigned
    *greedily* here: repeatedly hand the next-largest chromosome to whichever
    bucket (train/val/test) is currently furthest below its target gene-count
    share. A fixed assignment would produce very uneven splits at the much
    smaller scale used here (~1000 genes across 22 autosomes, where e.g.
    chr19 alone is unusually gene-dense).

    Args:
        gene_to_chrom: maps gene_id -> chromosome string (autosomes only;
            filter out sex chromosomes/MT before calling this).
        val_frac, test_frac: target gene-count fractions for val/test (the
            remainder targets train). Exact fractions are not achievable in
            general since chromosomes are indivisible; the greedy assignment
            gets as close as possible.
        seed: only used to break ties between same-size chromosomes, so the
            result is deterministic given the same input.

    Returns:
        `GeneSplit` with disjoint, chromosome-grouped train/val/test gene
        lists (sorted) and the resulting chrom -> split assignment.
    """
    chrom_to_genes: Dict[str, List[str]] = {}
    for gene, chrom in gene_to_chrom.items():
        chrom_to_genes.setdefault(chrom, []).append(gene)

    rng = np.random.RandomState(seed)
    chroms_by_count: Dict[int, List[str]] = {}
    for chrom, genes in chrom_to_genes.items():
        chroms_by_count.setdefault(len(genes), []).append(chrom)
    ordered_chroms: List[str] = []
    for count in sorted(chroms_by_count.keys(), reverse=True):
        group = chroms_by_count[count]
        rng.shuffle(group)
        ordered_chroms.extend(group)

    n_total = sum(len(v) for v in chrom_to_genes.values())
    targets = {
        "train": (1.0 - val_frac - test_frac) * n_total,
        "val": val_frac * n_total,
        "test": test_frac * n_total,
    }
    counts = {"train": 0, "val": 0, "test": 0}
    chrom_assignment: Dict[str, str] = {}

    for chrom in ordered_chroms:
        n_genes = len(chrom_to_genes[chrom])
        deficits = {bucket: targets[bucket] - counts[bucket] for bucket in counts}
        bucket = max(deficits, key=deficits.get)
        chrom_assignment[chrom] = bucket
        counts[bucket] += n_genes

    train, val, test = [], [], []
    for chrom, bucket in chrom_assignment.items():
        target_list = {"train": train, "val": val, "test": test}[bucket]
        target_list.extend(chrom_to_genes[chrom])

    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))
    return GeneSplit(train=sorted(train), val=sorted(val), test=sorted(test), chrom_assignment=chrom_assignment)
