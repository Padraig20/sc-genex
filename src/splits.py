"""Train/val/test splitting: donors (random) and genes (chromosome-bucket).

Two independent splits are combined to build the 4-way seen/unseen
evaluation matrix (see `src/train.py`):
- `split_donors`: random, seeded individual-level split. Default fractions
  (`val_frac=0.10`, `test_frac=0.10`) match the paper's actual ROSMAP split
  ("individuals are split randomly into train (n=689), validation (n=85),
  and test (n=85) sets to achieve an 80%/10%/10% split" -- Methods).
- `split_genes_by_paper_chromosomes` (**default**, used by
  `src/pretrain.py`/`src/train.py`): the paper's own *fixed* chromosome-range
  gene split ("train chromosomes = range(1, 17), validation chromosomes =
  [17, 18, 21, 22], test chromosomes = [19, 20]" -- Methods), which every
  gene set on a given chromosome always lands in the same split, letting
  results be compared directly against the paper's own per-chromosome
  breakdown (Supplementary Fig. 5).
- `split_genes_by_chromosome` (**fallback**, `--gene-split-scheme greedy`):
  a *dynamic*, greedy chromosome assignment that targets even split
  proportions regardless of gene-count-per-chromosome -- useful when the
  paper's fixed ranges would leave a split empty (e.g. a small top-N
  heritability-ranked gene subset that happens to have zero genes on
  chromosomes 19/20).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

DEFAULT_SEED = 0

# The paper's fixed, chromosome-range gene split (Methods, "Gene sets"):
# "We use the chromosome split: train chromosomes = range(1, 17), validation
# chromosomes = [17, 18, 21, 22], test chromosomes = [19, 20]" -- covers all
# 22 autosomes (16 + 4 + 2), allocating 14,786/2,126/2,008 genes (78%/11%/11%)
# of their ~19.8k-gene universe.
PAPER_TRAIN_CHROMS: tuple = tuple(str(c) for c in range(1, 17))
PAPER_VAL_CHROMS: tuple = ("17", "18", "21", "22")
PAPER_TEST_CHROMS: tuple = ("19", "20")


@dataclass
class DonorSplit:
    train: List[str]
    val: List[str]
    test: List[str]


def split_donors(
    donor_ids: Sequence[str],
    val_frac: float = 0.10,
    test_frac: float = 0.10,
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


def _normalize_chrom(chrom: str) -> str:
    """`"chr7"` / `"7"` -> `"7"`, for comparing against `PAPER_*_CHROMS`."""
    return chrom[3:] if chrom.lower().startswith("chr") else chrom


def split_genes_by_paper_chromosomes(
    gene_to_chrom: Dict[str, str],
    train_chroms: Sequence[str] = PAPER_TRAIN_CHROMS,
    val_chroms: Sequence[str] = PAPER_VAL_CHROMS,
    test_chroms: Sequence[str] = PAPER_TEST_CHROMS,
) -> GeneSplit:
    """The paper's own *fixed* chromosome-range gene split (Methods, "Gene sets"):
    `train_chroms = range(1, 17)`, `val_chroms = [17, 18, 21, 22]`,
    `test_chroms = [19, 20]` by default -- every autosome is assigned to
    exactly one split, so results are directly comparable to the paper's own
    per-chromosome breakdown (their Supplementary Fig. 5).

    Unlike `split_genes_by_chromosome`, this is *not* seeded/dynamic: a
    given chromosome always lands in the same split regardless of how many
    genes from it are present in `gene_to_chrom` (e.g. a top-N
    heritability-ranked gene subset), which is both the point (matching the
    paper's methodology exactly) and the risk -- a small/biased gene subset
    can easily end up with zero genes on the val/test chromosomes. Callers
    should check for empty splits and fall back to `split_genes_by_chromosome`
    (`--gene-split-scheme greedy`) if that happens.

    Genes on chromosomes not listed in any of the three sets (e.g. sex
    chromosomes, if not already filtered out upstream) are silently dropped,
    matching the paper's restriction of this specific split to the 22
    autosomes.
    """
    train_set, val_set, test_set = set(train_chroms), set(val_chroms), set(test_chroms)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("train_chroms/val_chroms/test_chroms must be disjoint")

    train, val, test = [], [], []
    for gene, chrom in gene_to_chrom.items():
        normalized = _normalize_chrom(str(chrom))
        if normalized in train_set:
            train.append(gene)
        elif normalized in val_set:
            val.append(gene)
        elif normalized in test_set:
            test.append(gene)
        # else: not on any of the 22 listed autosomes -- dropped, as in the paper.

    chrom_assignment: Dict[str, str] = {}
    for chrom in train_set:
        chrom_assignment[chrom] = "train"
    for chrom in val_set:
        chrom_assignment[chrom] = "val"
    for chrom in test_set:
        chrom_assignment[chrom] = "test"

    return GeneSplit(train=sorted(train), val=sorted(val), test=sorted(test), chrom_assignment=chrom_assignment)


GENE_SPLIT_SCHEMES = ("paper", "greedy")


def select_gene_split(
    gene_to_chrom: Dict[str, str],
    scheme: str = "paper",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> GeneSplit:
    """Dispatches to `split_genes_by_paper_chromosomes` (`scheme="paper"`, default) or
    `split_genes_by_chromosome` (`scheme="greedy"`), and raises a clear, actionable
    error if the chosen scheme leaves any split empty -- which `"paper"`'s *fixed*
    chromosome ranges can do for a small/biased gene subset (e.g. a top-N
    heritability-ranked gene list with no genes on chromosomes 19/20).
    """
    if scheme == "paper":
        gene_split = split_genes_by_paper_chromosomes(gene_to_chrom)
    elif scheme == "greedy":
        gene_split = split_genes_by_chromosome(gene_to_chrom, val_frac=val_frac, test_frac=test_frac, seed=seed)
    else:
        raise ValueError(f"Unknown gene_split_scheme={scheme!r}, expected one of {GENE_SPLIT_SCHEMES}")

    empty = [name for name, genes in (("train", gene_split.train), ("val", gene_split.val), ("test", gene_split.test)) if len(genes) == 0]
    if empty:
        raise ValueError(
            f"--gene-split-scheme={scheme!r} produced an empty {'/'.join(empty)} split (this gene set likely has no "
            f"genes on the relevant chromosomes){' -- try --gene-split-scheme greedy instead' if scheme == 'paper' else ''}"
        )
    return gene_split
