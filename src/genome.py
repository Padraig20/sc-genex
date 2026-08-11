"""Reference genome, gene-TSS annotation, and personalized-sequence construction.

This module reproduces the personalized-input-sequence step of Variformer
(https://github.com/shirondru/enformer_fine_tuning) without needing to
materialize per-haplotype consensus FASTA files via bcftools/samtools.

Variformer one-hot encodes each of a donor's two haplotypes and averages them,
so a heterozygous SNP becomes 0.5/0.5 in the input. For unphased, biallelic
SNPs, that average only depends on the alt-allele dosage (0, 1, or 2 copies),
not on which haplotype the alt allele sits on:

    onehot = (1 - dosage / 2) * onehot(ref) + (dosage / 2) * onehot(alt)

So we can build the same input directly from VCF genotypes with `pysam`,
without ever generating consensus FASTAs.
"""
from __future__ import annotations

import re
from bisect import bisect_left
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

try:
    import pysam
except ImportError:  # pragma: no cover - optional at import time for tooling
    pysam = None

BASE_TO_ONE_HOT: Dict[str, np.ndarray] = {
    "A": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "C": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "G": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    "T": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    "N": np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
}

ENFORMER_REFERENCE_SEQ_LEN = 196_608


def one_hot_encode_sequence(seq: str) -> np.ndarray:
    """One-hot encodes a DNA string into an [L, 4] float32 array (A, C, G, T)."""
    seq = seq.upper()
    out = np.zeros((len(seq), 4), dtype=np.float32)
    for base, vec in BASE_TO_ONE_HOT.items():
        if base == "N":
            continue
        out[np.frombuffer(seq.encode("ascii"), dtype=np.uint8) == ord(base)] = vec
    # anything not ACGT (N, ambiguity codes, etc.) -> uniform 0.25 to avoid
    # injecting spurious signal for ambiguous reference bases
    known_mask = np.isin(np.frombuffer(seq.encode("ascii"), dtype=np.uint8), [ord(b) for b in "ACGT"])
    out[~known_mask] = BASE_TO_ONE_HOT["N"]
    return out


@dataclass
class GeneRecord:
    gene_id: str
    gene_name: Optional[str]
    chrom: str
    start: int  # 0-based, GTF-style inclusive start converted to 0-based
    end: int  # exclusive
    strand: str

    @property
    def tss(self) -> int:
        """0-based transcription start site position."""
        return self.start if self.strand == "+" else self.end - 1


class GeneAnnotation:
    """Minimal GTF parser that extracts gene-level records (chrom, TSS, strand).

    Deliberately avoids a heavyweight GTF-parsing dependency: a POC only needs
    the `gene` feature rows, which is a handful of columns per gene.
    """

    _ATTR_RE = re.compile(r'(\w+)\s+"([^"]*)"')

    def __init__(self, gtf_path: str):
        self.gtf_path = gtf_path
        self._by_gene_id: Dict[str, GeneRecord] = {}
        self._by_gene_name: Dict[str, GeneRecord] = {}
        self._parse()

    def _parse(self) -> None:
        opener = _smart_open(self.gtf_path)
        with opener as fh:
            for line in fh:
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if not line or line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 9 or fields[2] != "gene":
                    continue
                chrom, _, _, start, end, _, strand = fields[0], fields[1], fields[2], fields[3], fields[4], fields[5], fields[6]
                attrs = dict(self._ATTR_RE.findall(fields[8]))
                gene_id = attrs.get("gene_id")
                if gene_id is None:
                    continue
                gene_id = gene_id.split(".")[0]  # strip Ensembl version suffix
                gene_name = attrs.get("gene_name")
                record = GeneRecord(
                    gene_id=gene_id,
                    gene_name=gene_name,
                    chrom=chrom,
                    start=int(start) - 1,  # GTF is 1-based inclusive -> 0-based
                    end=int(end),
                    strand=strand,
                )
                self._by_gene_id[gene_id] = record
                if gene_name:
                    self._by_gene_name[gene_name] = record

    def get(self, gene: str) -> GeneRecord:
        """Looks up a gene by Ensembl ID (preferred) or gene symbol."""
        gene_id = gene.split(".")[0]
        if gene_id in self._by_gene_id:
            return self._by_gene_id[gene_id]
        if gene in self._by_gene_name:
            return self._by_gene_name[gene]
        raise KeyError(f"Gene '{gene}' not found in GTF annotation at {self.gtf_path}")


def _smart_open(path: str):
    if path.endswith(".gz"):
        import gzip

        return gzip.open(path, "rt")
    return open(path, "r")


def get_tss_window(gene: GeneRecord, seq_len: int) -> tuple[int, int]:
    """Returns a TSS-centered [start, end) window of length `seq_len`, 0-based.

    Mirrors Variformer's `generate_train_batch_one_gene`: it always centers
    within the canonical Enformer receptive field before trimming to the
    (usually shorter) requested window, so the TSS bin lands in the same
    relative position regardless of `seq_len`.
    """
    region_center = gene.tss
    start = region_center - (seq_len // 2)
    end = region_center + (seq_len // 2)
    return start, end


class ReferenceGenome:
    """Thin wrapper around an indexed reference FASTA (e.g. hg38)."""

    def __init__(self, fasta_path: str):
        if pysam is None:
            raise ImportError("pysam is required to read the reference genome FASTA")
        self.fasta_path = fasta_path
        self._fasta = pysam.FastaFile(fasta_path)

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """Fetches [start, end) (0-based, half-open), upper-cased."""
        return self._fasta.fetch(chrom, start, end).upper()

    def close(self) -> None:
        self._fasta.close()


class VCFGenotypeReader:
    """Reads per-sample alt-allele dosage for biallelic SNPs in a region.

    Expects one bgzip+tabix-indexed VCF per chromosome, e.g. `chr1.vcf.gz`
    or `1.vcf.gz` under `vcf_dir` (configurable via `filename_template`).
    """

    def __init__(self, vcf_dir: str, filename_template: str = "{chrom}.vcf.gz"):
        if pysam is None:
            raise ImportError("pysam is required to read genotype VCFs")
        self.vcf_dir = vcf_dir
        self.filename_template = filename_template
        self._open_files: Dict[str, "pysam.VariantFile"] = {}

    def _get_vcf(self, chrom: str) -> "pysam.VariantFile":
        import os

        if chrom not in self._open_files:
            candidates = [chrom, chrom.lstrip("chr"), f"chr{chrom.lstrip('chr')}"]
            path = None
            for candidate in candidates:
                candidate_path = os.path.join(self.vcf_dir, self.filename_template.format(chrom=candidate))
                if os.path.exists(candidate_path):
                    path = candidate_path
                    break
            if path is None:
                raise FileNotFoundError(
                    f"No VCF found for chromosome '{chrom}' under {self.vcf_dir} "
                    f"(tried template '{self.filename_template}' with {candidates})"
                )
            self._open_files[chrom] = pysam.VariantFile(path)
        return self._open_files[chrom]

    def get_dosages_in_region(self, chrom: str, start: int, end: int, sample_id: str) -> Dict[int, tuple[str, str, int]]:
        """Returns {0-based position: (ref, alt, dosage)} for biallelic SNPs.

        `dosage` is the number of alt-allele copies (0, 1, 2) that `sample_id`
        carries. Multi-allelic sites, indels, and missing genotypes are skipped.
        """
        vcf = self._get_vcf(chrom)
        if sample_id not in vcf.header.samples:
            raise KeyError(f"Sample '{sample_id}' not found in VCF for chromosome '{chrom}'")

        out: Dict[int, tuple[str, str, int]] = {}
        # pysam region fetch uses 0-based start, exclusive end (matches our convention)
        for record in vcf.fetch(chrom, max(start, 0), end):
            ref = record.ref
            alts = record.alts
            if ref is None or alts is None or len(alts) != 1:
                continue  # skip multi-allelic sites for this POC
            alt = alts[0]
            if len(ref) != 1 or len(alt) != 1:
                continue  # SNPs only, no indels

            genotype = record.samples[sample_id].get("GT")
            if genotype is None or any(allele is None for allele in genotype):
                continue  # missing genotype

            dosage = sum(1 for allele in genotype if allele == 1)
            pos_0based = record.pos - 1  # VCF POS is 1-based
            out[pos_0based] = (ref.upper(), alt.upper(), dosage)
        return out

    def close(self) -> None:
        for vcf in self._open_files.values():
            vcf.close()
        self._open_files.clear()


def build_personalized_onehot(
    reference: ReferenceGenome,
    vcf_reader: VCFGenotypeReader,
    sample_id: str,
    chrom: str,
    start: int,
    end: int,
) -> np.ndarray:
    """Builds the dosage-averaged one-hot sequence for `sample_id` over [start, end).

    Equivalent to averaging the one-hot encodings of both unphased haplotypes,
    as in Variformer's `_one_hot_encode_diploid`, but computed directly from
    genotype dosage (see module docstring) instead of building consensus FASTAs.
    """
    ref_seq = reference.fetch(chrom, start, end)
    assert len(ref_seq) == end - start, (
        f"Reference fetch returned {len(ref_seq)} bases, expected {end - start} "
        f"for region {chrom}:{start}-{end}"
    )
    onehot = one_hot_encode_sequence(ref_seq)

    dosages = vcf_reader.get_dosages_in_region(chrom, start, end, sample_id)
    for pos, (ref_allele, alt_allele, dosage) in dosages.items():
        rel_pos = pos - start
        if rel_pos < 0 or rel_pos >= onehot.shape[0]:
            continue
        observed_ref = ref_seq[rel_pos]
        if observed_ref != ref_allele:
            # Reference genome disagrees with the VCF's REF allele at this site
            # (e.g. wrong genome build) -- skip rather than silently corrupt the input
            continue
        if dosage == 0:
            continue  # already reference at this position
        ref_vec = BASE_TO_ONE_HOT.get(ref_allele, BASE_TO_ONE_HOT["N"])
        alt_vec = BASE_TO_ONE_HOT.get(alt_allele, BASE_TO_ONE_HOT["N"])
        onehot[rel_pos] = (1.0 - dosage / 2.0) * ref_vec + (dosage / 2.0) * alt_vec

    return onehot


def onek1k_sample_id_from_donor_id(donor_id: str, prefix: str = "OneK1K_") -> str:
    """Maps an h5ad `donor_id` to the genotype-file sample ID for OneK1K.

    The h5ad's `obs['donor_id']` is formatted `"{X}_{Y}"` (e.g. `"10_10"`), but
    the genotype files (VCF/PLINK) use individual IDs formatted `"{prefix}{Y}"`
    (e.g. `"OneK1K_10"`) -- i.e. only the second, `Y`, component is used, not
    the first. This was confirmed against a real `.fam` file:

        0   OneK1K_1     0   0   2   -9
        0   OneK1K_10    0   0   2   -9
        0   OneK1K_100   0   0   0   -9

    Do not assume `donor_id` and the genotype sample ID are identical.
    """
    suffix = donor_id.rsplit("_", 1)[-1]
    return f"{prefix}{suffix}"


def chrom_for_gene(gene: GeneRecord, vcf_chrom_style: Optional[str] = None) -> str:
    """Normalizes chromosome naming between GTF ('chr1' or '1') and VCF conventions."""
    chrom = gene.chrom
    if vcf_chrom_style == "chr" and not chrom.startswith("chr"):
        return f"chr{chrom}"
    if vcf_chrom_style == "no_chr" and chrom.startswith("chr"):
        return chrom[3:]
    return chrom
