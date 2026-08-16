"""Reference genome, gene-TSS annotation, and personalized-sequence construction.

Two personalized-sequence representations live here:

1. `build_personalized_onehot` -- a **dosage-averaged**, phase-agnostic single
   sequence per donor (kept for the PrediXcan/heritability baseline in
   `src/heritability.py`, which only needs dosage, not phase).
2. `build_haplotype_onehots` -- **phased** maternal/paternal haplotype
   sequences, read directly from the VCF's phased `GT` field (e.g. `0|1`),
   matching pSAGE-net's `PersonalGenomeDataset` (see
   https://github.com/mostafavilabuw/SAGEnet). This is what the model's
   dataset (`src/dataset.py`) actually trains on. It assumes input VCFs are
   already phased (e.g. from imputation/statistical phasing upstream) -- no
   phasing is performed by this pipeline.

Both avoid materializing per-haplotype consensus FASTA files via
bcftools/samtools; genotypes are read directly with `pysam` and applied to an
already-fetched reference one-hot array.

Both are also always fetched in forward-strand (increasing-coordinate)
orientation; callers (`src/dataset.py`, `src/pretrain.py`) reverse-complement
`"-"` strand genes via `orient_onehot_by_strand` below before feeding them to
a CNN, matching the paper's strand-aware convention.
"""
from __future__ import annotations

import re
from bisect import bisect_left
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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


# BASE_TO_ONE_HOT's channel order is (A, C, G, T); reverse-complementing an
# already-one-hot-encoded array is therefore "reverse the length axis, then
# swap A<->T (channels 0, 3) and C<->G (channels 1, 2)" -- equivalent to, but
# cheaper than, reverse-complementing the raw string and re-encoding it.
_REVERSE_COMPLEMENT_CHANNEL_ORDER = [3, 2, 1, 0]


def reverse_complement_onehot(onehot: np.ndarray) -> np.ndarray:
    """Reverse-complements a one-hot encoded `[L, 4]` (A, C, G, T) sequence array."""
    return onehot[::-1, _REVERSE_COMPLEMENT_CHANNEL_ORDER]


def orient_onehot_by_strand(onehot: np.ndarray, strand: str) -> np.ndarray:
    """Reverse-complements negative-strand genes' one-hot sequences (no-op on `"+"`).

    Fetched windows are always read off the reference in forward (`+`
    strand/increasing-coordinate) orientation regardless of the gene's own
    strand, so a `"-"` strand gene's TSS otherwise lands at the *end* of the
    array pointing "backwards" relative to transcription. The paper's
    ablations found reverse-complementing `"-"` strand genes so every input
    is TSS-relative and 5'->3' in the gene's own transcriptional direction
    measurably improves mean-expression prediction; apply this consistently
    to every reference/haplotype array before it reaches the CNN trunk.
    """
    if strand == "-":
        return reverse_complement_onehot(onehot)
    return onehot


@dataclass
class GeneRecord:
    gene_id: str
    gene_name: Optional[str]
    chrom: str
    start: int  # 0-based, GTF-style inclusive start converted to 0-based
    end: int  # exclusive
    strand: str
    biotype: Optional[str] = None  # e.g. "protein_coding" (GTF `gene_type`/`gene_biotype`)

    @property
    def tss(self) -> int:
        """0-based transcription start site position."""
        return self.start if self.strand == "+" else self.end - 1


class GeneAnnotation:
    """Minimal GTF parser that extracts gene-level records (chrom, TSS, strand, biotype).

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
                # GENCODE uses "gene_type", Ensembl uses "gene_biotype" -- accept either.
                biotype = attrs.get("gene_type") or attrs.get("gene_biotype")
                record = GeneRecord(
                    gene_id=gene_id,
                    gene_name=gene_name,
                    chrom=chrom,
                    start=int(start) - 1,  # GTF is 1-based inclusive -> 0-based
                    end=int(end),
                    strand=strand,
                    biotype=biotype,
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

    def all_records(self) -> List[GeneRecord]:
        """Returns every parsed gene record (one per unique `gene_id`)."""
        return list(self._by_gene_id.values())


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

    def get_dosage_matrix_in_region(
        self, chrom: str, start: int, end: int, sample_ids: Sequence[str]
    ) -> Tuple[List[int], List[Tuple[str, str]], "np.ndarray"]:
        """Fetches all biallelic SNPs in [start, end) once and reads dosage for many samples.

        Unlike `get_dosages_in_region` (single sample, used by
        `build_personalized_onehot`), this reads each VCF record only once and
        extracts every requested sample's genotype from it -- meant for
        diagnostics/baselines that need a full donor x variant dosage matrix
        (see `src/diagnose.py`), where calling `get_dosages_in_region` once per
        sample would re-scan the region redundantly.

        Returns:
            positions: 0-based variant positions, in VCF order.
            ref_alt: (ref, alt) per variant, same order as `positions`.
            dosage: [len(sample_ids), n_variants] float32 array; NaN for
                missing/multi-allelic/indel-skipped genotypes.
        """
        vcf = self._get_vcf(chrom)
        missing_samples = [s for s in sample_ids if s not in vcf.header.samples]
        if missing_samples:
            preview = missing_samples[:5]
            suffix = "..." if len(missing_samples) > 5 else ""
            raise KeyError(
                f"{len(missing_samples)} sample(s) not found in VCF for chromosome '{chrom}': {preview}{suffix}"
            )

        positions: List[int] = []
        ref_alt: List[Tuple[str, str]] = []
        dosage_columns: List[np.ndarray] = []
        for record in vcf.fetch(chrom, max(start, 0), end):
            ref = record.ref
            alts = record.alts
            if ref is None or alts is None or len(alts) != 1:
                continue  # skip multi-allelic sites for this POC
            alt = alts[0]
            if len(ref) != 1 or len(alt) != 1:
                continue  # SNPs only, no indels

            column = np.full(len(sample_ids), np.nan, dtype=np.float32)
            for i, sample_id in enumerate(sample_ids):
                genotype = record.samples[sample_id].get("GT")
                if genotype is None or any(allele is None for allele in genotype):
                    continue
                column[i] = sum(1 for allele in genotype if allele == 1)

            positions.append(record.pos - 1)
            ref_alt.append((ref.upper(), alt.upper()))
            dosage_columns.append(column)

        if not dosage_columns:
            return [], [], np.zeros((len(sample_ids), 0), dtype=np.float32)

        dosage = np.stack(dosage_columns, axis=1)  # [n_samples, n_variants]
        return positions, ref_alt, dosage

    def get_phased_genotypes_in_region(
        self, chrom: str, start: int, end: int, sample_id: str
    ) -> Dict[int, Tuple[str, str, int, int, bool]]:
        """Returns {0-based position: (ref, alt, allele0, allele1, phased)} for biallelic SNPs.

        Unlike `get_dosages_in_region` (collapses a genotype to a single
        dosage count, 0/1/2), this keeps the two alleles -- and whether the
        call is marked `phased` -- separate, which is what's needed to build
        distinct maternal (`allele0`) and paternal (`allele1`) haplotype
        sequences (see `build_haplotype_onehots`). Multi-allelic sites,
        indels, and missing genotypes are skipped, same as
        `get_dosages_in_region`.
        """
        vcf = self._get_vcf(chrom)
        if sample_id not in vcf.header.samples:
            raise KeyError(f"Sample '{sample_id}' not found in VCF for chromosome '{chrom}'")

        out: Dict[int, tuple[str, str, int, int, bool]] = {}
        for record in vcf.fetch(chrom, max(start, 0), end):
            ref = record.ref
            alts = record.alts
            if ref is None or alts is None or len(alts) != 1:
                continue  # skip multi-allelic sites for this POC
            alt = alts[0]
            if len(ref) != 1 or len(alt) != 1:
                continue  # SNPs only, no indels

            sample = record.samples[sample_id]
            genotype = sample.get("GT")
            if genotype is None or len(genotype) != 2 or any(allele is None for allele in genotype):
                continue  # missing or non-diploid genotype
            allele0, allele1 = genotype
            if allele0 not in (0, 1) or allele1 not in (0, 1):
                continue  # defensive: shouldn't happen for a biallelic record

            pos_0based = record.pos - 1  # VCF POS is 1-based
            out[pos_0based] = (ref.upper(), alt.upper(), int(allele0), int(allele1), bool(sample.phased))
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


class UnphasedHeterozygousSiteError(ValueError):
    """Raised when a heterozygous site lacks phase information and `strict_phasing=True`."""


def build_haplotype_onehots(
    reference: ReferenceGenome,
    vcf_reader: VCFGenotypeReader,
    sample_id: str,
    chrom: str,
    start: int,
    end: int,
    strict_phasing: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Builds separate maternal/paternal one-hot sequences for `sample_id` over [start, end).

    This is pSAGE-net's personal-sequence construction (see
    `PersonalGenomeDataset.get_personal_tensor` at
    https://github.com/mostafavilabuw/SAGEnet/blob/main/SAGEnet/data.py):
    each haplotype is built independently by applying one allele per site
    (`allele0` -> maternal, `allele1` -> paternal, per the VCF's `GT` field
    order) to the reference sequence, *without* averaging them together --
    unlike `build_personalized_onehot`, which discards phase entirely.
    Restricted to biallelic SNPs (no indels), like the rest of this codebase.

    Homozygous sites (`allele0 == allele1`) don't need phase information --
    both haplotypes get the same allele regardless of which is "first" in the
    VCF record. Heterozygous sites do: if such a site isn't marked `phased`
    (pysam's `sample.phased`), assigning its alleles to a specific haplotype
    would fabricate information the data doesn't actually contain. By
    default (`strict_phasing=False`) such sites are skipped for both
    haplotypes (left at the reference base) with the assumption that input
    VCFs are already phased and this should be rare; pass
    `strict_phasing=True` to raise `UnphasedHeterozygousSiteError` instead,
    e.g. to fail loudly if you suspect your VCFs are not actually phased.

    Returns:
        (maternal_onehot, paternal_onehot), each `[end - start, 4]` float32.
    """
    ref_seq = reference.fetch(chrom, start, end)
    assert len(ref_seq) == end - start, (
        f"Reference fetch returned {len(ref_seq)} bases, expected {end - start} "
        f"for region {chrom}:{start}-{end}"
    )
    maternal = one_hot_encode_sequence(ref_seq)
    paternal = one_hot_encode_sequence(ref_seq)

    genotypes = vcf_reader.get_phased_genotypes_in_region(chrom, start, end, sample_id)
    for pos, (ref_allele, alt_allele, allele0, allele1, phased) in genotypes.items():
        rel_pos = pos - start
        if rel_pos < 0 or rel_pos >= maternal.shape[0]:
            continue
        observed_ref = ref_seq[rel_pos]
        if observed_ref != ref_allele:
            # Reference genome disagrees with the VCF's REF allele at this site
            # (e.g. wrong genome build) -- skip rather than silently corrupt the input
            continue

        is_heterozygous = allele0 != allele1
        if is_heterozygous and not phased:
            if strict_phasing:
                raise UnphasedHeterozygousSiteError(
                    f"Heterozygous, unphased genotype for sample '{sample_id}' at "
                    f"{chrom}:{pos + 1} -- input VCFs are expected to be phased "
                    "(pass strict_phasing=False to skip such sites instead)"
                )
            continue  # cannot safely assign this site to a haplotype -- skip both

        alt_vec = BASE_TO_ONE_HOT.get(alt_allele, BASE_TO_ONE_HOT["N"])
        if allele0 == 1:
            maternal[rel_pos] = alt_vec
        if allele1 == 1:
            paternal[rel_pos] = alt_vec

    return maternal, paternal


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
