"""PrediXcan-style heritability ranking: fit an elastic-net model per candidate
gene on local SNP dosages vs. pseudobulk expression, rank by held-out
(validation-donor) Pearson r, and select the top-N genes.

This is the same procedure pSAGE-net's paper uses to build its "top 1000
genes by heritability" training set (Methods / PrediXcan:
https://www.nature.com/articles/s41592-026-03124-8): one ElasticNet model per
gene (`l1_ratio` fixed, `alpha` auto-selected via `ElasticNetCV`), trained on
train donors, ranked by Pearson r on validation donors.

The cached per-gene models this produces (`out_csv`, refit on train+val by
`src/evaluate.py`) are also directly reused for the final model-vs-PrediXcan
comparison, since PrediXcan *is* this same elastic-net baseline.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.genome import (
    GeneAnnotation,
    GeneRecord,
    VCFGenotypeReader,
    chrom_for_gene,
    get_tss_window,
    onek1k_sample_id_from_donor_id,
)
from src.regression_utils import ElasticNetBaseline, filter_by_maf, impute_and_filter_dosage, is_constant, safe_pearsonr
from src.splits import DonorSplit

AUTOSOMES = {str(i) for i in range(1, 23)}
DEFAULT_PROTEIN_CODING_BIOTYPES = {"protein_coding"}


def autosomal_protein_coding_genes(
    annotation: GeneAnnotation,
    var_gene_ids: Sequence[str],
    biotypes: Sequence[str] = tuple(DEFAULT_PROTEIN_CODING_BIOTYPES),
) -> List[GeneRecord]:
    """Candidate gene universe: protein-coding, chr1-22, present in the h5ad `var` index.

    Mirrors the paper's Gencode-based filter (`gene_type == 'protein_coding'`,
    autosomes + sex chromosomes excluded here since OneK1K/GTF chrom naming
    for X/Y varies and the paper's own filter keeps 1-22 plus X/Y -- we keep
    strictly autosomal for simplicity, consistent with avoiding
    sex-chromosome dosage-coding edge cases in a from-scratch POC).
    """
    var_set = set(var_gene_ids)
    genes = []
    for record in annotation.all_records():
        if record.chrom.lstrip("chr") not in AUTOSOMES:
            continue
        if record.biotype not in biotypes:
            continue
        if record.gene_id not in var_set:
            continue
        genes.append(record)
    return genes


@dataclass
class GeneHeritabilityResult:
    gene_id: str
    chrom: str
    n_variants: int
    val_pearson_r: float
    alpha: float
    l1_ratio: float
    skipped_reason: Optional[str] = None


def fit_gene_elasticnet(
    gene_id: str,
    chrom: str,
    start: int,
    end: int,
    vcf_dir: str,
    vcf_filename_template: str,
    sample_ids_train: List[str],
    sample_ids_val: List[str],
    y_train: np.ndarray,
    y_val: np.ndarray,
    maf_min: float,
    l1_ratio: float,
    cv: int,
    seed: int,
    max_missing_frac: float,
) -> GeneHeritabilityResult:
    """Fits one gene's PrediXcan-style ElasticNet model and evaluates held-out Pearson r.

    Public so it can be reused outside the ranking loop -- `src/evaluate.py`
    calls this again per gene, with `sample_ids_train`/`y_train` = train+val
    donors and `sample_ids_val`/`y_val` = test donors, to get a PrediXcan
    baseline that's directly comparable to the sequence model's held-out
    test-donor performance. Safe to run in a worker process (pysam file
    handles aren't fork/pickle-safe to share, so a fresh `VCFGenotypeReader`
    is opened here).
    """
    vcf_reader = VCFGenotypeReader(vcf_dir, filename_template=vcf_filename_template)
    try:
        all_sample_ids = sample_ids_train + sample_ids_val
        positions, ref_alt, raw_dosage = vcf_reader.get_dosage_matrix_in_region(chrom, start, end, all_sample_ids)
    finally:
        vcf_reader.close()

    dosage, positions, ref_alt, allele_freq = impute_and_filter_dosage(raw_dosage, positions, ref_alt, max_missing_frac)
    if dosage.shape[1] > 0 and maf_min > 0:
        dosage, positions, ref_alt, allele_freq = filter_by_maf(dosage, positions, ref_alt, allele_freq, maf_min)

    if dosage.shape[1] == 0:
        return GeneHeritabilityResult(gene_id, chrom, 0, float("nan"), float("nan"), l1_ratio, skipped_reason="no usable variants in window")

    n_train = len(sample_ids_train)
    dosage_train, dosage_val = dosage[:n_train], dosage[n_train:]

    if is_constant(y_train):
        return GeneHeritabilityResult(
            gene_id, chrom, dosage.shape[1], float("nan"), float("nan"), l1_ratio, skipped_reason="constant train-donor pseudobulk expression"
        )

    model = ElasticNetBaseline.fit_cv(dosage_train, y_train, l1_ratios=[l1_ratio], cv=cv, seed=seed)
    pred_val = model.predict(dosage_val)
    r, _ = safe_pearsonr(y_val, pred_val)
    return GeneHeritabilityResult(gene_id, chrom, dosage.shape[1], r, model.alpha, model.l1_ratio)


def rank_genes_by_heritability(
    h5ad_path: str,
    cell_type: str,
    gtf_path: str,
    vcf_dir: str,
    donor_split: DonorSplit,
    vcf_filename_template: str = "{chrom}.vcf.gz",
    vcf_chrom_style: Optional[str] = None,
    vcf_sample_id_scheme: str = "onek1k",
    vcf_sample_id_prefix: str = "OneK1K_",
    donor_id_map: Optional[Dict[str, str]] = None,
    window: int = 40_000,
    maf_min: float = 0.01,
    l1_ratio: float = 0.5,
    cv: int = 5,
    seed: int = 0,
    max_missing_frac: float = 0.2,
    min_cells_per_donor: int = 5,
    n_workers: int = 1,
    out_csv: Optional[str] = None,
    save_every: int = 25,
) -> pd.DataFrame:
    """Ranks every protein-coding autosomal gene (present in the h5ad) by held-out Pearson r.

    Resumable: if `out_csv` already exists, genes already present in it are
    skipped -- so a long-running rank-everything pass can be safely
    restarted. Progress is saved to `out_csv` every `save_every` completed
    genes (as well as at the end).
    """
    # Local import: avoids a heritability.py <-> data.preprocess import cycle
    # at module load time (data/preprocess.py doesn't import src.heritability,
    # but keeping this import local makes that independence explicit).
    from data.preprocess import load_celltype_pseudobulk_matrix

    annotation = GeneAnnotation(gtf_path)

    import anndata as ad

    var_gene_ids = list(ad.read_h5ad(h5ad_path, backed="r").var.index.astype(str))
    candidate_genes = autosomal_protein_coding_genes(annotation, var_gene_ids)
    print(f"[heritability] {len(candidate_genes)} protein-coding autosomal candidate genes found in h5ad var")
    if len(candidate_genes) == 0:
        raise ValueError("No candidate genes found -- check --gtf biotype field and h5ad var gene IDs match")

    gene_ids = [g.gene_id for g in candidate_genes]
    pb = load_celltype_pseudobulk_matrix(h5ad_path, cell_type, gene_ids, min_cells_per_donor=min_cells_per_donor)
    print(f"[heritability] pseudobulk matrix: {len(pb.donor_ids)} donors x {len(pb.gene_ids)} genes")

    donor_id_map = donor_id_map or {}
    if vcf_sample_id_scheme == "onek1k":
        sample_id_fn = lambda d: onek1k_sample_id_from_donor_id(d, prefix=vcf_sample_id_prefix)
    else:
        sample_id_fn = lambda d: d
    resolve = lambda d: donor_id_map.get(d, sample_id_fn(d))

    pb_donor_set = set(pb.donor_ids)
    train_donors = [d for d in donor_split.train if d in pb_donor_set]
    val_donors = [d for d in donor_split.val if d in pb_donor_set]
    if len(train_donors) == 0 or len(val_donors) == 0:
        raise ValueError(
            f"Not enough train/val donors with >= {min_cells_per_donor} cells of type '{cell_type}' "
            f"(train={len(train_donors)}, val={len(val_donors)}) -- cannot rank heritability"
        )
    print(f"[heritability] using {len(train_donors)} train donors, {len(val_donors)} val donors")

    sample_ids_train = [resolve(d) for d in train_donors]
    sample_ids_val = [resolve(d) for d in val_donors]

    done_genes: set = set()
    results: List[dict] = []
    if out_csv is not None and os.path.exists(out_csv):
        existing_df = pd.read_csv(out_csv)
        done_genes = set(existing_df["gene_id"].astype(str))
        results = existing_df.to_dict("records")
        print(f"[heritability] resuming: {len(done_genes)} genes already ranked in {out_csv}")

    todo_genes = [g for g in candidate_genes if g.gene_id not in done_genes]
    print(f"[heritability] {len(todo_genes)} genes remaining to rank")

    def _save() -> None:
        if out_csv is not None:
            os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
            pd.DataFrame(results).to_csv(out_csv, index=False)

    if len(todo_genes) == 0:
        return pd.DataFrame(results)

    try:
        from tqdm import tqdm

        progress = tqdm(total=len(todo_genes), desc=f"heritability ranking ({cell_type})")
    except ImportError:
        progress = None

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for gene_record in todo_genes:
            chrom = chrom_for_gene(gene_record, vcf_chrom_style)
            start, end = get_tss_window(gene_record, window)
            y_train = pb.mean.loc[train_donors, gene_record.gene_id].to_numpy(dtype=np.float64)
            y_val = pb.mean.loc[val_donors, gene_record.gene_id].to_numpy(dtype=np.float64)
            future = executor.submit(
                fit_gene_elasticnet,
                gene_record.gene_id,
                chrom,
                start,
                end,
                vcf_dir,
                vcf_filename_template,
                sample_ids_train,
                sample_ids_val,
                y_train,
                y_val,
                maf_min,
                l1_ratio,
                cv,
                seed,
                max_missing_frac,
            )
            futures[future] = gene_record

        n_done = 0
        for future in as_completed(futures):
            gene_record = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - keep ranking robust to a single gene's failure
                result = GeneHeritabilityResult(
                    gene_record.gene_id, gene_record.chrom, 0, float("nan"), float("nan"), l1_ratio, skipped_reason=f"error: {exc}"
                )
            results.append(
                {
                    "gene_id": result.gene_id,
                    "chrom": result.chrom,
                    "n_variants": result.n_variants,
                    "val_pearson_r": result.val_pearson_r,
                    "alpha": result.alpha,
                    "l1_ratio": result.l1_ratio,
                    "skipped_reason": result.skipped_reason,
                }
            )
            n_done += 1
            if progress is not None:
                progress.update(1)
            if n_done % save_every == 0:
                _save()

    if progress is not None:
        progress.close()
    _save()
    return pd.DataFrame(results)


def select_top_genes(ranking_df: pd.DataFrame, top_n: int = 1000) -> pd.DataFrame:
    """Selects the top-N genes by validation-donor Pearson r (skipped/nan genes excluded)."""
    ranked = ranking_df.dropna(subset=["val_pearson_r"]).sort_values("val_pearson_r", ascending=False)
    return ranked.head(top_n).reset_index(drop=True)
