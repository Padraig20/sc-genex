# sc-genex: Variformer + Single-Cell Gaussian-NLL POC

A low-level proof-of-concept that fine-tunes [Enformer](https://www.nature.com/articles/s41592-021-01252-x)
per (gene, cell-type) on [OneK1K](https://onek1k.org/) single-cell data, following the
personalized-sequence approach of [Variformer](https://github.com/shirondru/enformer_fine_tuning),
with two changes:

1. **No pseudobulking at training time.** Every single cell of a donor contributes its own
   training example (same personalized DNA input, that cell's own expression value as target).
   Evaluation is still pseudobulk (per-donor mean), matching Variformer.
2. **Gaussian NLL loss with a learned `(mu, sigma)` head**, instead of MSE against a single value.
   `sigma` is meant to capture the aleatoric, cell-to-cell variability of a gene's expression for a
   given individual -- i.e. whether that gene is specifically/consistently expressed, or highly
   variable, across that person's cells of that type.

See `plan.md`-style discussion in the chat history for the full design rationale; the summary below
covers what you need to actually run this.

## What's implemented

| File | Responsibility |
|---|---|
| `data/preprocess.py` | Loads the OneK1K h5ad (backed mode), filters to one gene x cell-type, library-size-normalizes (`log1p(CP-median)`), groups per-cell values by donor, computes pseudobulk mean/std, splits donors into train/val/test. |
| `src/genome.py` | Minimal GTF parser (gene TSS lookup), reference-genome FASTA reader, per-chromosome VCF genotype reader, and the personalized one-hot sequence builder. |
| `src/dataset.py` | `SingleCellDonorDataset` (train: one item per donor with *all* of that donor's per-cell targets) and `PseudobulkEvalDataset` (val/test: one item per donor with the pseudobulk mean/std). |
| `src/model.py` | `VariformerGNLL`: pretrained Enformer + a 2-output head (`mu`, `sigma`) via `enformer-pytorch`'s `HeadAdapterWrapper`. |
| `src/loss.py` | Per-cell-weighted Gaussian NLL (see "How training works" below). |
| `src/metrics.py` | Pseudobulk Pearson r / R2, and the `sigma`-vs-empirical-std calibration correlation (the direct test of this POC's hypothesis). |
| `src/wandb_logger.py` | Optional W&B logging; no-op by default so nothing requires a W&B login. |
| `src/train.py` | CLI entrypoint tying everything together. |

## How training works

A donor's DNA input is identical across all of their cells (same genotype), so for fixed model
weights the network's prediction `(mu, sigma)` for that donor is deterministic. Instead of literally
repeating the forward pass once per cell, `src/train.py` forwards each donor's sequence **once** per
step, then `src/loss.py::per_cell_gaussian_nll` scores that single `(mu, sigma)` against *every one*
of the donor's actual per-cell target values, sums the per-cell NLLs across the whole batch, and
divides by the batch's total cell count (not by the number of donors). This is mathematically
identical to training on every individual cell independently with a shared input, but requires only
one forward pass per donor per step. It also means (by construction of the Gaussian NLL) that the
MLE recovered by training is each donor's own per-cell sample mean and sample standard deviation --
exactly the "no pseudobulking, but it still learns the mean like pseudobulking would" property this
POC is meant to test, now extended to also learn the spread.

This has been unit-tested (`per_cell_gaussian_nll` recovers each donor's true per-cell mean/std when
optimized directly) and integration-tested (a full `train.py` run against real OneK1K expression data
combined with a synthetic genome/VCF, see the "Testing performed" section).

## Data requirements

Only `data/onek1k_cellxgene_standardized.h5ad` exists in this repo. To actually run training you
must supply, as CLI arguments:

- `--vcf-dir`: a directory with one **bgzip + tabix-indexed** VCF per chromosome (biallelic SNPs,
  standard `GT` field), e.g. `chr1.vcf.gz` .. `chr22.vcf.gz` or `1.vcf.gz` .. `22.vcf.gz`
  (configurable via `--vcf-filename-template`).

  **Sample naming is *not* the identity mapping.** The h5ad's `obs['donor_id']` is formatted
  `"{X}_{Y}"` (e.g. `"10_10"`), but the real OneK1K genotype files use individual IDs formatted
  `"OneK1K_{Y}"` -- confirmed against an actual `.fam` file:

  ```
  0   OneK1K_1     0   0   2   -9
  0   OneK1K_10    0   0   2   -9
  0   OneK1K_100   0   0   0   -9
  ```

  i.e. only the second (`Y`) component of `donor_id` is used, prefixed with `OneK1K_`. This is the
  **default** (`--vcf-sample-id-scheme onek1k`, `--vcf-sample-id-prefix "OneK1K_"`). Pass
  `--vcf-sample-id-scheme identity` if your genotype files instead use the donor_id string directly,
  or `--donor-id-map path/to/mapping.csv` (2-column CSV: `h5ad_donor_id,vcf_sample_id`) for any
  donors that don't follow the default convention -- explicit mapping entries always take
  precedence. If a sample ID can't be found in the VCF, `VCFGenotypeReader` raises a `KeyError`
  naming the missing sample rather than silently falling back, so naming mismatches surface
  immediately instead of corrupting training.
- `--genome-fasta`: an indexed reference genome FASTA matching the VCFs' build (e.g. hg38,
  `samtools faidx` run once beforehand).
- `--gtf`: a GTF/GFF gene annotation (Ensembl/GENCODE-style) whose `gene_id` matches the h5ad's
  `var` index (Ensembl IDs); used only to look up each gene's TSS.

None of these three are auto-downloaded -- this was a deliberate choice to keep the POC free of
network calls at run time.

### Personalized-sequence construction, without consensus FASTAs

Variformer one-hot-encodes each of a donor's two haplotypes and averages them (heterozygous SNPs
become 0.5/0.5), which normally requires building per-haplotype consensus FASTA files with
bcftools/samtools. For unphased, biallelic SNPs that average only depends on the alt-allele
**dosage** (0, 1, or 2 copies), not on phase:

```
onehot = (1 - dosage / 2) * onehot(ref) + (dosage / 2) * onehot(alt)
```

So `src/genome.py` reads dosage straight from each VCF's `GT` field and applies it directly to the
reference sequence, with no consensus-FASTA generation step.

## Usage

```bash
uv sync  # installs torch, enformer-pytorch, anndata, pysam, etc. -- see pyproject.toml

uv run python src/train.py \
  --gene ENSG00000075624 --cell-type "naive B cell" \
  --h5ad-path data/onek1k_cellxgene_standardized.h5ad \
  --vcf-dir /path/to/genotypes \
  --genome-fasta /path/to/hg38.fa \
  --gtf /path/to/gencode.gtf.gz \
  --seq-len 49152 --batch-size 8 --epochs 20 --lr 5e-6 \
  --out-dir results/ACTB_naiveB
```

`--gene` accepts either an Ensembl ID (preferred, unambiguous) or a gene symbol.
`--cell-type` must exactly match a value in `obs['cell_type']` (see the 29 OneK1K categories, e.g.
`naive B cell`, `CD4-positive, alpha-beta T cell`, `CD14-positive monocyte`, ...).

Useful flags for constrained compute (no full fine-tune needed to sanity-check wiring):
- `--random-weights`: skips the ~350M-parameter pretrained Enformer download and uses a randomly
  initialized model instead. Not scientifically meaningful, but useful to verify the pipeline runs.
- `--freeze-enformer`: only trains the new `(mu, sigma)` head.
- `--finetune-last-n-layers N`: only trains the last `N` transformer blocks + head.

Outputs land in `--out-dir`: `predictions_val_epoch{N}.csv` each eval epoch, `predictions_test_final.csv`
at the end, and `best_model.pt` (the checkpoint with the best validation pseudobulk Pearson r). Each
predictions CSV has one row per donor: `donor_id, y_true_pseudobulk_mean, y_true_empirical_std,
y_pred_mu, y_pred_sigma, n_cells`.

## Testing performed

Real VCFs/reference genome/GPU were not available in the development environment, so full
biologically-meaningful training was not run. Instead, correctness was verified with:

- Unit tests of `src/genome.py`'s dosage-averaging logic against a synthetic FASTA + VCF (homozygous
  alt, heterozygous, and missing-genotype cases).
- Unit tests of `src/genome.py`'s GTF parsing and TSS-window centering.
- A unit test confirming `per_cell_gaussian_nll` recovers each donor's true per-cell sample mean and
  sample standard deviation when optimized directly (the core MLE property this POC relies on), and
  that its per-cell weighting matches a manual sum-of-NLLs-over-total-cells computation.
- Unit tests of the pseudobulk/sigma-calibration correlation metrics against known correlations.
- A forward-pass smoke test of `VariformerGNLL` (random weights) confirming output shapes and that
  `sigma > 0`.
- A full end-to-end integration run of `src/train.py` against the **real** OneK1K h5ad (a small cell
  type) combined with a **synthetic** genome/VCF (random genotypes, since real ones aren't available)
  and `--random-weights`: confirmed the CLI runs to completion, training loss decreases, and
  prediction CSVs / checkpoints are written correctly. This included a synthetic VCF using
  `OneK1K_{Y}`-style sample IDs (not identical to the h5ad `donor_id`), confirming the default
  `--vcf-sample-id-scheme onek1k` mapping resolves donors correctly with no explicit
  `--donor-id-map` needed, and that mismatched IDs fail loudly (`KeyError`) rather than silently.

## Out of scope for this POC

- Variformer's original contrastive pairwise loss term and multi-gene/multi-tissue batching.
- ROSMAP support, distributed training samplers.
- Auto-downloading the reference genome, GTF, or genotype data.
