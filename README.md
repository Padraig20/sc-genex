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
| `src/diagnose.py` | Standalone diagnostics for a (gene, cell-type) pair -- see "Diagnosing whether a result is 'too easy'" below. |

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
y_pred_mu, y_pred_sigma, n_cells`. Pass `--no-save-checkpoint` to skip writing `best_model.pt`
(prediction CSVs are still saved) -- useful for quick/exploratory runs where you don't need the
weights, e.g. when just checking whether a gene x cell-type pair is worth training for real.

## Diagnosing whether a result is "too easy"

A high pseudobulk Pearson r / R2 on the test set doesn't necessarily mean the model learned
subtle regulatory grammar from sequence -- it can also mean the gene has one strong, well-tagged
local eQTL (or even a structural presence/absence effect), or that the test set is small enough
for a decent-looking correlation to arise partly by chance. `src/diagnose.py` runs a battery of
checks for a given (gene, cell-type) pair to help distinguish these cases. It needs **no GPU and
no Enformer** -- only the h5ad, VCFs, and GTF -- so it's cheap to run before or after a `train.py`
run:

```bash
uv run python src/diagnose.py \
  --gene ENSG00000198502 --cell-type "memory B cell" \
  --h5ad-path data/onek1k_cellxgene_standardized.h5ad \
  --vcf-dir /path/to/genotypes --gtf /path/to/gencode.gtf.gz \
  --seq-len 49152 \
  --predictions-csv results/ACTB_memoryBcell/predictions_test_final.csv \
  --out-dir results/ACTB_memoryBcell/diagnostics
```

Pass the same `--seed`, `--min-cells-per-donor`, `--val-frac`, `--test-frac`, `--seq-len`, and VCF
sample-ID options you used for the corresponding `train.py` run, so the donor split and genomic
window line up; `--predictions-csv` is optional but lets it directly compare against your trained
model's actual test predictions.

It reports:

1. **Expression-distribution diagnostics**: zero-inflation/dropout rate, skew, and -- importantly
   -- whether `n_cells` per donor confounds the pseudobulk target (donors with very few cells get a
   noisier *estimate* of cell-cell std purely from small-sample variance, unrelated to biology; this
   shows up as a strong `n_cells` vs. `pseudobulk_std` correlation).
2. **An MHC/HLA region flag.** Genes inside the classical MHC region (chr6:28.48-33.45Mb, GRCh38)
   are known for unusually large, simple genetic effects on expression. The example gene from your
   command, `ENSG00000198502`, is actually **HLA-DRB5** (chr6:32.5Mb) -- and HLA-DRB5 is a
   textbook case: its presence in a person's genome is essentially tied to which `HLA-DRB1` haplotype
   they carry (it's entirely absent on some common haplotypes), tagged by a long, high-LD block of
   common SNPs. That's much closer to "classify a common haplotype" than "learn quantitative
   regulatory grammar," so amazing results there are quite plausibly explained by this alone, not by
   the sequence model learning something subtle.
3. **Top individual variants** in the training window ranked by raw correlation with pseudobulk
   expression -- a single variant with a large, significant correlation is itself a red flag for
   "simple eQTL, not complex grammar." Computed separately for `mu` (pseudobulk mean) and `sigma`
   (empirical cell-cell std) -- see point 4.
4. **An elastic-net baseline, for *both* `mu` and `sigma`**: elastic net regression (L1+L2, like a
   PrediXcan/TWAS model -- the standard approach for predicting a trait from many correlated,
   LD-linked SNPs) directly on raw SNP dosages in the same TSS window used for training, evaluated
   on the identical train/(val+)test donor split (`alpha`/`l1_ratio` are selected via
   `ElasticNetCV`'s internal k-fold CV on train+val). Run once against the pseudobulk mean (a
   standard eQTL-style question) and once against the empirical cell-cell std (a **vQTL-style**
   question: is cell-to-cell variability itself locally genetically determined, or does the
   sequence model have to work harder to predict it than it does for the mean?). If either
   baseline's test Pearson r/R2 is close to Enformer's (especially with `--predictions-csv`
   passed), the deep sequence model likely isn't adding much beyond what a standard eQTL/vQTL scan
   would find for that head. The script also prints a direct `mu` vs. `sigma` baseline comparison
   with an interpretation: comparable R2/significance for both suggests a genuine genetic
   component to expression variability at this locus (not just sampling noise); a much weaker
   `sigma` baseline suggests variability is harder to explain from local genotype alone.
5. **Permutation tests** (label-shuffling, not relying on any normality assumption): one for each
   of the `mu`/`sigma` baselines (shuffles the fit-set labels only, keeps the real test labels, to
   check whether the observed test performance exceeds what could happen by chance given how few
   donors and how many variants there are), and one for your actual model's predictions (each head
   separately) if you pass `--predictions-csv` (shuffles the pairing between predictions and true
   values on the fixed test set, to check whether the observed correlation exceeds chance given
   only `n_test` donors -- this matters a lot when `n_test` is ~20-30, which is common after an
   85/15 split on OneK1K's ~500-900 donors per cell type).
6. **Model prediction-quality stats and plots for *both* heads**, if `--predictions-csv` is given:
   Pearson r, Pearson p, R2, and the permutation p-value for `mu` (vs. true pseudobulk mean) *and*
   separately for `sigma` (vs. true empirical cell-cell std) -- since the whole point of this POC
   is `sigma`, not just `mu`. Printed with a short glossary of what each stat means:
   - **Pearson r**: linear correlation between predicted and true values, -1..1.
   - **Pearson p**: parametric significance of `r` (assumes bivariate normality); smaller = more
     significant, but can be unreliable for small/non-normal `n`.
   - **R2**: fraction of the true values' variance explained by the predictions; 1 = perfect,
     0 = no better than always predicting the mean, <0 = worse than that.
   - **permutation p**: the assumption-free version of Pearson p from point 5 above, applied to
     `mu` and `sigma` individually.

If `--out-dir` is given it also saves `diagnostics_summary.json`, `top_variants_mu.csv` /
`top_variants_sigma.csv`, a 6-panel `diagnostics.png` (top row: pseudobulk histogram, `n_cells`
confound scatter, top-`mu`-variant scatter; bottom row: `mu` baseline's permutation null
distribution, top-`sigma`-variant scatter, `sigma` baseline's permutation null distribution -- so
the `mu` and `sigma` diagnostics sit side by side for easy comparison), and -- if
`--predictions-csv` was given -- `model_predictions_vs_truth.png`: a 2-panel predicted-vs-true
scatter plot (test set), one panel for `mu` and one for `sigma`, each with a `y = x` reference line
and the Pearson r/p, R2, permutation-p stats annotated directly on the plot.

As a sanity check that a *real* gene x cell-type effect, a *simple* large eQTL/vQTL, and *pure
noise* are all distinguishable, `src/diagnose.py` was validated against two synthetic genotype
setups on the same real expression data: (a) random genotypes unlinked to expression, where the
elastic-net baselines and all permutation tests correctly reported no significant signal for
either `mu` or `sigma` (p ~ 0.2-0.9); and (b) genotypes with a deliberately planted, imperfect
(85%-penetrant) linear effect on `mu` from one set of variants and on `sigma` from a *different*
set of variants, where the baselines correctly recovered both effects (test R2 ~ 0.67-0.68,
permutation p = 0/300 for each) and the `mu`-vs-`sigma` comparison correctly flagged the
"comparable genetic signal for both" case.

**Constant targets are handled explicitly, not as a crash.** Pseudobulk targets (or empirical
stds) can legitimately be constant across donors -- e.g. an undetected gene, or a cell type where
almost every donor has only 1 cell of that type (so empirical std is 0 everywhere). Pearson
correlation, R2, skewness, and elastic-net fitting are all undefined or ill-posed in that case.
Both `src/train.py` (via `src/metrics.py`) and `src/diagnose.py` detect this up front (`np.std(...)
== 0`, not just relying on NaN propagation) and print an explicit "skipping" note instead of
raising or warning noisily (especially inside the `n_permutations`-iteration loops), independently
for `mu` and `sigma` -- so if, say, `sigma` happens to be constant but `mu` isn't, the `mu`
diagnostics still run normally and only the `sigma` section is skipped.

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
- A full end-to-end run of `src/diagnose.py` against real OneK1K expression (ACTB x naive B cell,
  which has genuine cross-donor variance in both `mu` and `sigma`) combined with a synthetic VCF of
  genotypes unlinked to expression: confirmed all sections (distribution stats, MHC flag, top
  variants, elastic-net baselines, permutation tests, plots/JSON/CSV output) execute correctly for
  *both* `mu` and `sigma` and report no false "significant" signal (p ~ 0.4-0.65) on genotypes with
  no real relationship to expression.
- The same setup repeated with a synthetic VCF containing a deliberately planted, imperfect linear
  effect -- a different set of variants tagging `mu` vs. `sigma` -- confirming the elastic-net
  baselines correctly recover both planted effects (test R2 ~ 0.67 for `mu`, ~0.68 for `sigma`,
  permutation p = 0/300 for each), that the top-variant tables correctly surface the *correct*
  tagging variants for each target, and that the `mu`-vs-`sigma` "can genotype predict variability
  as well as it predicts the mean?" comparison correctly reports the "comparable genetic signal"
  case.
- The `--predictions-csv` comparison path was exercised with a synthetic predictions CSV (true
  values plus Gaussian noise) built from the same real test-set donors, confirming the
  Enformer-vs-baseline comparison, JSON summary (`mu_baseline`/`sigma_baseline` keys), and both
  plot files render correctly.
- Also unit-tested `is_in_mhc_region` against real HLA-DRB5 and ACTB coordinates, and the
  elastic-net/permutation-test helpers directly against a synthetic case with a known linear
  relationship.
- The `mu`/`sigma` model prediction-quality stats and `model_predictions_vs_truth.png` plot were
  exercised against real `--predictions-csv` output from that same `--random-weights` run
  (confirming both panels render and the annotated stats match a manual recomputation) and against
  a synthetic predictions CSV with constant true values, confirming the "skipped (constant...)"
  path is taken cleanly (nan stats, degraded-but-non-crashing plot) rather than raising.

## Out of scope for this POC

- Variformer's original contrastive pairwise loss term and multi-gene/multi-tissue batching.
- ROSMAP support, distributed training samplers.
- Auto-downloading the reference genome, GTF, or genotype data.
