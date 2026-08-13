"""Combined training objective for `PSAGEnetSC`: MSE on the mean head, plus a
per-cell-weighted Gaussian NLL on the difference head (using the new
`diff_sigma` head to control its scale), matching the plan:

    d_ij = y_ij - m_g                                                  # cell j of donor i, gene g
    mean_loss = MSE(m_hat_gi, m_g)                                     # per (gene, donor) example
    diff_loss = per_cell_gaussian_nll(d_hat_gi, sigma_d_hat_gi, {d_ij}_j)
    loss = lam_ref * mean_loss + lam_diff * diff_loss                  # lam_ref=1, lam_diff=10 (paper defaults)

This mirrors pSAGE-net's own weighted two-term loss (`L = lambda_m *
MSE(mean) + lambda_d * MSE(diff)`, see the paper's Fig. 1a), with the
`diff` term upgraded from plain MSE (bulk RNA-seq, one value per individual)
to a Gaussian NLL over every single cell's own difference-from-mean value
(no pseudobulking at training time) -- exactly the "no pseudobulking, but it
still learns the mean via MLE" property from the original single-gene POC,
now applied specifically to the *difference* head rather than the raw
expression value.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

LOG_2PI = math.log(2 * math.pi)


def gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Elementwise Gaussian NLL: 0.5 * (log(2*pi*sigma^2) + ((y - mu) / sigma)^2)."""
    var = sigma.pow(2)
    return 0.5 * (LOG_2PI + torch.log(var) + (target - mu).pow(2) / var)


def per_cell_gaussian_nll(
    d_hat: torch.Tensor,
    sigma_d_hat: torch.Tensor,
    cell_diff_targets: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Batch-total Gaussian NLL over the difference head, weighted equally per cell.

    Args:
        d_hat: `[B]` predicted personal difference-from-mean, one value per
            `(gene, donor)` example (constant across that donor's cells,
            since the model only sees one DNA input per example).
        sigma_d_hat: `[B]` predicted cell-to-cell std of that difference,
            one value per example (`> 0`).
        cell_diff_targets: length-`B` sequence of 1D tensors, one per
            `(gene, donor)` example, each holding that donor's per-cell
            difference-from-population-mean values for that gene
            (`y_ij - m_g`, variable length across examples).

    Returns:
        Scalar loss = (sum over examples of sum over that example's cells of
        the per-cell NLL) / (total number of cells in the batch). Weighting
        by total cells (not by number of examples) makes this mathematically
        equivalent to training on every single cell independently with a
        shared per-example input -- for a fixed `sigma_d_hat`, minimizing
        w.r.t. `d_hat` recovers each `(gene, donor)`'s per-cell sample mean
        difference; jointly optimizing recovers the sample mean and (biased)
        sample std, i.e. the Gaussian MLE.
    """
    assert d_hat.shape[0] == sigma_d_hat.shape[0] == len(cell_diff_targets)
    total_nll = d_hat.new_zeros(())
    total_cells = 0
    for i, targets in enumerate(cell_diff_targets):
        if targets.numel() == 0:
            continue
        d_hat_i = d_hat[i].expand_as(targets)
        sigma_i = sigma_d_hat[i].expand_as(targets)
        total_nll = total_nll + gaussian_nll(d_hat_i, sigma_i, targets).sum()
        total_cells += targets.numel()
    if total_cells == 0:
        return total_nll
    return total_nll / total_cells


@dataclass
class LossComponents:
    total: torch.Tensor
    mean_loss: torch.Tensor
    diff_loss: torch.Tensor


def psagenet_sc_loss(
    m_hat: torch.Tensor,
    d_hat: torch.Tensor,
    sigma_d_hat: torch.Tensor,
    population_mean_targets: torch.Tensor,
    cell_diff_targets: Sequence[torch.Tensor],
    lam_ref: float = 1.0,
    lam_diff: float = 10.0,
) -> LossComponents:
    """Combined training objective: `lam_ref * MSE(mean) + lam_diff * GNLL(diff, diff_sigma)`.

    Args:
        m_hat: `[B]` predicted population mean, one per `(gene, donor)` example.
        d_hat, sigma_d_hat: `[B]` predicted difference / difference-std, one per example.
        population_mean_targets: `[B]` true population mean for each example's
            gene (the same value repeated across every donor of that gene --
            see `data.preprocess.compute_population_means`; always computed
            from TRAIN donors only, regardless of the example's own split).
        cell_diff_targets: per-example per-cell difference-from-population-mean
            targets, see `per_cell_gaussian_nll`.
        lam_ref, lam_diff: loss term weights (paper defaults: 1 and 10 --
            pSAGE-net upweights the "difference" term since it's the harder,
            per-individual signal, and the "mean" term would otherwise
            dominate early training).

    Returns:
        `LossComponents(total, mean_loss, diff_loss)` -- the unweighted
        `mean_loss`/`diff_loss` are included for logging each term separately.
    """
    mean_loss = torch.mean((m_hat - population_mean_targets) ** 2)
    diff_loss = per_cell_gaussian_nll(d_hat, sigma_d_hat, cell_diff_targets)
    total = lam_ref * mean_loss + lam_diff * diff_loss
    return LossComponents(total=total, mean_loss=mean_loss, diff_loss=diff_loss)


def pseudobulk_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Plain MSE, ignoring NaNs -- an eval-time sanity metric only (not used for training)."""
    mask = ~torch.isnan(target)
    return torch.mean((pred[mask] - target[mask]) ** 2)
