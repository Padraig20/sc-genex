"""Per-cell-weighted Gaussian negative log-likelihood loss.

Implements the training objective described in the plan: each donor's DNA
input is identical across their cells, so the model produces a single
`(mu, sigma)` per donor. Rather than literally repeating the forward pass once
per cell, we score that single `(mu, sigma)` against every one of the donor's
actual per-cell target values and sum the per-cell NLLs. Summing (instead of
averaging) within a donor, then dividing the whole batch's total loss by the
batch's total *cell* count (not by the number of donors), makes this exactly
equivalent to training on every single cell independently with a shared input
-- which is the "no pseudobulking, but still learns the mean via MLE" idea:
for a fixed sigma, minimizing this loss w.r.t. mu recovers the per-donor
sample mean; jointly optimizing mu and sigma recovers the per-donor sample
mean and (biased) sample variance, i.e. the Gaussian MLE.
"""
from __future__ import annotations

import math
from typing import List, Sequence

import torch

LOG_2PI = math.log(2 * math.pi)


def gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Elementwise Gaussian NLL: 0.5 * (log(2*pi*sigma^2) + ((y - mu) / sigma)^2)."""
    var = sigma.pow(2)
    return 0.5 * (LOG_2PI + torch.log(var) + (target - mu).pow(2) / var)


def per_cell_gaussian_nll(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    cell_targets: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Batch-total Gaussian NLL, weighted equally per cell (not per donor).

    Args:
        mu: [B] predicted mean per donor in the batch.
        sigma: [B] predicted std per donor in the batch (must be > 0).
        cell_targets: length-B sequence of 1D tensors, one per donor, each
            holding that donor's per-cell normalized expression values
            (variable length across donors).

    Returns:
        Scalar loss = (sum over donors of sum over that donor's cells of the
        per-cell NLL) / (total number of cells in the batch).
    """
    assert mu.shape[0] == sigma.shape[0] == len(cell_targets)
    total_nll = mu.new_zeros(())
    total_cells = 0
    for i, targets in enumerate(cell_targets):
        if targets.numel() == 0:
            continue
        mu_i = mu[i].expand_as(targets)
        sigma_i = sigma[i].expand_as(targets)
        total_nll = total_nll + gaussian_nll(mu_i, sigma_i, targets).sum()
        total_cells += targets.numel()
    if total_cells == 0:
        return total_nll
    return total_nll / total_cells


def pseudobulk_mse(mu: torch.Tensor, pseudobulk_target: torch.Tensor) -> torch.Tensor:
    """Plain MSE between predicted mu and the donor-level pseudobulk mean.

    Useful as an evaluation-time sanity metric (not used for training).
    """
    mask = ~torch.isnan(pseudobulk_target)
    return torch.mean((mu[mask] - pseudobulk_target[mask]) ** 2)
