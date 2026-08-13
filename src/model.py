"""A from-scratch, compact CNN model for personal-genome expression prediction,
directly inspired by pSAGE-net (https://github.com/mostafavilabuw/SAGEnet,
https://www.nature.com/articles/s41592-026-03124-8): a shared convolutional
trunk is applied to a gene's reference sequence and to a donor's maternal and
paternal haplotype sequences; a "mean" head predicts the gene's population
mean expression from the reference branch alone, and a "difference" head
predicts the donor's personal deviation from that mean from
`personal_features - reference_features`.

Two changes vs. pSAGE-net, per this project's plan:
  1. **New third head** (`diff_sigma`): predicts the donor's cell-to-cell
     std of their difference from the population mean -- i.e. how variable
     this gene's expression is across that donor's own cells of the given
     type. There is no single-cell analogue of this in pSAGE-net (which
     trains on bulk RNA-seq, one value per individual).
  2. The `diff` head is trained with a Gaussian NLL against per-cell targets
     (using `diff_sigma`), instead of pSAGE-net's plain MSE against a single
     bulk value -- see `src/loss.py`.

No pretrained weights are used anywhere in this model; everything is trained
from scratch on personal sequence + single-cell expression.
"""
from __future__ import annotations

from math import ceil
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SIGMA_EPS = 1e-3


def conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: str = "same",
    stride: int = 1,
    dilation: int = 1,
    bias: bool = True,
    batch_norm: bool = True,
) -> nn.Sequential:
    """Pre-activation conv block: BatchNorm -> Conv1d -> ReLU.

    Ported from pSAGE-net's `SAGEnet/nn.py::ConvBlock`.
    """
    return nn.Sequential(
        nn.BatchNorm1d(in_channels) if batch_norm else nn.Identity(),
        nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias),
        nn.ReLU(inplace=True),
    )


class Residual(nn.Module):
    """Ported from pSAGE-net's `SAGEnet/nn.py::Residual`."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)


class CNNTrunk(nn.Module):
    """Shared convolutional trunk (ported from pSAGE-net's `rSAGEnet`/`pSAGEnet`
    conv stack, `SAGEnet/models.py`), applied with shared weights to the
    reference sequence and to each haplotype of the personal sequence.

    Defaults are the paper's bolded hyperparameter choices (Methods,
    "R-SAGE-net training").
    """

    def __init__(
        self,
        input_length: int = 40_000,
        first_layer_kernel_number: int = 900,
        int_layers_kernel_number: int = 256,
        first_layer_kernel_size: int = 10,
        int_layers_kernel_size: int = 5,
        n_conv_blocks: int = 5,
        n_dilated_conv_blocks: int = 0,
        pooling_size: int = 10,
        pooling_type: str = "avg",
        batch_norm: bool = True,
        padding: str = "same",
        increasing_dilation: bool = False,
    ):
        super().__init__()
        # pSAGE-net hardcodes batch_norm=False for the first conv layer specifically
        # (unlike r-SAGEnet, which parameterizes it) -- kept to match.
        self.conv0 = conv_block(4, first_layer_kernel_number, first_layer_kernel_size, padding=padding, batch_norm=False)

        def pooling_layer() -> nn.Module:
            if pooling_type == "max":
                return nn.MaxPool1d(pooling_size, ceil_mode=True)
            if pooling_type == "avg":
                return nn.AvgPool1d(pooling_size, ceil_mode=True)
            raise ValueError("pooling_type must be 'max' or 'avg'")

        self.conv_layers = nn.ModuleList()
        fc_dim = input_length

        self.conv_layers.append(
            conv_block(first_layer_kernel_number, int_layers_kernel_number, int_layers_kernel_size, padding=padding, batch_norm=batch_norm)
        )
        self.conv_layers.append(pooling_layer())
        fc_dim = ceil(fc_dim / pooling_size)

        for _ in range(n_conv_blocks - 1):
            self.conv_layers.append(
                Residual(
                    conv_block(
                        int_layers_kernel_number, int_layers_kernel_number, int_layers_kernel_size, padding=padding, batch_norm=batch_norm
                    )
                )
            )
            self.conv_layers.append(pooling_layer())
            fc_dim = ceil(fc_dim / pooling_size)

        self.dilated_conv_layers = nn.ModuleList()
        for layer in range(n_dilated_conv_blocks):
            dilation = 2 ** (layer + 1) if increasing_dilation else 2
            self.dilated_conv_layers.append(
                Residual(
                    conv_block(
                        int_layers_kernel_number,
                        int_layers_kernel_number,
                        int_layers_kernel_size,
                        dilation=dilation,
                        padding=padding,
                        batch_norm=batch_norm,
                    )
                )
            )

        self.output_dim = fc_dim * int_layers_kernel_number

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: `[B, 4, L]` one-hot sequence (channel-first, for Conv1d). Returns flattened `[B, output_dim]`."""
        x = self.conv0(x)
        for layer in self.conv_layers:
            x = layer(x)
        for layer in self.dilated_conv_layers:
            x = layer(x)
        return x.flatten(1)


def _make_hidden_layers(input_dim: int, hidden_size: int, h_layers: int, dropout: float) -> Tuple[nn.Module, int]:
    """`h_layers` of `Linear -> ReLU -> Dropout`; first maps `input_dim -> hidden_size`,
    later ones `hidden_size -> hidden_size`. `h_layers=0` -> identity (output dim = `input_dim`).

    Returns `(module, output_dim)` so callers can size a final `Linear` head correctly
    even when `h_layers=0` (pSAGE-net's own code assumes `h_layers>=1` and would
    silently break on `h_layers=0` with a concatenated, `2x`-width input).
    """
    layers = []
    in_dim = input_dim
    for _ in range(max(h_layers, 0)):
        layers += [nn.Linear(in_dim, hidden_size), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        in_dim = hidden_size
    module = nn.Sequential(*layers) if layers else nn.Identity()
    return module, in_dim


class PSAGEnetSC(nn.Module):
    """pSAGE-net-inspired model for single-cell expression prediction, with a
    new third ("diff_sigma") head.

    A shared CNN trunk + shared feature-projection layer (`fc0`, matching
    pSAGE-net) is applied identically to the reference sequence and to each
    haplotype of the personal sequence. Three FC heads then branch off:
      - `mean`: from reference features alone -> `m_hat` (population mean).
      - `diff`: from `personal_features - reference_features` (or
        concatenated, via `subtract_or_concat`) -> `d_hat` (personal
        difference from the mean).
      - `diff_sigma` (**new**): shares `diff`'s hidden layers, separate
        output projection, `softplus`-activated -> `sigma_d_hat` (predicted
        cell-to-cell std of that difference), used by the Gaussian NLL loss
        in `src/loss.py`.
    """

    def __init__(
        self,
        input_length: int = 40_000,
        first_layer_kernel_number: int = 900,
        int_layers_kernel_number: int = 256,
        first_layer_kernel_size: int = 10,
        int_layers_kernel_size: int = 5,
        hidden_size: int = 256,
        n_conv_blocks: int = 5,
        n_dilated_conv_blocks: int = 0,
        h_layers: int = 1,
        pooling_size: int = 10,
        pooling_type: str = "avg",
        batch_norm: bool = True,
        padding: str = "same",
        dropout: float = 0.0,
        increasing_dilation: bool = False,
        subtract_or_concat: str = "subtract",
    ):
        super().__init__()
        if subtract_or_concat not in ("subtract", "concat"):
            raise ValueError("subtract_or_concat must be 'subtract' or 'concat'")
        self.subtract_or_concat = subtract_or_concat

        self.trunk = CNNTrunk(
            input_length=input_length,
            first_layer_kernel_number=first_layer_kernel_number,
            int_layers_kernel_number=int_layers_kernel_number,
            first_layer_kernel_size=first_layer_kernel_size,
            int_layers_kernel_size=int_layers_kernel_size,
            n_conv_blocks=n_conv_blocks,
            n_dilated_conv_blocks=n_dilated_conv_blocks,
            pooling_size=pooling_size,
            pooling_type=pooling_type,
            batch_norm=batch_norm,
            padding=padding,
            increasing_dilation=increasing_dilation,
        )

        # Shared projection (pSAGE-net's `fc0`): same weights applied to both the
        # reference branch and the (already haplotype-averaged) personal branch.
        self.feature_proj = nn.Sequential(nn.Linear(self.trunk.output_dim, hidden_size), nn.ReLU(inplace=True))

        self.mean_fc, mean_fc_out_dim = _make_hidden_layers(hidden_size, hidden_size, h_layers, dropout)
        self.mean_out = nn.Linear(mean_fc_out_dim, 1)

        diff_input_dim = hidden_size * 2 if subtract_or_concat == "concat" else hidden_size
        self.diff_fc, diff_fc_out_dim = _make_hidden_layers(diff_input_dim, hidden_size, h_layers, dropout)
        self.diff_out = nn.Linear(diff_fc_out_dim, 1)
        self.diff_sigma_out = nn.Linear(diff_fc_out_dim, 1)

    def _trunk_features(self, seq: torch.Tensor) -> torch.Tensor:
        """`seq`: `[B, L, 4]` one-hot -> raw (pre-projection) trunk features `[B, trunk.output_dim]`."""
        return self.trunk(seq.transpose(1, 2))  # Conv1d wants channels-first: [B, 4, L]

    def forward(
        self, ref_seq: torch.Tensor, mat_seq: torch.Tensor, pat_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            ref_seq: `[B, L, 4]` one-hot reference sequence (TSS-centered).
            mat_seq, pat_seq: `[B, L, 4]` one-hot maternal/paternal haplotype
                sequences for the same window (phased, see `src/genome.py`).

        Returns:
            m_hat: `[B]` predicted population mean expression (from `ref_seq` only).
            d_hat: `[B]` predicted personal difference from the population mean.
            sigma_d_hat: `[B]` predicted cell-to-cell std of that difference, `> 0`.
        """
        ref_raw = self._trunk_features(ref_seq)
        mat_raw = self._trunk_features(mat_seq)
        pat_raw = self._trunk_features(pat_seq)
        personal_raw = (mat_raw + pat_raw) / 2.0

        ref_feat = self.feature_proj(ref_raw)
        personal_feat = self.feature_proj(personal_raw)  # same (shared) weights as ref_feat

        if self.subtract_or_concat == "concat":
            diff_feat = torch.cat([personal_feat, ref_feat], dim=-1)
        else:
            diff_feat = personal_feat - ref_feat

        m_hat = self.mean_out(self.mean_fc(ref_feat)).squeeze(-1)

        diff_hidden = self.diff_fc(diff_feat)
        d_hat = self.diff_out(diff_hidden).squeeze(-1)
        sigma_d_hat = F.softplus(self.diff_sigma_out(diff_hidden)).squeeze(-1) + SIGMA_EPS

        return m_hat, d_hat, sigma_d_hat
