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


class ReferenceExpressionModel(nn.Module):
    """r-SAGE-net equivalent: reference-sequence-only model predicting a
    gene's (bulk) population-mean expression, with no personal genome/donor
    input at all -- mirrors the paper's `rSAGEnet` (own `fc0`/`fclayers`/
    `ref_out`), trained with plain MSE against per-gene population-mean
    targets (`src/pretrain.py`).

    Shares `CNNTrunk` and its hyperparameters with `PSAGEnetSC` so that this
    model's `trunk` submodule's `state_dict()` can be loaded directly into a
    `PSAGEnetSC.trunk` afterwards (`src/train.py --init-from-reference-model`)
    -- only the conv/pooling trunk transfers; `feature_proj`/`fc`/`out` here
    have no counterpart in `PSAGEnetSC` and are never loaded into it.
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
    ):
        super().__init__()
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
        self.feature_proj = nn.Sequential(nn.Linear(self.trunk.output_dim, hidden_size), nn.ReLU(inplace=True))
        self.fc, fc_out_dim = _make_hidden_layers(hidden_size, hidden_size, h_layers, dropout)
        self.out = nn.Linear(fc_out_dim, 1)

    def forward(self, ref_seq: torch.Tensor) -> torch.Tensor:
        """`ref_seq`: `[B, L, 4]` one-hot reference sequence (TSS-centered). Returns `m_hat`: `[B]`."""
        raw = self.trunk(ref_seq.transpose(1, 2))  # Conv1d wants channels-first: [B, 4, L]
        feat = self.feature_proj(raw)
        return self.out(self.fc(feat)).squeeze(-1)


# The subset of `ReferenceExpressionModel`/`PSAGEnetSC` constructor kwargs that
# actually determine `CNNTrunk`'s architecture (and therefore its state_dict
# shapes) -- shared by `src/pretrain.py` (saved alongside `reference_model.pt`)
# and `src/train.py --init-from-reference-model` (which checks the *current*
# run's `model_config` against it before loading trunk weights, since a
# shape/config mismatch would otherwise fail deep inside `load_state_dict`
# with a much less actionable error). `hidden_size`/`h_layers`/`dropout`/
# `subtract_or_concat` are deliberately excluded: they affect only
# `PSAGEnetSC`'s post-trunk heads (or `ReferenceExpressionModel`'s own
# never-transferred `feature_proj`/`fc`/`out`), not the trunk itself.
TRUNK_HYPERPARAM_KEYS: Tuple[str, ...] = (
    "input_length",
    "first_layer_kernel_number",
    "int_layers_kernel_number",
    "first_layer_kernel_size",
    "int_layers_kernel_size",
    "n_conv_blocks",
    "n_dilated_conv_blocks",
    "pooling_size",
    "pooling_type",
    "batch_norm",
    "padding",
    "increasing_dilation",
)


def validate_trunk_hyperparams(current_config: dict, reference_config: dict) -> None:
    """Raises `ValueError` listing every `TRUNK_HYPERPARAM_KEYS` entry that
    differs between `current_config` (this run's `PSAGEnetSC` `model_config`)
    and `reference_config` (a pretrained `ReferenceExpressionModel`'s saved
    `reference_model_config.json`) -- called by `src/train.py
    --init-from-reference-model` before loading trunk weights, since a
    shape mismatch would otherwise only surface as a much less actionable
    error deep inside `load_state_dict`.
    """
    mismatches = []
    for key in TRUNK_HYPERPARAM_KEYS:
        current_value = current_config.get(key)
        reference_value = reference_config.get(key)
        if current_value != reference_value:
            mismatches.append(f"  {key}: current={current_value!r} vs reference_model={reference_value!r}")
    if mismatches:
        raise ValueError(
            "Trunk hyperparameter mismatch between this run's model config and the "
            "--init-from-reference-model checkpoint's reference_model_config.json "
            "(trunk weights would not load correctly):\n" + "\n".join(mismatches)
        )


def load_trunk_from_reference_model(model: "PSAGEnetSC", reference_state_dict: dict) -> None:
    """Loads only the `trunk.*` keys of a `ReferenceExpressionModel.state_dict()`
    into `model.trunk` -- `feature_proj`/`mean_fc`/`mean_out`/`diff_fc`/
    `diff_out`/`diff_sigma_out` are left randomly initialized, matching the
    paper exactly ("all parameters in the convolutional and pooling layers of
    r-SAGE-net are loaded into p-SAGE-net" -- nothing else transfers).
    """
    trunk_prefix = "trunk."
    trunk_state = {k[len(trunk_prefix) :]: v for k, v in reference_state_dict.items() if k.startswith(trunk_prefix)}
    if not trunk_state:
        raise ValueError("No 'trunk.*' keys found in the reference model checkpoint's state_dict -- wrong checkpoint file?")
    model.trunk.load_state_dict(trunk_state, strict=True)
