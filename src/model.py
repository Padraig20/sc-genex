"""Variformer-style Enformer fine-tuning with a two-headed (mu, sigma) output.

Follows Variformer (https://github.com/shirondru/enformer_fine_tuning): loads
a pretrained Enformer, replaces its output head via `HeadAdapterWrapper`, and
reads off the prediction at the center bin of the (uncropped) output. The only
architectural change here is `num_tracks=2` instead of 1 per tissue: one track
for the predicted mean `mu`, one raw track that is transformed into a
strictly-positive `sigma` via softplus, to support the Gaussian NLL loss in
`loss.py`.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from enformer_pytorch import Enformer
from enformer_pytorch.finetune import HeadAdapterWrapper

PRETRAINED_MODEL_NAME = "EleutherAI/enformer-official-rough"
SIGMA_EPS = 1e-3


class VariformerGNLL(nn.Module):
    """Enformer fine-tuned to output a per-gene x cell-type (mu, sigma) pair.

    Args:
        random_weights: if True, initializes Enformer with random weights
            instead of downloading the ~350M-parameter pretrained checkpoint.
            Intended only for smoke-testing the pipeline on machines without
            GPU/bandwidth for the real checkpoint -- not scientifically
            meaningful.
        freeze_enformer: if True, freezes the whole Enformer trunk and only
            trains the new (mu, sigma) head. Useful to cut compute when full
            fine-tuning (Variformer's default, `freeze_enformer=False`) is
            too expensive.
        finetune_last_n_layers_only: if set, freezes all but the last N
            transformer blocks of Enformer (and the head). Ignored if
            `freeze_enformer=True`.
    """

    def __init__(
        self,
        random_weights: bool = False,
        freeze_enformer: bool = False,
        finetune_last_n_layers_only: Optional[int] = None,
    ):
        super().__init__()
        self.freeze_enformer = freeze_enformer
        self.finetune_last_n_layers_only = finetune_last_n_layers_only

        if random_weights:
            enformer = Enformer.from_hparams(
                dim=1536,
                depth=11,
                heads=8,
                output_heads=dict(human=5313, mouse=1643),
                target_length=-1,
            )
        else:
            enformer = Enformer.from_pretrained(PRETRAINED_MODEL_NAME, target_length=-1)

        self.head = HeadAdapterWrapper(
            enformer=enformer,
            num_tracks=2,  # [mu, raw_sigma]
            post_transformer_embed=False,  # required by HeadAdapterWrapper for this use case
            output_activation=nn.Identity(),
        )

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            seq: [B, L, 4] one-hot DNA sequence batch.

        Returns:
            mu: [B] predicted mean normalized expression.
            sigma: [B] predicted std of normalized expression across cells, > 0.
        """
        preds = self.head(
            seq,
            freeze_enformer=self.freeze_enformer,
            finetune_last_n_layers_only=self.finetune_last_n_layers_only,
        )  # [B, num_bins, 2]
        center_bin = preds.shape[1] // 2
        center = preds[:, center_bin, :]  # [B, 2]
        mu = center[:, 0]
        raw_sigma = center[:, 1]
        sigma = torch.nn.functional.softplus(raw_sigma) + SIGMA_EPS
        return mu, sigma
