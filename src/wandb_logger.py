"""Optional Weights & Biases logging wrapper.

`train.py` never requires a W&B account: pass `--wandb` to enable real
logging (requires `wandb` installed and logged in); otherwise everything
routes through a no-op logger so the POC runs standalone.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


class NoOpLogger:
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        pass

    def log_table(self, name: str, df: pd.DataFrame) -> None:
        pass

    def finish(self) -> None:
        pass


class WandbLogger:
    """Thin wrapper around `wandb`. Construction fails loudly if wandb is unavailable."""

    def __init__(self, project: str = "sc-genex-variformer-poc", **init_kwargs):
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Install it (`uv add wandb`) or run without --wandb."
            ) from exc
        self._wandb = wandb
        self._run = wandb.init(project=project, **init_kwargs)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        self._wandb.log(metrics, step=step)

    def log_table(self, name: str, df: pd.DataFrame) -> None:
        self._wandb.log({name: self._wandb.Table(dataframe=df)})

    def finish(self) -> None:
        self._wandb.finish()


def get_logger(use_wandb: bool, **init_kwargs):
    if use_wandb:
        return WandbLogger(**init_kwargs)
    return NoOpLogger()
