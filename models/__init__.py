import importlib
import pkgutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class TargetConfig:
    horizon: int = 100  # 50, 100, or 200[cite: 1]
    label_prefix: str = "y"  # "y" (quantile), "y_sd", or "y_onset"[cite: 1]
    leader_only: bool = True  # filter by mask_leader_{d}
    markout_col: str = (
        "mid_fwd_h100"  # e.g., "mid_fwd_h100" or "mid_fwd_ms5000"[cite: 1]
    )
    top_k_pcts: tuple = (1.0, 2.0, 5.0)

    def target_col(self, direction: str) -> str:
        return f"{self.label_prefix}_h{self.horizon}_{direction}"

    def mask_col(self, direction: str) -> str:
        return f"mask_leader_{direction}"


class BaseSpikeModel(ABC):
    """Plug-and-play base interface for all models"""

    name: str = "base"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def fit(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        cfg: TargetConfig,
    ) -> "BaseSpikeModel":
        """Fit model using CONTROLS-2026 train_df and optional val_df.

        Note: train_df contains full chunks up to the train split, so rolling
        features can be computed per chunk_id before filtering to split ==
        'train'.
        """

    @abstractmethod
    def predict_proba(
        self,
        df: pl.DataFrame,
        cfg: TargetConfig,
    ) -> pl.DataFrame:
        """Compute features causally on full chunks in `df`, then return a DataFrame

        containing at least: ['chunk_id', 'event_seq', 'prob_buyR',
        'prob_sellR'].
        """

    @staticmethod
    def valid_mask_expr(
        cfg: TargetConfig, direction: str, split: str | None = None
    ) -> pl.Expr:
        """Helper expression to filter valid labelled rows for a given direction."""
        y_col = cfg.target_col(direction)
        expr = pl.col(y_col).is_not_null()
        if cfg.leader_only:
            expr = expr & pl.col(cfg.mask_col(direction)).fill_null(False)
        if split is not None:
            expr = expr & (pl.col("split") == split)
        return expr


MODEL_REGISTRY: dict[str, type[BaseSpikeModel]] = {}


def register_model(name: str):
    """Decorator to register a model class into MODEL_REGISTRY."""

    def decorator(cls: type[BaseSpikeModel]):
        cls.name = name
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def discover_models() -> dict[str, type[BaseSpikeModel]]:
    """Dynamically import all .py modules inside models/ so @register_model runs."""
    pkg_dir = Path(__file__).resolve().parent
    for _, module_name, is_pkg in pkgutil.iter_modules([str(pkg_dir)]):
        if not is_pkg and not module_name.startswith("_"):
            importlib.import_module(f"{__name__}.{module_name}")
    return MODEL_REGISTRY


def get_model(name: str, **kwargs) -> BaseSpikeModel:
    discover_models()
    if name not in MODEL_REGISTRY:
        avail = ", ".join(sorted(MODEL_REGISTRY.keys())) or "<none>"
        raise ValueError(f"Unknown model '{name}'. Available models: [{avail}]")
    return MODEL_REGISTRY[name](**kwargs)
