"""
Shared model-class dispatcher for 1D training and evaluation.

Consolidates the dispatch logic that `scripts/dry_run.py::build_model`
hard-codes for short smoke-tests, so the Phase B sharded trainer
(`train_fno_sharded.py`) and any future trainer can dispatch all 11
baselines from the plan without duplicating the glue.

The `model_class` key comes from a sweep-generated per-run YAML
(see `configs/sweep_phase_b.yaml::model_configs.*`). Allowed values:

    mlp, fno1d, lemo,
    deeponet, memno, anie, nie, localno, lno, mfno, zfno, vanillafno

The kwargs live under `config["model"]` (shared by all classes); LEMO
additionally accepts a top-level `sigma`.
"""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from models import (
    MLPBaseline, create_fno1d, create_lemo, create_research_baseline,
)


_RESEARCH_BASELINES = {
    "deeponet", "memno", "anie", "nie",
    "localno", "lno", "mfno", "zfno", "vanillafno",
}


def build_model(config: dict, in_channels: int, out_channels: int,
                length: int) -> nn.Module:
    """Dispatch on `config['model_class']`. Returns an nn.Module.

    `config` is the full flat run-config (as written by
    slurm.sweep_phase_b); per-class kwargs are read from `config['model']`.
    """
    model_class: str = config.get("model_class", "fno1d").lower()
    model_cfg: dict[str, Any] = config.get("model", {})

    if model_class == "mlp":
        return MLPBaseline(
            seq_length=length,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=model_cfg.get("hidden_dim", 256),
            n_layers=model_cfg.get("n_layers", 4),
        )

    if model_class == "fno1d":
        return create_fno1d(
            in_channels, out_channels, model_cfg,
            use_residual=config.get("use_residual", False),
        )

    if model_class == "lemo":
        c = dict(config)
        c.setdefault("model", {})
        c["model"] = {**c["model"], "length": length}
        return create_lemo(in_channels, out_channels, c)

    # Round 2.23: ND-generic models (1D recovers when spatial_shape=())
    if model_class == "lemo_nd":
        from models.lemo_nd import create_lemo_nd
        return create_lemo_nd(in_channels, out_channels, config)

    if model_class == "lemo_pc_nd":
        from models.lemo_pc_nd import create_lemo_pc_nd
        return create_lemo_pc_nd(in_channels, out_channels, config, length=length)

    if model_class == "causal_lemo_pc_nd":
        from models.lemo_pc_nd import create_causal_lemo_pc_nd
        return create_causal_lemo_pc_nd(in_channels, out_channels, config, length=length)

    if model_class == "fno_nd":
        from models.fno_nd import create_fno_nd
        return create_fno_nd(in_channels, out_channels, config)

    if model_class == "markov_fno_nd":
        from models.baselines_nd import create_markov_fno_nd
        return create_markov_fno_nd(in_channels, out_channels, config, length=length)

    if model_class == "windowed_fno_nd":
        from models.baselines_nd import create_windowed_fno_nd
        return create_windowed_fno_nd(in_channels, out_channels, config, length=length)

    if model_class == "unet_nd":
        from models.baselines_nd import create_unet_nd
        return create_unet_nd(in_channels, out_channels, config, length=length)

    if model_class == "memno_nd":
        from models.baselines_nd import create_memno_nd
        return create_memno_nd(in_channels, out_channels, config, length=length)

    if model_class == "ffno_nd":
        from models.baselines_nd import create_ffno_nd
        return create_ffno_nd(in_channels, out_channels, config, length=length)

    if model_class in _RESEARCH_BASELINES:
        return create_research_baseline(
            model_class, in_channels, out_channels, length, config,
        )

    raise ValueError(
        f"unknown model_class: {model_class!r}. "
        f"Allowed: mlp, fno1d, lemo, lemo_nd, lemo_pc_nd, causal_lemo_pc_nd, fno_nd, "
        f"{sorted(_RESEARCH_BASELINES)}"
    )
