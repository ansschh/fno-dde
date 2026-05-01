"""Models module for 1D FNO, LEMO, and baseline architectures."""

from .fno1d import FNO1d, FNO1dResidual, SpectralConv1d, FNOBlock1d, create_fno1d, count_parameters
from .lemo import LEMO, ContinuousLagConv1d, PointwiseNorm, create_lemo
from .baselines import NaiveBaseline, TCN, MLPBaseline, LinearODEBaseline, create_baseline
from .shift_augmentation import apply_cyclic_shift
from .research_baselines import (
    DeepONet, MemNO, ANIE, LocalizedNO, LNO, MFNO, ZFNO, VanillaFNO,
    create_research_baseline,
)

__all__ = [
    "FNO1d",
    "FNO1dResidual",
    "SpectralConv1d",
    "FNOBlock1d",
    "create_fno1d",
    "count_parameters",
    "LEMO",
    "ContinuousLagConv1d",
    "PointwiseNorm",
    "create_lemo",
    "NaiveBaseline",
    "TCN",
    "MLPBaseline",
    "LinearODEBaseline",
    "create_baseline",
    "apply_cyclic_shift",
    "DeepONet",
    "MemNO",
    "ANIE",
    "LocalizedNO",
    "LNO",
    "MFNO",
    "ZFNO",
    "VanillaFNO",
    "create_research_baseline",
]
