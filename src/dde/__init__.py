"""DDE module for defining and solving delay differential equations."""

from .families import (
    DDEConfig,
    DDEFamily,
    Linear2DDE,
    HutchDDE,
    MackeyGlassDDE,
    VdPDDE,
    PredatorPreyDDE,
    DistUniformDDE,
    DistExpDDE,
    DDE_FAMILIES,
    get_family,
)

__all__ = [
    "DDEConfig",
    "DDEFamily",
    "Linear2DDE",
    "HutchDDE",
    "MackeyGlassDDE",
    "VdPDDE",
    "PredatorPreyDDE",
    "DistUniformDDE",
    "DistExpDDE",
    "DDE_FAMILIES",
    "get_family",
]
