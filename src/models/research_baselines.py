"""
Research-grade operator-learning baselines for the LEMO benchmark.

All six models implement the same forward contract as FNO1d:
    input  (batch, length, in_channels)
    output (batch, length, out_channels)

These are simplified, paper-internal implementations inspired by the cited
works. They are NOT exact replicas of the original authors' released code;
they preserve the defining inductive bias of each architecture while fitting
the (batch, length, channels) harness of the dde-fno benchmark. Each class
docstring cites the paper it is based on and notes the simplifications made.

Contents:
    DeepONet        -- Lu et al. 2021, Nature Mach. Intell. (deepxde backend)
    MemNO           -- Buitrago et al. ICLR 2025 (author architecture + S4D)
    ANIE            -- Zappala et al. Nat. Mach. Intell. 2024 (vendored IE_source/)
    LocalizedNO     -- Liu-Schiaffini et al. ICML 2024 (paper-description reimpl)
    LNO             -- Cao et al. Nat. Mach. Intell. 2024 (vendored 1D_Duffing)
    MFNO            -- Lee et al. arXiv 2025 (user's paper-faithful implementation)
    ZFNO, VanillaFNO -- MFNO paper's padding-ablation variants
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Helpers
# ============================================================================

class PointwiseMLP(nn.Module):
    """Pointwise MLP applied to each time step independently. Input
    (batch, length, in_c) -> (batch, length, out_c)."""
    def __init__(self, in_c: int, out_c: int, hidden: int = 64, depth: int = 2,
                 act: str = "gelu") -> None:
        super().__init__()
        act_mod = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}[act]
        layers = [nn.Linear(in_c, hidden), act_mod()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), act_mod()]
        layers.append(nn.Linear(hidden, out_c))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# 1. DeepONet (Lu et al., Nat. Mach. Intell. 2021)
# ============================================================================
#
# Author-faithful: backed by deepxde.nn.pytorch.deeponet.DeepONetCartesianProd.
# Requires `pip install deepxde` with DDE_BACKEND=pytorch.
# The branch/trunk MLPs, dot-product merge, and bias term follow the
# DeepXDE implementation verbatim; our adapter only handles the reshape
# from our (B, L, C_in) -> (B, L, C_out) benchmark contract into the
# Cartesian-product (branch=(B, L*C_in), trunk=(L, 1)) format.

def _import_deepxde_cp():
    """Lazy import. DeepXDE sets torch defaults at import time (e.g. default
    device) that conflict with our DataLoader generators. We save the
    default-generator device before import and restore it after.
    """
    import os as _os
    _os.environ.setdefault("DDE_BACKEND", "pytorch")
    # Snapshot pre-import torch defaults.
    _prev_default_device = None
    try:
        _prev_default_device = torch.get_default_device() if hasattr(torch, "get_default_device") else None
    except Exception:
        pass
    from deepxde.nn.pytorch.deeponet import DeepONetCartesianProd as _CP
    # Restore defaults.
    try:
        if hasattr(torch, "set_default_device"):
            torch.set_default_device(_prev_default_device or "cpu")
    except Exception:
        pass
    return _CP


class DeepONet(nn.Module):
    """Adapter over deepxde.nn.pytorch.deeponet.DeepONetCartesianProd.

    Paper: Lu et al., "Learning nonlinear operators via DeepONet based on the
    universal approximation theorem of operators," Nat. Mach. Intell. 2021.
    Implementation: https://github.com/lululxvi/deepxde (PyTorch backend).

    The DeepONet core (branch MLP + trunk MLP + dot-product merge + bias) is
    unmodified. Our adapter reshapes the benchmark's (batch, length, in_channels)
    input into the Cartesian-product format the reference code expects:
      branch input = flattened history (batch, length * in_channels)
      trunk  input = time coordinates on [0, 1] (length, 1)
    and stacks the per-output-channel scalar predictions into a single
    (batch, length, out_channels) tensor via `num_outputs=out_channels` with
    the "independent" multi-output strategy.
    """

    def __init__(self, in_channels: int, out_channels: int, length: int,
                 p: int = 128, branch_hidden: int = 128, trunk_hidden: int = 128,
                 depth: int = 4, activation: str = "gelu",
                 kernel_initializer: str = "Glorot normal",
                 **_ignored) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.length = length
        try:
            _DeepONetCP = _import_deepxde_cp()
        except Exception as e:
            raise RuntimeError(
                "deepxde with PyTorch backend is required for the DeepONet baseline. "
                f"Import failed: {e!r}. Install with `pip install deepxde`."
            )
        branch_sizes = [length * in_channels] + [branch_hidden] * (depth - 2) + [p]
        trunk_sizes = [1] + [trunk_hidden] * (depth - 2) + [p]
        self._net = _DeepONetCP(
            layer_sizes_branch=branch_sizes,
            layer_sizes_trunk=trunk_sizes,
            activation=activation,
            kernel_initializer=kernel_initializer,
            num_outputs=out_channels,
            multi_output_strategy=None if out_channels == 1 else "independent",
        )
        self.register_buffer(
            "_trunk_grid",
            torch.linspace(0.0, 1.0, length, dtype=torch.float32).view(length, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        branch_in = x.reshape(B, L * C)
        trunk_in = self._trunk_grid.to(x.device)
        y = self._net((branch_in, trunk_in))
        if y.dim() == 2:
            y = y.unsqueeze(-1)
        return y


# ============================================================================
# 2. MemNO (Buitrago et al., ICLR 2025)
# ============================================================================
#
# Paper-faithful architectural pattern lives in memno_paper.py. The
# paper's Python code (rearrange-based markovian/memory interleaving)
# is reproduced exactly for our S=1 1D time-series contract, and the
# memory_layer is a minimal S4D (Gu 2022) — a strict subset of the
# paper's S4. Authors have not released public code as of April 2026.

from .memno_paper import MemNO as _MemNO, S4D  # noqa: F401  (S4D exported for testing)


class MemNO(nn.Module):
    """Adapter: author-faithful MemNO from memno_paper."""
    def __init__(self, in_channels: int, out_channels: int, length: int,
                 hidden: int = 64, n_markovian: int = 4, memory_position: int = 2,
                 s4_state: int = 64, **_ignored) -> None:
        super().__init__()
        self._net = _MemNO(
            in_channels=in_channels,
            out_channels=out_channels,
            length=length,
            hidden=hidden,
            n_markovian=n_markovian,
            memory_position=memory_position,
            s4_state=s4_state,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)


# ============================================================================
# 3. ANIE (Attentional Neural Integral Equations; Zappala et al., Nat. Mach. Intell. 2024)
# ============================================================================
#
# Author-faithful: uses the vendored IE_source/ tree from
# https://github.com/emazap7/ANIE. Specifically:
#   - IE_source/kernels.py:model_blocks   (attention encoder used as Encoder=)
#   - IE_source/Attentional_IE_solver.py:Integral_attention_solver_multbatch
#
# Construction pattern mirrors NIE_Attention_Burgers_longT.ipynb cell 67:
#     model = model_blocks(dim+1, dim_emb, n_head, n_blocks, n_ff,
#                           attention_type, Final_block=False, dropout=0.1)
# and solver call pattern mirrors the train loop in experiments.py:
#     Integral_attention_solver_multbatch(
#         ts, y_0, Encoder=model, sampling_points=L,
#         max_iterations=args.max_iterations,
#         smoothing_factor=args.smoothing_factor,
#     ).solve()
#
# Requires:  pip install git+https://github.com/patrick-kidger/torchcubicspline.git
#            pip install torchdiffeq
# Vendored third_party/anie/IE_source/  (6 files, 254 KB).


def _anie_thirdparty_dir():
    """Return the repo-root-relative path to the vendored ANIE tree."""
    import pathlib
    return str(pathlib.Path(__file__).resolve().parent.parent.parent / "third_party" / "anie")


def _import_anie():
    """Lazy import of ANIE. Vendored files + external deps must be present.
    Monkey-patches numpy compatibility shims for API removals the authors'
    code uses that were dropped in NumPy 2.x (np.ComplexWarning).
    """
    import sys as _sys
    import numpy as _np
    # NumPy 2 compat: np.ComplexWarning moved / was removed. The ANIE solver
    # uses it in a warnings.filterwarnings() call; give it a harmless alias
    # if missing. (Upstream code unchanged.)
    if not hasattr(_np, "ComplexWarning"):
        try:
            _np.ComplexWarning = _np.exceptions.ComplexWarning   # type: ignore[attr-defined]
        except Exception:
            class _ComplexWarning(Warning):
                pass
            _np.ComplexWarning = _ComplexWarning                  # type: ignore[attr-defined]
    tp = _anie_thirdparty_dir()
    if tp not in _sys.path:
        _sys.path.insert(0, tp)
    from IE_source.kernels import model_blocks as _mb
    from IE_source.Attentional_IE_solver import Integral_attention_solver_multbatch as _solver
    return _mb, _solver


class ANIE(nn.Module):
    """Author-faithful ANIE: attention encoder (``kernels.model_blocks``) +
    iterative fixed-point solver (``Integral_attention_solver_multbatch``).

    The attention encoder owns the trainable parameters. On each forward
    pass we instantiate a fresh solver object (the solver is a non-Module
    class that takes the encoder as a callable). Training loops `solve()`
    for ``max_iterations`` fixed-point steps per forward pass, which is
    substantially slower than feed-forward baselines: expect ~``max_iterations``x
    the wall-clock of a comparable feed-forward model.

    Paper: Zappala et al., "Learning integral operators via neural integral
    equations," Nat. Mach. Intell. 2024.
    """
    def __init__(self, in_channels: int, out_channels: int, length: int,
                 dim_emb: int = 64, n_head: int = 4, n_blocks: int = 3,
                 n_ff: int = 64, attention_type: str = "galerkin",
                 max_iterations: int = 3, smoothing_factor: float = 0.5,
                 internal_dim: Optional[int] = None, **_ignored) -> None:
        super().__init__()
        model_blocks, solver_cls = _import_anie()
        self._solver_cls = solver_cls

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.length = length
        self.dim = internal_dim or max(in_channels, out_channels, 4)
        self.max_iterations = max_iterations
        self.smoothing_factor = smoothing_factor

        # Input / output projections to / from ANIE's internal dim. Internal
        # feature width is `self.dim`; model_blocks adds +1 for the time
        # channel that the solver concatenates into its forward.
        self.in_proj = nn.Linear(in_channels, self.dim)
        self.out_proj = nn.Linear(self.dim, out_channels)

        # Attention encoder: model_blocks signature is
        #   (dimension, dim_emb, n_head, n_blocks, n_ff, attention_type,
        #    dim_out=2, Final_block=False, dropout=0.1).
        # dimension here is the non-time feature width; the solver concatenates
        # a time channel before calling the encoder, so model_blocks internally
        # handles the +1.
        self.encoder = model_blocks(
            dimension=self.dim,
            dim_emb=dim_emb,
            n_head=n_head,
            n_blocks=n_blocks,
            n_ff=n_ff,
            attention_type=attention_type,
            dim_out=self.dim,
            Final_block=False,
            dropout=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, in_channels)
        B, L, _ = x.shape
        h = self.in_proj(x)                                  # (B, L, dim)
        ts = torch.linspace(0.0, 1.0, L, device=x.device)    # (L,)
        y_0 = h[:, 0:1, :]                                   # (B, 1, dim)

        solver = self._solver_cls(
            x=ts,
            y_0=y_0,
            Encoder=self.encoder,
            sampling_points=L,
            max_iterations=self.max_iterations,
            smoothing_factor=self.smoothing_factor,
            accumulate_grads=True,  # keep encoder grads alive through solve()
            use_support=False,
        )
        y = solver.solve()                                    # (B, L, dim)
        return self.out_proj(y)


# ============================================================================
# 4. LocalizedNO (Liu-Schiaffini et al., ICML 2024)
# ============================================================================
#
# Author-faithful: uses the official `neuralop.models.local_no.LocalNO`
# class from https://github.com/neuraloperator/neuraloperator.
# Install: `pip install neuraloperator torch-harmonics` (or rebuild
# torch-harmonics from source if binary ABI mismatches the installed
# PyTorch, as was needed on our H100 cluster).
#
# LocalNO is **2D-native** — its `default_in_shape` is a 2D (H, W) tuple
# and the DISCO / differential layers operate on spatial grids. This
# adapter accepts 2D data in `(batch, H, W, in_channels)` or
# `(batch, in_channels, H, W)` form. Passing 1D data raises a clear
# runtime error pointing the user to the 2D benchmark.


def _import_localno():
    """Lazy import of LocalNO; path is
    `neuralop.models.local_no.LocalNO`, not the top-level models export,
    because LocalNO is gated behind the same optional-deps try/except as
    SFNO in `neuralop.models.__init__`.
    """
    from neuralop.models.local_no import LocalNO
    return LocalNO


# Paper hyperparameters (Table 4 + Appendix C of Liu-Schiaffini et al. 2024).
# Keys are (task, variant); task in {"darcy","navier_stokes","diffusion_reaction"},
# variant in {"fno","diff","integral","both"}. Each value is a
# (task-specific-extras, variant-specific-config) pair that we merge.
_LOCALNO_CONFIGS = {
    "darcy": {
        "extra": dict(
            n_layers=4, conv_padding_mode="reflect", fin_diff_kernel_size=3,
            radius_cutoff=0.007, disco_kernel_shape=[2, 4],
            domain_length=[2, 2], positional_embedding="grid",
        ),
        "variants": {
            "fno":      dict(n_modes=(20, 20), hidden_channels=41, diff_layers=False, disco_layers=False),
            "diff":     dict(n_modes=(12, 12), hidden_channels=65, diff_layers=True,  disco_layers=False),
            "integral": dict(n_modes=(20, 20), hidden_channels=40, diff_layers=False, disco_layers=True),
            "both":     dict(n_modes=(12, 12), hidden_channels=64, diff_layers=True,  disco_layers=True),
        },
    },
    "navier_stokes": {
        "extra": dict(
            n_layers=4, conv_padding_mode="periodic", fin_diff_kernel_size=3,
            radius_cutoff=0.05 * 3.141592653589793, disco_kernel_shape=[2, 4],
            domain_length=[2 * 3.141592653589793, 2 * 3.141592653589793],
            positional_embedding="grid",
        ),
        "variants": {
            "fno":      dict(n_modes=(40, 40), hidden_channels=65,  diff_layers=False, disco_layers=False),
            "diff":     dict(n_modes=(40, 40), hidden_channels=65,  diff_layers=True,  disco_layers=False),
            "integral": dict(n_modes=(20, 20), hidden_channels=129, diff_layers=False, disco_layers=True),
            "both":     dict(n_modes=(20, 20), hidden_channels=127, diff_layers=True,  disco_layers=True),
        },
    },
    "diffusion_reaction": {
        "extra": dict(
            n_layers=4, conv_padding_mode="reflect", fin_diff_kernel_size=3,
            radius_cutoff=None, disco_kernel_shape=[2, 4],
            domain_length=[2, 2], positional_embedding="grid",
        ),
        "variants": {
            "fno":      dict(n_modes=(16, 16), hidden_channels=29, diff_layers=False, disco_layers=False),
            "diff":     dict(n_modes=(16, 16), hidden_channels=29, diff_layers=True,  disco_layers=False),
            "integral": dict(n_modes=(16, 16), hidden_channels=29, diff_layers=False, disco_layers=True),
            "both":     dict(n_modes=(16, 16), hidden_channels=29, diff_layers=True,  disco_layers=True),
        },
    },
}


class LocalizedNO(nn.Module):
    """Author-faithful LocalNO from Liu-Schiaffini et al. ICML 2024.

    Backed by `neuralop.models.local_no.LocalNO`. 2D-native:
    `default_in_shape` is `(H, W)` and inputs must be 2D spatial grids.
    Paper-level hyperparameter presets are available via
    ``task`` ∈ {"darcy", "navier_stokes", "diffusion_reaction"} ×
    ``variant`` ∈ {"fno", "diff", "integral", "both"}.

    This class accepts either `(batch, H, W, in_channels)` (our 2D
    benchmark contract) or `(batch, in_channels, H, W)` (neuralop's
    native contract); the adapter auto-detects by channel-axis position
    and permutes if needed.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        length: int,
        default_in_shape=(32, 32),
        task: str = "darcy",
        variant: str = "diff",
        override: Optional[dict] = None,
        **_ignored,
    ) -> None:
        super().__init__()
        if isinstance(default_in_shape, int):
            default_in_shape = (default_in_shape, default_in_shape)
        if len(default_in_shape) != 2:
            raise ValueError(
                f"LocalNO requires a 2D default_in_shape (H, W); got {default_in_shape}. "
                "For 1D benchmarks, LocalNO is not applicable — the paper's architecture "
                "is defined over 2D spatial grids. Use TCN or a 1D FNO variant instead."
            )
        self.default_in_shape = tuple(default_in_shape)
        self.in_channels = in_channels
        self.out_channels = out_channels

        LocalNO = _import_localno()
        cfg_bundle = _LOCALNO_CONFIGS.get(task.lower())
        if cfg_bundle is None:
            raise ValueError(
                f"Unknown LocalNO task preset: {task!r}. "
                f"Expected one of {list(_LOCALNO_CONFIGS)}."
            )
        variant_cfg = cfg_bundle["variants"].get(variant.lower())
        if variant_cfg is None:
            raise ValueError(
                f"Unknown LocalNO variant: {variant!r}. "
                f"Expected one of {list(cfg_bundle['variants'])} for task {task!r}."
            )

        kwargs = {**cfg_bundle["extra"], **variant_cfg}
        if override:
            kwargs.update(override)

        self._net = LocalNO(
            in_channels=in_channels,
            out_channels=out_channels,
            default_in_shape=self.default_in_shape,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            raise RuntimeError(
                "LocalNO received a 3D input (batch, length, channels). It is a 2D "
                "operator and does not support 1D sequence data; use a 1D baseline "
                "(TCN, FNO1d, LEMO) instead. 2D benchmark data should be shaped "
                "(batch, H, W, channels) or (batch, channels, H, W)."
            )
        if x.dim() != 4:
            raise RuntimeError(f"Expected 4D input, got shape {tuple(x.shape)}")

        # Auto-detect channel axis.
        B = x.shape[0]
        if x.shape[-1] == self.in_channels:
            x = x.permute(0, 3, 1, 2).contiguous()      # (B, C, H, W)
        elif x.shape[1] == self.in_channels:
            pass                                         # already (B, C, H, W)
        else:
            raise RuntimeError(
                f"LocalNO: cannot locate the in_channels={self.in_channels} axis in "
                f"input of shape {tuple(x.shape)}"
            )

        y = self._net(x)                                 # (B, out_C, H, W)
        # Return in (B, H, W, out_C) to match our 2D harness contract.
        return y.permute(0, 2, 3, 1).contiguous()


# ============================================================================
# 5. LNO (Laplace Neural Operator; Cao et al., Nat. Mach. Intell. 2024)
# ============================================================================
#
# Author-faithful implementation vendored from
# https://github.com/qianyingcao/Laplace-Neural-Operator (1D_Duffing_c0/main.py).
# See lno_paper.py for the pole-residue Laplace layer and LNO1d class,
# which is the upstream file's code with two surgical changes:
#   (1) the `t` time grid is passed explicitly into PR.forward instead of
#       pulled from a global `grid_x_train`, and
#   (2) in_channels / out_channels are parameters of LNO1d (upstream
#       hardcoded to 1).
# The Laplace-layer math (pole-residue parameterization, FFT/IFFT flow,
# transient + steady-state decomposition) is byte-identical.

from .lno_paper import LNO1d as _LNO1d, PR  # noqa: F401 (PR exported for testing)


class LNO(nn.Module):
    """Adapter: author-faithful LNO1d from lno_paper. Default
    hyperparameters match the 1D Duffing config of Cao et al.: width=4,
    modes=16, sin activation in the projection head.
    """

    def __init__(self, in_channels: int, out_channels: int, length: int,
                 width: int = 4, modes: int = 16, **_ignored) -> None:
        super().__init__()
        self._net = _LNO1d(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            modes=modes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)


# ============================================================================
# 6. MFNO (Mirror-padded FNO; Lee, Kim, Park, arXiv 2507.17887, 2025)
# ============================================================================
#
# Paper-faithful implementation lives in mfno_paper.py (parameter-count
# matched to 664,961 for the scalar-SDE configuration: 32 latent channels,
# 5 Fourier layers, modes=64, GELU, mirror padding). This module only
# provides thin dde-fno-harness adapters.
#
# The paper's input is (xi_const_path, brownian_path) with in_channels=2;
# our DDE data has more channels (signal, mask, t_norm, parameters). We
# accept arbitrary in_channels and pass through to the FNO1dCore, matching
# the paper's ARCHITECTURE exactly while adapting the INPUT INTERFACE to
# the benchmark's data format. This is the standard way to port a
# reference architecture to a different data regime.

from .mfno_paper import (
    MFNO1d as _MFNO1d,
    ZFNO1d as _ZFNO1d,
    VanillaFNO1d as _VanillaFNO1d,
)


class MFNO(nn.Module):
    """Mirror-padded FNO, author-faithful to Lee-Kim-Park 2025.

    Default hyperparameters match the paper's Table 3 for scalar SDE:
    latent_channels=32, modes=64, num_layers=5. These give ~664k parameters
    on in_channels=2; our DDE data has more input channels so the exact
    parameter count is slightly different.
    """
    def __init__(self, in_channels: int, out_channels: int, length: int,
                 latent_channels: int = 32, modes: int = 64, num_layers: int = 5,
                 projection_channels: int = 128, activation: str = "gelu",
                 **_ignored) -> None:
        super().__init__()
        self._length = length
        self._net = _MFNO1d(
            input_channels=in_channels,
            output_channels=out_channels,
            latent_channels=latent_channels,
            modes=modes,
            num_layers=num_layers,
            projection_channels=projection_channels,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)


class ZFNO(nn.Module):
    """Zero-padded FNO (paper's ablation variant using zero rather than mirror padding).
    Same FNO core and hyperparameters as MFNO; differs only in the padding scheme."""
    def __init__(self, in_channels: int, out_channels: int, length: int,
                 latent_channels: int = 32, modes: int = 64, num_layers: int = 5,
                 projection_channels: int = 128, activation: str = "gelu",
                 **_ignored) -> None:
        super().__init__()
        self._net = _ZFNO1d(
            input_channels=in_channels, output_channels=out_channels,
            latent_channels=latent_channels, modes=modes, num_layers=num_layers,
            projection_channels=projection_channels, activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)


class VanillaFNO(nn.Module):
    """Vanilla (no-padding) FNO using the same core; paper's ablation baseline.
    Included so the benchmark can exercise the exact MFNO-vs-ZFNO-vs-FNO comparison
    the paper uses for mirror-padding ablations.
    """
    def __init__(self, in_channels: int, out_channels: int, length: int,
                 latent_channels: int = 32, modes: int = 64, num_layers: int = 5,
                 projection_channels: int = 128, activation: str = "gelu",
                 **_ignored) -> None:
        super().__init__()
        self._net = _VanillaFNO1d(
            input_channels=in_channels, output_channels=out_channels,
            latent_channels=latent_channels, modes=modes, num_layers=num_layers,
            projection_channels=projection_channels, activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)


# ============================================================================
# Factory
# ============================================================================

def create_research_baseline(name: str, in_channels: int, out_channels: int,
                               length: int, config: dict) -> nn.Module:
    """Factory. `name` is one of: deeponet, memno, nie, localno, lno, mfno,
    zfno, vanillafno.
    """
    mcfg = config.get("model", {})
    kwargs = {k: v for k, v in mcfg.items() if k != "sigma"}  # strip LEMO-only key
    if name == "deeponet":
        return DeepONet(in_channels, out_channels, length, **kwargs)
    elif name == "memno":
        return MemNO(in_channels, out_channels, length, **kwargs)
    elif name == "anie" or name == "nie":
        # `nie` kept as backward-compat alias for `anie`
        return ANIE(in_channels, out_channels, length, **kwargs)
    elif name == "localno":
        return LocalizedNO(in_channels, out_channels, length, **kwargs)
    elif name == "lno":
        return LNO(in_channels, out_channels, length, **kwargs)
    elif name == "mfno":
        return MFNO(in_channels, out_channels, length, **kwargs)
    elif name == "zfno":
        return ZFNO(in_channels, out_channels, length, **kwargs)
    elif name == "vanillafno":
        return VanillaFNO(in_channels, out_channels, length, **kwargs)
    else:
        raise ValueError(f"unknown research baseline: {name!r}")
