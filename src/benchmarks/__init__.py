"""Benchmark utilities for DDE-FNO."""

from .solver_health import SolverHealthLog, compute_solver_stats
from .label_fidelity import LabelFidelityBenchmark
from .residual_check import ResidualBenchmark
from .diversity_metrics import DiversityMetrics
from .model_metrics import ModelMetrics, compute_all_metrics

__all__ = [
    "SolverHealthLog",
    "compute_solver_stats",
    "LabelFidelityBenchmark",
    "ResidualBenchmark",
    "DiversityMetrics",
    "ModelMetrics",
    "compute_all_metrics",
]
