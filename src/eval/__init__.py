"""Unified evaluation module for DDE-FNO."""
from .unified_eval import (
    EvalMetrics,
    compute_relative_l2_original_space,
    evaluate_model_on_loader,
    evaluate_model_on_dataset,
    compute_error_vs_time,
    save_metrics_json,
    print_metrics_table,
)

__all__ = [
    "EvalMetrics",
    "compute_relative_l2_original_space",
    "evaluate_model_on_loader",
    "evaluate_model_on_dataset",
    "compute_error_vs_time",
    "save_metrics_json",
    "print_metrics_table",
]
