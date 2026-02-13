"""Training module for FNO models."""

# Lazy imports to avoid hard dependency on tensorboard
def __getattr__(name):
    if name in ("Trainer", "masked_mse_loss", "relative_l2_error", "evaluate_model"):
        from .train_fno import Trainer, masked_mse_loss, relative_l2_error, evaluate_model
        _exports = {
            "Trainer": Trainer,
            "masked_mse_loss": masked_mse_loss,
            "relative_l2_error": relative_l2_error,
            "evaluate_model": evaluate_model,
        }
        return _exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Trainer", "masked_mse_loss", "relative_l2_error", "evaluate_model"]
