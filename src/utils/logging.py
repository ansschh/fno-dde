"""
Logging Utilities

Provides consistent logging setup for training and evaluation.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = 'dde_fno',
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    Setup a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to save log file (optional)
        level: Logging level
        console: Whether to log to console
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'{name}_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TrainingLogger:
    """
    Helper class for logging training progress.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.epoch_metrics = {}
    
    def log_epoch(self, epoch: int, metrics: dict):
        """Log metrics for an epoch."""
        metrics_str = ' | '.join([f'{k}={v:.6f}' for k, v in metrics.items()])
        self.logger.info(f'Epoch {epoch:4d} | {metrics_str}')
        
        # Store for later analysis
        for k, v in metrics.items():
            if k not in self.epoch_metrics:
                self.epoch_metrics[k] = []
            self.epoch_metrics[k].append(v)
    
    def log_info(self, message: str):
        """Log an info message."""
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """Log an error message."""
        self.logger.error(message)
    
    def get_best_epoch(self, metric: str, mode: str = 'min') -> int:
        """Get the epoch with best value for a metric."""
        if metric not in self.epoch_metrics:
            return -1

        values = self.epoch_metrics[metric]
        if mode == 'min':
            return int(np.argmin(values)) + 1
        else:
            return int(np.argmax(values)) + 1


class ExperimentLogger:
    """
    Structured experiment logger with JSON-lines output and auto training curve plots.

    Writes per-epoch metrics to a JSONL file and generates training curve plots
    on flush/close. Designed for both single-GPU and DDP training.
    """

    def __init__(self, output_dir: str | Path, experiment_name: str = "train",
                 rank: int = 0):
        self.output_dir = Path(output_dir)
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.rank = rank
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.jsonl"
        self.history: dict[str, list[float]] = {}
        self._epoch_count = 0

        # Only rank 0 writes
        if rank == 0:
            self._fh = open(self.metrics_file, "a")
        else:
            self._fh = None

    def log_epoch(self, epoch: int, metrics: dict[str, float]):
        """Log metrics for a single epoch."""
        record = {"epoch": epoch, **metrics}
        if self._fh is not None:
            import json as _json
            self._fh.write(_json.dumps(record) + "\n")
            self._fh.flush()

        for k, v in metrics.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)
        self._epoch_count = epoch

    def log_eval(self, split_name: str, metrics: dict):
        """Log evaluation results for a split."""
        record = {"event": "eval", "split": split_name, **metrics}
        if self._fh is not None:
            import json as _json
            self._fh.write(_json.dumps(record, default=str) + "\n")
            self._fh.flush()

    def log_config(self, config: dict):
        """Log experiment configuration."""
        record = {"event": "config", **config}
        if self._fh is not None:
            import json as _json
            self._fh.write(_json.dumps(record, default=str) + "\n")
            self._fh.flush()

    def plot_training_curves(self):
        """Generate training curve plots from accumulated history."""
        if self.rank != 0 or not self.history:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        plot_dir = self.log_dir / "plots"
        plot_dir.mkdir(exist_ok=True)

        epochs = list(range(1, self._epoch_count + 1))

        # Loss curves
        fig, ax = plt.subplots(figsize=(8, 5))
        if "train_loss" in self.history:
            ax.plot(epochs[:len(self.history["train_loss"])],
                    self.history["train_loss"], label="Train Loss", linewidth=1.5)
        if "val_loss" in self.history:
            ax.plot(epochs[:len(self.history["val_loss"])],
                    self.history["val_loss"], label="Val Loss", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title(f"{self.experiment_name} — Loss Curves")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "loss_curves.png", dpi=150)
        plt.close(fig)

        # relL2 curves
        if "val_rel_l2" in self.history:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs[:len(self.history["val_rel_l2"])],
                    self.history["val_rel_l2"], label="Val relL2",
                    linewidth=1.5, color="#31a354")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Relative L2 Error")
            ax.set_title(f"{self.experiment_name} — Validation relL2")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / "val_rel_l2.png", dpi=150)
            plt.close(fig)

        # fRMSE curves (if available)
        frmse_keys = [k for k in self.history if k.startswith("frmse_")]
        if frmse_keys:
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = {"frmse_low": "#2c7fb8", "frmse_mid": "#41b6c4", "frmse_high": "#e34a33"}
            for k in sorted(frmse_keys):
                ax.plot(epochs[:len(self.history[k])], self.history[k],
                        label=k.replace("_", " ").title(), linewidth=1.5,
                        color=colors.get(k, None))
            ax.set_xlabel("Epoch")
            ax.set_ylabel("fRMSE")
            ax.set_title(f"{self.experiment_name} — Frequency-Binned RMSE")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / "frmse_curves.png", dpi=150)
            plt.close(fig)

        # LR schedule
        if "lr" in self.history:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs[:len(self.history["lr"])], self.history["lr"],
                    linewidth=1.5, color="#756bb1")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Learning Rate")
            ax.set_yscale("log")
            ax.set_title(f"{self.experiment_name} — Learning Rate Schedule")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_dir / "lr_schedule.png", dpi=150)
            plt.close(fig)

    def close(self):
        """Flush, generate plots, and close file handle."""
        self.plot_training_curves()
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __del__(self):
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass


def format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f'{seconds:.1f}s'
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f'{int(minutes)}m {int(secs)}s'
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f'{int(hours)}h {int(minutes)}m'


def print_model_summary(model, input_shape: tuple):
    """Print a summary of model architecture."""
    print("\nModel Summary")
    print("=" * 60)
    print(f"{'Layer':<30} {'Output Shape':<20} {'Params':<10}")
    print("-" * 60)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        params = param.numel()
        total_params += params
        if param.requires_grad:
            trainable_params += params
        
        shape_str = str(list(param.shape))
        print(f"{name:<30} {shape_str:<20} {params:<10,}")
    
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print()
