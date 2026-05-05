# Model Report: vdp

**Generated:** 2025-12-29

## Configuration

- **Model:** FNO1dResidual
- **Modes:** 12
- **Width:** 48
- **Layers:** 3
- **Epochs:** 150
- **Best Epoch:** 133

## Performance Summary

| Split | N | Median | P90 | P95 |
|-------|---|--------|-----|-----|
| id | 2000 | 0.8519 | 1.0726 | 1.1686 |
| ood_delay | 2000 | 1.0388 | 1.3967 | 1.4661 |
| ood_history | 2000 | 1.3358 | 1.8821 | 1.9894 |
| ood_horizon | 2000 | 1.2165 | 1.3719 | 1.4197 |

## OOD Gaps

| Split | Gap (vs ID) |
|-------|-------------|
| ood_delay | 1.22x |
| ood_history | 1.57x |
| ood_horizon | 1.43x |

## Plots

- `training_curves.png` - Loss and validation metrics over training
- `split_boxplot.png` - Performance comparison across splits
- `error_vs_time_comparison.png` - Error evolution over time
- `overlays_random_*.png` - Random trajectory samples
- `overlays_worst_*.png` - Worst-case trajectories