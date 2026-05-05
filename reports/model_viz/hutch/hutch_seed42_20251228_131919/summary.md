# Model Report: hutch

**Generated:** 2025-12-29

## Configuration

- **Model:** FNO1dResidual
- **Modes:** 12
- **Width:** 48
- **Layers:** 3
- **Epochs:** 150
- **Best Epoch:** 72

## Performance Summary

| Split | N | Median | P90 | P95 |
|-------|---|--------|-----|-----|
| id | 2000 | 0.8343 | 1.0570 | 1.5343 |
| ood_delay | 2000 | 1.0010 | 1.5426 | 2.3259 |
| ood_history | 2000 | 0.9411 | 1.4774 | 1.9980 |
| ood_horizon | 2000 | 0.7200 | 1.1276 | 1.3149 |

## OOD Gaps

| Split | Gap (vs ID) |
|-------|-------------|
| ood_delay | 1.20x |
| ood_history | 1.13x |
| ood_horizon | 0.86x |

## Plots

- `training_curves.png` - Loss and validation metrics over training
- `split_boxplot.png` - Performance comparison across splits
- `error_vs_time_comparison.png` - Error evolution over time
- `overlays_random_*.png` - Random trajectory samples
- `overlays_worst_*.png` - Worst-case trajectories