# Model Report: dist_exp

**Generated:** 2025-12-29

## Configuration

- **Model:** FNO1dResidual
- **Modes:** 12
- **Width:** 48
- **Layers:** 3
- **Epochs:** 150
- **Best Epoch:** 121

## Performance Summary

| Split | N | Median | P90 | P95 |
|-------|---|--------|-----|-----|
| id | 2000 | 0.4490 | 0.7365 | 0.8043 |
| ood_delay | 2000 | 0.6343 | 0.9622 | 1.0481 |
| ood_history | 2000 | 0.5150 | 1.0061 | 1.1091 |
| ood_horizon | 2000 | 0.3772 | 0.6984 | 0.7752 |

## OOD Gaps

| Split | Gap (vs ID) |
|-------|-------------|
| ood_delay | 1.41x |
| ood_history | 1.15x |
| ood_horizon | 0.84x |

## Plots

- `training_curves.png` - Loss and validation metrics over training
- `split_boxplot.png` - Performance comparison across splits
- `error_vs_time_comparison.png` - Error evolution over time
- `overlays_random_*.png` - Random trajectory samples
- `overlays_worst_*.png` - Worst-case trajectories