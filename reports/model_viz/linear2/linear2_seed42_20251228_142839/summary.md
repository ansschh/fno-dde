# Model Report: linear2

**Generated:** 2025-12-29

## Configuration

- **Model:** FNO1dResidual
- **Modes:** 12
- **Width:** 48
- **Layers:** 3
- **Epochs:** 150
- **Best Epoch:** 79

## Performance Summary

| Split | N | Median | P90 | P95 |
|-------|---|--------|-----|-----|
| id | 2000 | 0.9971 | 1.1455 | 1.5529 |
| ood_delay | 2000 | 1.0004 | 1.1721 | 1.6721 |
| ood_history | 2000 | 1.0002 | 1.1954 | 1.6740 |
| ood_horizon | 2000 | 1.0140 | 1.1671 | 2.1302 |

## OOD Gaps

| Split | Gap (vs ID) |
|-------|-------------|
| ood_delay | 1.00x |
| ood_history | 1.00x |
| ood_horizon | 1.02x |

## Plots

- `training_curves.png` - Loss and validation metrics over training
- `split_boxplot.png` - Performance comparison across splits
- `error_vs_time_comparison.png` - Error evolution over time
- `overlays_random_*.png` - Random trajectory samples
- `overlays_worst_*.png` - Worst-case trajectories