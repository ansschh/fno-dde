# Model Report: dist_uniform

**Generated:** 2025-12-29

## Configuration

- **Model:** FNO1dResidual
- **Modes:** 12
- **Width:** 48
- **Layers:** 3
- **Epochs:** 150
- **Best Epoch:** 118

## Performance Summary

| Split | N | Median | P90 | P95 |
|-------|---|--------|-----|-----|
| id | 2000 | 0.4295 | 0.6638 | 0.7317 |
| ood_delay | 2000 | 0.6474 | 1.0557 | 1.1832 |
| ood_history | 2000 | 1.0479 | 1.9265 | 2.4517 |
| ood_horizon | 2000 | 0.4632 | 0.8246 | 0.9040 |

## OOD Gaps

| Split | Gap (vs ID) |
|-------|-------------|
| ood_delay | 1.51x |
| ood_history | 2.44x |
| ood_horizon | 1.08x |

## Plots

- `training_curves.png` - Loss and validation metrics over training
- `split_boxplot.png` - Performance comparison across splits
- `error_vs_time_comparison.png` - Error evolution over time
- `overlays_random_*.png` - Random trajectory samples
- `overlays_worst_*.png` - Worst-case trajectories