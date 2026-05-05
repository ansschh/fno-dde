# Baseline v1 Results

## Summary Table

| Family | Split | N | Median | P95 | Mean±Std |
|--------|-------|---|--------|-----|----------|
| hutch | id | 2000 | 0.1306 | 0.5884 | 0.1998±0.1895 |
| hutch | ood_delay | 2000 | 0.1945 | 0.7268 | 0.2713±0.2471 |
| hutch | ood_delay_hole | 2000 | 0.2120 | 0.5864 | 0.2524±0.1747 |
| hutch | ood_history | 2000 | 0.1958 | 1.1234 | 0.3517±0.3818 |
| linear2 | id | 2000 | 0.5735 | 1.3912 | 0.6672±0.5163 |
| linear2 | ood_delay | 2000 | 0.4955 | 1.0211 | 0.5475±0.5693 |
| linear2 | ood_delay_hole | 2000 | 0.5667 | 1.4563 | 0.6760±0.6386 |
| linear2 | ood_history | 2000 | 0.8213 | 1.9223 | 1.0167±1.3590 |

## OOD Gaps

### hutch

- **ood_delay**: 1.49x
- **ood_delay_hole**: 1.62x
- **ood_history**: 1.50x

### linear2

- **ood_delay**: 0.86x
- **ood_delay_hole**: 0.99x
- **ood_history**: 1.43x
