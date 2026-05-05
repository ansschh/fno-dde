# Baseline Comparison: FNO vs Naive (Persistence)

| Family | Naive (Median) | FNO (Median) | Improvement | Naive (P95) | FNO (P95) | P95 Improve |
|--------|---------------:|-------------:|------------:|------------:|----------:|------------:|
| hutch | 1.1636 | 0.1209 | 9.6x | 3.9130 | 0.5520 | 7.1x |
| linear2 | 6.0758 | 0.6118 | 9.9x | 48.6402 | 1.5317 | 31.8x |
| vdp | 1.3519 | 0.2962 | 4.6x | 2.2987 | 0.8974 | 2.6x |
| dist_uniform | 0.7066 | 0.0886 | 8.0x | 1.8447 | 0.4060 | 4.5x |
| dist_exp | 0.5666 | 0.0770 | 7.4x | 1.9687 | 0.6832 | 2.9x |

**Metric:** Relative L2 error in original space, future region only (with loss_mask).

**Naive baseline:** Persistence predictor y(t) = y(0) for all t.

**Interpretation:** FNO improvement factor shows how much better FNO is than simply predicting the initial condition.