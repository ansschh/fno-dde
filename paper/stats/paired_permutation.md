# Paired-permutation analysis — LEMO_PC vs baselines

**Method:** Paired-sample permutation test (10,000 iterations) on per-cell test relL2, paired by (family, regime, seed). Bootstrap 95% CI on improvement ratio (10,000 resamples).  Holm-Bonferroni correction across 4 main baselines.

## Aggregate results (all 5 fams × 3 regimes × 3 seeds = up to 45 paired cells)

| baseline | n_pairs | LEMO_PC mean | baseline mean | improvement % | 95% CI | p (perm) | Holm p< | Hedges g |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 45 | 0.0224 | 0.0730 | 69.3% | [66.2, 72.2] | 9.999e-05 | 0.0125 | 5.08 |
| MarkovFNO | 45 | 0.0224 | 0.1111 | 79.8% | [77.7, 81.8] | 9.999e-05 | 0.0167 | 5.90 |
| WindFNO | 45 | 0.0224 | 0.1120 | 80.0% | [77.9, 82.0] | 9.999e-05 | 0.0250 | 6.01 |
| UNet | 45 | 0.0224 | 0.0249 | 10.0% | [3.4, 16.0] | 0.0042 | 0.0500 | 0.45 |
| LEMO_ND_ablation | 45 | 0.0224 | 0.4276 | 94.8% | [94.2, 95.3] | 9.999e-05 | N/A | 23.43 |

## Per-regime breakdown

### vs FNO
| regime | n_pairs | LEMO mean | baseline mean | improv. % | 95% CI | p |
|---|---:|---:|---:|---:|---:|---:|
| clean | 15 | 0.0188 | 0.0730 | 74.3% | [69.8, 78.7] | 9.999e-05 |
| lowres | 15 | 0.0294 | 0.0730 | 59.6% | [56.7, 62.5] | 9.999e-05 |
| noisy | 15 | 0.0190 | 0.0730 | 73.9% | [69.6, 78.2] | 9.999e-05 |

### vs MarkovFNO
| regime | n_pairs | LEMO mean | baseline mean | improv. % | 95% CI | p |
|---|---:|---:|---:|---:|---:|---:|
| clean | 15 | 0.0188 | 0.1111 | 83.1% | [79.9, 86.1] | 9.999e-05 |
| lowres | 15 | 0.0294 | 0.1111 | 73.5% | [71.2, 75.6] | 9.999e-05 |
| noisy | 15 | 0.0190 | 0.1111 | 82.9% | [79.8, 85.8] | 9.999e-05 |

### vs WindFNO
| regime | n_pairs | LEMO mean | baseline mean | improv. % | 95% CI | p |
|---|---:|---:|---:|---:|---:|---:|
| clean | 15 | 0.0188 | 0.1119 | 83.2% | [80.1, 86.2] | 9.999e-05 |
| lowres | 15 | 0.0294 | 0.1119 | 73.7% | [71.4, 75.8] | 9.999e-05 |
| noisy | 15 | 0.0190 | 0.1120 | 83.0% | [79.9, 85.9] | 9.999e-05 |

### vs UNet
| regime | n_pairs | LEMO mean | baseline mean | improv. % | 95% CI | p |
|---|---:|---:|---:|---:|---:|---:|
| clean | 15 | 0.0188 | 0.0247 | 24.1% | [19.7, 28.9] | 9.999e-05 |
| lowres | 15 | 0.0294 | 0.0251 | -17.5% | [-25.6, -10.7] | 9.999e-05 |
| noisy | 15 | 0.0190 | 0.0249 | 23.6% | [19.0, 28.2] | 9.999e-05 |

### vs LEMO_ND_ablation
| regime | n_pairs | LEMO mean | baseline mean | improv. % | 95% CI | p |
|---|---:|---:|---:|---:|---:|---:|
| clean | 15 | 0.0188 | 0.4273 | 95.6% | [94.8, 96.4] | 9.999e-05 |
| lowres | 15 | 0.0294 | 0.4280 | 93.1% | [92.5, 93.7] | 9.999e-05 |
| noisy | 15 | 0.0190 | 0.4273 | 95.5% | [94.8, 96.3] | 9.999e-05 |
