# Table 4: Baseline Model & Training Settings

| Setting | Value |
|---------|-------|
| **Architecture** | FNO1d-Residual |
| Fourier modes | 12 |
| Width (hidden dim) | 48 |
| Layers | 3 |
| Activation | GELU |
| Dropout | 0.1 |
| **Training** | |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| LR scheduler | ReduceLROnPlateau (patience=10, factor=0.5) |
| Weight decay | 1e-4 |
| Batch size | 32 |
| Max epochs | 150 |
| Early stopping | patience=20 |
| **Normalization** | |
| Input/output | Per-channel mean/std from training set |
| Applied to | All splits (train stats only) |

**Total parameters:** ~93k

**Location:** Main paper or Appendix