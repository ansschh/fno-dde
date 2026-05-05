"""Patch eval_equivariance_dense.py to use full-roll (proper T1 test)
instead of state-only-shift."""
from pathlib import Path

p = Path(__file__).resolve().parent / "eval_equivariance_dense.py"
src = p.read_text()

old = '''def cyclic_shift_state_only(x: torch.Tensor, k: int, n_state_channels: int) -> torch.Tensor:
    if k % x.shape[1] == 0:
        return x.clone()
    state = x[..., :n_state_channels]
    rest = x[..., n_state_channels:]
    state_shifted = torch.roll(state, shifts=k, dims=1)
    return torch.cat([state_shifted, rest], dim=-1)'''

new = '''def cyclic_shift_full(x: torch.Tensor, k: int, n_state_channels: int) -> torch.Tensor:
    """Roll ALL input channels along lag axis (axis 1).

    This is the proper T1 cyclic-shift test: every channel (state, mask,
    time, params) moves together along the lag axis. The earlier
    `cyclic_shift_state_only` is a stronger property that the LEMO-PC
    architecture is NOT designed for and is misleading vs the T1 theorem.
    """
    if k % x.shape[1] == 0:
        return x.clone()
    return torch.roll(x, shifts=k, dims=1)


def cyclic_shift_state_only(x: torch.Tensor, k: int, n_state_channels: int) -> torch.Tensor:
    if k % x.shape[1] == 0:
        return x.clone()
    state = x[..., :n_state_channels]
    rest = x[..., n_state_channels:]
    state_shifted = torch.roll(state, shifts=k, dims=1)
    return torch.cat([state_shifted, rest], dim=-1)'''

if old in src:
    src = src.replace(old, new)
    src = src.replace(
        "sk_x = cyclic_shift_state_only(x, int(k), n_state_channels)",
        "sk_x = cyclic_shift_full(x, int(k), n_state_channels)",
    )
    p.write_text(src)
    print("PATCHED")
else:
    print("OLD STRING NOT FOUND - aborting")
