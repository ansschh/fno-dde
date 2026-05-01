"""One-shot patcher: extend gen_dde_pde_ood.py to support dist_kernel families."""
from pathlib import Path

p = Path("scripts/gen_dde_pde_ood.py")
s = p.read_text()

# 1) Extend imports
s = s.replace(
    "    KuramotoParams, simulate_kuramoto,\n    write_shard,",
    "    KuramotoParams, simulate_kuramoto,\n"
    "    DistKernelRDParams, simulate_dist_kernel_rd,\n"
    "    sample_dist_kernel_rd_param_set,\n"
    "    write_shard,",
)

# 2) Add OOD_TAUS entries
s = s.replace(
    '    "kuramoto_2d":      [0.1, 1.5],\n}',
    '    "kuramoto_2d":      [0.1, 1.5],\n'
    '    "dist_exp_rd_2d":      [0.1, 1.5],\n'
    '    "dist_gaussian_rd_2d": [0.1, 1.5],\n'
    '    "dist_gamma_rd_2d":    [0.1, 1.5],\n'
    '    "dist_uniform_rd_2d":  [0.1, 1.5],\n'
    '    "dist_powerlaw_rd_2d": [0.1, 1.5],\n'
    '}',
)

# 3) Add dist_kernel make_ood_params branch
new_branch = (
    '    if family.startswith("dist_") and family.endswith("_rd_2d") and family != "dist_delay_rd_2d":\n'
    '        kernel_type = family.replace("dist_", "").replace("_rd_2d", "")\n'
    '        p_train = sample_dist_kernel_rd_param_set(rng, kernel_type)\n'
    '        new_tau_max = max(4.0 * tau, 0.4)\n'
    '        new_tau_max = round(new_tau_max / dt) * dt\n'
    '        return DistKernelRDParams(\n'
    '            kernel_type=kernel_type, A=p_train.A, tau=tau, tau_max=new_tau_max,\n'
    '            kernel_extra=p_train.kernel_extra, D=p_train.D,\n'
    '            T_total=T_total, dt=dt, n_grid=n_grid, L=L)\n'
)
s = s.replace("    raise ValueError(family)", new_branch + "    raise ValueError(family)")

# 4) Extend simulate dispatcher
s = s.replace(
    '    if family == "kuramoto_2d":       return simulate_kuramoto(p, rng, grid)\n',
    '    if family == "kuramoto_2d":       return simulate_kuramoto(p, rng, grid)\n'
    '    if family.startswith("dist_") and family.endswith("_rd_2d") and family != "dist_delay_rd_2d":\n'
    '        return simulate_dist_kernel_rd(p, rng, grid)\n',
)

p.write_text(s)
print("patched gen_dde_pde_ood.py")
