#=
History Function Sampling

Provides utilities for generating diverse, smooth history functions:
- Random Fourier series (smooth, cheap, reproducible)
- Consistent 2D histories (e.g., x and v=dx/dt for VdP)
- Positive-constrained histories
=#

module HistorySampling

using Random
using DataInterpolations

export sample_fourier_history, sample_positive_history
export sample_vdp_history, sample_2d_positive_history
export HistoryInterp, build_history_interpolant
export trapz

# ============================================================================
# Trapezoidal integration (for computing initial auxiliary variables)
# ============================================================================
function trapz(t::AbstractVector, y::AbstractVector)
    @assert length(t) == length(y)
    return sum((t[i+1] - t[i]) * (y[i+1] + y[i]) / 2 for i in 1:length(t)-1)
end

# ============================================================================
# History interpolator wrapper
# ============================================================================
struct HistoryInterp
    interps::Vector{Any}
end

function (hh::HistoryInterp)(t::Real)
    return [hh.interps[i](t) for i in eachindex(hh.interps)]
end

"""
Build a cubic spline interpolant for history data.

Arguments:
- t_hist: time grid on [-τmax, 0], must be sorted ascending
- phi: history values, shape (d, N_hist) where d is state dimension
"""
function build_history_interpolant(t_hist::Vector{Float64}, phi::Matrix{Float64})
    d = size(phi, 1)
    interps = Any[]
    for i in 1:d
        # CubicSpline expects (values, times)
        push!(interps, CubicSpline(vec(phi[i, :]), t_hist))
    end
    return HistoryInterp(interps)
end

# ============================================================================
# Fourier series history (scalar)
# ============================================================================
"""
Sample a smooth scalar history using random Fourier series.

Returns: (values, coefficients)
- values: array of φ(t) on t_hist
- coefficients: NamedTuple with (c0, a, b, L) for reproducibility
"""
function sample_fourier_history(
    rng::AbstractRNG,
    t_hist::Vector{Float64};
    K::Int = 8,
    amp::Float64 = 1.0
)
    L = abs(first(t_hist))  # ≈ τmax
    
    # Random coefficients (decay with k for smoothness)
    c0 = randn(rng)
    a = randn(rng, K) ./ (1:K)
    b = randn(rng, K) ./ (1:K)
    
    # Evaluate on grid
    vals = similar(t_hist)
    for (j, t) in enumerate(t_hist)
        s = c0
        for k in 1:K
            ω = 2π * k / L
            s += a[k] * cos(ω * t) + b[k] * sin(ω * t)
        end
        vals[j] = amp * s
    end
    
    coeffs = (c0 = c0, a = a, b = b, L = L, K = K, amp = amp)
    return vals, coeffs
end

"""
Sample a positive scalar history using exp transform.

Returns: (values, coefficients)
"""
function sample_positive_history(
    rng::AbstractRNG,
    t_hist::Vector{Float64};
    K::Int = 8,
    amp::Float64 = 0.6,
    eps::Float64 = 1e-3
)
    vals, coeffs = sample_fourier_history(rng, t_hist; K = K, amp = amp)
    # exp transform guarantees positivity
    pos_vals = exp.(vals) .+ eps
    return pos_vals, coeffs
end

# ============================================================================
# Van der Pol history (x and v = dx/dt consistent)
# ============================================================================
"""
Sample 2D history for Van der Pol: (x(t), v(t) = dx/dt).

v(t) is computed analytically from the Fourier series derivative.

Returns: (x, v, coefficients)
"""
function sample_vdp_history(
    rng::AbstractRNG,
    t_hist::Vector{Float64};
    K::Int = 8,
    amp::Float64 = 1.0
)
    L = abs(first(t_hist))
    
    c0 = randn(rng)
    a = randn(rng, K) ./ (1:K)
    b = randn(rng, K) ./ (1:K)
    
    x = similar(t_hist)
    v = similar(t_hist)
    
    for (j, t) in enumerate(t_hist)
        sx = c0
        sv = 0.0
        for k in 1:K
            ω = 2π * k / L
            sx += a[k] * cos(ω * t) + b[k] * sin(ω * t)
            sv += -a[k] * ω * sin(ω * t) + b[k] * ω * cos(ω * t)
        end
        x[j] = amp * sx
        v[j] = amp * sv
    end
    
    coeffs = (c0 = c0, a = a, b = b, L = L, K = K, amp = amp)
    return x, v, coeffs
end

"""
Sample 2D positive history (e.g., for predator-prey).

Returns: (x, y, coefficients_x, coefficients_y)
"""
function sample_2d_positive_history(
    rng::AbstractRNG,
    t_hist::Vector{Float64};
    K::Int = 8,
    amp::Float64 = 0.6,
    eps::Float64 = 1e-3
)
    x, coeffs_x = sample_positive_history(rng, t_hist; K = K, amp = amp, eps = eps)
    y, coeffs_y = sample_positive_history(rng, t_hist; K = K, amp = amp, eps = eps)
    return x, y, coeffs_x, coeffs_y
end

# ============================================================================
# Compute initial auxiliary variables for distributed delays
# ============================================================================
"""
Compute m(0) for uniform kernel: m(0) = (1/τ) ∫_{-τ}^0 x(s) ds
"""
function compute_uniform_aux_init(
    t_hist::Vector{Float64},
    x_hist::Vector{Float64},
    τ::Float64
)
    # Find indices where t >= -τ
    idx = findall(t -> t >= -τ, t_hist)
    if isempty(idx)
        # τ > τmax, use full history
        idx = 1:length(t_hist)
    end
    return trapz(t_hist[idx], x_hist[idx]) / τ
end

"""
Compute z(0) for exponential kernel: z(0) = (1/C) ∫_{-τ}^0 exp(λs) x(s) ds
where C = (1 - exp(-λτ))/λ
"""
function compute_exp_aux_init(
    t_hist::Vector{Float64},
    x_hist::Vector{Float64},
    τ::Float64,
    λ::Float64
)
    C = (1.0 - exp(-λ * τ)) / λ
    idx = findall(t -> t >= -τ, t_hist)
    if isempty(idx)
        idx = 1:length(t_hist)
    end
    # ∫ exp(λs) x(s) ds
    integrand = exp.(λ .* t_hist[idx]) .* x_hist[idx]
    return trapz(t_hist[idx], integrand) / C
end

end # module
