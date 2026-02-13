#=
DDE Sample Generator

Main entry point for generating individual DDE samples.
Combines families, history sampling, and robust solving.
=#

module DDEGenerator

using Random
using DifferentialEquations
using DelayDiffEq

include("families.jl")
include("history.jl")
include("solver.jl")

using .DDEFamilies
using .HistorySampling
using .DDESolver

export gen_sample, GeneratedSample

"""
A generated DDE sample with all metadata.
"""
struct GeneratedSample
    t_hist::Vector{Float64}
    t_out::Vector{Float64}
    phi::Matrix{Float64}      # (N_hist, d_hist)
    y::Matrix{Float64}        # (N_out, d_state)
    params::Vector{Float64}
    param_names::Vector{Symbol}
    lags::Vector{Float64}
    attempt::Int              # which solver succeeded
    success::Bool
    fail_reason::String
end

"""
Sample parameters for a family.
"""
function sample_params(rng::AbstractRNG, spec::FamilySpec)
    params = Dict{Symbol, Float64}()
    for name in spec.param_names
        lo, hi = spec.param_ranges[name]
        if lo == hi
            params[name] = lo  # fixed parameter
        else
            params[name] = rand(rng) * (hi - lo) + lo
        end
    end
    return params
end

"""
Get delay values from parameters.
"""
function get_lags(params::Dict{Symbol, Float64}, spec::FamilySpec)
    return Float64[params[τ] for τ in spec.lag_params]
end

"""
Generate one sample for a given family.

Arguments:
- family: Symbol (:linear2, :hutch, :mackey_glass, :vdp, :predator_prey, :dist_uniform, :dist_exp)
- rng: random number generator
- τmax: maximum delay (defines history window)
- T: solution horizon
- dt_out: output time step
- N_hist: number of history grid points
- reltol, abstol: solver tolerances

Returns: GeneratedSample
"""
function gen_sample(
    family::Symbol,
    rng::AbstractRNG;
    τmax::Float64 = 2.0,
    T::Float64 = 20.0,
    dt_out::Float64 = 0.05,
    N_hist::Int = 256,
    reltol::Float64 = 1e-6,
    abstol::Float64 = 1e-8
)
    spec = get_family_spec(family)
    
    # Time grids
    t_hist = collect(range(-τmax, 0.0; length = N_hist))
    t_out = collect(0.0:dt_out:T)
    
    # Sample parameters
    params = sample_params(rng, spec)
    lags = get_lags(params, spec)
    
    # Convert params to NamedTuple for RHS
    param_nt = NamedTuple{Tuple(spec.param_names)}(Tuple(params[k] for k in spec.param_names))
    
    # Sample history and setup problem based on family
    phi, u0, hfun = setup_history_and_ic(family, spec, rng, t_hist, params)
    
    # Create DDE problem
    prob = DDEProblem(
        spec.rhs,
        u0,
        hfun,
        (0.0, T),
        param_nt;
        constant_lags = lags
    )
    
    # Solve
    result = robust_solve(prob; lags = lags, saveat = t_out, reltol = reltol, abstol = abstol)
    
    if !result.success
        return GeneratedSample(
            t_hist, t_out, phi, zeros(length(t_out), spec.state_dim),
            collect(values(params)), collect(keys(params)), lags,
            0, false, result.message
        )
    end
    
    # Extract solution
    y = Array(result.sol)'  # (N_out, d_state)
    
    # Convert params to vector (in order)
    param_vec = Float64[params[k] for k in spec.param_names]
    
    return GeneratedSample(
        t_hist, t_out, phi, y,
        param_vec, spec.param_names, lags,
        result.attempt, true, ""
    )
end

"""
Setup history function and initial condition based on family type.
"""
function setup_history_and_ic(
    family::Symbol,
    spec::FamilySpec,
    rng::AbstractRNG,
    t_hist::Vector{Float64},
    params::Dict{Symbol, Float64}
)
    if family == :linear2
        # Scalar, no positivity
        phi_vals, _ = sample_fourier_history(rng, t_hist; K = 10, amp = 1.0)
        phi = reshape(phi_vals, :, 1)  # (N_hist, 1)
        hist = build_history_interpolant(t_hist, phi')
        hfun(p, t) = t <= 0 ? hist(t) : hist(0.0)
        u0 = [phi_vals[end]]
        return phi, u0, hfun
        
    elseif family == :hutch || family == :mackey_glass
        # Scalar, positive
        phi_vals, _ = sample_positive_history(rng, t_hist; K = 10, amp = 0.6)
        phi = reshape(phi_vals, :, 1)
        hist = build_history_interpolant(t_hist, phi')
        hfun(p, t) = t <= 0 ? hist(t) : hist(0.0)
        u0 = [phi_vals[end]]
        return phi, u0, hfun
        
    elseif family == :vdp
        # 2D: (x, v) with v = dx/dt
        x, v, _ = sample_vdp_history(rng, t_hist; K = 10, amp = 1.0)
        phi = hcat(x, v)  # (N_hist, 2)
        hist = build_history_interpolant(t_hist, phi')
        hfun(p, t) = t <= 0 ? hist(t) : hist(0.0)
        u0 = [x[end], v[end]]
        return phi, u0, hfun
        
    elseif family == :predator_prey
        # 2D positive
        x, y_hist, _, _ = sample_2d_positive_history(rng, t_hist; K = 10, amp = 0.6)
        phi = hcat(x, y_hist)  # (N_hist, 2)
        hist = build_history_interpolant(t_hist, phi')
        hfun(p, t) = t <= 0 ? hist(t) : hist(0.0)
        u0 = [x[end], y_hist[end]]
        return phi, u0, hfun
        
    elseif family == :dist_uniform
        # x positive, m = auxiliary (uniform kernel)
        x_hist, _ = sample_positive_history(rng, t_hist; K = 10, amp = 0.6)
        τ = params[:τ]
        m0 = compute_uniform_aux_init(t_hist, x_hist, τ)
        # History: store x, m is constant = m0 for t <= 0
        m_hist = fill(m0, length(t_hist))
        phi = hcat(x_hist, m_hist)  # (N_hist, 2)
        hist = build_history_interpolant(t_hist, phi')
        hfun(p, t) = t <= 0 ? hist(t) : hist(0.0)
        u0 = [x_hist[end], m0]
        return phi, u0, hfun
        
    elseif family == :dist_exp
        # x positive, z = auxiliary (exponential kernel)
        x_hist, _ = sample_positive_history(rng, t_hist; K = 10, amp = 0.6)
        τ = params[:τ]
        λ = params[:λ]
        z0 = compute_exp_aux_init(t_hist, x_hist, τ, λ)
        z_hist = fill(z0, length(t_hist))
        phi = hcat(x_hist, z_hist)
        hist = build_history_interpolant(t_hist, phi')
        hfun(p, t) = t <= 0 ? hist(t) : hist(0.0)
        u0 = [x_hist[end], z0]
        return phi, u0, hfun
        
    else
        error("Unknown family: $family")
    end
end

end # module
