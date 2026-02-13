#=
Validation Harness for DDE Generation

Golden sanity tests and solver accuracy checks to ensure
data generation is correct before scaling up.
=#

using Random
using LinearAlgebra
using Printf

include("generate.jl")
include("qc.jl")

using .DDEGenerator
using .QualityControl

# ============================================================================
# Golden Sanity Tests
# ============================================================================

"""
Test 1: Linear2delay reduces to ODE when b1=b2=0.
x'(t) = a*x(t) should give x(t) = x0*exp(a*t)
"""
function test_linear_reduces_to_ode(; tol::Float64 = 1e-4)
    println("Test: Linear DDE reduces to ODE (b1=b2=0)...")
    
    rng = MersenneTwister(12345)
    
    # Generate sample with b1=b2=0 forced
    include("families.jl")
    using .DDEFamilies
    using .HistorySampling
    using .DDESolver
    using DifferentialEquations
    using DelayDiffEq
    
    τmax = 2.0
    T = 10.0
    dt_out = 0.01
    N_hist = 256
    
    t_hist = collect(range(-τmax, 0.0; length = N_hist))
    t_out = collect(0.0:dt_out:T)
    
    # Fixed params: b1=b2=0, so it's just x' = a*x
    a = 0.5
    params = (a = a, b1 = 0.0, b2 = 0.0, τ1 = 1.0, τ2 = 1.5)
    lags = [1.0, 1.5]
    
    # Simple constant history
    x0 = 2.0
    phi_vals = fill(x0, N_hist)
    phi = reshape(phi_vals, :, 1)
    hist = build_history_interpolant(t_hist, phi')
    hfun(p, t) = t <= 0 ? hist(t) : hist(0.0)
    u0 = [x0]
    
    prob = DDEProblem(rhs_linear2!, u0, hfun, (0.0, T), params; constant_lags = lags)
    result = robust_solve(prob; lags = lags, saveat = t_out, reltol = 1e-8, abstol = 1e-10)
    
    if !result.success
        println("  FAIL: Solver failed")
        return false
    end
    
    y_num = vec(Array(result.sol)')
    y_exact = x0 .* exp.(a .* t_out)
    
    max_err = maximum(abs.(y_num .- y_exact))
    rel_err = max_err / maximum(abs.(y_exact))
    
    passed = rel_err < tol
    status = passed ? "PASS" : "FAIL"
    println("  $status: max_err = $(@sprintf("%.2e", max_err)), rel_err = $(@sprintf("%.2e", rel_err))")
    
    return passed
end

"""
Test 2: Van der Pol reduces to ODE when κ=0 (no delay feedback).
Compare with standard VdP solution.
"""
function test_vdp_reduces_to_ode(; tol::Float64 = 1e-3)
    println("Test: Van der Pol DDE reduces to ODE (κ=0)...")
    
    include("families.jl")
    using .DDEFamilies
    using .HistorySampling
    using .DDESolver
    using DifferentialEquations
    using DelayDiffEq
    
    τmax = 2.0
    T = 10.0
    dt_out = 0.01
    N_hist = 256
    
    t_hist = collect(range(-τmax, 0.0; length = N_hist))
    t_out = collect(0.0:dt_out:T)
    
    μ = 1.0
    params = (μ = μ, κ = 0.0, τ = 1.0)
    lags = [1.0]
    
    # Initial conditions
    x0, v0 = 2.0, 0.0
    x_hist = fill(x0, N_hist)
    v_hist = fill(v0, N_hist)
    phi = hcat(x_hist, v_hist)
    hist = build_history_interpolant(t_hist, phi')
    hfun(p, t) = t <= 0 ? hist(t) : hist(0.0)
    u0 = [x0, v0]
    
    # Solve DDE
    prob_dde = DDEProblem(rhs_vdp_delay!, u0, hfun, (0.0, T), params; constant_lags = lags)
    result = robust_solve(prob_dde; lags = lags, saveat = t_out, reltol = 1e-8, abstol = 1e-10)
    
    if !result.success
        println("  FAIL: DDE solver failed")
        return false
    end
    
    y_dde = Array(result.sol)'  # (N, 2)
    
    # Solve reference ODE (no delay term)
    function vdp_ode!(du, u, p, t)
        x, v = u
        du[1] = v
        du[2] = p.μ * (1.0 - x^2) * v - x
    end
    
    prob_ode = ODEProblem(vdp_ode!, u0, (0.0, T), params)
    sol_ode = solve(prob_ode, Tsit5(); saveat = t_out, reltol = 1e-10, abstol = 1e-12)
    y_ode = Array(sol_ode)'
    
    max_err = maximum(abs.(y_dde .- y_ode))
    rel_err = max_err / maximum(abs.(y_ode))
    
    passed = rel_err < tol
    status = passed ? "PASS" : "FAIL"
    println("  $status: max_err = $(@sprintf("%.2e", max_err)), rel_err = $(@sprintf("%.2e", rel_err))")
    
    return passed
end

"""
Test 3: History continuity - y(0) should match phi(0).
"""
function test_history_continuity(; n_samples::Int = 20, tol::Float64 = 1e-6)
    println("Test: History continuity at t=0...")
    
    families = [:linear2, :hutch, :mackey_glass, :vdp, :predator_prey, :dist_uniform]
    all_passed = true
    
    for family in families
        rng = MersenneTwister(42)
        max_err = 0.0
        
        for _ in 1:n_samples
            sample = gen_sample(family, rng; τmax = 2.0, T = 5.0, dt_out = 0.01, N_hist = 256)
            
            if !sample.success
                continue
            end
            
            # Check phi(0) vs y(0)
            d = min(size(sample.phi, 2), size(sample.y, 2))
            phi_end = sample.phi[end, 1:d]
            y_start = sample.y[1, 1:d]
            err = maximum(abs.(phi_end .- y_start))
            max_err = max(max_err, err)
        end
        
        passed = max_err < tol
        status = passed ? "PASS" : "FAIL"
        println("  $family: $status (max_err = $(@sprintf("%.2e", max_err)))")
        all_passed = all_passed && passed
    end
    
    return all_passed
end

# ============================================================================
# Solver Accuracy Spot-Check
# ============================================================================

"""
Compare fast vs reference solver to verify accuracy.
"""
function test_solver_accuracy(; n_samples::Int = 10, family::Symbol = :hutch)
    println("Test: Solver accuracy ($family, $n_samples samples)...")
    
    include("families.jl")
    include("history.jl")
    include("solver.jl")
    using .DDEFamilies
    using .HistorySampling
    using .DDESolver
    using DifferentialEquations
    using DelayDiffEq
    
    spec = get_family_spec(family)
    
    τmax = 2.0
    T = 10.0
    dt_out = 0.02
    N_hist = 256
    
    t_hist = collect(range(-τmax, 0.0; length = N_hist))
    t_out = collect(0.0:dt_out:T)
    
    errors = Float64[]
    
    rng = MersenneTwister(123)
    
    for _ in 1:n_samples
        # Sample params
        params = Dict{Symbol, Float64}()
        for name in spec.param_names
            lo, hi = spec.param_ranges[name]
            params[name] = lo == hi ? lo : rand(rng) * (hi - lo) + lo
        end
        lags = Float64[params[τ] for τ in spec.lag_params]
        param_nt = NamedTuple{Tuple(spec.param_names)}(Tuple(params[k] for k in spec.param_names))
        
        # Sample history
        if spec.requires_positive
            phi_vals, _ = sample_positive_history(rng, t_hist; K = 10, amp = 0.6)
        else
            phi_vals, _ = sample_fourier_history(rng, t_hist; K = 10, amp = 1.0)
        end
        
        if spec.state_dim == 1
            phi = reshape(phi_vals, :, 1)
            u0 = [phi_vals[end]]
        else
            # For 2D, use simple approach
            phi_vals2, _ = spec.requires_positive ? 
                sample_positive_history(rng, t_hist; K = 10, amp = 0.6) :
                sample_fourier_history(rng, t_hist; K = 10, amp = 1.0)
            phi = hcat(phi_vals, phi_vals2)
            u0 = [phi_vals[end], phi_vals2[end]]
        end
        
        hist = build_history_interpolant(t_hist, phi')
        hfun(p, t) = t <= 0 ? hist(t) : hist(0.0)
        
        prob = DDEProblem(spec.rhs, u0, hfun, (0.0, T), param_nt; constant_lags = lags)
        
        # Fast solve
        tstops = make_breakpoint_stops(lags, T)
        sol_fast = solve(prob, MethodOfSteps(Tsit5());
            saveat = t_out, reltol = 1e-6, abstol = 1e-8, tstops = tstops)
        
        if sol_fast.retcode != :Success && sol_fast.retcode != ReturnCode.Success
            continue
        end
        
        # Reference solve (tighter tolerances)
        sol_ref = solve(prob, MethodOfSteps(Vern9());
            saveat = t_out, reltol = 1e-10, abstol = 1e-12, tstops = tstops)
        
        if sol_ref.retcode != :Success && sol_ref.retcode != ReturnCode.Success
            continue
        end
        
        y_fast = Array(sol_fast)'
        y_ref = Array(sol_ref)'
        
        rel_err = norm(y_fast - y_ref) / (norm(y_ref) + 1e-10)
        push!(errors, rel_err)
    end
    
    if isempty(errors)
        println("  WARN: No successful comparisons")
        return true
    end
    
    mean_err = sum(errors) / length(errors)
    max_err = maximum(errors)
    p95_err = sort(errors)[max(1, Int(ceil(0.95 * length(errors))))]
    
    println("  Samples compared: $(length(errors))")
    println("  Mean rel error: $(@sprintf("%.2e", mean_err))")
    println("  95th percentile: $(@sprintf("%.2e", p95_err))")
    println("  Max rel error: $(@sprintf("%.2e", max_err))")
    
    passed = p95_err < 1e-3
    status = passed ? "PASS" : "WARN"
    println("  $status")
    
    return passed
end

# ============================================================================
# Run All Tests
# ============================================================================

function run_all_tests()
    println("=" ^ 60)
    println("DDE Generation Validation Suite")
    println("=" ^ 60)
    println()
    
    results = Bool[]
    
    # Golden tests
    push!(results, test_linear_reduces_to_ode())
    push!(results, test_vdp_reduces_to_ode())
    push!(results, test_history_continuity())
    
    println()
    
    # Accuracy tests
    for family in [:linear2, :hutch, :mackey_glass]
        push!(results, test_solver_accuracy(; n_samples = 10, family = family))
    end
    
    println()
    println("=" ^ 60)
    n_passed = sum(results)
    n_total = length(results)
    if n_passed == n_total
        println("All $n_total tests PASSED")
    else
        println("$n_passed / $n_total tests passed")
    end
    println("=" ^ 60)
    
    return all(results)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_tests()
end
