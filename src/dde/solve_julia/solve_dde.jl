# DDE Solver using Julia's DifferentialEquations.jl (SciML)
# 
# This script provides functions to solve DDEs and save results for Python consumption.
# Usage: julia solve_dde.jl <config_file.json>

using DifferentialEquations
using DelimitedFiles
using JSON
using NPZ

"""
Solve a scalar linear DDE: x'(t) = a*x(t) + b*x(t-τ)
"""
function solve_linear_dde(params, history_t, history_y, T; n_points=256)
    a = params["a"]
    b = params["b"]
    τ = params["tau"]
    
    # Create history function via interpolation
    history_interp = linear_interpolation(history_t, history_y)
    h(p, t) = t <= 0 ? history_interp(t) : history_y[end]
    
    # Define DDE
    function f!(du, u, h, p, t)
        du[1] = a * u[1] + b * h(p, t - τ)[1]
    end
    
    # Initial condition
    u0 = [h(nothing, 0.0)]
    
    # Solve
    lags = [τ]
    prob = DDEProblem(f!, u0, h, (0.0, T), nothing; constant_lags=lags)
    sol = solve(prob, MethodOfSteps(Tsit5()), saveat=range(0, T, length=n_points))
    
    return sol.t, reduce(hcat, sol.u)'
end

"""
Solve Hutchinson equation: x'(t) = r*x(t)*(1 - x(t-τ)/K)
"""
function solve_hutchinson_dde(params, history_t, history_y, T; n_points=256)
    r = params["r"]
    K = params["K"]
    τ = params["tau"]
    
    history_interp = linear_interpolation(history_t, history_y)
    h(p, t) = t <= 0 ? history_interp(t) : history_y[end]
    
    function f!(du, u, h, p, t)
        x_delayed = h(p, t - τ)[1]
        du[1] = r * u[1] * (1 - x_delayed / K)
    end
    
    u0 = [h(nothing, 0.0)]
    lags = [τ]
    prob = DDEProblem(f!, u0, h, (0.0, T), nothing; constant_lags=lags)
    sol = solve(prob, MethodOfSteps(Tsit5()), saveat=range(0, T, length=n_points))
    
    return sol.t, reduce(hcat, sol.u)'
end

"""
Solve Mackey-Glass: x'(t) = β*x(t-τ)/(1+x(t-τ)^n) - γ*x(t)
"""
function solve_mackey_glass_dde(params, history_t, history_y, T; n_points=256, n_exp=10)
    β = params["beta"]
    γ = params["gamma"]
    τ = params["tau"]
    
    history_interp = linear_interpolation(history_t, history_y)
    h(p, t) = t <= 0 ? history_interp(t) : history_y[end]
    
    function f!(du, u, h, p, t)
        x_delayed = max(h(p, t - τ)[1], 1e-10)
        du[1] = β * x_delayed / (1 + x_delayed^n_exp) - γ * u[1]
    end
    
    u0 = [h(nothing, 0.0)]
    lags = [τ]
    prob = DDEProblem(f!, u0, h, (0.0, T), nothing; constant_lags=lags)
    sol = solve(prob, MethodOfSteps(Tsit5()), saveat=range(0, T, length=n_points))
    
    return sol.t, reduce(hcat, sol.u)'
end

"""
Simple linear interpolation helper
"""
function linear_interpolation(xs, ys)
    function interp(x)
        if x <= xs[1]
            return ys[1, :]
        elseif x >= xs[end]
            return ys[end, :]
        end
        
        # Find interval
        i = searchsortedlast(xs, x)
        i = max(1, min(i, length(xs) - 1))
        
        # Linear interpolation
        t = (x - xs[i]) / (xs[i+1] - xs[i])
        return (1 - t) * ys[i, :] + t * ys[i+1, :]
    end
    return interp
end

# Main entry point for batch processing
function main()
    if length(ARGS) < 1
        println("Usage: julia solve_dde.jl <config_file.json>")
        return
    end
    
    config = JSON.parsefile(ARGS[1])
    
    family = config["family"]
    params = config["params"]
    history_t = Float64.(config["history_t"])
    history_y = Float64.(hcat(config["history_y"]...)')
    T = config["T"]
    n_points = get(config, "n_points", 256)
    output_file = config["output_file"]
    
    # Dispatch to appropriate solver
    t, y = if family == "linear"
        solve_linear_dde(params, history_t, history_y, T; n_points=n_points)
    elseif family == "hutchinson"
        solve_hutchinson_dde(params, history_t, history_y, T; n_points=n_points)
    elseif family == "mackey_glass"
        solve_mackey_glass_dde(params, history_t, history_y, T; n_points=n_points)
    else
        error("Unknown family: $family")
    end
    
    # Save results
    npzwrite(output_file, Dict("t" => collect(t), "y" => y))
    println("Saved solution to $output_file")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
