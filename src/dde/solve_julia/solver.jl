#=
Robust DDE Solver

Implements a multi-attempt solver with fallback ladder:
1. Fast: MethodOfSteps(Tsit5())
2. Accurate: MethodOfSteps(Vern6())
3. Stiff: MethodOfSteps(Rosenbrock23())
4. Constrained: MethodOfSteps(Tsit5(); constrained=true) with dtmax

Uses tstops and d_discontinuities for proper discontinuity handling.
=#

module DDESolver

using DifferentialEquations
using DelayDiffEq
using OrdinaryDiffEq

export robust_solve, make_breakpoint_stops, SolveResult

"""
Result of a DDE solve attempt.
"""
struct SolveResult
    sol::Any           # the ODESolution
    success::Bool
    attempt::Int       # which solver attempt succeeded (1-4)
    retcode::Symbol
    message::String
end

"""
Generate tstops at multiples of delays for discontinuity handling.
"""
function make_breakpoint_stops(lags::Vector{Float64}, T::Float64)
    stops = Float64[0.0]
    for τ in lags
        if τ > 0
            kmax = floor(Int, T / τ)
            for k in 1:kmax
                push!(stops, k * τ)
            end
        end
    end
    sort!(unique!(stops))
    # Filter to [0, T]
    filter!(t -> 0.0 <= t <= T, stops)
    return stops
end

"""
Robust solve with fallback ladder.

Arguments:
- prob: DDEProblem
- lags: vector of delay values
- saveat: times to save solution
- reltol, abstol: tolerances
- maxiters: maximum iterations

Returns: SolveResult
"""
function robust_solve(
    prob;
    lags::Vector{Float64},
    saveat,
    reltol::Float64 = 1e-6,
    abstol::Float64 = 1e-8,
    maxiters::Int = 10_000_000
)
    T = prob.tspan[2]
    tstops = make_breakpoint_stops(lags, T)
    ddisc = copy(tstops)  # derivative discontinuities at same points
    
    min_lag = minimum(filter(x -> x > 0, lags); init = 1.0)
    
    # Solver attempts in order of preference
    attempts = [
        (name = "Tsit5",
         alg = MethodOfSteps(Tsit5()),
         dtmax = Inf),
        (name = "Vern6",
         alg = MethodOfSteps(Vern6()),
         dtmax = Inf),
        (name = "Rosenbrock23",
         alg = MethodOfSteps(Rosenbrock23()),
         dtmax = Inf),
        (name = "Tsit5_constrained",
         alg = MethodOfSteps(Tsit5(); constrained = true),
         dtmax = min_lag / 5),
    ]
    
    last_error = ""
    
    for (k, att) in enumerate(attempts)
        try
            sol = solve(
                prob,
                att.alg;
                saveat = saveat,
                reltol = reltol,
                abstol = abstol,
                tstops = tstops,
                d_discontinuities = ddisc,
                dtmax = att.dtmax,
                maxiters = maxiters
            )
            
            if sol.retcode == :Success || sol.retcode == ReturnCode.Success
                return SolveResult(sol, true, k, :Success, att.name)
            else
                last_error = "retcode=$(sol.retcode)"
            end
        catch e
            last_error = sprint(showerror, e)
        end
    end
    
    # All attempts failed
    return SolveResult(nothing, false, 0, :Failed, last_error)
end

end # module
