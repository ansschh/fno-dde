#=
DDE Family Definitions

Each family defines:
- RHS function with signature f!(du, u, h, p, t)
- Parameter sampling
- State dimension info
- Required positivity constraints
=#

module DDEFamilies

export FamilySpec, get_family_spec
export rhs_linear2!, rhs_hutch!, rhs_mackey_glass!, rhs_vdp_delay!
export rhs_predator_prey!, rhs_dist_uniform!, rhs_dist_exp!

"""
Specification for a DDE family.
"""
struct FamilySpec
    name::Symbol
    rhs::Function
    state_dim::Int
    hist_dim::Int          # dimension of history needed (may differ for auxiliary vars)
    param_names::Vector{Symbol}
    param_ranges::Dict{Symbol, Tuple{Float64, Float64}}
    lag_params::Vector{Symbol}  # which params are delays
    requires_positive::Bool
    y_clip::Float64        # max allowed |y| for QC
end

# ============================================================================
# Family 1: Linear DDE with two discrete delays
# x'(t) = a*x(t) + b1*x(t-τ1) + b2*x(t-τ2)
# ============================================================================
function rhs_linear2!(du, u, h, p, t)
    x = u[1]
    x_τ1 = h(p, t - p.τ1)[1]
    x_τ2 = h(p, t - p.τ2)[1]
    du[1] = p.a * x + p.b1 * x_τ1 + p.b2 * x_τ2
end

const LINEAR2_SPEC = FamilySpec(
    :linear2,
    rhs_linear2!,
    1,  # state_dim
    1,  # hist_dim
    [:a, :b1, :b2, :τ1, :τ2],
    Dict(:a => (-2.0, 1.0), :b1 => (-2.0, 2.0), :b2 => (-2.0, 2.0),
         :τ1 => (0.1, 2.0), :τ2 => (0.1, 2.0)),
    [:τ1, :τ2],
    false,
    1e6
)

# ============================================================================
# Family 2: Hutchinson (delayed logistic) equation
# x'(t) = r*x(t)*(1 - x(t-τ)/K)
# ============================================================================
function rhs_hutch!(du, u, h, p, t)
    x = u[1]
    x_τ = h(p, t - p.τ)[1]
    du[1] = p.r * x * (1.0 - x_τ / p.K)
end

const HUTCH_SPEC = FamilySpec(
    :hutch,
    rhs_hutch!,
    1,
    1,
    [:r, :K, :τ],
    Dict(:r => (0.5, 3.0), :K => (0.5, 2.0), :τ => (0.1, 2.0)),
    [:τ],
    true,
    100.0
)

# ============================================================================
# Family 3: Mackey-Glass equation
# x'(t) = β*x(t-τ)/(1 + x(t-τ)^n) - γ*x(t)
# ============================================================================
function rhs_mackey_glass!(du, u, h, p, t)
    x = u[1]
    x_τ = max(h(p, t - p.τ)[1], 1e-10)  # protect against negative
    du[1] = p.β * x_τ / (1.0 + x_τ^p.n) - p.γ * x
end

const MACKEY_GLASS_SPEC = FamilySpec(
    :mackey_glass,
    rhs_mackey_glass!,
    1,
    1,
    [:β, :γ, :τ, :n],
    Dict(:β => (1.0, 4.0), :γ => (0.5, 3.0), :τ => (0.2, 2.0), :n => (10.0, 10.0)),  # n fixed
    [:τ],
    true,
    50.0
)

# ============================================================================
# Family 4: Van der Pol oscillator with delayed feedback
# x'(t) = v(t)
# v'(t) = μ*(1 - x²)*v - x + κ*x(t-τ)
# ============================================================================
function rhs_vdp_delay!(du, u, h, p, t)
    x, v = u[1], u[2]
    x_τ = h(p, t - p.τ)[1]
    du[1] = v
    du[2] = p.μ * (1.0 - x^2) * v - x + p.κ * x_τ
end

const VDP_SPEC = FamilySpec(
    :vdp,
    rhs_vdp_delay!,
    2,
    2,
    [:μ, :κ, :τ],
    Dict(:μ => (0.5, 3.0), :κ => (-2.0, 2.0), :τ => (0.1, 2.0)),
    [:τ],
    false,
    50.0
)

# ============================================================================
# Family 5: 2D Predator-Prey with two delays
# x'(t) = x(t)*(α - β*y(t-τ1))
# y'(t) = y(t)*(-δ + γ*x(t-τ2))
# ============================================================================
function rhs_predator_prey!(du, u, h, p, t)
    x, y = u[1], u[2]
    y_τ1 = h(p, t - p.τ1)[2]
    x_τ2 = h(p, t - p.τ2)[1]
    du[1] = x * (p.α - p.β * y_τ1)
    du[2] = y * (-p.δ + p.γ * x_τ2)
end

const PREDATOR_PREY_SPEC = FamilySpec(
    :predator_prey,
    rhs_predator_prey!,
    2,
    2,
    [:α, :β, :γ, :δ, :τ1, :τ2],
    Dict(:α => (0.5, 2.0), :β => (0.5, 2.0), :γ => (0.5, 2.0), :δ => (0.5, 2.0),
         :τ1 => (0.1, 2.0), :τ2 => (0.1, 2.0)),
    [:τ1, :τ2],
    true,
    100.0
)

# ============================================================================
# Family 6: Distributed delay - uniform kernel (moving average)
# x'(t) = r*x(t)*(1 - m(t)/K)
# m(t) = (1/τ) ∫_{t-τ}^t x(s) ds
# Auxiliary: m'(t) = (x(t) - x(t-τ))/τ
# ============================================================================
function rhs_dist_uniform!(du, u, h, p, t)
    x, m = u[1], u[2]
    x_τ = h(p, t - p.τ)[1]
    du[1] = p.r * x * (1.0 - m / p.K)
    du[2] = (x - x_τ) / p.τ
end

const DIST_UNIFORM_SPEC = FamilySpec(
    :dist_uniform,
    rhs_dist_uniform!,
    2,  # state: [x, m]
    1,  # hist: only x needed (m computed from integral)
    [:r, :K, :τ],
    Dict(:r => (0.5, 2.5), :K => (0.5, 2.0), :τ => (0.1, 2.0)),
    [:τ],
    true,
    100.0
)

# ============================================================================
# Family 7: Distributed delay - exponential kernel
# x'(t) = r*x(t)*(1 - z(t)/K)
# z(t) = (1/C) ∫_{t-τ}^t exp(-λ(t-s)) x(s) ds
# where C = (1 - exp(-λτ))/λ (normalization)
# Auxiliary: z'(t) = -λ*z + (x - exp(-λτ)*x(t-τ))/C
# ============================================================================
function rhs_dist_exp!(du, u, h, p, t)
    x, z = u[1], u[2]
    x_τ = h(p, t - p.τ)[1]
    C = (1.0 - exp(-p.λ * p.τ)) / p.λ
    du[1] = p.r * x * (1.0 - z / p.K)
    du[2] = -p.λ * z + (x - exp(-p.λ * p.τ) * x_τ) / C
end

const DIST_EXP_SPEC = FamilySpec(
    :dist_exp,
    rhs_dist_exp!,
    2,
    1,
    [:r, :K, :τ, :λ],
    Dict(:r => (0.5, 2.5), :K => (0.5, 2.0), :τ => (0.1, 2.0), :λ => (0.5, 5.0)),
    [:τ],
    true,
    100.0
)

# ============================================================================
# Registry
# ============================================================================
const FAMILY_SPECS = Dict{Symbol, FamilySpec}(
    :linear2 => LINEAR2_SPEC,
    :hutch => HUTCH_SPEC,
    :mackey_glass => MACKEY_GLASS_SPEC,
    :vdp => VDP_SPEC,
    :predator_prey => PREDATOR_PREY_SPEC,
    :dist_uniform => DIST_UNIFORM_SPEC,
    :dist_exp => DIST_EXP_SPEC,
)

function get_family_spec(name::Symbol)::FamilySpec
    haskey(FAMILY_SPECS, name) || error("Unknown family: $name. Available: $(keys(FAMILY_SPECS))")
    return FAMILY_SPECS[name]
end

end # module
