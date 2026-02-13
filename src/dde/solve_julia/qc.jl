#=
Quality Control for Generated Samples

Checks:
- No NaN/Inf values
- Bounded amplitude (family-specific)
- Positivity for positive-required families
- Continuity at t=0
=#

module QualityControl

include("families.jl")
using .DDEFamilies

export QCResult, run_qc, is_acceptable

"""
Result of quality control checks.
"""
struct QCResult
    passed::Bool
    finite::Bool
    bounded::Bool
    positive::Bool
    continuous::Bool
    max_val::Float64
    min_val::Float64
    continuity_error::Float64
    fail_reasons::Vector{String}
end

"""
Run quality control on a generated sample.

Arguments:
- y: solution array (N_out, d_state)
- phi: history array (N_hist, d_hist)
- family: family symbol
- y_clip: maximum allowed |y|
- cont_tol: tolerance for continuity check at t=0

Returns: QCResult
"""
function run_qc(
    y::Matrix{Float64},
    phi::Matrix{Float64},
    family::Symbol;
    y_clip::Float64 = 100.0,
    cont_tol::Float64 = 1e-4
)
    spec = get_family_spec(family)
    fail_reasons = String[]
    
    # 1. Finite check
    finite_check = all(isfinite, y)
    if !finite_check
        push!(fail_reasons, "non_finite_values")
    end
    
    # 2. Bounded check
    max_val = finite_check ? maximum(abs.(y)) : Inf
    min_val = finite_check ? minimum(y) : -Inf
    bounded_check = max_val < y_clip
    if !bounded_check
        push!(fail_reasons, "amplitude_exceeded:max=$(max_val)")
    end
    
    # 3. Positivity check (for families that require it)
    positive_check = true
    if spec.requires_positive && finite_check
        # Allow tiny numerical noise
        positive_check = min_val >= -1e-6
        if !positive_check
            push!(fail_reasons, "negative_state:min=$(min_val)")
        end
    end
    
    # 4. Continuity at t=0: |phi(0) - y(0)|
    cont_error = 0.0
    continuous_check = true
    if finite_check && size(phi, 2) > 0 && size(y, 1) > 0
        # phi[end, :] should match y[1, :] (for overlapping dimensions)
        d_check = min(size(phi, 2), size(y, 2))
        if d_check > 0
            # Handle case where y has more dims (auxiliary vars)
            phi_end = phi[end, 1:d_check]
            y_start = y[1, 1:d_check]
            cont_error = maximum(abs.(phi_end .- y_start))
            continuous_check = cont_error < cont_tol
            if !continuous_check
                push!(fail_reasons, "discontinuity_at_t0:err=$(cont_error)")
            end
        end
    end
    
    passed = finite_check && bounded_check && positive_check && continuous_check
    
    return QCResult(
        passed,
        finite_check,
        bounded_check,
        positive_check,
        continuous_check,
        max_val,
        min_val,
        cont_error,
        fail_reasons
    )
end

"""
Simple check if a QC result is acceptable.
"""
function is_acceptable(qc::QCResult)
    return qc.passed
end

end # module
