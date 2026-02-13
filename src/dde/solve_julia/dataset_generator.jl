#=
Sharded Dataset Generator

Main script for generating datasets with:
- Sharded storage (resumable)
- Quality control with rejection sampling
- Comprehensive logging
- Manifest tracking
=#

using Random
using NPZ
using JSON3
using Dates
using Pkg

include("generate.jl")
include("qc.jl")

using .DDEGenerator
using .QualityControl

# ============================================================================
# Configuration
# ============================================================================
@kwdef struct DatasetConfig
    family::Symbol
    τmax::Float64 = 2.0
    T::Float64 = 20.0
    dt_out::Float64 = 0.05
    N_hist::Int = 256
    reltol::Float64 = 1e-6
    abstol::Float64 = 1e-8
    
    # Dataset sizes
    n_train::Int = 800
    n_val::Int = 100
    n_test::Int = 100
    
    # Sharding
    shard_size::Int = 64
    
    # QC
    y_clip::Float64 = 100.0
    max_retries::Int = 10
    
    # Output
    output_dir::String = "data"
    seed::Int = 42
end

# ============================================================================
# Shard Writer
# ============================================================================
struct Shard
    t_hist::Vector{Float64}
    t_out::Vector{Float64}
    phi::Array{Float64, 3}      # (B, N_hist, d_hist)
    y::Array{Float64, 3}        # (B, N_out, d_state)
    params::Matrix{Float64}     # (B, P)
    lags::Matrix{Float64}       # (B, L)
    attempts::Vector{Int}       # (B,)
    count::Int
end

function create_empty_shard(config::DatasetConfig, spec, B::Int)
    N_hist = config.N_hist
    N_out = length(0.0:config.dt_out:config.T)
    d_state = spec.state_dim
    d_hist = spec.hist_dim
    n_params = length(spec.param_names)
    n_lags = length(spec.lag_params)
    
    return Shard(
        collect(range(-config.τmax, 0.0; length = N_hist)),
        collect(0.0:config.dt_out:config.T),
        zeros(B, N_hist, d_hist),
        zeros(B, N_out, d_state),
        zeros(B, n_params),
        zeros(B, n_lags),
        zeros(Int, B),
        0
    )
end

function add_to_shard!(shard::Shard, sample::GeneratedSample, idx::Int)
    shard.phi[idx, :, :] = sample.phi[:, 1:size(shard.phi, 3)]
    shard.y[idx, :, :] = sample.y
    shard.params[idx, :] = sample.params
    shard.lags[idx, :] = sample.lags
    shard.attempts[idx] = sample.attempt
end

function write_shard(path::String, shard::Shard, meta::Dict)
    NPZ.npzwrite(path, Dict(
        "t_hist" => shard.t_hist,
        "t_out" => shard.t_out,
        "phi" => shard.phi,
        "y" => shard.y,
        "params" => shard.params,
        "lags" => shard.lags,
        "attempts" => shard.attempts,
        "meta_json" => JSON3.write(meta)
    ))
end

# ============================================================================
# Failure Logger
# ============================================================================
mutable struct FailureLog
    path::String
    io::IOStream
    counts::Dict{String, Int}
end

function FailureLog(path::String)
    io = open(path, "a")
    return FailureLog(path, io, Dict{String, Int}())
end

function log_failure!(log::FailureLog, reason::String, params::Vector{Float64})
    entry = Dict("reason" => reason, "params" => params, "time" => string(now()))
    println(log.io, JSON3.write(entry))
    flush(log.io)
    log.counts[reason] = get(log.counts, reason, 0) + 1
end

function close_log!(log::FailureLog)
    close(log.io)
end

# ============================================================================
# Main Generation Loop
# ============================================================================
function generate_split(
    config::DatasetConfig,
    split::Symbol,
    n_samples::Int,
    base_seed::Int,
    spec
)
    split_dir = joinpath(config.output_dir, string(config.family), string(split))
    mkpath(split_dir)
    
    fail_log = FailureLog(joinpath(split_dir, "failures.jsonl"))
    
    n_shards = cld(n_samples, config.shard_size)
    total_generated = 0
    
    for shard_id in 0:(n_shards - 1)
        shard_path = joinpath(split_dir, "shard_$(lpad(shard_id, 3, '0')).npz")
        
        # Resume: skip existing shards
        if isfile(shard_path)
            println("  Shard $shard_id already exists, skipping...")
            total_generated += config.shard_size
            continue
        end
        
        # Determine shard size (last shard may be smaller)
        remaining = n_samples - shard_id * config.shard_size
        B = min(config.shard_size, remaining)
        
        shard = create_empty_shard(config, spec, B)
        shard_rng = MersenneTwister(base_seed + shard_id * 1000)
        
        idx = 1
        while idx <= B
            # Generate sample
            sample = gen_sample(
                config.family,
                shard_rng;
                τmax = config.τmax,
                T = config.T,
                dt_out = config.dt_out,
                N_hist = config.N_hist,
                reltol = config.reltol,
                abstol = config.abstol
            )
            
            # Check success
            if !sample.success
                log_failure!(fail_log, "solver_failed:" * sample.fail_reason, sample.params)
                continue
            end
            
            # Run QC
            qc = run_qc(sample.y, sample.phi, config.family; y_clip = config.y_clip)
            
            if !qc.passed
                for reason in qc.fail_reasons
                    log_failure!(fail_log, reason, sample.params)
                end
                continue
            end
            
            # Accept sample
            add_to_shard!(shard, sample, idx)
            idx += 1
        end
        
        # Write shard
        meta = Dict(
            "family" => string(config.family),
            "split" => string(split),
            "shard_id" => shard_id,
            "n_samples" => B,
            "config" => Dict(
                "τmax" => config.τmax,
                "T" => config.T,
                "dt_out" => config.dt_out,
                "N_hist" => config.N_hist,
                "reltol" => config.reltol,
                "abstol" => config.abstol,
            ),
            "seed" => base_seed + shard_id * 1000,
            "timestamp" => string(now()),
            "julia_version" => string(VERSION),
        )
        
        write_shard(shard_path, shard, meta)
        total_generated += B
        println("  Shard $shard_id: wrote $B samples")
    end
    
    close_log!(fail_log)
    
    # Print failure summary
    if !isempty(fail_log.counts)
        println("  Failure summary:")
        for (reason, count) in fail_log.counts
            println("    $reason: $count")
        end
    end
    
    return total_generated
end

function generate_dataset(config::DatasetConfig)
    include("families.jl")
    using .DDEFamilies
    
    spec = get_family_spec(config.family)
    
    println("=" ^ 60)
    println("Generating dataset for family: $(config.family)")
    println("=" ^ 60)
    println("Config:")
    println("  τmax = $(config.τmax), T = $(config.T), dt_out = $(config.dt_out)")
    println("  N_hist = $(config.N_hist)")
    println("  train: $(config.n_train), val: $(config.n_val), test: $(config.n_test)")
    println("  shard_size = $(config.shard_size)")
    println()
    
    family_dir = joinpath(config.output_dir, string(config.family))
    mkpath(family_dir)
    
    # Generate each split
    splits = [
        (:train, config.n_train, config.seed),
        (:val, config.n_val, config.seed + 100_000),
        (:test, config.n_test, config.seed + 200_000),
    ]
    
    manifest = Dict{String, Any}(
        "family" => string(config.family),
        "config" => Dict(
            "τmax" => config.τmax,
            "T" => config.T,
            "dt_out" => config.dt_out,
            "N_hist" => config.N_hist,
            "reltol" => config.reltol,
            "abstol" => config.abstol,
            "y_clip" => config.y_clip,
        ),
        "param_names" => [string(n) for n in spec.param_names],
        "param_ranges" => Dict(string(k) => v for (k, v) in spec.param_ranges),
        "state_dim" => spec.state_dim,
        "splits" => Dict{String, Any}(),
        "created" => string(now()),
        "seed" => config.seed,
    )
    
    for (split, n, seed) in splits
        println("\nGenerating $split split ($n samples)...")
        n_gen = generate_split(config, split, n, seed, spec)
        manifest["splits"][string(split)] = Dict(
            "n_samples" => n_gen,
            "n_shards" => cld(n, config.shard_size),
        )
    end
    
    # Write manifest
    manifest_path = joinpath(family_dir, "manifest.json")
    open(manifest_path, "w") do io
        JSON3.pretty(io, manifest)
    end
    println("\nManifest written to: $manifest_path")
    
    println("\n" * "=" ^ 60)
    println("Dataset generation complete!")
    println("=" ^ 60)
end

# ============================================================================
# CLI Entry Point
# ============================================================================
function main()
    if length(ARGS) < 1
        println("Usage: julia dataset_generator.jl <family> [options]")
        println("Families: linear2, hutch, mackey_glass, vdp, predator_prey, dist_uniform, dist_exp")
        println("Options:")
        println("  --n_train=N     Number of training samples (default: 800)")
        println("  --n_val=N       Number of validation samples (default: 100)")
        println("  --n_test=N      Number of test samples (default: 100)")
        println("  --shard_size=N  Samples per shard (default: 64)")
        println("  --output_dir=D  Output directory (default: data)")
        println("  --seed=N        Random seed (default: 42)")
        println("  --T=X           Solution horizon (default: 20.0)")
        println("  --dt_out=X      Output time step (default: 0.05)")
        return
    end
    
    family = Symbol(ARGS[1])
    
    # Parse optional arguments
    kwargs = Dict{Symbol, Any}(:family => family)
    
    for arg in ARGS[2:end]
        if startswith(arg, "--")
            parts = split(arg[3:end], "=")
            if length(parts) == 2
                key = Symbol(parts[1])
                val_str = parts[2]
                # Parse value
                if key in (:n_train, :n_val, :n_test, :shard_size, :seed, :N_hist)
                    kwargs[key] = parse(Int, val_str)
                elseif key in (:T, :dt_out, :τmax, :reltol, :abstol, :y_clip)
                    kwargs[key] = parse(Float64, val_str)
                else
                    kwargs[key] = val_str
                end
            end
        end
    end
    
    config = DatasetConfig(; kwargs...)
    generate_dataset(config)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
