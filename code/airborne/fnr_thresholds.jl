#!/usr/bin/env julia

using CSV
using DataFrames
using FiniteDiff
using HSSModel
using HSSModel.DerivedQuantities
using HSSModel.SteadyStateSolvers

const NROWS = 9
const NCOLS = 10

const R_GAS = 8.314e6 / 6.022e23
const NAIR = 101325 / (R_GAS * 298)
const ALPHA_SWEEP = 0.04

function usage()
    println(stderr, """
    Usage: julia fnr_thresholds.jl [--recompute] [--length N] [--state PATH]

      --recompute     Overwrite the FNR npz files. Without this, exit.
      --length N      NOx and VOCR sweep length (default 100)
      --state PATH    Bin-state CSV (default: paper/data/fnr_bin_state.csv)
    """)
end

function parse_args(args)
    opts = Dict{String,Any}(
        "recompute" => false,
        "length" => 100,
        "state" => joinpath(@__DIR__, "..", "..", "data", "fnr_bin_state.csv"),
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a in ("--help", "-h")
            usage()
            exit(0)
        elseif a == "--recompute"
            opts["recompute"] = true
            i += 1
        elseif a == "--length"
            opts["length"] = parse(Int, args[i + 1])
            i += 2
        elseif a == "--state"
            opts["state"] = args[i + 1]
            i += 2
        else
            error("Unknown argument: $a")
        end
    end
    return opts
end

function po3_ppbhr(nox_ppb, vocr; phox, rno2)
    nox_nd = nox_ppb * 1e-9 * NAIR
    no2_nd = rno2 * nox_nd
    no_nd = (1 - rno2) * nox_nd
    res = hox_ss_solver(no_nd, no2_nd, phox, vocr, ALPHA_SWEEP)
    return DerivedQuantities.ozone_prod(res) / NAIR * 1e9 * 3600
end

function dP_dNOx_central(nox_grid, vocr; phox, rno2)
    y = collect(nox_grid)
    ny = length(y)
    P = [po3_ppbhr(yi, vocr; phox=phox, rno2=rno2) for yi in y]
    f = similar(P)
    for i in 2:(ny - 1)
        f[i] = (P[i + 1] - P[i - 1]) / (y[i + 1] - y[i - 1])
    end
    f[1] = (P[2] - P[1]) / (y[2] - y[1])
    f[ny] = (P[ny] - P[ny - 1]) / (y[ny] - y[ny - 1])
    return f
end

function saturated_cross(nox_grid, vocr; phox, rno2)
    y = collect(nox_grid)
    f = dP_dNOx_central(y, vocr; phox=phox, rno2=rno2)
    for i in 1:(length(y) - 1)
        f1, f2 = f[i], f[i + 1]
        if isfinite(f1) && isfinite(f2) && ((f1 == 0) || (f1 * f2 < 0))
            t = -f1 / (f2 - f1)
            return y[i] + t * (y[i + 1] - y[i])
        end
    end
    return missing
end

function limited_cross(nox_grid, vocr; phox, rno2)
    y = collect(nox_grid)
    f = dP_dNOx_central(y, vocr; phox=phox, rno2=rno2)
    g = [
        FiniteDiff.finite_difference_derivative(
            t -> po3_ppbhr(yi, t; phox=phox, rno2=rno2), vocr; relstep=1e-4
        ) for yi in y
    ]
    h = y .* f .- vocr .* g
    good = (f .> 0) .& (g .> 0)
    for i in 1:(length(y) - 1)
        if (good[i] || good[i + 1]) && isfinite(h[i]) && isfinite(h[i + 1]) &&
           ((h[i] == 0) || (h[i] * h[i + 1] < 0))
            t = -h[i] / (h[i + 1] - h[i])
            return y[i] + t * (y[i + 1] - y[i])
        end
    end
    return missing
end

function fit_b(x_vocr, y_nox)
    if length(x_vocr) < 2
        return NaN
    end
    s = sqrt.(x_vocr)
    denom = sum(s .* s)
    denom == 0 && return NaN
    return sum(s .* y_nox) / denom
end

function crossovers(phox, rno2, vocr_vec, nox_ppb)
    if !(isfinite(phox) && isfinite(rno2) && 0 < rno2 < 1)
        return (b_lim=NaN, b_sat=NaN)
    end
    sat = [saturated_cross(nox_ppb, v; phox=phox, rno2=rno2) for v in vocr_vec]
    lim = [limited_cross(nox_ppb, v; phox=phox, rno2=rno2) for v in vocr_vec]
    mask_sat = .!ismissing.(sat)
    mask_lim = .!ismissing.(lim)
    b_sat = fit_b(vocr_vec[mask_sat], Float64.(sat[mask_sat]))
    b_lim = fit_b(vocr_vec[mask_lim], Float64.(lim[mask_lim]))
    return (b_lim=b_lim, b_sat=b_sat)
end

function read_hcho_slope(data_dir)
    path = joinpath(data_dir, "fnr_hcho_slope.txt")
    if !isfile(path)
        error("Missing $path. Run fnr_thresholds.py first.")
    end
    return parse(Float64, split(readline(path))[1])
end

function main(args=ARGS)
    opts = parse_args(args)
    data_dir = joinpath(@__DIR__, "..", "..", "data")
    sweep_path = joinpath(data_dir, "fnr_sweep.csv")
    if !opts["recompute"]
        println("FNR grids left unchanged. Pass --recompute to regenerate the sweep.")
        println("  $(joinpath(data_dir, "fnr_noxlim_thresholds.npz"))")
        println("  $(joinpath(data_dir, "fnr_noxsat_thresholds.npz"))")
        return
    end

    state = CSV.read(opts["state"], DataFrame)
    slope = read_hcho_slope(data_dir)
    println("HCHO vs sqrt(VOCR) slope = $slope")
    println("Sweeping $(nrow(state)) bins on $(Threads.nthreads()) threads")
    n = opts["length"]
    nox_ppb = exp10.(range(-2.0, stop=1.3, length=n))
    vocr_vec = collect(range(1e-3, stop=60.0, length=n))

    b_lim = fill(NaN, NROWS, NCOLS)
    b_sat = fill(NaN, NROWS, NCOLS)
    rno2_grid = fill(NaN, NROWS, NCOLS)
    rows = collect(eachrow(state))
    Threads.@threads for row in rows
        i, j = Int(row.i), Int(row.j)
        println("Bin ($i, $j), n=$(row.count)")
        curves = crossovers(Float64(row.phox), Float64(row.rno2), vocr_vec, nox_ppb)
        b_lim[i, j] = 1 / curves.b_lim
        b_sat[i, j] = 1 / curves.b_sat
        rno2_grid[i, j] = row.rno2
    end

    noxlim = b_lim .* (1 ./ rno2_grid) .* slope
    noxsat = b_sat .* (1 ./ rno2_grid) .* slope
    mkpath(data_dir)
    open(sweep_path, "w") do io
        println(io, "i,j,noxlim,noxsat")
        for i in 1:NROWS, j in 1:NCOLS
            println(io, "$i,$j,$(noxlim[i, j]),$(noxsat[i, j])")
        end
    end
    println("Wrote $sweep_path")
    println("Pack the grids with: python $(joinpath(@__DIR__, "fnr_thresholds.py")) --pack")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
