#!/usr/bin/env julia

using CSV
using DataFrames
using Dates
using FiniteDiff
using HSSModel
using HSSModel.DerivedQuantities
using HSSModel.SteadyStateSolvers

const SteadyStateResult = SteadyStateSolvers.SteadyStateResult

function usage()
    println(stderr, """
    Usage: julia steady_state_po3.jl [options]

      --input,  -i PATH   Input CSV  (default: ../../data/hox_nox_steady_state.csv)
      --output, -o PATH   Output CSV (default: ../../data/hox_nox_steady_state_results.csv)
      --regimes           Classify NOx-sat / transitional / NOx-lim + thresholds
      --limit N           Only first N rows (testing)
      --help,   -h        Show this help
    """)
end

function parse_commandline(args)
    opts = Dict{String,Any}(
        "input" => joinpath(@__DIR__, "..", "..", "data", "hox_nox_steady_state.csv"),
        "output" => joinpath(@__DIR__, "..", "..", "data", "hox_nox_steady_state_results.csv"),
        "regimes" => false,
        "limit" => 0,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a in ("--help", "-h")
            usage()
            exit(0)
        elseif a in ("--input", "-i")
            opts["input"] = args[i + 1]; i += 2
        elseif a in ("--output", "-o")
            opts["output"] = args[i + 1]; i += 2
        elseif a == "--regimes"
            opts["regimes"] = true; i += 1
        elseif a == "--limit"
            opts["limit"] = parse(Int, args[i + 1]); i += 2
        else
            error("Unknown argument: $a (use --help)")
        end
    end
    return opts
end

function safe_hox_ss_solver(no, no2, phox, vocr, alpha)
    try
        return hox_ss_solver(no, no2, phox, vocr, alpha)
    catch e
        @warn "hox_ss_solver failed" exception = (e, catch_backtrace())
        return nothing
    end
end

ozone_prod_rate(res::SteadyStateResult) = DerivedQuantities.ozone_prod(res)

ppbhr_from_native(P_native::Float64, M::Float64) = P_native / M * 1e9 * 3600

function PO3_native(no::Float64, no2::Float64, phox::Float64, vocr::Float64, alpha::Float64)
    res = safe_hox_ss_solver(no, no2, phox, vocr, alpha)
    return res === nothing ? NaN : ozone_prod_rate(res)
end

function dP_dNOx_keep_ratio_native(no::Float64, no2::Float64, phox::Float64, vocr::Float64, alpha::Float64)
    NOx0 = no + no2
    if !(isfinite(NOx0)) || NOx0 <= 0
        return NaN
    end
    rNO, rNO2 = no / NOx0, no2 / NOx0
    f(NOxTot) = PO3_native(rNO * NOxTot, rNO2 * NOxTot, phox, vocr, alpha)
    return FiniteDiff.finite_difference_derivative(f, NOx0; relstep = 1e-4)
end

function dP_dVOCR_native(no::Float64, no2::Float64, phox::Float64, vocr::Float64, alpha::Float64)
    g(v) = PO3_native(no, no2, phox, v, alpha)
    return FiniteDiff.finite_difference_derivative(g, vocr; relstep = 1e-4)
end

function classify_regime_row_native(
    no::Float64, no2::Float64, phox::Float64, vocr::Float64, alpha::Float64, M::Float64,
)
    Pn = PO3_native(no, no2, phox, vocr, alpha)
    dPdx = dP_dNOx_keep_ratio_native(no, no2, phox, vocr, alpha)
    dPdv = dP_dVOCR_native(no, no2, phox, vocr, alpha)

    if !(isfinite(Pn) && isfinite(dPdx) && isfinite(dPdv))
        return (;
            regime = :missing,
            P_native = Pn,
            P_ppbhr = NaN,
            dP_dNOx = dPdx,
            dP_dVOCR = dPdv,
            E_NOx = NaN,
            E_VOCR = NaN,
        )
    end

    NOx0 = no + no2
    E_NOx = NOx0 * dPdx
    E_VOC = vocr * dPdv

    regime =
        dPdx < 0 ? :nox_saturated :
        ((dPdx > 0) && (dPdv > 0) && (E_NOx > E_VOC)) ? :nox_limited :
        :transitional

    return (;
        regime,
        P_native = Pn,
        P_ppbhr = ppbhr_from_native(Pn, M),
        dP_dNOx = dPdx,
        dP_dVOCR = dPdv,
        E_NOx = E_NOx,
        E_VOCR = E_VOC,
    )
end

function threshold_saturated(no::Float64, no2::Float64, phox::Float64, vocr::Float64, alpha::Float64)
    NOx0 = no + no2
    if !(isfinite(NOx0)) || NOx0 <= 0
        return (NOx_star = NaN, b_sat = NaN)
    end
    rNO, rNO2 = no / NOx0, no2 / NOx0

    function dfdNOx(N)
        f(Ntot) = PO3_native(rNO * Ntot, rNO2 * Ntot, phox, vocr, alpha)
        return FiniteDiff.finite_difference_derivative(f, N; relstep = 1e-4)
    end

    facs = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0, 8.0, 16.0)
    xs = Float64[]
    fs = Float64[]
    for fac in facs
        x = NOx0 * fac
        val = dfdNOx(x)
        if isfinite(val)
            push!(xs, x)
            push!(fs, val)
        end
    end
    length(xs) < 2 && return (NOx_star = NaN, b_sat = NaN)

    L = nothing
    R = nothing
    for i in 1:(length(xs) - 1)
        if fs[i] * fs[i + 1] <= 0
            L = (xs[i], fs[i])
            R = (xs[i + 1], fs[i + 1])
            break
        end
    end
    L === nothing && return (NOx_star = NaN, b_sat = NaN)

    xL, fL = L
    xR, fR = R
    for _ in 1:40
        xM = 0.5 * (xL + xR)
        fM = dfdNOx(xM)
        !isfinite(fM) && break
        if fL * fM <= 0
            xR, fR = xM, fM
        else
            xL, fL = xM, fM
        end
    end
    NOx_star = 0.5 * (xL + xR)
    return (NOx_star = NOx_star, b_sat = NOx_star / sqrt(vocr))
end

function main(args=ARGS)
    opts = parse_commandline(args)
    input_path = abspath(opts["input"])
    output_path = abspath(opts["output"])
    do_regimes = opts["regimes"]
    row_limit = opts["limit"]

    println("Loading $input_path ...")
    df = CSV.read(input_path, DataFrame)
    if row_limit > 0
        df = df[1:min(row_limit, nrow(df)), :]
        println("Limited to $(nrow(df)) rows")
    end

    solver_cols = [:no_nd, :no2_nd, :VOCR, :alpha, :PHOx, :M]
    df = dropmissing(df, solver_cols)
    df = filter(df) do row
        all(solver_cols) do col
            value = row[col]
            value isa Number && isfinite(value) && value >= 0
        end
    end
    println("$(nrow(df)) rows after cleaning")

    println("Solving HOx steady state for $(nrow(df)) points ...")
    results = Vector{Union{SteadyStateResult,Nothing}}(undef, nrow(df))
    n_fail = 0
    for i in 1:nrow(df)
        if i % 5000 == 0
            println("  solved $i / $(nrow(df))")
        end
        results[i] = safe_hox_ss_solver(
            df.no_nd[i], df.no2_nd[i], df.PHOx[i], df.VOCR[i], df.alpha[i],
        )
        if results[i] === nothing
            n_fail += 1
        end
    end
    ok = [r !== nothing for r in results]
    df = df[ok, :]
    results = results[ok]
    println("Kept $(nrow(df)) successful solves ($(n_fail) failed)")

    df.oh = getfield.(results, :oh)
    df.ho2 = getfield.(results, :ho2)
    df.ro2 = getfield.(results, :ro2)
    df.pO3 = ozone_prod_rate.(results) ./ df.M .* 1e9 .* 3600
    df.nox_lifetime = string.(nox_lifetime.(results))

    if eltype(df.timestamp) <: AbstractString
        df.timestamp = DateTime.(df.timestamp, dateformat"yyyy-mm-dd HH:MM:SS")
    end

    if do_regimes
        println("Classifying ozone-formation regimes ...")
        N = nrow(df)
        regime = Vector{Symbol}(undef, N)
        P_nat = Vector{Float64}(undef, N)
        P_ppbhr = Vector{Float64}(undef, N)
        dNOx_v = Vector{Float64}(undef, N)
        dVOC_v = Vector{Float64}(undef, N)
        E_NOx_v = Vector{Float64}(undef, N)
        E_VOCR_v = Vector{Float64}(undef, N)

        for i in 1:N
            if i % 2000 == 0
                println("  regime $i / $N")
            end
            out = classify_regime_row_native(
                Float64(df.no_nd[i]),
                Float64(df.no2_nd[i]),
                Float64(df.PHOx[i]),
                Float64(df.VOCR[i]),
                Float64(df.alpha[i]),
                Float64(df.M[i]),
            )
            regime[i] = out.regime
            P_nat[i] = out.P_native
            P_ppbhr[i] = out.P_ppbhr
            dNOx_v[i] = out.dP_dNOx
            dVOC_v[i] = out.dP_dVOCR
            E_NOx_v[i] = out.E_NOx
            E_VOCR_v[i] = out.E_VOCR
        end

        df.P_O3_native = P_nat
        df.P_O3_ppbhr = P_ppbhr
        df.regime = String.(regime)
        df.dP_dNOx = dNOx_v
        df.dP_dVOCR = dVOC_v
        df.E_NOx = E_NOx_v
        df.E_VOCR = E_VOCR_v

        cond = (df.dP_dNOx .> 0) .& (df.dP_dVOCR .> 0)
        b_lim_calc = (df.VOCR .* df.dP_dVOCR) ./ df.dP_dNOx ./ sqrt.(df.VOCR)
        df.b_lim_native = ifelse.(cond, b_lim_calc, NaN)

        println("Computing NOx-saturated thresholds ...")
        b_sat = similar(df.VOCR, Float64)
        for i in eachindex(df.VOCR)
            t = threshold_saturated(
                Float64(df.no_nd[i]),
                Float64(df.no2_nd[i]),
                Float64(df.PHOx[i]),
                Float64(df.VOCR[i]),
                Float64(df.alpha[i]),
            )
            b_sat[i] = t.b_sat
        end
        df.b_sat_native = b_sat
        df.b_point_native = (df.no_nd .+ df.no2_nd) ./ sqrt.(df.VOCR)
        df.delta_b_lim = df.b_point_native .- df.b_lim_native
        df.delta_b_sat = df.b_point_native .- df.b_sat_native

        df.NOx_ppb = (df.no_nd .+ df.no2_nd) ./ df.M .* 1e9
        df.NOx_lim_ppb = df.b_lim_native .* sqrt.(df.VOCR) ./ df.M .* 1e9
        df.NOx_sat_ppb = df.b_sat_native .* sqrt.(df.VOCR) ./ df.M .* 1e9
        df.b_point_ppb = df.NOx_ppb ./ sqrt.(df.VOCR)

        println(combine(groupby(df, :regime), nrow => :count))
    end

    out_dir = dirname(output_path)
    !isempty(out_dir) && mkpath(out_dir)
    CSV.write(output_path, df)
    println("Wrote $(nrow(df)) rows to $output_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
