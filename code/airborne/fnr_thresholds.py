#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic_2d

HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE.parents[1] / "data" / "hox_nox_steady_state_results.csv"
STATE_CSV = HERE.parents[1] / "data" / "fnr_bin_state.csv"
SLOPE_TXT = HERE.parents[1] / "data" / "fnr_hcho_slope.txt"
SWEEP_CSV = HERE.parents[1] / "data" / "fnr_sweep.csv"
LIM_NPZ = HERE.parents[1] / "data" / "fnr_noxlim_thresholds.npz"
SAT_NPZ = HERE.parents[1] / "data" / "fnr_noxsat_thresholds.npz"

AGE_BINS_H = np.linspace(0.0, np.sqrt(10.0), 10) ** 2
YHAT_BINS = np.arange(0.0, 1.1, 0.1)
MIN_COUNT = 30

def pack_sweep():
    sweep = pd.read_csv(SWEEP_CSV)
    grids = {}
    for name in ("noxlim", "noxsat"):
        grid = np.full((9, 10), np.nan)
        for row in sweep.itertuples(index=False):
            grid[int(row.i) - 1, int(row.j) - 1] = getattr(row, name)
        grids[name] = grid
    np.savez(LIM_NPZ, grid=grids["noxlim"])
    np.savez(SAT_NPZ, grid=grids["noxsat"])
    print(f"Wrote {LIM_NPZ}")
    print(f"Wrote {SAT_NPZ}")

def hcho_vocr_slope(df):
    sample = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["VOCR", "HCHO"])
    sample = sample[(sample["VOCR"] > 0) & (sample["HCHO"] > 0) & (sample["HCHO"] < 200)]
    sqrt_vocr = np.sqrt(sample["VOCR"].to_numpy())
    hcho = sample["HCHO"].to_numpy()

    def through_origin(x, slope):
        return slope * x

    slope, _ = curve_fit(through_origin, sqrt_vocr, hcho)
    predicted = through_origin(sqrt_vocr, slope[0])
    ss_res = np.sum((hcho - predicted) ** 2)
    ss_tot = np.sum((hcho - hcho.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    return float(slope[0]), float(r2), int(len(sample))

def write_bin_state():
    df = pd.read_csv(
        INPUT_CSV,
        usecols=[
            "smoke_age",
            "normalized_distance_to_center",
            "NO",
            "NO2",
            "PHOx",
            "alpha",
            "VOCR",
            "HCHO",
            "M",
        ],
    )
    slope, r2, n_fit = hcho_vocr_slope(df)
    SLOPE_TXT.write_text(f"{slope}\n")
    print(f"HCHO vs sqrt(VOCR) slope = {slope:.6f} (R^2 = {r2:.3f}, n = {n_fit})")
    print(f"Wrote {SLOPE_TXT}")
    df = df.dropna()
    df = df[(df["NO"] + df["NO2"]) > 0]
    df["age_hr"] = df["smoke_age"] / 3600.0
    df["rno2"] = df["NO2"] / (df["NO"] + df["NO2"])

    def binned(values):
        stat, _, _, _ = binned_statistic_2d(
            df["age_hr"],
            df["normalized_distance_to_center"],
            values,
            statistic="mean",
            bins=[AGE_BINS_H, YHAT_BINS],
        )
        return stat

    count, _, _, _ = binned_statistic_2d(
        df["age_hr"],
        df["normalized_distance_to_center"],
        df["PHOx"],
        statistic="count",
        bins=[AGE_BINS_H, YHAT_BINS],
    )
    phox = binned(df["PHOx"])
    rno2 = binned(df["rno2"])
    alpha = binned(df["alpha"])
    rows = []
    for i in range(phox.shape[0]):
        for j in range(phox.shape[1]):
            if count[i, j] < MIN_COUNT or not np.isfinite(phox[i, j]):
                continue
            rows.append(
                {
                    "i": i + 1,
                    "j": j + 1,
                    "phox": phox[i, j],
                    "rno2": rno2[i, j],
                    "alpha": alpha[i, j],
                    "count": int(count[i, j]),
                }
            )
    out = pd.DataFrame(rows)
    STATE_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(STATE_CSV, index=False)
    print(f"Wrote {len(out)} occupied bins to {STATE_CSV}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack",
        action="store_true",
        help="Write the npz grids from fnr_sweep.csv",
    )
    args = parser.parse_args()
    if args.pack:
        pack_sweep()
    else:
        write_bin_state()

if __name__ == "__main__":
    main()
