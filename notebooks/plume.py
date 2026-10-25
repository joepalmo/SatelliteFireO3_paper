from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AIRBORNE = ROOT / "airborne"
RESULTS_CSV = ROOT / "paper" / "data" / "hox_nox_steady_state_results.csv"

AGE_BINS_H = np.linspace(0.0, np.sqrt(10.0), 10) ** 2
YHAT_BINS = np.arange(0.0, 1.1, 0.1)

def load_plume_table() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV, parse_dates=["timestamp"]).set_index("timestamp")
    df["NO2/NOx"] = df["NO2"] / (df["NO2"] + df["NO"])
    df["NOx/NOy"] = (df["NO2"] + df["NO"]) / df["NOy"]
    df["Ox/CO"] = (df["NO2"] + df["O3"]) / df["CO"]
    df["age_hr"] = df["smoke_age"] / 3600.0
    return df

def binned_mean(df: pd.DataFrame, column: str, min_count: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.stats import binned_statistic_2d

    tmp = df.dropna(subset=["age_hr", "normalized_distance_to_center", column])
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        tmp["age_hr"],
        tmp["normalized_distance_to_center"],
        tmp[column],
        statistic="mean",
        bins=[AGE_BINS_H, YHAT_BINS],
    )
    count, _, _, _ = binned_statistic_2d(
        tmp["age_hr"],
        tmp["normalized_distance_to_center"],
        tmp[column],
        statistic="count",
        bins=[AGE_BINS_H, YHAT_BINS],
    )
    stat = stat.astype(float)
    stat[count < min_count] = np.nan
    return stat, x_edges, y_edges
