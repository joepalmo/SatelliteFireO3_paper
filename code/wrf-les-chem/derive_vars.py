#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
WRFOUT = Path(
    "/home/jpalmo/fs09/Datasets/WRF/wrf-chem-les/wrfout/wrfout/"
    "wrfout_d01_0001-08-04_00:00:00"
)
MODEL = HERE.parents[1] / "data" / "po3_nn_model.keras"
SCALER = HERE.parents[1] / "data" / "po3_scaler.pkl"
OUTPUT = HERE.parents[1] / "data" / "les_williams.nc"
SS_PO3 = HERE.parents[1] / "data" / "ss_po3_results.nc"

KB = 1.380649e-23
G = 9.80665
P0 = 1.0e5
KAPPA = 0.2854

def _time0(ds, name):
    da = ds[name]
    if "Time" in da.dims:
        da = da.isel(Time=0)
    return np.asarray(da.values, dtype=np.float64)

def steady_state_po3():
    if not SS_PO3.is_file():
        raise SystemExit(
            f"Missing {SS_PO3}"
        )
    with xr.open_dataset(SS_PO3) as ds:
        da = ds["PO3"]
        if "Time" in da.dims:
            da = da.isel(Time=0)
        return np.asarray(da.values, dtype=np.float64)

def _nanmean_z(field, mask):
    masked = np.where(mask, field, np.nan)
    return np.nanmean(masked, axis=0)

def predict_column(age, yhat, hcho_ppt, no2_ppb):
    try:
        import pickle

        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        print("TensorFlow is not importable. column_pO3_pred left empty.")
        return None

    valid = (
        np.isfinite(age)
        & np.isfinite(yhat)
        & np.isfinite(hcho_ppt)
        & np.isfinite(no2_ppb)
    )
    pred = np.full(age.shape, np.nan)
    if not np.any(valid):
        return pred
    frame = pd.DataFrame(
        {
            "smoke_age": age[valid],
            "normalized_distance_to_center": yhat[valid] / 2.0,
            "CH2O": hcho_ppt[valid],
            "NO2": no2_ppb[valid],
        }
    )
    with SCALER.open("rb") as handle:
        scaler = pickle.load(handle)
    model = keras.models.load_model(MODEL)
    scaled = scaler.transform(frame)
    pred[valid] = np.exp(model.predict(scaled, verbose=0).ravel()) - 1.0
    del tf
    return pred

def main():
    needed = [
        "co", "no2", "hcho", "o3", "T", "P", "PB", "PH", "PHB",
        "AgeTracer1", "AgeTracer2", "oc_srf_area",
    ]
    print(f"Opening {WRFOUT}")
    raw = xr.open_dataset(WRFOUT, decode_times=False)
    present = [name for name in needed if name in raw]
    dx = float(raw.attrs["DX"])
    dy = float(raw.attrs["DY"])
    ds = raw[present].load()
    raw.close()

    pressure = _time0(ds, "P") + _time0(ds, "PB")
    theta = _time0(ds, "T") + 300.0
    temperature = theta * (pressure / P0) ** KAPPA
    n_air_m3 = pressure / (KB * temperature)

    z_stag = (_time0(ds, "PH") + _time0(ds, "PHB")) / G
    dz = np.diff(z_stag, axis=0)
    z_mid = 0.5 * (z_stag[:-1] + z_stag[1:])

    co = _time0(ds, "co")
    no2 = _time0(ds, "no2")
    hcho = _time0(ds, "hcho")
    po3_ppbhr = steady_state_po3()
    if po3_ppbhr.shape != co.shape:
        raise SystemExit(
            f"P(O3) shape {po3_ppbhr.shape} does not match CO {co.shape} in {SS_PO3}"
        )
    mask = co > np.nanquantile(co, 0.95)

    def column(ppm):
        chi = ppm * 1e-6
        return np.nansum(chi * n_air_m3 * dz, axis=0) / 1e4

    co_col = column(co)
    no2_col = column(no2)
    hcho_col = column(hcho)
    column_po3 = _nanmean_z(po3_ppbhr, mask)
    oc = _nanmean_z(_time0(ds, "oc_srf_area"), mask)
    no2_ppb = _nanmean_z(no2 * 1e3, mask)
    hcho_ppb = _nanmean_z(hcho * 1e3, mask)
    age1 = _nanmean_z(_time0(ds, "AgeTracer1"), mask)
    age2 = _nanmean_z(_time0(ds, "AgeTracer2"), mask)
    with np.errstate(divide="ignore", invalid="ignore"):
        plume_age = -np.log(age1 / age2) * 3600.0

    nz, ny, nx = co.shape
    y = (np.arange(ny) - (ny - 1) / 2.0) * dy
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx
    z = np.nanmean(z_mid, axis=(1, 2))
    dist = np.abs(y)[:, None] * np.ones((1, nx))
    plume_col = co_col > np.nanquantile(co_col, 0.85)
    half = np.nanmax(np.where(plume_col, dist, np.nan), axis=0)
    yhat = dist / half
    yhat = np.where(yhat <= 1.0, yhat, np.nan)

    pred = predict_column(plume_age, yhat, hcho_ppb * 1000.0, no2_ppb)
    if pred is None:
        pred = np.full(column_po3.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (pred - column_po3) / column_po3 * 100.0

    section = np.where(mask, po3_ppbhr, np.nan)
    out = xr.Dataset(
        {
            "O3_production_rate_ppbhr": (("z", "y", "x"), section.astype(np.float32)),
            "column_pO3": (("y", "x"), column_po3.astype(np.float32)),
            "column_pO3_pred": (("y", "x"), pred.astype(np.float32)),
            "column_pO3_pct_bias": (("y", "x"), pct.astype(np.float32)),
            "oc_srf_area": (("y", "x"), oc.astype(np.float32)),
            "no2_column_molec_cm2": (("y", "x"), no2_col.astype(np.float32)),
            "hcho_column_molec_cm2": (("y", "x"), hcho_col.astype(np.float32)),
            "co_column_molec_cm2": (("y", "x"), co_col.astype(np.float32)),
            "plume_age": (("y", "x"), plume_age.astype(np.float32)),
            "normalized_distance_to_center": (("y", "x"), yhat.astype(np.float32)),
            "no2_plume_ppb": (("y", "x"), no2_ppb.astype(np.float32)),
            "hcho_plume_ppb": (("y", "x"), hcho_ppb.astype(np.float32)),
        },
        coords={"z": z.astype(np.float32), "y": y.astype(np.float32), "x": x.astype(np.float32)},
    )
    out["z"].attrs["units"] = "m"
    out["y"].attrs["units"] = "m"
    out["x"].attrs["units"] = "m"
    out.attrs["source"] = str(WRFOUT)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(OUTPUT)
    print(f"Wrote {OUTPUT}")

if __name__ == "__main__":
    main()
