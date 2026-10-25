#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import Transformer

HERE = Path(__file__).resolve().parent
L3 = HERE.parents[2] / "tropomi" / "L3"
OUTPUT = HERE.parents[1] / "data" / "tropomi_williams_20190803.nc"
GRANULE = "20190803T201219"

CENTER_LAT = 47.964712
CENTER_LON = -118.5446
WIND_DIR_DEG = 75.0
WIND_SPEED_MS = 5.0
NBINS_ALONG = 10
CO_PLUME = 3e18
BOX = dict(longitude=slice(-119.0, -116.0), latitude=slice(47.0, 49.0))

def open_granule(product):
    def start_stamp(path):
        times = [part for part in path.name.split("_") if len(part) == 15 and "T" in part]
        return times[0]

    files = [path for path in sorted((L3 / product).glob("*.nc")) if start_stamp(path) == GRANULE]
    if len(files) != 1:
        raise SystemExit(f"Expected one {product} file starting {GRANULE}, found {files}")
    return xr.open_dataset(files[0])

def as_molec(da):
    scale = 1e15 if float(np.nanmedian(da)) < 1e6 else 1.0
    return da * scale

def local_crosswind(lat, lon, mask):
    proj = (
        f"+proj=aeqd +lat_0={CENTER_LAT} +lon_0={CENTER_LON} "
        "+ellps=WGS84 +units=m +no_defs"
    )
    transformer = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
    lon2d, lat2d = np.meshgrid(lon, lat)
    x2d, y2d = transformer.transform(lon2d, lat2d)
    x0, y0 = transformer.transform(CENTER_LON, CENTER_LAT)
    dx = x2d - x0
    dy = y2d - y0
    mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    mx, my = mercator.transform(lon2d, lat2d)
    mx0, my0 = mercator.transform(CENTER_LON, CENTER_LAT)
    distance_m = np.hypot(mx - mx0, my - my0)
    theta = np.deg2rad(WIND_DIR_DEG)
    u_x, u_y = np.sin(theta), np.cos(theta)
    n_x, n_y = -u_y, u_x
    d_along = dx * u_x + dy * u_y
    d_perp = dx * n_x + dy * n_y

    d_along_m = np.where(mask, d_along, np.nan)
    d_perp_m = np.where(mask, d_perp, np.nan)
    flat_along = d_along_m.ravel()
    flat_perp = d_perp_m.ravel()
    valid = np.isfinite(flat_perp)
    bins = np.linspace(np.nanmin(flat_along), np.nanmax(flat_along), NBINS_ALONG + 1)
    bin_idx = np.digitize(flat_along, bins) - 1
    norm = np.full(flat_perp.shape, np.nan)
    for b in range(NBINS_ALONG):
        sel = valid & (bin_idx == b)
        if not np.any(sel):
            continue
        half = np.nanmax(np.abs(flat_perp[sel]))
        if not np.isfinite(half) or half == 0:
            continue
        norm[sel] = flat_perp[sel] / half
    return np.abs(norm.reshape(mask.shape)), distance_m

def main():
    def field(ds, name):
        da = as_molec(ds[name])
        if "time" in da.dims:
            da = da.isel(time=0, drop=True)
        return da

    no2 = open_granule("NO2")
    hcho = open_granule("HCHO")
    co = open_granule("CO")
    ds = xr.Dataset(
        {
            "tropospheric_NO2_column_number_density": field(
                no2, "tropospheric_NO2_column_number_density"
            ),
            "tropospheric_HCHO_column_number_density": field(
                hcho, "tropospheric_HCHO_column_number_density"
            ),
            "CO_column_number_density": field(co, "CO_column_number_density"),
        }
    )
    ds = ds.assign_coords(latitude=no2.latitude, longitude=no2.longitude)
    lat = ds.latitude.values
    lon = ds.longitude.values
    lat_slice = slice(39.05, 50.85) if lat[0] < lat[-1] else slice(50.85, 39.05)
    lon_slice = slice(-127.95, -107.15) if lon[0] < lon[-1] else slice(-107.15, -127.95)
    ds = ds.sel(latitude=lat_slice, longitude=lon_slice)

    hcho_da = ds["tropospheric_HCHO_column_number_density"]
    no2_da = ds["tropospheric_NO2_column_number_density"]
    ds["FNR"] = (hcho_da - np.nanquantile(hcho_da, 0.05)) / (
        no2_da - np.nanquantile(no2_da, 0.05)
    )

    lat = ds.latitude.values
    lon = ds.longitude.values
    lon2d, lat2d = np.meshgrid(lon, lat)
    in_box = (
        (lon2d >= BOX["longitude"].start)
        & (lon2d <= BOX["longitude"].stop)
        & (lat2d >= BOX["latitude"].start)
        & (lat2d <= BOX["latitude"].stop)
        & (ds["CO_column_number_density"].values > CO_PLUME)
    )
    yhat, distance_m = local_crosswind(lat, lon, in_box)
    age_h = distance_m / WIND_SPEED_MS / 3600.0
    ds["t"] = xr.DataArray(
        np.where(in_box, age_h, np.nan),
        coords={"latitude": ds.latitude, "longitude": ds.longitude},
        dims=("latitude", "longitude"),
    )
    ds["yhat"] = xr.DataArray(
        np.where(in_box, yhat, np.nan),
        coords={"latitude": ds.latitude, "longitude": ds.longitude},
        dims=("latitude", "longitude"),
    )
    ds.attrs["granule"] = GRANULE
    ds.attrs["wind_speed_m_s"] = WIND_SPEED_MS
    ds.attrs["wind_dir_deg"] = WIND_DIR_DEG
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(OUTPUT)
    print(f"Wrote {OUTPUT}")
    print(f"  plume pixels with age: {int(np.isfinite(ds['t'].values).sum())}")

if __name__ == "__main__":
    main()
