#!/usr/bin/env python3

from pathlib import Path
import re

import numpy as np
import pandas as pd
import geopandas as gpd

HERE = Path(__file__).resolve().parent
MERGE_CSV = HERE.parents[2] / "airborne" / "mrg01_cleaned.csv"
VOC_TABLE = HERE / "config" / "Xu2021_VOCs_clean.csv"
OH_CLOCK_TABLE = HERE / "config" / "phenol_benzene.csv"
OUTPUT_CSV = HERE.parents[1] / "data" / "hox_nox_steady_state.csv"
VOCR_CSV = HERE.parents[1] / "data" / "vocr_contributions.csv"

PPX = {"ppm": 2.46e13, "ppb": 2.46e10, "ppt": 2.46e7}
UNITS = {
    "ISAF": "ppt",
    "PTR": "ppb",
    "ACES": "ppb",
    "CAMS": "ppt",
    "CIMS": "ppt",
    "iWAS": "ppb",
    "DACOM": "ppb",
}

def ppx_to_numden(mixing_ratio, unit, pressure_atm, temp_k):
    return mixing_ratio * PPX[unit] * pressure_atm * (298.0 / temp_k)

def bimolecular_rate_law(A, n, B, T):
    return A * (T / 298.0) ** n * np.exp(-B / T)

def phenol_benzene_oh(phenol, benzene, temperature_k, smoke_age):
    rates = pd.read_csv(OH_CLOCK_TABLE)
    R = 8.314472e-3

    def constants(name):
        hit = rates.loc[rates["Core_Name"] == name].iloc[0]
        A, n, B, EF = (float(hit["A"]), float(hit["n"]), float(hit["B"]), float(hit["EF"]))
        k = A * (temperature_k / 298.0) ** n * np.exp(-B / (R * temperature_k))
        return k, EF

    k_phenol, ef_phenol = constants("Phenol")
    k_benzene, ef_benzene = constants("Benzene")
    with np.errstate(divide="ignore", invalid="ignore"):
        exposure = (1.0 / (k_phenol - k_benzene)) * (
            np.log(ef_phenol / ef_benzene) - np.log(phenol / benzene)
        )
        oh = exposure / smoke_age
    return oh

print("Loading data...")
df = pd.read_csv(MERGE_CSV, engine="pyarrow")
df = df.set_index("timestamp")
df.index = pd.to_datetime(df.index)

df = df.loc[df["transect_number"].notna()].copy()

df["pressure_atm"] = df["Static_Pressure"] / 101325.0 * 100.0
df["temp_K"] = df["Static_Air_Temp"] + 273.15
df["M"] = 2.46e19 * df["pressure_atm"] * (298.0 / df["temp_K"])

print("Computing distance to plume center...")
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
    crs="EPSG:4326",
)

local_time = gdf.index - pd.Timedelta(hours=8)
gdf["day_transect"] = [
    f"{ts.date()}_{transect}_{fire}"
    for ts, transect, fire in zip(local_time, gdf["transect_number"], gdf["fire_id"])
]

distance_to_center = []
for _, group in gdf.groupby("day_transect"):
    if (group["transect_number"] == 0).all():
        group = group.copy()
        group["distance_to_center_km"] = np.nan
        group["distance_to_center_max_km"] = np.nan
        group["normalized_distance_to_center"] = np.nan
        distance_to_center.append(group)
        continue

    projected = group.to_crs(epsg=3857)
    centroid = projected.unary_union.centroid
    co = projected["CO_DACOM"]
    if co.notna().any():
        max_co_point = projected.loc[co.idxmax()].geometry
    else:
        max_co_point = centroid

    group = group.copy()
    group["distance_to_center_km"] = projected.geometry.distance(centroid) / 1000.0
    group["distance_to_center_max_km"] = projected.geometry.distance(max_co_point) / 1000.0
    span = group["distance_to_center_max_km"].max()
    if span > 0:
        group["normalized_distance_to_center"] = group["distance_to_center_max_km"] / span
    else:
        group["normalized_distance_to_center"] = np.nan
    distance_to_center.append(group)

distance_gdf = pd.concat(distance_to_center).sort_index()

print("Preparing VOC table...")
voc_df = pd.read_csv(VOC_TABLE)
voc_df["VOC_Instrument"] = voc_df["VOC"] + "_" + voc_df["Instrument"]
voc_df["Factor"] = 1.0

def parse_note(note):
    if isinstance(note, str):
        match = re.match(r"(\d+)%\s+of\s+([A-Za-z0-9]+)", note)
        if match:
            return int(match.group(1)) / 100.0, match.group(2)
    return None, None

voc_df[["percent", "formula"]] = voc_df["note"].apply(lambda x: pd.Series(parse_note(x)))
voc_df["Factor"] = np.where(voc_df["percent"].notna(), voc_df["percent"], 1.0).astype(float)
voc_df["VOC_Instrument"] = np.where(
    voc_df["formula"].notna(),
    voc_df["formula"] + "_" + voc_df["Instrument"],
    voc_df["VOC_Instrument"],
)

for old, new in [
    ("PTR", "NOAAUIOPTR"),
    ("HCHO", "CH2O"),
    ("Ethane_CAMS", "C2H6_CAMS"),
    ("Phenol_CIMS", "PHENOL_CITCIMS"),
    ("isoprene_iWAS", "Isoprene_iWAS"),
    ("t2Butene_iWAS", "t2butene_iWAS"),
    ("Acetone_NOAAUIOPTR", "AcetonePropanal_NOAAUIOPTR"),
    ("C10H16_NOAAUIOPTR", "Monoterpenes_NOAAUIOPTR"),
    ("C4H8O_NOAAUIOPTR", "C4Carbonyls_NOAAUIOPTR"),
    ("C6H6O2_NOAAUIOPTR", "Catecholx5MeFurfural_NOAAUIOPTR"),
    ("C2H4O2_NOAAUIOPTR", "Glycolaldehyde_NOAAUIOPTR"),
    ("C4H6O_NOAAUIOPTR", "C4Carbonyls_NOAAUIOPTR"),
    ("C8H10_NOAAUIOPTR", "C8Aromatics_NOAAUIOPTR"),
    ("HydroxyBenzoquinone_NOAAUIOPTR", "C6H4O3_NOAAUIOPTR"),
    ("MaleicAnhydride_NOAAUIOPTR", "MaleicAnhyd_NOAAUIOPTR"),
    ("Trimethylbenzene_NOAAUIOPTR", "C9Aromatics_NOAAUIOPTR"),
    ("x25Dimethylfuran_NOAAUIOPTR", "DimeFurans_NOAAUIOPTR"),
    ("C5H6O_NOAAUIOPTR", "x2MeFuranx3MeFuran_NOAAUIOPTR"),
]:
    voc_df["VOC_Instrument"] = voc_df["VOC_Instrument"].str.replace(old, new, regex=False)

voc_df["Unit"] = voc_df["Instrument"].map(UNITS)
voc_df.loc[voc_df["VOC"] == "isoprene", ["Instrument", "VOC_Instrument", "Unit"]] = [
    "PTR",
    "Isoprene_NOAAUIOPTR",
    "ppb",
]
voc_df = voc_df[voc_df["Instrument"] != "iWAS"].copy()

print("Computing VOCR and alpha...")
contributions = {}
vocr = np.zeros(len(df), dtype=float)
temp_k = df["temp_K"].to_numpy()
pressure_atm = df["pressure_atm"].to_numpy()

for row in voc_df.itertuples(index=False):
    mixing = pd.to_numeric(df[row.VOC_Instrument], errors="coerce").to_numpy(dtype=float)
    mixing = np.where(np.isnan(mixing), 0.0, mixing)
    mixing = np.clip(mixing, 0.0, None)
    conc = ppx_to_numden(mixing, row.Unit, pressure_atm, temp_k)
    k = bimolecular_rate_law(float(row.A), float(row.n), float(row.B), temp_k)
    contrib = float(row.Factor) * conc * k
    contributions[row.VOC] = contrib
    vocr = vocr + contrib

contrib_df = pd.DataFrame(contributions, index=df.index)
contrib_df.index.name = "timestamp"
alpha_of = voc_df.drop_duplicates("VOC").set_index("VOC")["alpha"].astype(float)
weight = contrib_df.sum(axis=1)
alpha = contrib_df.mul(alpha_of, axis=1).sum(axis=1) / weight.replace(0.0, np.nan)

contrib_df["VOCR"] = vocr
VOCR_CSV.parent.mkdir(parents=True, exist_ok=True)
contrib_df.to_csv(VOCR_CSV)
print(f"Wrote VOC contributions to {VOCR_CSV}")

print("Computing P(HOx)...")
o2 = 0.21 * df["M"]
n2 = 0.78 * df["M"]
k_h2o = bimolecular_rate_law(1.63e-10, 1.0, -60.0, df["temp_K"])
k_o2 = bimolecular_rate_law(3.3e-11, 1.0, -55.0, df["temp_K"])
k_n2 = bimolecular_rate_law(2.15e-11, 1.0, -110.0, df["temp_K"])

hono = ppx_to_numden(df["HNO2_NOAACIMS"], "ppt", df["pressure_atm"], df["temp_K"])
hcho = ppx_to_numden(df["CH2O_ISAF"], "ppt", df["pressure_atm"], df["temp_K"])
ch3cho = ppx_to_numden(df["CH3CHO_NOAAUIOPTR"], "ppb", df["pressure_atm"], df["temp_K"])
o3 = ppx_to_numden(df["O3_CL"], "ppb", df["pressure_atm"], df["temp_K"])
h2o = ppx_to_numden(df["H2O_DLH"], "ppm", df["pressure_atm"], df["temp_K"])

phi_oh = (2.0 * k_h2o * h2o) / (k_h2o * h2o + k_o2 * o2 + k_n2 * n2)
phox = (
    df["jHNO2_OH_NO_CAFS"] * hono
    + df["jCH2O_H2_CO_CAFS"] * hcho
    + df["jCH3CHO_CH3_HCO_CAFS"] * ch3cho
    + phi_oh * df["jO3_O2_O1D_CAFS"] * o3
)

print("Saving results...")
out = pd.DataFrame(index=df.index)
out.index.name = "timestamp"
out["Latitude"] = df["Latitude"]
out["Longitude"] = df["Longitude"]
out["transect_number"] = df["transect_number"]
out["fire_id"] = df["fire_id"]
out["smoke_age"] = df["smoke_age"]
out["day_transect"] = distance_gdf["day_transect"]
out["distance_to_center_km"] = distance_gdf["distance_to_center_km"]
out["distance_to_center_max_km"] = distance_gdf["distance_to_center_max_km"]
out["normalized_distance_to_center"] = distance_gdf["normalized_distance_to_center"]
out["no_nd"] = ppx_to_numden(df["NO_CL"], "ppb", df["pressure_atm"], df["temp_K"])
out["no2_nd"] = ppx_to_numden(df["NO2_CL"], "ppb", df["pressure_atm"], df["temp_K"])
out["VOCR"] = vocr
out["alpha"] = alpha
out["PHOx"] = phox
out["M"] = df["M"]
out["CO"] = df["CO_DACOM"]
out["NO"] = df["NO_CL"]
out["NO2"] = df["NO2_CL"]
out["NOy"] = df["NOy_CL"]
out["O3"] = df["O3_CL"]
out["HCHO"] = df["CH2O_ISAF"] * 1e-3
out["CH2O"] = df["CH2O_ISAF"]
out["jNO2"] = df["jNO2_NO_O3P_CAFS"]
out["f_hono"] = df["jHNO2_OH_NO_CAFS"] * hono / phox.replace(0.0, np.nan)
out["oh_phenol_benzene"] = phenol_benzene_oh(
    df["Phenol_NOAAUIOPTR"],
    df["Benzene_NOAAUIOPTR"],
    df["temp_K"],
    df["smoke_age"],
)

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT_CSV, index=True)
print(f"Wrote {len(out)} rows to {OUTPUT_CSV}")
