"""
Step 16J - Daily cold-air advection and subtropical-jet diagnostics for every October-March day.

Why
    Steps 10B (advection) and 10C (jet/waveguide) saved only event-window means, so they cannot be
    used for day-by-day onset timelines or in the Step 16D hazard model. This step derives the same
    physical quantities as DAILY series from the existing Step 9B standardized files.

Diagnostics (00 UTC)
    850-hPa horizontal temperature advection  -(u dT/dx + v dT/dy), K day-1, computed on the native
    grid and averaged (cosine-latitude weights) over:
        bd    Bangladesh box         20.5-26.8N, 88-93E
        nigp  NW Indo-Gangetic box   25-32N, 75-88E   (upstream cold-air source region)
    Subtropical jet at 200 hPa over 70-100E: the longitude-mean zonal wind profile between 15 and
    45N; jet-core latitude (latitude of the maximum) and jet-core speed (the maximum).

Inputs (read only): 03_intermediate/era5_preprocessed/pressure_level_standardized/ERA5_PL_STD_YYYY_MM.nc
Output: 03_intermediate/era5_box_daily/era5_mechanism_daily_oct_mar.csv (+ policy, report, checksums)

Run from the project root:
    python3 07_scripts/15j_extract_advection_jet_daily.py 2>&1 | tee 09_logs/step16j_advection_jet.log
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

_spec = importlib.util.spec_from_file_location(
    "term_common", Path(__file__).resolve().parent / "15_termination_common.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

PL_DIR = C.PROJECT_ROOT / "03_intermediate" / "era5_preprocessed" / "pressure_level_standardized"
OUT = C.PROJECT_ROOT / "03_intermediate" / "era5_box_daily" / "era5_mechanism_daily_oct_mar.csv"
REPORT = C.TERM_QC / "step16j_advection_jet_report.txt"

EARTH_RADIUS = 6.371e6
SECONDS_PER_DAY = 86400.0
ADV_BOXES = {"bd": (20.5, 26.8, 88.0, 93.0), "nigp": (25.0, 32.0, 75.0, 88.0)}
JET_LEVEL, JET_LON, JET_LAT = 200.0, (70.0, 100.0), (15.0, 45.0)
MONTHS = [10, 11, 12, 1, 2, 3]
FIRST_WINTER, LAST_WINTER = 1985, 2024

POLICY = {
    "step": "Step 16J", "title": "Daily 850-hPa temperature advection and 200-hPa jet diagnostics",
    "time": "00 UTC daily snapshots, October-March, winters 1985/86-2024/25",
    "advection": {"level_hpa": 850, "formula": "-(u dT/dx + v dT/dy)",
                  "derivatives": "centred differences on the native 0.25-degree grid (xarray.differentiate)",
                  "units": "K per day", "boxes": ADV_BOXES, "averaging": "cosine-latitude weighted"},
    "jet": {"level_hpa": JET_LEVEL, "longitude_band": JET_LON, "latitude_band": JET_LAT,
            "core_latitude": "latitude of maximum longitude-mean zonal wind",
            "core_speed": "maximum longitude-mean zonal wind"},
    "inputs": "Step 9B standardized pressure-level files (read only)",
}


def box(da: xr.DataArray, lat0: float, lat1: float, lon0: float, lon1: float) -> xr.DataArray:
    lat = da["lat"].values
    lat_slice = slice(lat1, lat0) if lat[0] > lat[-1] else slice(lat0, lat1)
    return da.sel(lat=lat_slice, lon=slice(lon0, lon1))


def wmean(da: xr.DataArray) -> np.ndarray:
    w = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(w).mean(dim=("lat", "lon")).values


def main() -> None:
    C.ensure_dirs()
    C.write_policy("step16j_advection_jet_policy.json", POLICY)
    frames, missing = [], []
    for winter in range(FIRST_WINTER, LAST_WINTER + 1):
        for month in MONTHS:
            year = winter if month >= 10 else winter + 1
            f = PL_DIR / f"ERA5_PL_STD_{year}_{month:02d}.nc"
            if not f.exists():
                missing.append(f.name)
                print("MISSING", f.name)
                continue
            with xr.open_dataset(f) as ds:
                # pad the region slightly so centred differences are valid at box edges
                sub = ds.sel(level_hpa=850.0)
                lat = sub["lat"].values
                lat_slice = slice(34.0, 18.0) if lat[0] > lat[-1] else slice(18.0, 34.0)
                sub = sub.sel(lat=lat_slice, lon=slice(72.0, 96.0))
                T = sub["air_temperature_k"].load()
                u = sub["u_wind_ms"].load()
                v = sub["v_wind_ms"].load()
                coslat = np.cos(np.deg2rad(T["lat"]))
                dTdx = T.differentiate("lon") * (180.0 / np.pi) / (EARTH_RADIUS * coslat)
                dTdy = T.differentiate("lat") * (180.0 / np.pi) / EARTH_RADIUS
                adv = -(u * dTdx + v * dTdy) * SECONDS_PER_DAY
                cols = {f"{k}_adv850_k_day": wmean(box(adv, *b)) for k, b in ADV_BOXES.items()}

                uj = ds["u_wind_ms"].sel(level_hpa=JET_LEVEL)
                uj = box(uj, JET_LAT[0], JET_LAT[1], JET_LON[0], JET_LON[1]).mean("lon").load()
                idx = uj.argmax("lat")
                cols["jet200_core_lat_70_100e"] = uj["lat"].values[idx.values]
                cols["jet200_core_speed_70_100e"] = uj.max("lat").values
                times = pd.to_datetime(ds["time"].values).normalize()
            df = pd.DataFrame(cols, index=times)
            df.index.name = "date"
            frames.append(df)
            print(f"done {year}-{month:02d}")
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, float_format="%.5f")
    C.write_checksums("step16j_output_sha256.txt", [OUT])
    desc = out.describe().T[["mean", "std", "min", "max"]].round(3)
    text = (f"Step 16J - advection and jet daily diagnostics\nRows: {len(out)} (expected 7290)\n"
            f"Missing files: {missing if missing else 'none'}\n\n{desc.to_string()}\n")
    REPORT.write_text(text)
    (C.TERM_QC / "step16j_advection_jet_summary.json").write_text(
        json.dumps({"rows": len(out), "missing_files": missing, "completed_utc": C.now_utc()}, indent=2))
    print(text)


if __name__ == "__main__":
    main()
