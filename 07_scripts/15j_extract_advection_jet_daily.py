"""
Step 16J - Daily cold-air advection and subtropical-jet diagnostics for every October-March day.

Why
    Steps 10B (advection) and 10C (jet/waveguide) saved only event-window means, so they cannot be
    used for day-by-day onset timelines or in the Step 16D hazard model. This step derives the same
    physical quantities as DAILY series from the existing Step 9B standardized files.

Diagnostics (00 UTC)
    850-hPa horizontal temperature advection  -(u dT/dx + v dT/dy), K day-1, computed on the native
    grid and averaged (cosine-latitude weights) over the valid (above-ground) grid points of:
        bd    Bangladesh box         20.5-26.8N, 88-93E
        nigp  NW Indo-Gangetic box   25-32N, 75-88E   (upstream cold-air source region)
    Subtropical jet at 200 hPa over 70-100E: the longitude-mean zonal wind profile between 15 and
    45N; jet-core latitude (latitude of the maximum) and jet-core speed (the maximum).

Terrain mask (Step 18 revision, October 2026)
    850 hPa lies below ground over the Himalaya and Tibet, which fill about 40% of the NW IGP box; ERA5
    values there are extrapolated. Grid points whose ETOPO1 cell-mean elevation exceeds the threshold
    (default 1,000 m, 15_termination_common.TERRAIN_MAX_M), and their four neighbours (used by the centred
    differences), are excluded before the box averages. The previous, unmasked series is written alongside.

Inputs (read only): 03_intermediate/era5_preprocessed/pressure_level_standardized/ERA5_PL_STD_YYYY_MM.nc
                    01_raw_data/geospatial/etopo1_south_asia_0p1deg.nc
Output: 03_intermediate/era5_box_daily/era5_mechanism_daily_oct_mar.csv           (terrain-masked, primary)
        03_intermediate/era5_box_daily/era5_mechanism_daily_oct_mar_unmasked.csv  (previous definition)
        (+ policy, report, checksums)
    With --terrain-max-m 1500 (pre-stated sensitivity) the masked series goes to
        era5_mechanism_daily_oct_mar_terrain1500.csv and the primary file is left untouched.

Run from the project root:
    python3 07_scripts/15j_extract_advection_jet_daily.py 2>&1 | tee 09_logs/step18_15j_advection_jet.log
    python3 07_scripts/15j_extract_advection_jet_daily.py --terrain-max-m 1500 2>&1 | tee 09_logs/step18_15j_terrain1500.log
"""
from __future__ import annotations

import argparse
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
    "terrain_mask": {"source": "ETOPO1 cell-mean elevation on the ERA5 grid (01_raw_data/geospatial)",
                     "threshold_m": None, "neighbours": "four neighbours of every masked point also excluded",
                     "applies_to": "850-hPa advection box means; the 200-hPa jet is not masked",
                     "revision": "Step 18 (October 2026); unmasked series kept as *_unmasked.csv"},
}


def box(da: xr.DataArray, lat0: float, lat1: float, lon0: float, lon1: float) -> xr.DataArray:
    lat = da["lat"].values
    lat_slice = slice(lat1, lat0) if lat[0] > lat[-1] else slice(lat0, lat1)
    return da.sel(lat=lat_slice, lon=slice(lon0, lon1))


def wmean(da: xr.DataArray) -> np.ndarray:
    w = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(w).mean(dim=("lat", "lon")).values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain-max-m", type=float, default=C.TERRAIN_MAX_M,
                    help="ETOPO1 cell-mean elevation above which 850-hPa points are masked (default %(default)s)")
    a = ap.parse_args()
    primary = abs(a.terrain_max_m - C.TERRAIN_MAX_M) < 1e-9
    out_masked = OUT if primary else OUT.with_name(f"{OUT.stem}_terrain{a.terrain_max_m:.0f}.csv")
    out_unmasked = OUT.with_name(f"{OUT.stem}_unmasked.csv")
    C.ensure_dirs()
    policy = dict(POLICY)
    policy["terrain_mask"] = dict(POLICY["terrain_mask"], threshold_m=a.terrain_max_m)
    C.write_policy("step16j_advection_jet_policy.json" if primary else
                   f"step18_advection_terrain{a.terrain_max_m:.0f}_policy.json", policy)
    frames, frames_raw, missing = [], [], []
    valid_frac = {}
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
                raw = {f"{k}_adv850_k_day": wmean(box(adv, *b)) for k, b in ADV_BOXES.items()}
                bad = C.dilate_mask(C.below_ground_mask(T["lat"].values, T["lon"].values, 850.0,
                                                        a.terrain_max_m))
                valid = xr.DataArray(~bad, coords={"lat": T["lat"], "lon": T["lon"]}, dims=("lat", "lon"))
                adv_m = adv.where(valid)
                cols = {f"{k}_adv850_k_day": wmean(box(adv_m, *b)) for k, b in ADV_BOXES.items()}
                if not valid_frac:
                    for k, b in ADV_BOXES.items():
                        vb = box(valid, *b)
                        w = np.cos(np.deg2rad(vb["lat"])) * xr.ones_like(vb, dtype=float)
                        valid_frac[k] = float((w * vb).sum() / w.sum())

                uj = ds["u_wind_ms"].sel(level_hpa=JET_LEVEL)
                uj = box(uj, JET_LAT[0], JET_LAT[1], JET_LON[0], JET_LON[1]).mean("lon").load()
                idx = uj.argmax("lat")
                cols["jet200_core_lat_70_100e"] = uj["lat"].values[idx.values]
                cols["jet200_core_speed_70_100e"] = uj.max("lat").values
                times = pd.to_datetime(ds["time"].values).normalize()
            df = pd.DataFrame(cols, index=times)
            df.index.name = "date"
            frames.append(df)
            jet = {c: cols[c] for c in cols if c.startswith("jet")}
            dr = pd.DataFrame({**raw, **jet}, index=times)
            dr.index.name = "date"
            frames_raw.append(dr)
            print(f"done {year}-{month:02d}")
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated()]
    out_raw = pd.concat(frames_raw).sort_index()
    out_raw = out_raw[~out_raw.index.duplicated()][out.columns]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_masked, float_format="%.5f")
    out_raw.to_csv(out_unmasked, float_format="%.5f")
    C.write_checksums("step16j_output_sha256.txt" if primary else
                      f"step18_advection_terrain{a.terrain_max_m:.0f}_sha256.txt", [out_masked, out_unmasked])
    desc = out.describe().T[["mean", "std", "min", "max"]].round(3)
    djf = out.index.month.isin([12, 1, 2])
    cmp_lines = []
    for c in [c for c in out.columns if "adv850" in c]:
        r = np.corrcoef(out.loc[djf, c], out_raw.loc[djf, c])[0, 1]
        cmp_lines.append(f"  {c:20s} DJF mean {out.loc[djf, c].mean():+.3f} (unmasked {out_raw.loc[djf, c].mean():+.3f}),"
                         f" SD {out.loc[djf, c].std():.3f} (unmasked {out_raw.loc[djf, c].std():.3f}),"
                         f" correlation with unmasked {r:.3f}")
    text = (f"Step 16J - advection and jet daily diagnostics (terrain-masked, Step 18)\n"
            f"Terrain threshold: {a.terrain_max_m:.0f} m (ETOPO1 cell mean; masked points and their 4 neighbours removed)\n"
            f"Valid (unmasked) area fraction: " + ", ".join(f"{k} {v:.1%}" for k, v in valid_frac.items()) + "\n"
            f"Output: {out_masked.name}; unmasked copy: {out_unmasked.name}\n"
            f"Rows: {len(out)} (expected 7290)\nMissing files: {missing if missing else 'none'}\n\n"
            f"Masked versus unmasked (DJF):\n" + "\n".join(cmp_lines) + f"\n\n{desc.to_string()}\n")
    report = REPORT if primary else REPORT.with_name(f"step18_advection_terrain{a.terrain_max_m:.0f}_report.txt")
    report.write_text(text)
    if primary:
        (C.TERM_QC / "step16j_advection_jet_summary.json").write_text(
            json.dumps({"rows": len(out), "missing_files": missing, "terrain_threshold_m": a.terrain_max_m,
                        "valid_area_fraction": valid_frac, "completed_utc": C.now_utc()}, indent=2))
    print(text)


if __name__ == "__main__":
    main()
