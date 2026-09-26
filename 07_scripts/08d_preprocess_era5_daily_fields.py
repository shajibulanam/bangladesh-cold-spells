from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PL_IN = PROJECT_ROOT / "01_raw_data" / "era5" / "pressure_level"
SL_INSTANT_IN = PROJECT_ROOT / "01_raw_data" / "era5" / "single_level_extracted" / "instant"
SL_ACCUM_IN = PROJECT_ROOT / "01_raw_data" / "era5" / "single_level_extracted" / "accum"

OUT_ROOT = PROJECT_ROOT / "03_intermediate" / "era5_preprocessed"
PL_OUT = OUT_ROOT / "pressure_level_standardized"
SL_INSTANT_OUT = OUT_ROOT / "single_level_instant_standardized"
SL_ACCUM_OUT = OUT_ROOT / "single_level_accum_standardized"

REPORT_DIR = PROJECT_ROOT / "05_qc_reports" / "era5"
ADMIN_DIR = PROJECT_ROOT / "00_admin"

PL_OUT.mkdir(parents=True, exist_ok=True)
SL_INSTANT_OUT.mkdir(parents=True, exist_ok=True)
SL_ACCUM_OUT.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
ADMIN_DIR.mkdir(parents=True, exist_ok=True)

PREPROCESS_LOG = REPORT_DIR / "step9b_era5_preprocessing_log.csv"
PREPROCESS_REPORT = REPORT_DIR / "step9b_era5_preprocessing_report.txt"
PREPROCESS_SUMMARY = REPORT_DIR / "step9b_era5_preprocessing_summary.json"

GRAVITY = 9.80665


def detect_year_month(path: Path) -> tuple[int, int]:
    match = re.search(r"(19|20)\d{2}[_-](0[1-9]|1[0-2])", path.name)
    if match:
        text = match.group(0).replace("-", "_")
        year, month = text.split("_")
        return int(year), int(month)

    match = re.search(r"(19|20)\d{2}(0[1-9]|1[0-2])", path.name)
    if match:
        text = match.group(0)
        return int(text[:4]), int(text[4:6])

    raise ValueError(f"Could not detect year/month from {path.name}")


def standardize_common_coords(ds: xr.Dataset) -> xr.Dataset:
    rename = {}

    if "valid_time" in ds.coords:
        rename["valid_time"] = "time"

    if "pressure_level" in ds.coords:
        rename["pressure_level"] = "level_hpa"
    elif "level" in ds.coords:
        rename["level"] = "level_hpa"
    elif "isobaricInhPa" in ds.coords:
        rename["isobaricInhPa"] = "level_hpa"

    if "latitude" in ds.coords:
        rename["latitude"] = "lat"

    if "longitude" in ds.coords:
        rename["longitude"] = "lon"

    ds = ds.rename(rename)

    if "time" in ds.coords:
        ds["time"].attrs.update({
            "standard_name": "time",
            "description": "Daily 00 UTC ERA5 valid time"
        })

    if "lat" in ds.coords:
        ds["lat"].attrs.update({
            "standard_name": "latitude",
            "units": "degrees_north"
        })

    if "lon" in ds.coords:
        ds["lon"].attrs.update({
            "standard_name": "longitude",
            "units": "degrees_east"
        })

    if "level_hpa" in ds.coords:
        ds["level_hpa"].attrs.update({
            "long_name": "pressure level",
            "units": "hPa"
        })

    return ds


def add_global_attrs(ds: xr.Dataset, source_path: Path, group_name: str) -> xr.Dataset:
    ds.attrs.clear()

    ds.attrs.update({
        "title": f"Standardized ERA5 {group_name} daily 00 UTC fields",
        "source_file": str(source_path),
        "preprocessing_step": "Step 9B",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "raw_data_policy": "Raw ERA5 files were not modified.",
        "temporal_resolution": "daily 00 UTC fields, not daily means",
        "domain": "10N-70N, 30E-120E",
        "project": "Bangladesh persistent cold-spell analysis",
    })

    return ds


def compression_encoding(ds: xr.Dataset) -> dict:
    encoding = {}

    for name in ds.data_vars:
        encoding[name] = {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        }

    return encoding


def preprocess_pressure_level(path: Path) -> tuple[Path, dict]:
    year, month = detect_year_month(path)
    out_path = PL_OUT / f"ERA5_PL_STD_{year}_{month:02d}.nc"

    row = {
        "input_file": path.name,
        "output_file": out_path.name,
        "group": "pressure_level",
        "year": year,
        "month": month,
        "status": "started",
        "error": "",
    }

    with xr.open_dataset(path, decode_times=True) as raw:
        ds = standardize_common_coords(raw)

        out_vars = {}

        if "z" in ds:
            out_vars["geopotential_height_m"] = (ds["z"] / GRAVITY).astype("float32")
            out_vars["geopotential_height_m"].attrs.update({
                "long_name": "geopotential height",
                "units": "m",
                "conversion": "ERA5 geopotential divided by 9.80665",
            })

        if "t" in ds:
            out_vars["air_temperature_k"] = ds["t"].astype("float32")
            out_vars["air_temperature_k"].attrs.update({
                "long_name": "air temperature",
                "units": "K",
            })

        if "u" in ds:
            out_vars["u_wind_ms"] = ds["u"].astype("float32")
            out_vars["u_wind_ms"].attrs.update({
                "long_name": "eastward wind component",
                "units": "m s-1",
            })

        if "v" in ds:
            out_vars["v_wind_ms"] = ds["v"].astype("float32")
            out_vars["v_wind_ms"].attrs.update({
                "long_name": "northward wind component",
                "units": "m s-1",
            })

        if "vo" in ds:
            out_vars["relative_vorticity_s-1"] = ds["vo"].astype("float32")
            out_vars["relative_vorticity_s-1"].attrs.update({
                "long_name": "relative vorticity",
                "units": "s-1",
            })

        if "pv" in ds:
            out_vars["potential_vorticity_pvu"] = (ds["pv"] * 1_000_000.0).astype("float32")
            out_vars["potential_vorticity_pvu"].attrs.update({
                "long_name": "potential vorticity",
                "units": "PVU",
                "conversion": "ERA5 SI-unit PV multiplied by 1e6",
            })

        out = xr.Dataset(out_vars, coords=ds.coords)
        out = add_global_attrs(out, path, "pressure-level")

        out.to_netcdf(
            out_path,
            engine="netcdf4",
            encoding=compression_encoding(out),
        )

    row["status"] = "written"
    row["output_size_mb"] = round(out_path.stat().st_size / 1024 / 1024, 3)

    return out_path, row


def preprocess_single_instant(path: Path) -> tuple[Path, dict]:
    year, month = detect_year_month(path)
    out_path = SL_INSTANT_OUT / f"ERA5_SL_INSTANT_STD_{year}_{month:02d}.nc"

    row = {
        "input_file": path.name,
        "output_file": out_path.name,
        "group": "single_level_instant",
        "year": year,
        "month": month,
        "status": "started",
        "error": "",
    }

    with xr.open_dataset(path, decode_times=True) as raw:
        ds = standardize_common_coords(raw)

        out_vars = {}

        if "t2m" in ds:
            out_vars["t2m_c"] = (ds["t2m"] - 273.15).astype("float32")
            out_vars["t2m_c"].attrs.update({
                "long_name": "2 metre temperature",
                "units": "degree_Celsius",
                "conversion": "K minus 273.15",
            })

        if "msl" in ds:
            out_vars["msl_hpa"] = (ds["msl"] / 100.0).astype("float32")
            out_vars["msl_hpa"].attrs.update({
                "long_name": "mean sea-level pressure",
                "units": "hPa",
                "conversion": "Pa divided by 100",
            })

        if "sp" in ds:
            out_vars["surface_pressure_hpa"] = (ds["sp"] / 100.0).astype("float32")
            out_vars["surface_pressure_hpa"].attrs.update({
                "long_name": "surface pressure",
                "units": "hPa",
                "conversion": "Pa divided by 100",
            })

        if "u10" in ds:
            out_vars["u10_ms"] = ds["u10"].astype("float32")
            out_vars["u10_ms"].attrs.update({
                "long_name": "10 metre eastward wind",
                "units": "m s-1",
            })

        if "v10" in ds:
            out_vars["v10_ms"] = ds["v10"].astype("float32")
            out_vars["v10_ms"].attrs.update({
                "long_name": "10 metre northward wind",
                "units": "m s-1",
            })

        out = xr.Dataset(out_vars, coords=ds.coords)
        out = add_global_attrs(out, path, "single-level instant")

        out.to_netcdf(
            out_path,
            engine="netcdf4",
            encoding=compression_encoding(out),
        )

    row["status"] = "written"
    row["output_size_mb"] = round(out_path.stat().st_size / 1024 / 1024, 3)

    return out_path, row


def preprocess_single_accum(path: Path) -> tuple[Path, dict]:
    year, month = detect_year_month(path)
    out_path = SL_ACCUM_OUT / f"ERA5_SL_ACCUM_STD_{year}_{month:02d}.nc"

    row = {
        "input_file": path.name,
        "output_file": out_path.name,
        "group": "single_level_accum",
        "year": year,
        "month": month,
        "status": "started",
        "error": "",
    }

    with xr.open_dataset(path, decode_times=True) as raw:
        ds = standardize_common_coords(raw)

        out_vars = {}

        if "tisr" in ds:
            out_vars["tisr_j_m2"] = ds["tisr"].astype("float32")
            out_vars["tisr_j_m2"].attrs.update({
                "long_name": "TOA incident solar radiation",
                "units": "J m-2",
            })

        out = xr.Dataset(out_vars, coords=ds.coords)
        out = add_global_attrs(out, path, "single-level accumulated")

        out.to_netcdf(
            out_path,
            engine="netcdf4",
            encoding=compression_encoding(out),
        )

    row["status"] = "written"
    row["output_size_mb"] = round(out_path.stat().st_size / 1024 / 1024, 3)

    return out_path, row


def process_group(group_name: str, files: list[Path], function) -> list[dict]:
    rows = []

    total = len(files)

    for index, path in enumerate(files, start=1):
        print(f"[{group_name}] {index}/{total}: {path.name}", flush=True)

        try:
            out_path, row = function(path)

            # Quick readability check
            with xr.open_dataset(out_path) as ds:
                row["output_variables"] = ",".join(sorted(ds.data_vars))
                row["time_count"] = int(ds.sizes.get("time", 0))
                row["lat_count"] = int(ds.sizes.get("lat", 0))
                row["lon_count"] = int(ds.sizes.get("lon", 0))
                row["level_count"] = int(ds.sizes.get("level_hpa", 0)) if "level_hpa" in ds.sizes else 0

        except Exception as error:
            row = {
                "input_file": path.name,
                "output_file": "",
                "group": group_name,
                "year": None,
                "month": None,
                "status": "failed",
                "error": repr(error),
            }

        rows.append(row)

    return rows


def main() -> None:
    pl_files = sorted(PL_IN.glob("ERA5_PL_*.nc"))
    sl_instant_files = sorted(SL_INSTANT_IN.glob("ERA5_SL_instant_*.nc"))
    sl_accum_files = sorted(SL_ACCUM_IN.glob("ERA5_SL_accum_*.nc"))

    all_rows = []

    all_rows.extend(
        process_group(
            "pressure_level",
            pl_files,
            preprocess_pressure_level,
        )
    )

    all_rows.extend(
        process_group(
            "single_level_instant",
            sl_instant_files,
            preprocess_single_instant,
        )
    )

    all_rows.extend(
        process_group(
            "single_level_accum",
            sl_accum_files,
            preprocess_single_accum,
        )
    )

    log = pd.DataFrame(all_rows)
    log.to_csv(PREPROCESS_LOG, index=False)

    failed = log.loc[~log["status"].eq("written")]

    group_summary = (
        log.groupby("group")
        .agg(
            input_files=("input_file", "count"),
            written_files=("status", lambda s: int((s == "written").sum())),
            failed_files=("status", lambda s: int((s != "written").sum())),
            total_output_size_mb=("output_size_mb", "sum"),
        )
        .reset_index()
    )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_note": "Standardized ERA5 daily 00 UTC files created for regional analysis.",
        "pressure_level_output": str(PL_OUT),
        "single_level_instant_output": str(SL_INSTANT_OUT),
        "single_level_accum_output": str(SL_ACCUM_OUT),
        "total_files_written": int(log["status"].eq("written").sum()),
        "total_files_failed": int((~log["status"].eq("written")).sum()),
        "group_summary": group_summary.to_dict(orient="records"),
        "preprocessing_log": str(PREPROCESS_LOG),
    }

    PREPROCESS_SUMMARY.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "STEP 9B: ERA5 DAILY FIELD PREPROCESSING",
        "=" * 48,
        "Raw ERA5 files were not modified.",
        "Output files are standardized NetCDF files for regional analysis.",
        "",
        group_summary.to_string(index=False),
        "",
        f"Total files written: {summary['total_files_written']}",
        f"Total files failed: {summary['total_files_failed']}",
        "",
        f"Pressure-level output: {PL_OUT}",
        f"Single-level instant output: {SL_INSTANT_OUT}",
        f"Single-level accumulated output: {SL_ACCUM_OUT}",
        f"Preprocessing log: {PREPROCESS_LOG}",
    ]

    if not failed.empty:
        lines.extend(
            [
                "",
                "FAILED FILES:",
                failed[["group", "input_file", "error"]].to_string(index=False),
            ]
        )

    PREPROCESS_REPORT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))

    if not failed.empty:
        raise SystemExit("\nSTEP 9B PREPROCESSING FAILED. Check the report.")

    print("\nSTEP 9B-1 PASSED. ERA5 STANDARDIZED DAILY FILES CREATED.")


if __name__ == "__main__":
    main()
