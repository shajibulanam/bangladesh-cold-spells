"""
Step 16A - Daily ERA5 box-mean extraction for the termination analysis.

Purpose
    Build daily (00 UTC) area-weighted box means for the Bangladesh and South Asia
    boxes for every October-March day, winters 1985/86-2024/25, from the existing
    Step 9B standardized ERA5 files. These daily series are the ERA5 covariates for
    the Step 16 leakage-free persistence tests and discrete-time hazard model.

Inputs (read only, never modified)
    03_intermediate/era5_preprocessed/pressure_level_standardized/ERA5_PL_STD_YYYY_MM.nc
    03_intermediate/era5_preprocessed/single_level_instant_standardized/ERA5_SL_INSTANT_STD_YYYY_MM.nc

Outputs
    00_admin/step16a_era5_box_daily_policy.json
    03_intermediate/era5_box_daily/era5_box_daily_oct_mar.csv
    05_qc_reports/termination/step16a_era5_box_extraction_report.txt
    05_qc_reports/termination/step16a_era5_box_extraction_summary.json
    05_qc_reports/termination/step16a_missing_months.csv
    00_admin/step16a_output_sha256.txt

Run from the project root:
    python3 07_scripts/15a_extract_era5_box_daily.py 2>&1 | tee 09_logs/step16a_extract_era5_box_daily.log
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PL_DIR = PROJECT_ROOT / "03_intermediate" / "era5_preprocessed" / "pressure_level_standardized"
SL_DIR = PROJECT_ROOT / "03_intermediate" / "era5_preprocessed" / "single_level_instant_standardized"

OUT_DIR = PROJECT_ROOT / "03_intermediate" / "era5_box_daily"
OUT_CSV = OUT_DIR / "era5_box_daily_oct_mar.csv"

QC_DIR = PROJECT_ROOT / "05_qc_reports" / "termination"
REPORT_TXT = QC_DIR / "step16a_era5_box_extraction_report.txt"
SUMMARY_JSON = QC_DIR / "step16a_era5_box_extraction_summary.json"
MISSING_CSV = QC_DIR / "step16a_missing_months.csv"

POLICY_JSON = PROJECT_ROOT / "00_admin" / "step16a_era5_box_daily_policy.json"
SHA_TXT = PROJECT_ROOT / "00_admin" / "step16a_output_sha256.txt"

FIRST_WINTER = 1985
LAST_WINTER = 2024
MONTHS = [10, 11, 12, 1, 2, 3]

# Same box definitions as Step 9G / Step 9H (08l_compute_era5_bootstrap_significance.py)
BOXES = {
    "bd": {"name": "Bangladesh box", "lat_min": 20.5, "lat_max": 26.8, "lon_min": 88.0, "lon_max": 93.0},
    "sa": {"name": "South Asia box", "lat_min": 15.0, "lat_max": 35.0, "lon_min": 70.0, "lon_max": 100.0},
}

PL_FIELDS = {
    "geopotential_height_m": [1000, 850, 500, 200],
    "air_temperature_k": [1000, 850, 700, 500],
    "u_wind_ms": [850, 500, 200],
    "v_wind_ms": [850, 500, 200],
}
SL_FIELDS = ["t2m_c", "msl_hpa", "u10_ms", "v10_ms"]

POLICY = {
    "step": "Step 16A",
    "title": "Daily ERA5 box-mean extraction for the termination analysis",
    "inputs": {
        "pressure_level": str(PL_DIR.relative_to(PROJECT_ROOT)),
        "single_level_instant": str(SL_DIR.relative_to(PROJECT_ROOT)),
    },
    "temporal_resolution": "daily 00 UTC snapshots (06 local time), not daily means",
    "season": "October-March",
    "winters": f"{FIRST_WINTER}/{str(FIRST_WINTER + 1)[-2:]} to {LAST_WINTER}/{str(LAST_WINTER + 1)[-2:]}",
    "boxes": BOXES,
    "area_weighting": "cosine-latitude weighted mean over all grid points inside the box",
    "pressure_level_fields": PL_FIELDS,
    "single_level_fields": SL_FIELDS,
    "derived_fields": {"wspd10_ms": "10-m wind speed computed at each grid point, then box-averaged"},
    "anomalies": "Not computed here. Calendar-day anomalies are computed in Step 16B/16D from these series.",
    "raw_data_policy": "Input NetCDF files are read only and never modified.",
}


def box_subset(ds: xr.Dataset, box: dict) -> xr.Dataset:
    lat = ds["lat"].values
    if lat[0] > lat[-1]:
        lat_slice = slice(box["lat_max"], box["lat_min"])
    else:
        lat_slice = slice(box["lat_min"], box["lat_max"])
    return ds.sel(lat=lat_slice, lon=slice(box["lon_min"], box["lon_max"]))


def weighted_box_mean(da: xr.DataArray) -> np.ndarray:
    weights = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(weights).mean(dim=("lat", "lon")).load().values


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    started = datetime.now(timezone.utc)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)

    policy = dict(POLICY)
    policy["created_utc"] = started.isoformat()
    POLICY_JSON.write_text(json.dumps(policy, indent=2))
    print(f"Policy written: {POLICY_JSON.relative_to(PROJECT_ROOT)}")

    frames: list[pd.DataFrame] = []
    missing: list[dict] = []
    missing_levels: set[str] = set()

    for winter in range(FIRST_WINTER, LAST_WINTER + 1):
        for month in MONTHS:
            year = winter if month >= 10 else winter + 1
            pl_file = PL_DIR / f"ERA5_PL_STD_{year}_{month:02d}.nc"
            sl_file = SL_DIR / f"ERA5_SL_INSTANT_STD_{year}_{month:02d}.nc"

            absent = [p.name for p in (pl_file, sl_file) if not p.exists()]
            if absent:
                missing.append({"winter_start_year": winter, "year": year, "month": month,
                                "missing_files": ";".join(absent)})
                print(f"MISSING {year}-{month:02d}: {', '.join(absent)}")
                continue

            columns: dict[str, np.ndarray] = {}
            with xr.open_dataset(pl_file) as pl, xr.open_dataset(sl_file) as sl:
                available_levels = {float(v) for v in pl["level_hpa"].values}
                pl_times = pd.to_datetime(pl["time"].values).normalize()
                sl_times = pd.to_datetime(sl["time"].values).normalize()
                if not pl_times.equals(sl_times):
                    raise RuntimeError(f"Time mismatch between PL and SL files for {year}-{month:02d}")

                for key, box in BOXES.items():
                    pl_box = box_subset(pl, box)
                    sl_box = box_subset(sl, box)

                    for var, levels in PL_FIELDS.items():
                        for level in levels:
                            if float(level) not in available_levels:
                                missing_levels.add(f"{var}_{level}")
                                continue
                            columns[f"{key}_{var}_{level}"] = weighted_box_mean(
                                pl_box[var].sel(level_hpa=float(level)))

                    for var in SL_FIELDS:
                        if var in sl_box:
                            columns[f"{key}_{var}"] = weighted_box_mean(sl_box[var])

                    if "u10_ms" in sl_box and "v10_ms" in sl_box:
                        speed = np.sqrt(sl_box["u10_ms"] ** 2 + sl_box["v10_ms"] ** 2)
                        columns[f"{key}_wspd10_ms"] = weighted_box_mean(speed)

            frame = pd.DataFrame(columns, index=pl_times)
            frame.index.name = "date"
            frames.append(frame)
            print(f"done {year}-{month:02d}  days={len(frame)}")

    if not frames:
        raise RuntimeError("No ERA5 months were processed. Check the input directories.")

    out = pd.concat(frames).sort_index()
    duplicated = int(out.index.duplicated().sum())
    out = out[~out.index.duplicated(keep="first")]
    out.to_csv(OUT_CSV, float_format="%.4f")

    pd.DataFrame(missing, columns=["winter_start_year", "year", "month", "missing_files"]).to_csv(
        MISSING_CSV, index=False)

    # Validation
    expected_days = sum(
        len(pd.date_range(f"{w}-10-01", f"{w + 1}-03-31")) for w in range(FIRST_WINTER, LAST_WINTER + 1))
    nan_counts = out.isna().sum()
    t2m_range = (float(out["bd_t2m_c"].min()), float(out["bd_t2m_c"].max())) if "bd_t2m_c" in out else None
    issues = []
    if len(out) != expected_days:
        issues.append(f"Row count {len(out)} differs from expected {expected_days}.")
    if nan_counts.sum() > 0:
        issues.append("Some columns contain missing values: "
                      + ", ".join(f"{c}={n}" for c, n in nan_counts[nan_counts > 0].items()))
    if missing_levels:
        issues.append("Requested pressure levels not present: " + ", ".join(sorted(missing_levels)))
    if duplicated:
        issues.append(f"{duplicated} duplicated dates removed.")

    digest = sha256(OUT_CSV)
    SHA_TXT.write_text(f"{digest}  {OUT_CSV.relative_to(PROJECT_ROOT)}\n")

    summary = {
        "step": "Step 16A",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "output_csv": str(OUT_CSV.relative_to(PROJECT_ROOT)),
        "rows": int(len(out)),
        "expected_rows": int(expected_days),
        "columns": list(out.columns),
        "first_date": str(out.index.min().date()),
        "last_date": str(out.index.max().date()),
        "missing_month_count": len(missing),
        "bangladesh_box_t2m_range_c": t2m_range,
        "issues": issues,
        "sha256": digest,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    lines = [
        "Step 16A - Daily ERA5 box-mean extraction",
        "=" * 44,
        f"Completed (UTC): {summary['completed_utc']}",
        f"Output: {summary['output_csv']}",
        f"Rows: {summary['rows']} (expected {summary['expected_rows']})",
        f"Period: {summary['first_date']} to {summary['last_date']}",
        f"Columns ({len(out.columns)}): {', '.join(out.columns)}",
        f"Missing months: {len(missing)}",
        f"Bangladesh-box T2m range (degC): {t2m_range}",
        f"SHA-256: {digest}",
        "",
        "Validation issues:" if issues else "Validation: PASS - no issues found.",
    ] + [f"  - {i}" for i in issues]
    REPORT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
