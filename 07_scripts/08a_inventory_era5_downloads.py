from __future__ import annotations

import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRESSURE_DIR = PROJECT_ROOT / "01_raw_data" / "era5" / "pressure_level"
SINGLE_DIR = PROJECT_ROOT / "01_raw_data" / "era5" / "single_level"

OUT_DIR = PROJECT_ROOT / "03_intermediate" / "era5_inventory"
REPORT_DIR = PROJECT_ROOT / "05_qc_reports" / "era5"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

INVENTORY_CSV = OUT_DIR / "era5_file_inventory.csv"
MONTH_COVERAGE_CSV = OUT_DIR / "era5_month_coverage.csv"
ISSUES_CSV = REPORT_DIR / "era5_inventory_issues.csv"
SUMMARY_JSON = REPORT_DIR / "era5_inventory_summary.json"
REPORT_TXT = REPORT_DIR / "era5_inventory_report.txt"


def detect_year_month(path: Path) -> tuple[int | None, int | None]:
    text = path.name

    match = re.search(r"(19|20)\d{2}[_-](0[1-9]|1[0-2])", text)

    if match:
        year_month = match.group(0).replace("-", "_")
        year, month = year_month.split("_")
        return int(year), int(month)

    match = re.search(r"(19|20)\d{2}(0[1-9]|1[0-2])", text)

    if match:
        value = match.group(0)
        return int(value[:4]), int(value[4:6])

    return None, None


def classify_file(path: Path) -> str:
    try:
        if zipfile.is_zipfile(path):
            return "zip_archive"
    except Exception:
        pass

    try:
        with path.open("rb") as handle:
            signature = handle.read(8)

        if signature.startswith(b"\x89HDF"):
            return "netcdf_hdf5"

        if signature.startswith(b"CDF"):
            return "netcdf_classic"

    except Exception:
        return "unknown"

    return "unknown"


def open_dataset_summary(path: Path) -> dict:
    summary = {
        "readable": False,
        "variables": "",
        "coordinates": "",
        "dimensions": "",
        "time_start": "",
        "time_end": "",
        "time_count": np.nan,
        "latitude_min": np.nan,
        "latitude_max": np.nan,
        "longitude_min": np.nan,
        "longitude_max": np.nan,
        "grid_lat_count": np.nan,
        "grid_lon_count": np.nan,
        "pressure_levels": "",
        "pressure_level_count": np.nan,
        "has_missing_values": "",
        "error": "",
    }

    try:
        with xr.open_dataset(path, decode_times=True) as ds:
            summary["readable"] = True

            summary["variables"] = ",".join(sorted(list(ds.data_vars)))
            summary["coordinates"] = ",".join(sorted(list(ds.coords)))
            summary["dimensions"] = json.dumps({k: int(v) for k, v in ds.sizes.items()})

            time_name = None
            for candidate in ["valid_time", "time"]:
                if candidate in ds.coords:
                    time_name = candidate
                    break

            if time_name is not None:
                times = pd.to_datetime(ds[time_name].values)
                if len(times) > 0:
                    summary["time_start"] = str(times.min())
                    summary["time_end"] = str(times.max())
                    summary["time_count"] = int(len(times))

            lat_name = None
            for candidate in ["latitude", "lat"]:
                if candidate in ds.coords:
                    lat_name = candidate
                    break

            lon_name = None
            for candidate in ["longitude", "lon"]:
                if candidate in ds.coords:
                    lon_name = candidate
                    break

            if lat_name is not None:
                lat = ds[lat_name].values
                summary["latitude_min"] = float(np.nanmin(lat))
                summary["latitude_max"] = float(np.nanmax(lat))
                summary["grid_lat_count"] = int(len(lat))

            if lon_name is not None:
                lon = ds[lon_name].values
                summary["longitude_min"] = float(np.nanmin(lon))
                summary["longitude_max"] = float(np.nanmax(lon))
                summary["grid_lon_count"] = int(len(lon))

            level_name = None
            for candidate in ["pressure_level", "level", "isobaricInhPa"]:
                if candidate in ds.coords:
                    level_name = candidate
                    break

            if level_name is not None:
                levels = ds[level_name].values
                levels = [float(x) for x in levels]
                summary["pressure_levels"] = ",".join(str(int(x)) if float(x).is_integer() else str(x) for x in levels)
                summary["pressure_level_count"] = int(len(levels))

            missing_found = False
            for variable in ds.data_vars:
                arr = ds[variable]
                try:
                    if bool(arr.isnull().any().compute()):
                        missing_found = True
                        break
                except Exception:
                    if bool(arr.isnull().any()):
                        missing_found = True
                        break

            summary["has_missing_values"] = str(missing_found)

    except Exception as error:
        summary["error"] = repr(error)

    return summary


def inspect_zip(path: Path) -> dict:
    info = {
        "zip_member_count": 0,
        "zip_members": "",
        "zip_contains_nc": False,
    }

    try:
        with zipfile.ZipFile(path, "r") as zf:
            members = zf.namelist()
            info["zip_member_count"] = len(members)
            info["zip_members"] = "|".join(members)
            info["zip_contains_nc"] = any(member.endswith(".nc") for member in members)
    except Exception:
        pass

    return info


def main() -> None:
    rows = []
    issues = []

    for data_type, folder in [
        ("pressure_level", PRESSURE_DIR),
        ("single_level", SINGLE_DIR),
    ]:
        if not folder.exists():
            issues.append({
                "data_type": data_type,
                "file": str(folder),
                "issue": "folder_not_found",
            })
            continue

        files = sorted([p for p in folder.iterdir() if p.is_file()])

        for path in files:
            year, month = detect_year_month(path)
            file_type = classify_file(path)

            row = {
                "data_type": data_type,
                "file_name": path.name,
                "file_path": str(path),
                "file_size_mb": round(path.stat().st_size / 1024 / 1024, 3),
                "detected_year": year,
                "detected_month": month,
                "file_type": file_type,
            }

            if file_type == "zip_archive":
                zip_info = inspect_zip(path)
                row.update(zip_info)

                issues.append({
                    "data_type": data_type,
                    "file": path.name,
                    "issue": "zip_archive_named_as_nc_or_raw_file",
                })

            else:
                row.update({
                    "zip_member_count": 0,
                    "zip_members": "",
                    "zip_contains_nc": False,
                })

                nc_summary = open_dataset_summary(path)
                row.update(nc_summary)

                if not nc_summary["readable"]:
                    issues.append({
                        "data_type": data_type,
                        "file": path.name,
                        "issue": f"not_readable_as_netcdf: {nc_summary['error']}",
                    })

                if year is None or month is None:
                    issues.append({
                        "data_type": data_type,
                        "file": path.name,
                        "issue": "year_month_not_detected_from_filename",
                    })

            rows.append(row)

    inventory = pd.DataFrame(rows)
    inventory.to_csv(INVENTORY_CSV, index=False)

    # Expected months: Nov-Dec 1985, Jan-Feb 1986 ... Nov-Dec 2024, Jan-Feb 2025
    expected = []

    for winter_start in range(1985, 2025):
        for year, month in [
            (winter_start, 11),
            (winter_start, 12),
            (winter_start + 1, 1),
            (winter_start + 1, 2),
        ]:
            expected.append({
                "winter_start_year": winter_start,
                "year": year,
                "month": month,
            })

    expected_df = pd.DataFrame(expected)

    coverage_rows = []

    for _, row in expected_df.iterrows():
        year = int(row["year"])
        month = int(row["month"])

        for data_type in ["pressure_level", "single_level"]:
            subset = inventory.loc[
                inventory["data_type"].eq(data_type)
                & inventory["detected_year"].eq(year)
                & inventory["detected_month"].eq(month)
            ]

            coverage_rows.append({
                "winter_start_year": int(row["winter_start_year"]),
                "year": year,
                "month": month,
                "data_type": data_type,
                "file_count": len(subset),
                "normal_netcdf_count": int(subset["file_type"].isin(["netcdf_hdf5", "netcdf_classic"]).sum()) if not subset.empty else 0,
                "zip_count": int(subset["file_type"].eq("zip_archive").sum()) if not subset.empty else 0,
                "status": "present" if len(subset) > 0 else "missing",
            })

            if len(subset) == 0:
                issues.append({
                    "data_type": data_type,
                    "file": f"{year}_{month:02d}",
                    "issue": "expected_month_missing",
                })

    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(MONTH_COVERAGE_CSV, index=False)

    issues_df = pd.DataFrame(issues, columns=["data_type", "file", "issue"])
    issues_df.to_csv(ISSUES_CSV, index=False)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pressure_folder": str(PRESSURE_DIR),
        "single_folder": str(SINGLE_DIR),
        "total_files": int(len(inventory)),
        "pressure_level_files": int(inventory["data_type"].eq("pressure_level").sum()) if not inventory.empty else 0,
        "single_level_files": int(inventory["data_type"].eq("single_level").sum()) if not inventory.empty else 0,
        "zip_files": int(inventory["file_type"].eq("zip_archive").sum()) if not inventory.empty else 0,
        "normal_netcdf_files": int(inventory["file_type"].isin(["netcdf_hdf5", "netcdf_classic"]).sum()) if not inventory.empty else 0,
        "missing_expected_month_records": int(issues_df["issue"].eq("expected_month_missing").sum()) if not issues_df.empty else 0,
        "issue_count": int(len(issues_df)),
        "inventory_csv": str(INVENTORY_CSV),
        "coverage_csv": str(MONTH_COVERAGE_CSV),
        "issues_csv": str(ISSUES_CSV),
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "STEP 9A-1: ERA5 DOWNLOAD INVENTORY",
        "=" * 42,
        f"Pressure-level folder: {PRESSURE_DIR}",
        f"Single-level folder: {SINGLE_DIR}",
        "",
        f"Total files: {summary['total_files']}",
        f"Pressure-level files: {summary['pressure_level_files']}",
        f"Single-level files: {summary['single_level_files']}",
        f"Normal NetCDF files: {summary['normal_netcdf_files']}",
        f"ZIP archive files: {summary['zip_files']}",
        f"Missing expected month records: {summary['missing_expected_month_records']}",
        f"Total inventory issues: {summary['issue_count']}",
        "",
        f"Inventory CSV: {INVENTORY_CSV}",
        f"Month coverage CSV: {MONTH_COVERAGE_CSV}",
        f"Issues CSV: {ISSUES_CSV}",
    ]

    REPORT_TXT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n".join(report_lines))
    print("\nSTEP 9A-1 INVENTORY COMPLETE.")


if __name__ == "__main__":
    main()
