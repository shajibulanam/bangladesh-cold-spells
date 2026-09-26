from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PL_DIR = PROJECT_ROOT / "01_raw_data" / "era5" / "pressure_level"
SL_INSTANT_DIR = PROJECT_ROOT / "01_raw_data" / "era5" / "single_level_extracted" / "instant"
SL_ACCUM_DIR = PROJECT_ROOT / "01_raw_data" / "era5" / "single_level_extracted" / "accum"

REPORT_DIR = PROJECT_ROOT / "05_qc_reports" / "era5"
OUT_DIR = PROJECT_ROOT / "03_intermediate" / "era5_inventory"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_CSV = OUT_DIR / "era5_oct_mar_extended_file_validation.csv"
MONTH_MATRIX_CSV = OUT_DIR / "era5_oct_mar_extended_month_matrix.csv"
ISSUES_CSV = REPORT_DIR / "era5_oct_mar_extended_validation_issues.csv"
SUMMARY_JSON = REPORT_DIR / "era5_oct_mar_extended_validation_summary.json"
REPORT_TXT = REPORT_DIR / "era5_oct_mar_extended_validation_report.txt"


EXPECTED_PL_VARIABLES = {"z", "t", "u", "v", "vo", "pv"}
EXPECTED_SL_INSTANT_VARIABLES = {"t2m", "msl", "u10", "v10", "sp"}
EXPECTED_SL_ACCUM_VARIABLES = {"tisr"}

EXPECTED_PRESSURE_LEVELS = [1000, 850, 700, 500, 300, 250, 200, 100, 70, 50, 30, 10]


def detect_year_month(path: Path) -> tuple[int | None, int | None]:
    match = re.search(r"(19|20)\d{2}[_-](0[1-9]|1[0-2])", path.name)
    if match:
        text = match.group(0).replace("-", "_")
        year, month = text.split("_")
        return int(year), int(month)
    return None, None


def expected_months() -> pd.DataFrame:
    records = []

    for winter_start_year in range(1985, 2025):
        for year, month in [
            (winter_start_year, 10),
            (winter_start_year, 11),
            (winter_start_year, 12),
            (winter_start_year + 1, 1),
            (winter_start_year + 1, 2),
            (winter_start_year + 1, 3),
        ]:
            records.append({
                "winter_start_year": winter_start_year,
                "year": year,
                "month": month,
            })

    return pd.DataFrame(records)


def inspect_file(path: Path, group: str, expected_vars: set[str], need_levels: bool) -> tuple[dict, list[dict]]:
    issues = []
    year, month = detect_year_month(path)

    row = {
        "group": group,
        "file_name": path.name,
        "year": year,
        "month": month,
        "file_size_mb": round(path.stat().st_size / 1024 / 1024, 3),
        "readable": False,
        "variables": "",
        "missing_variables": "",
        "pressure_levels": "",
        "pressure_levels_ok": "" if not need_levels else False,
        "time_count": np.nan,
        "time_count_expected": np.nan,
        "time_count_ok": False,
        "all_times_00utc": False,
        "lat_min": np.nan,
        "lat_max": np.nan,
        "lon_min": np.nan,
        "lon_max": np.nan,
        "domain_ok": False,
        "error": "",
    }

    try:
        with xr.open_dataset(path, decode_times=True) as ds:
            row["readable"] = True

            variables = set(ds.data_vars)
            missing = sorted(expected_vars - variables)

            row["variables"] = ",".join(sorted(variables))
            row["missing_variables"] = ",".join(missing)

            if missing:
                issues.append({
                    "group": group,
                    "file": path.name,
                    "issue": "missing_variables: " + ",".join(missing),
                })

            tname = "valid_time" if "valid_time" in ds.coords else "time"

            times = pd.to_datetime(ds[tname].values)
            row["time_count"] = len(times)

            if year and month:
                expected_days = pd.Period(f"{year}-{month:02d}", freq="M").days_in_month
                row["time_count_expected"] = expected_days
                row["time_count_ok"] = len(times) == expected_days

                if not row["time_count_ok"]:
                    issues.append({
                        "group": group,
                        "file": path.name,
                        "issue": f"time_count_not_equal_days_in_month: {len(times)}",
                    })

            row["all_times_00utc"] = bool((times.hour == 0).all())

            if not row["all_times_00utc"]:
                issues.append({
                    "group": group,
                    "file": path.name,
                    "issue": "not_all_times_are_00utc",
                })

            lat_name = "latitude" if "latitude" in ds.coords else "lat"
            lon_name = "longitude" if "longitude" in ds.coords else "lon"

            lat = ds[lat_name].values
            lon = ds[lon_name].values

            row["lat_min"] = float(np.nanmin(lat))
            row["lat_max"] = float(np.nanmax(lat))
            row["lon_min"] = float(np.nanmin(lon))
            row["lon_max"] = float(np.nanmax(lon))

            row["domain_ok"] = (
                abs(row["lat_min"] - 10.0) < 1e-6
                and abs(row["lat_max"] - 70.0) < 1e-6
                and abs(row["lon_min"] - 30.0) < 1e-6
                and abs(row["lon_max"] - 120.0) < 1e-6
            )

            if not row["domain_ok"]:
                issues.append({
                    "group": group,
                    "file": path.name,
                    "issue": f"domain_mismatch lat {row['lat_min']}..{row['lat_max']} lon {row['lon_min']}..{row['lon_max']}",
                })

            if need_levels:
                lname = None
                for candidate in ["pressure_level", "level", "isobaricInhPa"]:
                    if candidate in ds.coords:
                        lname = candidate
                        break

                if lname is None:
                    issues.append({
                        "group": group,
                        "file": path.name,
                        "issue": "missing_pressure_level_coordinate",
                    })
                else:
                    levels = [int(float(x)) for x in ds[lname].values]
                    row["pressure_levels"] = ",".join(str(x) for x in levels)
                    row["pressure_levels_ok"] = levels == EXPECTED_PRESSURE_LEVELS

                    if not row["pressure_levels_ok"]:
                        issues.append({
                            "group": group,
                            "file": path.name,
                            "issue": "pressure_levels_mismatch: " + ",".join(str(x) for x in levels),
                        })

    except Exception as error:
        row["error"] = repr(error)
        issues.append({
            "group": group,
            "file": path.name,
            "issue": "not_readable_as_netcdf: " + repr(error),
        })

    return row, issues


def validate_group(group: str, folder: Path, pattern: str, expected_vars: set[str], need_levels: bool):
    rows = []
    issues = []

    for path in sorted(folder.glob(pattern)):
        row, file_issues = inspect_file(path, group, expected_vars, need_levels)
        rows.append(row)
        issues.extend(file_issues)

    return rows, issues


def main() -> None:
    rows = []
    issues = []

    group_specs = [
        ("pressure_level", PL_DIR, "ERA5_PL_*.nc", EXPECTED_PL_VARIABLES, True),
        ("single_level_instant", SL_INSTANT_DIR, "ERA5_SL_instant_*.nc", EXPECTED_SL_INSTANT_VARIABLES, False),
        ("single_level_accum", SL_ACCUM_DIR, "ERA5_SL_accum_*.nc", EXPECTED_SL_ACCUM_VARIABLES, False),
    ]

    for spec in group_specs:
        group_rows, group_issues = validate_group(*spec)
        rows.extend(group_rows)
        issues.extend(group_issues)

    validation = pd.DataFrame(rows)
    validation.to_csv(VALIDATION_CSV, index=False)

    matrix_rows = []

    expected = expected_months()

    for _, expected_row in expected.iterrows():
        year = int(expected_row["year"])
        month = int(expected_row["month"])

        for group in ["pressure_level", "single_level_instant", "single_level_accum"]:
            subset = validation.loc[
                validation["group"].eq(group)
                & validation["year"].eq(year)
                & validation["month"].eq(month)
            ]

            readable_count = int(subset["readable"].sum()) if not subset.empty else 0

            status = "present_readable_unique" if len(subset) == 1 and readable_count == 1 else "problem"

            matrix_rows.append({
                "winter_start_year": int(expected_row["winter_start_year"]),
                "year": year,
                "month": month,
                "group": group,
                "file_count": len(subset),
                "readable_count": readable_count,
                "status": status,
            })

            if len(subset) == 0:
                issues.append({
                    "group": group,
                    "file": f"{year}_{month:02d}",
                    "issue": "expected_month_missing",
                })

            if len(subset) > 1:
                issues.append({
                    "group": group,
                    "file": f"{year}_{month:02d}",
                    "issue": f"duplicate_month_files: {len(subset)}",
                })

    matrix = pd.DataFrame(matrix_rows)
    matrix.to_csv(MONTH_MATRIX_CSV, index=False)

    issues_df = pd.DataFrame(issues, columns=["group", "file", "issue"]).drop_duplicates()
    issues_df.to_csv(ISSUES_CSV, index=False)

    group_summary = (
        validation
        .groupby("group")
        .agg(
            file_count=("file_name", "count"),
            readable_count=("readable", "sum"),
            unique_months=("year", lambda s: len(set(zip(s, validation.loc[s.index, "month"])))),
            time_count_all_ok=("time_count_ok", "all"),
            all_times_00utc=("all_times_00utc", "all"),
            domain_all_ok=("domain_ok", "all"),
        )
        .reset_index()
    )

    expected_files_per_group = 240
    ready = (
        issues_df.empty
        and matrix["status"].eq("present_readable_unique").all()
        and group_summary["file_count"].eq(expected_files_per_group).all()
    )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_ready_oct_mar_extended": bool(ready),
        "expected_files_per_group": expected_files_per_group,
        "issue_count": int(len(issues_df)),
        "groups": group_summary.to_dict(orient="records"),
        "validation_csv": str(VALIDATION_CSV),
        "month_matrix_csv": str(MONTH_MATRIX_CSV),
        "issues_csv": str(ISSUES_CSV),
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "STEP 9B-extra: ERA5 OCTOBER-MARCH EXTENDED VALIDATION",
        "=" * 68,
        f"Extended archive ready: {ready}",
        f"Expected files per group: {expected_files_per_group}",
        f"Validation issue count: {len(issues_df)}",
        "",
        group_summary.to_string(index=False),
        "",
        f"Validation CSV: {VALIDATION_CSV}",
        f"Month matrix CSV: {MONTH_MATRIX_CSV}",
        f"Issues CSV: {ISSUES_CSV}",
        f"Summary JSON: {SUMMARY_JSON}",
    ]

    REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))

    if not ready:
        raise SystemExit("\nExtended ERA5 validation failed. Check issues CSV.")

    print("\nSTEP 9B-extra PASSED. OCTOBER-MARCH ERA5 ARCHIVE IS READY.")


if __name__ == "__main__":
    main()
