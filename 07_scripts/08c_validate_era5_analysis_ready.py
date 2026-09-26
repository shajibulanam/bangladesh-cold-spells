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

OUT_DIR = PROJECT_ROOT / "03_intermediate" / "era5_inventory"
REPORT_DIR = PROJECT_ROOT / "05_qc_reports" / "era5"
ADMIN_DIR = PROJECT_ROOT / "00_admin"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
ADMIN_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_CSV = OUT_DIR / "era5_analysis_ready_file_validation.csv"
MONTH_MATRIX_CSV = OUT_DIR / "era5_analysis_ready_month_matrix.csv"
ISSUES_CSV = REPORT_DIR / "era5_analysis_ready_issues.csv"
SUMMARY_JSON = REPORT_DIR / "era5_analysis_ready_summary.json"
REPORT_TXT = REPORT_DIR / "era5_analysis_ready_report.txt"
DATASET_REGISTRY = ADMIN_DIR / "step9a_era5_dataset_registry.json"


EXPECTED_PL_VARIABLES = {
    "z",
    "t",
    "u",
    "v",
    "vo",
    "pv",
}

EXPECTED_SL_INSTANT_VARIABLES = {
    "t2m",
    "msl",
    "u10",
    "v10",
    "sp",
}

EXPECTED_SL_ACCUM_VARIABLES = {
    "tisr",
}

EXPECTED_PRESSURE_LEVELS = [
    1000,
    850,
    700,
    500,
    300,
    250,
    200,
    100,
    70,
    50,
    30,
    10,
]

EXPECTED_LAT_MIN = 10.0
EXPECTED_LAT_MAX = 70.0
EXPECTED_LON_MIN = 30.0
EXPECTED_LON_MAX = 120.0


def detect_year_month(path: Path) -> tuple[int | None, int | None]:
    text = path.name

    match = re.search(r"(19|20)\d{2}[_-](0[1-9]|1[0-2])", text)
    if match:
        ym = match.group(0).replace("-", "_")
        year, month = ym.split("_")
        return int(year), int(month)

    match = re.search(r"(19|20)\d{2}(0[1-9]|1[0-2])", text)
    if match:
        value = match.group(0)
        return int(value[:4]), int(value[4:6])

    return None, None


def expected_month_records() -> pd.DataFrame:
    records = []

    for winter_start_year in range(1985, 2025):
        for year, month in [
            (winter_start_year, 11),
            (winter_start_year, 12),
            (winter_start_year + 1, 1),
            (winter_start_year + 1, 2),
        ]:
            records.append(
                {
                    "winter_start_year": winter_start_year,
                    "year": year,
                    "month": month,
                }
            )

    return pd.DataFrame(records)


def time_coord_name(ds: xr.Dataset) -> str | None:
    for name in ["valid_time", "time"]:
        if name in ds.coords:
            return name
    return None


def level_coord_name(ds: xr.Dataset) -> str | None:
    for name in ["pressure_level", "level", "isobaricInhPa"]:
        if name in ds.coords:
            return name
    return None


def inspect_one_file(
    path: Path,
    data_group: str,
    expected_variables: set[str],
    expect_pressure_levels: bool,
) -> tuple[dict, list[dict]]:
    issues = []
    year, month = detect_year_month(path)

    row = {
        "data_group": data_group,
        "file_name": path.name,
        "file_path": str(path),
        "year": year,
        "month": month,
        "file_size_mb": round(path.stat().st_size / 1024 / 1024, 3),
        "readable": False,
        "variables": "",
        "variable_count": np.nan,
        "expected_variables_present": False,
        "unexpected_variables": "",
        "missing_variables": "",
        "pressure_levels": "",
        "pressure_levels_ok": "" if not expect_pressure_levels else False,
        "time_start": "",
        "time_end": "",
        "time_count": np.nan,
        "time_count_expected": np.nan,
        "time_count_ok": False,
        "all_times_00utc": False,
        "latitude_min": np.nan,
        "latitude_max": np.nan,
        "longitude_min": np.nan,
        "longitude_max": np.nan,
        "domain_ok": False,
        "lat_count": np.nan,
        "lon_count": np.nan,
        "error": "",
    }

    if year is None or month is None:
        issues.append(
            {
                "data_group": data_group,
                "file": path.name,
                "issue": "year_month_not_detected_from_filename",
            }
        )

    try:
        with xr.open_dataset(path, decode_times=True) as ds:
            row["readable"] = True

            variables = set(str(v) for v in ds.data_vars)
            row["variables"] = ",".join(sorted(variables))
            row["variable_count"] = len(variables)

            missing_variables = sorted(expected_variables - variables)
            unexpected_variables = sorted(variables - expected_variables)

            row["expected_variables_present"] = len(missing_variables) == 0
            row["missing_variables"] = ",".join(missing_variables)
            row["unexpected_variables"] = ",".join(unexpected_variables)

            if missing_variables:
                issues.append(
                    {
                        "data_group": data_group,
                        "file": path.name,
                        "issue": f"missing_variables: {','.join(missing_variables)}",
                    }
                )

            if unexpected_variables:
                issues.append(
                    {
                        "data_group": data_group,
                        "file": path.name,
                        "issue": f"unexpected_variables: {','.join(unexpected_variables)}",
                    }
                )

            tname = time_coord_name(ds)
            if tname is None:
                issues.append(
                    {
                        "data_group": data_group,
                        "file": path.name,
                        "issue": "missing_time_coordinate",
                    }
                )
            else:
                times = pd.to_datetime(ds[tname].values)
                row["time_count"] = int(len(times))

                if len(times) > 0:
                    row["time_start"] = str(times.min())
                    row["time_end"] = str(times.max())

                    expected_days = pd.Period(
                        f"{year}-{month:02d}",
                        freq="M",
                    ).days_in_month if year is not None and month is not None else np.nan

                    row["time_count_expected"] = expected_days

                    row["time_count_ok"] = (
                        len(times) == expected_days
                        if not pd.isna(expected_days)
                        else False
                    )

                    # ERA5 files may be timezone-naive; checking hour is enough here.
                    row["all_times_00utc"] = bool((times.hour == 0).all())

                    if not row["time_count_ok"]:
                        issues.append(
                            {
                                "data_group": data_group,
                                "file": path.name,
                                "issue": f"time_count_not_equal_days_in_month: {len(times)}",
                            }
                        )

                    if not row["all_times_00utc"]:
                        issues.append(
                            {
                                "data_group": data_group,
                                "file": path.name,
                                "issue": "not_all_times_are_00utc",
                            }
                        )

            if "latitude" in ds.coords:
                lat = ds["latitude"].values
            elif "lat" in ds.coords:
                lat = ds["lat"].values
            else:
                lat = None

            if "longitude" in ds.coords:
                lon = ds["longitude"].values
            elif "lon" in ds.coords:
                lon = ds["lon"].values
            else:
                lon = None

            if lat is None or lon is None:
                issues.append(
                    {
                        "data_group": data_group,
                        "file": path.name,
                        "issue": "missing_latitude_or_longitude_coordinate",
                    }
                )
            else:
                row["latitude_min"] = float(np.nanmin(lat))
                row["latitude_max"] = float(np.nanmax(lat))
                row["longitude_min"] = float(np.nanmin(lon))
                row["longitude_max"] = float(np.nanmax(lon))
                row["lat_count"] = int(len(lat))
                row["lon_count"] = int(len(lon))

                row["domain_ok"] = (
                    abs(row["latitude_min"] - EXPECTED_LAT_MIN) < 1e-6
                    and abs(row["latitude_max"] - EXPECTED_LAT_MAX) < 1e-6
                    and abs(row["longitude_min"] - EXPECTED_LON_MIN) < 1e-6
                    and abs(row["longitude_max"] - EXPECTED_LON_MAX) < 1e-6
                )

                if not row["domain_ok"]:
                    issues.append(
                        {
                            "data_group": data_group,
                            "file": path.name,
                            "issue": (
                                "domain_mismatch: "
                                f"lat {row['latitude_min']}..{row['latitude_max']}, "
                                f"lon {row['longitude_min']}..{row['longitude_max']}"
                            ),
                        }
                    )

            if expect_pressure_levels:
                lname = level_coord_name(ds)

                if lname is None:
                    issues.append(
                        {
                            "data_group": data_group,
                            "file": path.name,
                            "issue": "missing_pressure_level_coordinate",
                        }
                    )
                else:
                    levels = [int(float(x)) for x in ds[lname].values]
                    row["pressure_levels"] = ",".join(str(x) for x in levels)
                    row["pressure_levels_ok"] = levels == EXPECTED_PRESSURE_LEVELS

                    if not row["pressure_levels_ok"]:
                        issues.append(
                            {
                                "data_group": data_group,
                                "file": path.name,
                                "issue": (
                                    "pressure_levels_mismatch: "
                                    + ",".join(str(x) for x in levels)
                                ),
                            }
                        )

    except Exception as error:
        row["error"] = repr(error)
        issues.append(
            {
                "data_group": data_group,
                "file": path.name,
                "issue": f"not_readable_as_netcdf: {repr(error)}",
            }
        )

    return row, issues


def validate_group(
    data_group: str,
    folder: Path,
    glob_pattern: str,
    expected_variables: set[str],
    expect_pressure_levels: bool,
) -> tuple[list[dict], list[dict]]:
    rows = []
    issues = []

    if not folder.exists():
        issues.append(
            {
                "data_group": data_group,
                "file": str(folder),
                "issue": "folder_not_found",
            }
        )
        return rows, issues

    files = sorted(folder.glob(glob_pattern))

    for path in files:
        row, file_issues = inspect_one_file(
            path=path,
            data_group=data_group,
            expected_variables=expected_variables,
            expect_pressure_levels=expect_pressure_levels,
        )
        rows.append(row)
        issues.extend(file_issues)

    return rows, issues


def main() -> None:
    all_rows = []
    all_issues = []

    checks = [
        {
            "data_group": "pressure_level",
            "folder": PL_DIR,
            "glob_pattern": "ERA5_PL_*.nc",
            "expected_variables": EXPECTED_PL_VARIABLES,
            "expect_pressure_levels": True,
        },
        {
            "data_group": "single_level_instant",
            "folder": SL_INSTANT_DIR,
            "glob_pattern": "ERA5_SL_instant_*.nc",
            "expected_variables": EXPECTED_SL_INSTANT_VARIABLES,
            "expect_pressure_levels": False,
        },
        {
            "data_group": "single_level_accum",
            "folder": SL_ACCUM_DIR,
            "glob_pattern": "ERA5_SL_accum_*.nc",
            "expected_variables": EXPECTED_SL_ACCUM_VARIABLES,
            "expect_pressure_levels": False,
        },
    ]

    for check in checks:
        rows, issues = validate_group(**check)
        all_rows.extend(rows)
        all_issues.extend(issues)

    validation = pd.DataFrame(all_rows)
    validation.to_csv(VALIDATION_CSV, index=False)

    expected = expected_month_records()

    matrix_rows = []

    for _, expected_row in expected.iterrows():
        year = int(expected_row["year"])
        month = int(expected_row["month"])

        for data_group in [
            "pressure_level",
            "single_level_instant",
            "single_level_accum",
        ]:
            subset = validation.loc[
                validation["data_group"].eq(data_group)
                & validation["year"].eq(year)
                & validation["month"].eq(month)
            ]

            readable_count = int(subset["readable"].sum()) if not subset.empty else 0

            matrix_rows.append(
                {
                    "winter_start_year": int(expected_row["winter_start_year"]),
                    "year": year,
                    "month": month,
                    "data_group": data_group,
                    "file_count": len(subset),
                    "readable_count": readable_count,
                    "status": (
                        "present_readable_unique"
                        if len(subset) == 1 and readable_count == 1
                        else "problem"
                    ),
                }
            )

            if len(subset) == 0:
                all_issues.append(
                    {
                        "data_group": data_group,
                        "file": f"{year}_{month:02d}",
                        "issue": "expected_month_missing",
                    }
                )

            elif len(subset) > 1:
                all_issues.append(
                    {
                        "data_group": data_group,
                        "file": f"{year}_{month:02d}",
                        "issue": f"duplicate_month_files: {len(subset)}",
                    }
                )

            elif readable_count != 1:
                all_issues.append(
                    {
                        "data_group": data_group,
                        "file": f"{year}_{month:02d}",
                        "issue": "month_file_not_readable",
                    }
                )

    matrix = pd.DataFrame(matrix_rows)
    matrix.to_csv(MONTH_MATRIX_CSV, index=False)

    issues_df = pd.DataFrame(
        all_issues,
        columns=["data_group", "file", "issue"],
    ).drop_duplicates()

    issues_df.to_csv(ISSUES_CSV, index=False)

    group_summary = (
        validation
        .groupby("data_group")
        .agg(
            file_count=("file_name", "count"),
            readable_count=("readable", "sum"),
            unique_months=("year", lambda s: len(set(zip(s, validation.loc[s.index, "month"])))),
            expected_variables_all_present=("expected_variables_present", "all"),
            time_count_all_ok=("time_count_ok", "all"),
            all_times_00utc=("all_times_00utc", "all"),
            domain_all_ok=("domain_ok", "all"),
        )
        .reset_index()
    )

    pressure_rows = validation.loc[validation["data_group"].eq("pressure_level")]
    pressure_levels_all_ok = (
        bool(pressure_rows["pressure_levels_ok"].all())
        if not pressure_rows.empty
        else False
    )

    month_matrix_ok = bool(matrix["status"].eq("present_readable_unique").all())
    no_validation_issues = issues_df.empty

    analysis_ready = bool(
        month_matrix_ok
        and no_validation_issues
        and pressure_levels_all_ok
    )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_ready": analysis_ready,
        "month_matrix_ok": month_matrix_ok,
        "pressure_levels_all_ok": pressure_levels_all_ok,
        "issue_count": int(len(issues_df)),
        "expected_months_per_group": 160,
        "groups": group_summary.to_dict(orient="records"),
        "pressure_level_directory": str(PL_DIR),
        "single_level_instant_directory": str(SL_INSTANT_DIR),
        "single_level_accum_directory": str(SL_ACCUM_DIR),
        "validation_csv": str(VALIDATION_CSV),
        "month_matrix_csv": str(MONTH_MATRIX_CSV),
        "issues_csv": str(ISSUES_CSV),
        "note": (
            "ERA5 files are daily 00 UTC fields, not daily means. "
            "The validated domain is 10N-70N and 30E-120E."
        ),
    }

    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    registry = {
        "era5_pressure_level": {
            "directory": str(PL_DIR),
            "file_pattern": "ERA5_PL_YYYY_MM.nc",
            "variables": sorted(EXPECTED_PL_VARIABLES),
            "pressure_levels_hpa": EXPECTED_PRESSURE_LEVELS,
            "domain": {
                "north": EXPECTED_LAT_MAX,
                "west": EXPECTED_LON_MIN,
                "south": EXPECTED_LAT_MIN,
                "east": EXPECTED_LON_MAX,
            },
            "temporal_resolution": "daily 00 UTC",
            "period": "November-February winters 1985/86-2024/25",
        },
        "era5_single_level_instant": {
            "directory": str(SL_INSTANT_DIR),
            "file_pattern": "ERA5_SL_instant_YYYY_MM.nc",
            "variables": sorted(EXPECTED_SL_INSTANT_VARIABLES),
            "domain": {
                "north": EXPECTED_LAT_MAX,
                "west": EXPECTED_LON_MIN,
                "south": EXPECTED_LAT_MIN,
                "east": EXPECTED_LON_MAX,
            },
            "temporal_resolution": "daily 00 UTC",
            "period": "November-February winters 1985/86-2024/25",
        },
        "era5_single_level_accum": {
            "directory": str(SL_ACCUM_DIR),
            "file_pattern": "ERA5_SL_accum_YYYY_MM.nc",
            "variables": sorted(EXPECTED_SL_ACCUM_VARIABLES),
            "domain": {
                "north": EXPECTED_LAT_MAX,
                "west": EXPECTED_LON_MIN,
                "south": EXPECTED_LAT_MIN,
                "east": EXPECTED_LON_MAX,
            },
            "temporal_resolution": "daily 00 UTC",
            "period": "November-February winters 1985/86-2024/25",
        },
        "important_limitations": [
            "The files are daily 00 UTC fields, not daily means.",
            "The domain is regional, not full hemispheric/global.",
            "November-February coverage is complete; October/March are not included in this validation.",
            "Full NAM/polar-vortex/zonal-mean diagnostics require a hemispheric or global dataset.",
        ],
    }

    DATASET_REGISTRY.write_text(
        json.dumps(registry, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "STEP 9A: FINAL ERA5 ANALYSIS-READINESS VALIDATION",
        "=" * 58,
        f"Analysis ready: {analysis_ready}",
        "",
        "Validated data groups:",
    ]

    for item in group_summary.to_dict(orient="records"):
        report_lines.extend(
            [
                f"",
                f"{item['data_group']}:",
                f"  Files: {int(item['file_count'])}",
                f"  Readable files: {int(item['readable_count'])}",
                f"  Unique year-months: {int(item['unique_months'])}",
                f"  Expected variables present in all files: {bool(item['expected_variables_all_present'])}",
                f"  Time count OK in all files: {bool(item['time_count_all_ok'])}",
                f"  All times are 00 UTC: {bool(item['all_times_00utc'])}",
                f"  Domain OK in all files: {bool(item['domain_all_ok'])}",
            ]
        )

    report_lines.extend(
        [
            "",
            f"Pressure levels OK in all pressure-level files: {pressure_levels_all_ok}",
            f"Month matrix OK: {month_matrix_ok}",
            f"Validation issue count: {len(issues_df)}",
            "",
            "Important interpretation:",
            "  These ERA5 files are daily 00 UTC fields, not daily means.",
            "  The validated spatial domain is 10N-70N and 30E-120E.",
            "  The validated temporal coverage is November-February for winters 1985/86-2024/25.",
            "  This is suitable for regional ERA5 composite and circulation analysis.",
            "  It is not sufficient for full hemispheric NAM/polar-vortex/zonal-mean diagnostics.",
            "",
            f"Validation CSV: {VALIDATION_CSV}",
            f"Month matrix CSV: {MONTH_MATRIX_CSV}",
            f"Issues CSV: {ISSUES_CSV}",
            f"Summary JSON: {SUMMARY_JSON}",
            f"Dataset registry: {DATASET_REGISTRY}",
        ]
    )

    REPORT_TXT.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(report_lines))

    if not analysis_ready:
        raise SystemExit(
            "\nSTEP 9A VALIDATION FAILED. Check the issues CSV before proceeding."
        )

    print("\nSTEP 9A PASSED. ERA5 DATA ARE READY FOR REGIONAL ANALYSIS.")


if __name__ == "__main__":
    main()
