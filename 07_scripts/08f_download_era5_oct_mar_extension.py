from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cdsapi
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PL_DIR = PROJECT_ROOT / "01_raw_data" / "era5" / "pressure_level"
SL_DIR = PROJECT_ROOT / "01_raw_data" / "era5" / "single_level"

REPORT_DIR = PROJECT_ROOT / "05_qc_reports" / "era5"
LOG_DIR = PROJECT_ROOT / "09_logs"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_CSV = REPORT_DIR / "step9b_extra_era5_oct_mar_download_manifest.csv"
SUMMARY_JSON = REPORT_DIR / "step9b_extra_era5_oct_mar_download_summary.json"
REPORT_TXT = REPORT_DIR / "step9b_extra_era5_oct_mar_download_report.txt"


PL_DATASET = "reanalysis-era5-pressure-levels"
SL_DATASET = "reanalysis-era5-single-levels"

AREA = [70, 30, 10, 120]  # North, West, South, East

PRESSURE_LEVELS = [
    "1000",
    "850",
    "700",
    "500",
    "300",
    "250",
    "200",
    "100",
    "70",
    "50",
    "30",
    "10",
]

PL_VARIABLES = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vorticity",
    "potential_vorticity",
]

SL_VARIABLES = [
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "toa_incident_solar_radiation",
]


def month_days(year: int, month: int) -> list[str]:
    if month in [1, 3, 5, 7, 8, 10, 12]:
        count = 31
    elif month in [4, 6, 9, 11]:
        count = 30
    elif month == 2:
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        count = 29 if leap else 28
    else:
        raise ValueError(month)

    return [f"{day:02d}" for day in range(1, count + 1)]


def extension_months() -> list[tuple[int, int]]:
    months = []

    for winter_start_year in range(1985, 2025):
        months.append((winter_start_year, 10))
        months.append((winter_start_year + 1, 3))

    return months


def pl_request(year: int, month: int) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": PL_VARIABLES,
        "pressure_level": PRESSURE_LEVELS,
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": month_days(year, month),
        "time": ["00:00"],
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def sl_request(year: int, month: int) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": SL_VARIABLES,
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": month_days(year, month),
        "time": ["00:00"],
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def safe_download(
    client: cdsapi.Client,
    dataset: str,
    request: dict,
    target: Path,
    data_group: str,
    year: int,
    month: int,
) -> dict:
    row = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_group": data_group,
        "dataset": dataset,
        "year": year,
        "month": month,
        "target_file": str(target),
        "target_name": target.name,
        "status": "",
        "file_size_mb": None,
        "error": "",
    }

    if target.exists() and target.stat().st_size > 1024 * 1024:
        row["status"] = "skipped_existing"
        row["file_size_mb"] = round(target.stat().st_size / 1024 / 1024, 3)
        print(f"SKIP existing {target.name} ({row['file_size_mb']} MB)", flush=True)
        return row

    tmp_target = target.with_suffix(target.suffix + ".part")

    if tmp_target.exists():
        tmp_target.unlink()

    try:
        print(f"DOWNLOAD {data_group} {year}-{month:02d} -> {target.name}", flush=True)
        client.retrieve(dataset, request, str(tmp_target))

        if not tmp_target.exists():
            raise FileNotFoundError(f"Temporary download file not found: {tmp_target}")

        if tmp_target.stat().st_size < 1024 * 1024:
            raise RuntimeError(f"Downloaded file is suspiciously small: {tmp_target.stat().st_size} bytes")

        tmp_target.replace(target)

        row["status"] = "downloaded"
        row["file_size_mb"] = round(target.stat().st_size / 1024 / 1024, 3)

        print(f"DONE {target.name} ({row['file_size_mb']} MB)", flush=True)

    except Exception as error:
        row["status"] = "failed"
        row["error"] = repr(error)

        if tmp_target.exists():
            try:
                tmp_target.unlink()
            except Exception:
                pass

        print(f"FAILED {data_group} {year}-{month:02d}: {repr(error)}", flush=True)

    return row


def main() -> None:
    PL_DIR.mkdir(parents=True, exist_ok=True)
    SL_DIR.mkdir(parents=True, exist_ok=True)

    months = extension_months()

    print("STEP 9B-extra: ERA5 OCTOBER/MARCH EXTENSION DOWNLOAD")
    print("=" * 64)
    print(f"Extension month count: {len(months)}")
    print(f"Pressure-level target folder: {PL_DIR}")
    print(f"Single-level target folder: {SL_DIR}")
    print("")

    client = cdsapi.Client()

    rows = []

    for index, (year, month) in enumerate(months, start=1):
        print("")
        print(f"MONTH {index}/{len(months)}: {year}-{month:02d}")
        print("-" * 40)

        pl_target = PL_DIR / f"ERA5_PL_{year}_{month:02d}.nc"
        sl_target = SL_DIR / f"ERA5_SL_{year}_{month:02d}.nc"

        rows.append(
            safe_download(
                client=client,
                dataset=PL_DATASET,
                request=pl_request(year, month),
                target=pl_target,
                data_group="pressure_level",
                year=year,
                month=month,
            )
        )

        # Short pause to avoid hammering request submission.
        time.sleep(2)

        rows.append(
            safe_download(
                client=client,
                dataset=SL_DATASET,
                request=sl_request(year, month),
                target=sl_target,
                data_group="single_level",
                year=year,
                month=month,
            )
        )

        time.sleep(2)

        manifest = pd.DataFrame(rows)
        manifest.to_csv(MANIFEST_CSV, index=False)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST_CSV, index=False)

    status_counts = manifest["status"].value_counts().to_dict()

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "extension_month_count": len(months),
        "expected_pressure_level_files": len(months),
        "expected_single_level_files": len(months),
        "status_counts": status_counts,
        "manifest_csv": str(MANIFEST_CSV),
        "pressure_level_folder": str(PL_DIR),
        "single_level_folder": str(SL_DIR),
    }

    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    failed = manifest.loc[manifest["status"].eq("failed")]

    report_lines = [
        "STEP 9B-extra: ERA5 OCTOBER/MARCH EXTENSION DOWNLOAD",
        "=" * 64,
        f"Extension months: {len(months)}",
        f"Expected pressure-level files: {len(months)}",
        f"Expected single-level files: {len(months)}",
        "",
        "Download status counts:",
        manifest["status"].value_counts().to_string(),
        "",
        f"Manifest CSV: {MANIFEST_CSV}",
        f"Summary JSON: {SUMMARY_JSON}",
    ]

    if not failed.empty:
        report_lines.extend(
            [
                "",
                "FAILED REQUESTS:",
                failed[["data_group", "year", "month", "target_name", "error"]].to_string(index=False),
            ]
        )

    REPORT_TXT.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("")
    print("\n".join(report_lines))

    if not failed.empty:
        raise SystemExit(
            "\nSome ERA5 extension downloads failed. Rerun the script later; existing files will be skipped."
        )

    print("\nSTEP 9B-extra DOWNLOAD COMPLETE.")


if __name__ == "__main__":
    main()
