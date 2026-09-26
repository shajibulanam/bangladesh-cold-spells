from __future__ import annotations

import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SINGLE_DIR = PROJECT_ROOT / "01_raw_data" / "era5" / "single_level"

EXTRACT_ROOT = PROJECT_ROOT / "01_raw_data" / "era5" / "single_level_extracted"
INSTANT_DIR = EXTRACT_ROOT / "instant"
ACCUM_DIR = EXTRACT_ROOT / "accum"
UNKNOWN_DIR = EXTRACT_ROOT / "unknown"

REPORT_DIR = PROJECT_ROOT / "05_qc_reports" / "era5"
INVENTORY_DIR = PROJECT_ROOT / "03_intermediate" / "era5_inventory"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
INVENTORY_DIR.mkdir(parents=True, exist_ok=True)
INSTANT_DIR.mkdir(parents=True, exist_ok=True)
ACCUM_DIR.mkdir(parents=True, exist_ok=True)
UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)

EXTRACTION_LOG = REPORT_DIR / "era5_single_level_extraction_log.csv"
EXTRACTION_REPORT = REPORT_DIR / "era5_single_level_extraction_report.txt"


def detect_year_month(path: Path) -> tuple[int, int]:
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

    raise ValueError(
        f"Could not detect year/month from filename: {path.name}"
    )


def classify_member(member_name: str) -> str:
    lower = member_name.lower()

    if "instant" in lower or "steptype-instant" in lower:
        return "instant"

    if "accum" in lower or "steptype-accum" in lower:
        return "accum"

    return "unknown"


def main() -> None:
    rows = []
    files = sorted(
        [path for path in SINGLE_DIR.iterdir() if path.is_file()]
    )

    zip_files = [
        path for path in files if zipfile.is_zipfile(path)
    ]

    if not zip_files:
        raise SystemExit(
            f"No ZIP-style files found in {SINGLE_DIR}"
        )

    for zip_path in zip_files:
        year, month = detect_year_month(zip_path)

        with zipfile.ZipFile(zip_path, "r") as archive:
            members = [
                member for member in archive.namelist()
                if not member.endswith("/")
            ]

            for member in members:
                if not member.lower().endswith(".nc"):
                    rows.append(
                        {
                            "source_zip": zip_path.name,
                            "year": year,
                            "month": month,
                            "member": member,
                            "member_type": "non_nc_skipped",
                            "output_file": "",
                            "status": "skipped",
                        }
                    )
                    continue

                member_type = classify_member(member)

                if member_type == "instant":
                    target_dir = INSTANT_DIR
                    output_name = f"ERA5_SL_instant_{year}_{month:02d}.nc"
                elif member_type == "accum":
                    target_dir = ACCUM_DIR
                    output_name = f"ERA5_SL_accum_{year}_{month:02d}.nc"
                else:
                    target_dir = UNKNOWN_DIR
                    safe_member = Path(member).name
                    output_name = f"ERA5_SL_unknown_{year}_{month:02d}_{safe_member}"

                output_path = target_dir / output_name

                with archive.open(member) as source, output_path.open("wb") as target:
                    shutil.copyfileobj(source, target)

                rows.append(
                    {
                        "source_zip": zip_path.name,
                        "year": year,
                        "month": month,
                        "member": member,
                        "member_type": member_type,
                        "output_file": str(output_path),
                        "status": "extracted",
                    }
                )

    log = pd.DataFrame(rows)
    log.to_csv(EXTRACTION_LOG, index=False)

    extracted = log.loc[log["status"].eq("extracted")]

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_zip_count": len(zip_files),
        "extracted_file_count": len(extracted),
        "instant_file_count": int(extracted["member_type"].eq("instant").sum()),
        "accum_file_count": int(extracted["member_type"].eq("accum").sum()),
        "unknown_file_count": int(extracted["member_type"].eq("unknown").sum()),
        "instant_directory": str(INSTANT_DIR),
        "accum_directory": str(ACCUM_DIR),
        "unknown_directory": str(UNKNOWN_DIR),
    }

    report_lines = [
        "STEP 9A-2: SINGLE-LEVEL ERA5 ZIP EXTRACTION",
        "=" * 52,
        f"Source ZIP files: {summary['source_zip_count']}",
        f"Extracted NetCDF files: {summary['extracted_file_count']}",
        f"Instant files: {summary['instant_file_count']}",
        f"Accumulated files: {summary['accum_file_count']}",
        f"Unknown files: {summary['unknown_file_count']}",
        "",
        f"Instant directory: {INSTANT_DIR}",
        f"Accum directory: {ACCUM_DIR}",
        f"Extraction log: {EXTRACTION_LOG}",
    ]

    EXTRACTION_REPORT.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(report_lines))

    if summary["unknown_file_count"] > 0:
        raise SystemExit(
            "WARNING: Some extracted files could not be classified."
        )

    print("\nSTEP 9A-2 EXTRACTION COMPLETE.")


if __name__ == "__main__":
    main()
