from __future__ import annotations

import calendar
import csv
import json
import math
import re
import unicodedata
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = (
    PROJECT_ROOT
    / "01_raw_data"
    / "Daily_Minimum_Temperature_2022_2025_raw.csv"
)

CROSSWALK_FILE = (
    PROJECT_ROOT
    / "02_metadata"
    / "new_tmin_station_crosswalk.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "new_tmin_2022_2025_daily_standardized.parquet"
)

TEMP_OUTPUT_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / ".new_tmin_2022_2025_daily_standardized.tmp.parquet"
)

REPORT_DIRECTORY = PROJECT_ROOT / "05_qc_reports"
ADMIN_DIRECTORY = PROJECT_ROOT / "00_admin"

EXPECTED_MONTHLY_ROWS = 2_242
EXPECTED_DAILY_ROWS = 68_243
EXPECTED_STATIONS = 48
EXPECTED_OFFICIAL_METADATA = 43
EXPECTED_PENDING_METADATA = 5
EXPECTED_NUMERIC_TMIN = 67_009
EXPECTED_MISSING_TMIN = 1_234
EXPECTED_ZERO_TMIN = 15
EXPECTED_MISSING_STATION_MONTHS = 62

BATCH_SIZE = 50_000

MISSING_CODES = {
    "",
    "**",
    "****",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
}


SCHEMA = pa.schema(
    [
        pa.field("source_dataset", pa.string()),
        pa.field("source_file", pa.string()),
        pa.field("source_row", pa.int32()),
        pa.field("source_day_column", pa.string()),
        pa.field("source_cell", pa.string()),

        pa.field("station_name_raw", pa.string()),
        pa.field("station_name_key", pa.string()),
        pa.field("station_uid", pa.string()),
        pa.field("station_id", pa.string()),
        pa.field("station_name_official", pa.string()),
        pa.field("latitude", pa.float64()),
        pa.field("longitude", pa.float64()),
        pa.field("metadata_status", pa.string()),
        pa.field(
            "usable_for_spatial_analysis",
            pa.bool_(),
        ),

        pa.field("year_raw", pa.string()),
        pa.field("month_raw", pa.string()),
        pa.field("year", pa.int16()),
        pa.field("month", pa.int8()),
        pa.field("day", pa.int8()),
        pa.field("date", pa.date32()),

        pa.field("tmin_raw", pa.string()),
        pa.field("tmin_new_csv", pa.float32()),
        pa.field("tmin_parse_status", pa.string()),
        pa.field("tmin_zero_flag", pa.bool_()),

        pa.field("structural_status", pa.string()),
    ]
)


def normalize_name(value: Any) -> str:
    if value is None:
        return ""

    text = unicodedata.normalize(
        "NFKD",
        str(value).strip(),
    )

    text = (
        text.encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )

    return re.sub(r"[^a-z0-9]+", "", text)


def raw_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float):
        if math.isnan(value):
            return ""

        if value.is_integer():
            return str(int(value))

    return str(value).strip()


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def parse_integer(
    value: Any,
) -> tuple[int | None, str]:
    text = raw_text(value)

    if text.lower() in MISSING_CODES:
        return None, "missing"

    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return None, "non_numeric"

    if not math.isfinite(numeric):
        return None, "non_numeric"

    if not numeric.is_integer():
        return None, "non_integer"

    return int(numeric), "valid_integer"


def parse_temperature(
    value: Any,
) -> tuple[float | None, str]:
    text = raw_text(value)

    if value is None or text == "":
        return None, "missing_cell"

    if text.lower() in MISSING_CODES:
        return None, "missing_code"

    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return None, "non_numeric"

    if not math.isfinite(numeric):
        return None, "non_numeric"

    return numeric, "numeric"


def find_header_line(path: Path) -> int:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file_handle:
        for line_number, line in enumerate(
            file_handle,
            start=1,
        ):
            if line.lstrip().startswith(
                "Station,Year,Month,Day1"
            ):
                return line_number

    raise ValueError(
        "Could not locate the Tmin CSV header."
    )


def load_station_crosswalk() -> dict[str, dict[str, Any]]:
    with CROSSWALK_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file_handle:
        rows = list(csv.DictReader(file_handle))

    if len(rows) != EXPECTED_STATIONS:
        raise ValueError(
            f"Expected {EXPECTED_STATIONS} crosswalk rows, "
            f"found {len(rows)}."
        )

    mapping: dict[str, dict[str, Any]] = {}

    for row in rows:
        key = normalize_name(
            row["raw_station_name"]
        )

        if key in mapping:
            raise ValueError(
                "Duplicate normalized station name "
                f"in crosswalk: {row['raw_station_name']}"
            )

        latitude_text = row["latitude"].strip()
        longitude_text = row["longitude"].strip()

        mapping[key] = {
            "station_uid":
                row["station_uid"].strip(),
            "station_id":
                row["station_id"].strip(),
            "station_name_official":
                row[
                    "official_station_name"
                ].strip(),
            "latitude":
                (
                    float(latitude_text)
                    if latitude_text
                    else None
                ),
            "longitude":
                (
                    float(longitude_text)
                    if longitude_text
                    else None
                ),
            "metadata_status":
                row["metadata_status"].strip(),
            "usable_for_spatial_analysis":
                parse_bool(
                    row[
                        "usable_for_spatial_analysis"
                    ]
                ),
        }

    return mapping


def empty_batch() -> dict[str, list[Any]]:
    return {
        field.name: []
        for field in SCHEMA
    }


def append_record(
    batch: dict[str, list[Any]],
    record: dict[str, Any],
) -> None:
    for field_name in batch:
        batch[field_name].append(
            record[field_name]
        )


def write_batch(
    writer: pq.ParquetWriter,
    batch: dict[str, list[Any]],
) -> int:
    row_count = len(batch["source_row"])

    if row_count == 0:
        return 0

    table = pa.Table.from_pydict(
        batch,
        schema=SCHEMA,
    )

    writer.write_table(table)

    return row_count


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    for required_file in [
        RAW_FILE,
        CROSSWALK_FILE,
    ]:
        if not required_file.exists():
            raise FileNotFoundError(
                f"Required file not found: {required_file}"
            )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    ADMIN_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    if TEMP_OUTPUT_FILE.exists():
        TEMP_OUTPUT_FILE.unlink()

    station_mapping = load_station_crosswalk()
    header_line = find_header_line(RAW_FILE)

    expected_header = [
        "Station",
        "Year",
        "Month",
        *[
            f"Day{day}"
            for day in range(1, 32)
        ],
    ]

    monthly_rows = 0
    daily_rows_written = 0
    blank_csv_rows_skipped = 0

    unique_station_names: set[str] = set()
    unique_station_uids: set[str] = set()
    observed_station_months: set[
        tuple[str, int, int]
    ] = set()

    metadata_status_counts: Counter[str] = Counter()
    parse_status_counts: Counter[str] = Counter()
    structural_status_counts: Counter[str] = Counter()
    station_daily_counts: Counter[str] = Counter()

    duplicate_month_rows: list[dict[str, Any]] = []
    invalid_month_rows: list[dict[str, Any]] = []
    unmatched_station_rows: list[dict[str, Any]] = []
    noncalendar_value_cells: list[
        dict[str, Any]
    ] = []
    nonnumeric_tmin_rows: list[dict[str, Any]] = []
    zero_tmin_rows: list[dict[str, Any]] = []

    noncalendar_cells_checked = 0

    writer: pq.ParquetWriter | None = None

    try:
        writer = pq.ParquetWriter(
            TEMP_OUTPUT_FILE,
            SCHEMA,
            compression="zstd",
            use_dictionary=True,
        )

        batch = empty_batch()

        with RAW_FILE.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file_handle:
            for _ in range(header_line - 1):
                next(file_handle)

            reader = csv.reader(file_handle)

            observed_header = next(reader)

            if observed_header != expected_header:
                raise ValueError(
                    "Unexpected CSV header.\n"
                    f"Expected: {expected_header}\n"
                    f"Observed: {observed_header}"
                )

            for source_row, values in enumerate(
                reader,
                start=header_line + 1,
            ):
                if not values or all(
                    not str(value).strip()
                    for value in values
                ):
                    blank_csv_rows_skipped += 1
                    continue

                if len(values) != len(expected_header):
                    raise ValueError(
                        f"CSV row {source_row} has "
                        f"{len(values)} columns; expected "
                        f"{len(expected_header)}."
                    )

                monthly_rows += 1

                row = dict(
                    zip(
                        expected_header,
                        values,
                        strict=True,
                    )
                )

                station_name_raw = (
                    row["Station"].strip()
                )

                station_name_key = normalize_name(
                    station_name_raw
                )

                unique_station_names.add(
                    station_name_raw
                )

                station = station_mapping.get(
                    station_name_key
                )

                if station is None:
                    unmatched_station_rows.append(
                        {
                            "source_row": source_row,
                            "station_name_raw":
                                station_name_raw,
                            "station_name_key":
                                station_name_key,
                        }
                    )
                    continue

                station_uid = station[
                    "station_uid"
                ]

                unique_station_uids.add(
                    station_uid
                )

                metadata_status_counts[
                    station["metadata_status"]
                ] += 1

                year, year_status = parse_integer(
                    row["Year"]
                )

                month, month_status = parse_integer(
                    row["Month"]
                )

                if (
                    year_status != "valid_integer"
                    or month_status
                    != "valid_integer"
                    or year is None
                    or month is None
                    or year < 2022
                    or year > 2025
                    or month < 1
                    or month > 12
                ):
                    invalid_month_rows.append(
                        {
                            "source_row": source_row,
                            "station_name_raw":
                                station_name_raw,
                            "year_raw": row["Year"],
                            "month_raw": row["Month"],
                            "year_status": year_status,
                            "month_status":
                                month_status,
                        }
                    )
                    continue

                station_month_key = (
                    station_uid,
                    year,
                    month,
                )

                if (
                    station_month_key
                    in observed_station_months
                ):
                    duplicate_month_rows.append(
                        {
                            "source_row": source_row,
                            "station_uid": station_uid,
                            "station_name_raw":
                                station_name_raw,
                            "year": year,
                            "month": month,
                        }
                    )

                observed_station_months.add(
                    station_month_key
                )

                days_in_month = calendar.monthrange(
                    year,
                    month,
                )[1]

                for day_number in range(1, 32):
                    day_column = (
                        f"Day{day_number}"
                    )

                    raw_value = row[day_column]

                    if day_number > days_in_month:
                        noncalendar_cells_checked += 1

                        value_text = raw_text(
                            raw_value
                        )

                        if (
                            value_text.lower()
                            not in MISSING_CODES
                        ):
                            noncalendar_value_cells.append(
                                {
                                    "source_row":
                                        source_row,
                                    "station_uid":
                                        station_uid,
                                    "station_name_raw":
                                        station_name_raw,
                                    "year": year,
                                    "month": month,
                                    "source_day_column":
                                        day_column,
                                    "raw_value":
                                        value_text,
                                }
                            )

                        continue

                    parsed_date = date(
                        year,
                        month,
                        day_number,
                    )

                    (
                        tmin_numeric,
                        tmin_parse_status,
                    ) = parse_temperature(
                        raw_value
                    )

                    parse_status_counts[
                        tmin_parse_status
                    ] += 1

                    zero_flag = (
                        tmin_parse_status
                        == "numeric"
                        and tmin_numeric == 0.0
                    )

                    if (
                        tmin_parse_status
                        == "non_numeric"
                    ):
                        nonnumeric_tmin_rows.append(
                            {
                                "source_row":
                                    source_row,
                                "source_day_column":
                                    day_column,
                                "station_uid":
                                    station_uid,
                                "station_name_raw":
                                    station_name_raw,
                                "date":
                                    parsed_date.isoformat(),
                                "tmin_raw":
                                    raw_text(
                                        raw_value
                                    ),
                            }
                        )

                    if zero_flag:
                        zero_tmin_rows.append(
                            {
                                "source_row":
                                    source_row,
                                "source_day_column":
                                    day_column,
                                "station_uid":
                                    station_uid,
                                "station_name_raw":
                                    station_name_raw,
                                "date":
                                    parsed_date.isoformat(),
                                "tmin_raw":
                                    raw_text(
                                        raw_value
                                    ),
                            }
                        )

                    if (
                        tmin_parse_status
                        == "non_numeric"
                    ):
                        structural_status = (
                            "review_required"
                        )
                    elif (
                        station["metadata_status"]
                        == "official_metadata_missing"
                    ):
                        structural_status = (
                            "ready_for_qc_metadata_pending"
                        )
                    else:
                        structural_status = (
                            "ready_for_qc"
                        )

                    structural_status_counts[
                        structural_status
                    ] += 1

                    station_daily_counts[
                        station_uid
                    ] += 1

                    record = {
                        "source_dataset":
                            "new_tmin_csv_2022_2025",
                        "source_file":
                            RAW_FILE.name,
                        "source_row":
                            source_row,
                        "source_day_column":
                            day_column,
                        "source_cell":
                            f"row{source_row}:{day_column}",

                        "station_name_raw":
                            station_name_raw,
                        "station_name_key":
                            station_name_key,
                        "station_uid":
                            station_uid,
                        "station_id":
                            station[
                                "station_id"
                            ],
                        "station_name_official":
                            station[
                                "station_name_official"
                            ],
                        "latitude":
                            station["latitude"],
                        "longitude":
                            station["longitude"],
                        "metadata_status":
                            station[
                                "metadata_status"
                            ],
                        "usable_for_spatial_analysis":
                            station[
                                "usable_for_spatial_analysis"
                            ],

                        "year_raw":
                            row["Year"].strip(),
                        "month_raw":
                            row["Month"].strip(),
                        "year": year,
                        "month": month,
                        "day": day_number,
                        "date": parsed_date,

                        "tmin_raw":
                            raw_text(raw_value),
                        "tmin_new_csv":
                            tmin_numeric,
                        "tmin_parse_status":
                            tmin_parse_status,
                        "tmin_zero_flag":
                            zero_flag,

                        "structural_status":
                            structural_status,
                    }

                    append_record(batch, record)

                    if (
                        len(batch["source_row"])
                        >= BATCH_SIZE
                    ):
                        daily_rows_written += (
                            write_batch(
                                writer,
                                batch,
                            )
                        )

                        print(
                            "Written "
                            f"{daily_rows_written:,} "
                            "daily records..."
                        )

                        batch = empty_batch()

        if batch["source_row"]:
            daily_rows_written += write_batch(
                writer,
                batch,
            )

    except Exception:
        if writer is not None:
            writer.close()
            writer = None

        if TEMP_OUTPUT_FILE.exists():
            TEMP_OUTPUT_FILE.unlink()

        raise

    finally:
        if writer is not None:
            writer.close()

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    TEMP_OUTPUT_FILE.replace(
        OUTPUT_FILE
    )

    all_expected_months = {
        (
            station_uid,
            year,
            month,
        )
        for station_uid in unique_station_uids
        for year in range(2022, 2026)
        for month in range(1, 13)
    }

    missing_station_month_keys = sorted(
        all_expected_months
        - observed_station_months
    )

    uid_to_station_name = {
        details["station_uid"]:
            raw_name
        for raw_name, details
        in [
            (
                key,
                station_mapping[key],
            )
            for key in station_mapping
        ]
    }

    missing_station_month_rows = []

    station_uid_to_raw_name = {}

    for raw_name_key, details in station_mapping.items():
        station_uid_to_raw_name[
            details["station_uid"]
        ] = raw_name_key

    for station_uid, year, month in (
        missing_station_month_keys
    ):
        missing_station_month_rows.append(
            {
                "station_uid": station_uid,
                "station_name_key":
                    station_uid_to_raw_name.get(
                        station_uid,
                        "",
                    ),
                "year": year,
                "month": month,
            }
        )

    station_count_rows = [
        {
            "station_uid": station_uid,
            "daily_calendar_rows": count,
        }
        for station_uid, count
        in sorted(
            station_daily_counts.items()
        )
    ]

    write_csv(
        REPORT_DIRECTORY
        / "step3b_duplicate_station_months.csv",
        duplicate_month_rows,
        [
            "source_row",
            "station_uid",
            "station_name_raw",
            "year",
            "month",
        ],
    )

    write_csv(
        REPORT_DIRECTORY
        / "step3b_invalid_station_month_rows.csv",
        invalid_month_rows,
        [
            "source_row",
            "station_name_raw",
            "year_raw",
            "month_raw",
            "year_status",
            "month_status",
        ],
    )

    write_csv(
        REPORT_DIRECTORY
        / "step3b_unmatched_station_rows.csv",
        unmatched_station_rows,
        [
            "source_row",
            "station_name_raw",
            "station_name_key",
        ],
    )

    write_csv(
        REPORT_DIRECTORY
        / "step3b_noncalendar_value_cells.csv",
        noncalendar_value_cells,
        [
            "source_row",
            "station_uid",
            "station_name_raw",
            "year",
            "month",
            "source_day_column",
            "raw_value",
        ],
    )

    write_csv(
        REPORT_DIRECTORY
        / "step3b_nonnumeric_tmin_rows.csv",
        nonnumeric_tmin_rows,
        [
            "source_row",
            "source_day_column",
            "station_uid",
            "station_name_raw",
            "date",
            "tmin_raw",
        ],
    )

    write_csv(
        REPORT_DIRECTORY
        / "step3b_zero_tmin_rows.csv",
        zero_tmin_rows,
        [
            "source_row",
            "source_day_column",
            "station_uid",
            "station_name_raw",
            "date",
            "tmin_raw",
        ],
    )

    write_csv(
        REPORT_DIRECTORY
        / "step3b_missing_station_months.csv",
        missing_station_month_rows,
        [
            "station_uid",
            "station_name_key",
            "year",
            "month",
        ],
    )

    write_csv(
        REPORT_DIRECTORY
        / "step3b_station_daily_counts.csv",
        station_count_rows,
        [
            "station_uid",
            "daily_calendar_rows",
        ],
    )

    parquet_file = pq.ParquetFile(
        OUTPUT_FILE
    )

    parquet_rows = (
        parquet_file.metadata.num_rows
    )

    official_station_count = sum(
        1
        for details in station_mapping.values()
        if details["metadata_status"]
        == "official_metadata_available"
    )

    pending_station_count = sum(
        1
        for details in station_mapping.values()
        if details["metadata_status"]
        == "official_metadata_missing"
    )

    numeric_count = parse_status_counts[
        "numeric"
    ]

    missing_count = (
        parse_status_counts["missing_cell"]
        + parse_status_counts["missing_code"]
    )

    summary = {
        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "source_file":
            str(RAW_FILE),
        "header_line":
            header_line,
        "monthly_rows":
            monthly_rows,
        "daily_rows_written":
            daily_rows_written,
        "parquet_rows":
            parquet_rows,
        "blank_csv_rows_skipped":
            blank_csv_rows_skipped,
        "unique_station_names":
            len(unique_station_names),
        "unique_station_uids":
            len(unique_station_uids),
        "official_metadata_stations":
            official_station_count,
        "pending_metadata_stations":
            pending_station_count,
        "observed_station_months":
            len(observed_station_months),
        "missing_station_months":
            len(
                missing_station_month_rows
            ),
        "duplicate_station_month_rows":
            len(duplicate_month_rows),
        "invalid_station_month_rows":
            len(invalid_month_rows),
        "unmatched_station_rows":
            len(unmatched_station_rows),
        "noncalendar_cells_checked":
            noncalendar_cells_checked,
        "noncalendar_nonmissing_cells":
            len(noncalendar_value_cells),
        "tmin_parse_status_counts":
            dict(parse_status_counts),
        "numeric_tmin_values":
            numeric_count,
        "missing_tmin_values":
            missing_count,
        "zero_tmin_values":
            len(zero_tmin_rows),
        "nonnumeric_tmin_values":
            len(nonnumeric_tmin_rows),
        "structural_status_counts":
            dict(
                structural_status_counts
            ),
        "output_file":
            str(OUTPUT_FILE),
        "output_size_bytes":
            OUTPUT_FILE.stat().st_size,
    }

    (
        REPORT_DIRECTORY
        / "step3b_ingestion_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (
        ADMIN_DIRECTORY
        / "step3b_parquet_schema.json"
    ).write_text(
        json.dumps(
            {
                field.name: str(field.type)
                for field in SCHEMA
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "STEP 3B: UPDATED TMIN CSV INGESTION",
        "=" * 41,
        (
            "Header line: "
            f"{header_line}"
        ),
        (
            "Monthly source rows: "
            f"{monthly_rows:,}"
        ),
        (
            "Daily calendar rows written: "
            f"{daily_rows_written:,}"
        ),
        (
            "Parquet rows: "
            f"{parquet_rows:,}"
        ),
        (
            "Unique stations: "
            f"{len(unique_station_names)}"
        ),
        (
            "Official-metadata stations: "
            f"{official_station_count}"
        ),
        (
            "Pending-metadata stations: "
            f"{pending_station_count}"
        ),
        (
            "Observed station-months: "
            f"{len(observed_station_months):,}"
        ),
        (
            "Missing station-months: "
            f"{len(missing_station_month_rows):,}"
        ),
        (
            "Duplicate station-month rows: "
            f"{len(duplicate_month_rows):,}"
        ),
        (
            "Invalid station-month rows: "
            f"{len(invalid_month_rows):,}"
        ),
        (
            "Unmatched station rows: "
            f"{len(unmatched_station_rows):,}"
        ),
        (
            "Noncalendar cells checked: "
            f"{noncalendar_cells_checked:,}"
        ),
        (
            "Noncalendar cells containing data: "
            f"{len(noncalendar_value_cells):,}"
        ),
        (
            "Numeric Tmin values: "
            f"{numeric_count:,}"
        ),
        (
            "Missing Tmin values: "
            f"{missing_count:,}"
        ),
        (
            "Nonnumeric Tmin values: "
            f"{len(nonnumeric_tmin_rows):,}"
        ),
        (
            "Zero Tmin values flagged: "
            f"{len(zero_tmin_rows):,}"
        ),
        "",
        "Tmin parse statuses:",
        *[
            f"- {key}: {value:,}"
            for key, value
            in sorted(
                parse_status_counts.items()
            )
        ],
        "",
        "Structural statuses:",
        *[
            f"- {key}: {value:,}"
            for key, value
            in sorted(
                structural_status_counts.items()
            )
        ],
        "",
        f"Output: {OUTPUT_FILE}",
    ]

    (
        REPORT_DIRECTORY
        / "step3b_ingestion_report.txt"
    ).write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("\n".join(report_lines))

    failures: list[str] = []

    if monthly_rows != EXPECTED_MONTHLY_ROWS:
        failures.append(
            f"Expected {EXPECTED_MONTHLY_ROWS:,} "
            f"monthly rows; found {monthly_rows:,}."
        )

    if daily_rows_written != EXPECTED_DAILY_ROWS:
        failures.append(
            f"Expected {EXPECTED_DAILY_ROWS:,} "
            "daily rows; wrote "
            f"{daily_rows_written:,}."
        )

    if parquet_rows != daily_rows_written:
        failures.append(
            "Parquet row count differs from "
            "daily rows written."
        )

    if (
        len(unique_station_names)
        != EXPECTED_STATIONS
    ):
        failures.append(
            f"Expected {EXPECTED_STATIONS} stations; "
            f"found {len(unique_station_names)}."
        )

    if (
        official_station_count
        != EXPECTED_OFFICIAL_METADATA
    ):
        failures.append(
            "Unexpected official-metadata "
            f"station count: {official_station_count}."
        )

    if (
        pending_station_count
        != EXPECTED_PENDING_METADATA
    ):
        failures.append(
            "Unexpected pending-metadata "
            f"station count: {pending_station_count}."
        )

    if duplicate_month_rows:
        failures.append(
            "Duplicate station-month rows were found."
        )

    if invalid_month_rows:
        failures.append(
            "Invalid station-month rows were found."
        )

    if unmatched_station_rows:
        failures.append(
            "Unmatched station rows were found."
        )

    if noncalendar_value_cells:
        failures.append(
            "One or more impossible calendar cells "
            "contain data."
        )

    if nonnumeric_tmin_rows:
        failures.append(
            "Unexpected nonnumeric Tmin values "
            "were found."
        )

    if numeric_count != EXPECTED_NUMERIC_TMIN:
        failures.append(
            f"Expected {EXPECTED_NUMERIC_TMIN:,} "
            f"numeric Tmin values; found "
            f"{numeric_count:,}."
        )

    if missing_count != EXPECTED_MISSING_TMIN:
        failures.append(
            f"Expected {EXPECTED_MISSING_TMIN:,} "
            f"missing Tmin values; found "
            f"{missing_count:,}."
        )

    if len(zero_tmin_rows) != EXPECTED_ZERO_TMIN:
        failures.append(
            f"Expected {EXPECTED_ZERO_TMIN} "
            "zero Tmin values; found "
            f"{len(zero_tmin_rows)}."
        )

    if (
        len(missing_station_month_rows)
        != EXPECTED_MISSING_STATION_MONTHS
    ):
        failures.append(
            f"Expected {EXPECTED_MISSING_STATION_MONTHS} "
            "missing station-months; found "
            f"{len(missing_station_month_rows)}."
        )

    if failures:
        raise SystemExit(
            "\nSTEP 3B FAILED:\n- "
            + "\n- ".join(failures)
        )

    print("\nSTEP 3B PASSED.")


if __name__ == "__main__":
    main()
