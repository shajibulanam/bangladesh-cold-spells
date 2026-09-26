from __future__ import annotations

import csv
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NEW_TMIN_FILE = (
    PROJECT_ROOT
    / "01_raw_data"
    / "Daily_Minimum_Temperature_2022_2025_raw.csv"
)

EXISTING_STATION_MASTER = (
    PROJECT_ROOT
    / "02_metadata"
    / "station_master.csv"
)

ALIAS_FILE = (
    PROJECT_ROOT
    / "02_metadata"
    / "new_tmin_station_aliases.csv"
)

OUTPUT_DIRECTORY = PROJECT_ROOT / "02_metadata"
REPORT_DIRECTORY = PROJECT_ROOT / "05_qc_reports"

EXPECTED_NEW_TMIN_STATIONS = 48
EXPECTED_MATCHED_TO_METADATA = 43
EXPECTED_MISSING_METADATA = 5
EXPECTED_EXTENDED_MASTER_ROWS = 51
EXPECTED_STATIONS_WITH_ANY_DATA = 48
EXPECTED_OFFICIAL_METADATA_ONLY = 3

EXPECTED_MISSING_NAMES = {
    "Ambagan(CTG)",
    "Bagabari",
    "Koyra",
    "Manikganj",
    "Narayanganj",
}


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


def clean_station_id(value: Any) -> str:
    text = str(value or "").strip()

    if text.endswith(".0"):
        text = text[:-2]

    return text


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def read_csv_records(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file_handle:
        return list(csv.DictReader(file_handle))


def find_new_tmin_header_line(path: Path) -> int:
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
        "Could not locate the new Tmin CSV header."
    )


def read_new_tmin_station_names(
    path: Path,
) -> list[str]:
    header_line = find_new_tmin_header_line(path)

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file_handle:
        for _ in range(header_line - 1):
            next(file_handle)

        reader = csv.DictReader(file_handle)

        required_columns = {
            "Station",
            "Year",
            "Month",
            "Day1",
            "Day31",
        }

        missing_columns = required_columns.difference(
            reader.fieldnames or []
        )

        if missing_columns:
            raise ValueError(
                "New Tmin file is missing columns: "
                f"{sorted(missing_columns)}"
            )

        stations: set[str] = set()

        for row in reader:
            station = str(
                row.get("Station", "")
            ).strip()

            if station:
                stations.add(station)

    return sorted(stations, key=str.casefold)


def read_aliases(path: Path) -> dict[str, str]:
    records = read_csv_records(path)

    mapping: dict[str, str] = {}

    for record in records:
        raw_name = record[
            "raw_station_name"
        ].strip()

        official_name = record[
            "official_station_name"
        ].strip()

        raw_key = normalize_name(raw_name)
        official_key = normalize_name(
            official_name
        )

        if raw_key in mapping:
            raise ValueError(
                f"Duplicate alias for {raw_name!r}."
            )

        mapping[raw_key] = official_key

    return mapping


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    required_files = [
        NEW_TMIN_FILE,
        EXISTING_STATION_MASTER,
        ALIAS_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_master = read_csv_records(
        EXISTING_STATION_MASTER
    )

    if len(existing_master) != 46:
        raise ValueError(
            "Expected 46 rows in the completed "
            f"Step 2 station master; found "
            f"{len(existing_master)}."
        )

    metadata_by_key: dict[
        str,
        dict[str, str],
    ] = {}

    metadata_by_id: dict[
        str,
        dict[str, str],
    ] = {}

    for record in existing_master:
        official_name = record[
            "official_station_name"
        ].strip()

        station_id = clean_station_id(
            record["station_id"]
        )

        key = normalize_name(official_name)

        if key in metadata_by_key:
            raise ValueError(
                "Duplicate official station name "
                f"in metadata: {official_name}"
            )

        if station_id in metadata_by_id:
            raise ValueError(
                "Duplicate official station ID "
                f"in metadata: {station_id}"
            )

        metadata_by_key[key] = record
        metadata_by_id[station_id] = record

    aliases = read_aliases(ALIAS_FILE)

    for raw_key, official_key in aliases.items():
        if official_key not in metadata_by_key:
            raise ValueError(
                "Alias points to a station absent "
                "from the metadata: "
                f"{official_key}"
            )

    new_station_names = (
        read_new_tmin_station_names(
            NEW_TMIN_FILE
        )
    )

    crosswalk_rows: list[
        dict[str, Any]
    ] = []

    matched_station_ids: set[str] = set()
    missing_metadata_rows: list[
        dict[str, Any]
    ] = []

    new_raw_names_by_station_id: dict[
        str,
        list[str],
    ] = {}

    for raw_name in new_station_names:
        raw_key = normalize_name(raw_name)

        if raw_key in aliases:
            target_key = aliases[raw_key]
            match_method = "manual_alias"
        else:
            target_key = raw_key
            match_method = "normalized_exact"

        metadata_record = metadata_by_key.get(
            target_key
        )

        if metadata_record is not None:
            station_id = clean_station_id(
                metadata_record["station_id"]
            )

            official_name = metadata_record[
                "official_station_name"
            ].strip()

            station_uid = f"BMD_{station_id}"

            row = {
                "source_dataset":
                    "new_tmin_2022_2025",
                "raw_station_name": raw_name,
                "raw_name_key": raw_key,
                "station_uid": station_uid,
                "station_id": station_id,
                "official_station_name":
                    official_name,
                "latitude":
                    metadata_record["latitude"],
                "longitude":
                    metadata_record["longitude"],
                "metadata_status":
                    "official_metadata_available",
                "match_method": match_method,
                "usable_for_spatial_analysis":
                    True,
                "action_required": False,
            }

            matched_station_ids.add(
                station_id
            )

            new_raw_names_by_station_id.setdefault(
                station_id,
                [],
            ).append(raw_name)

        else:
            station_uid = (
                "PENDING_"
                + normalize_name(
                    raw_name
                ).upper()
            )

            row = {
                "source_dataset":
                    "new_tmin_2022_2025",
                "raw_station_name": raw_name,
                "raw_name_key": raw_key,
                "station_uid": station_uid,
                "station_id": "",
                "official_station_name": "",
                "latitude": "",
                "longitude": "",
                "metadata_status":
                    "official_metadata_missing",
                "match_method":
                    "unmatched_metadata",
                "usable_for_spatial_analysis":
                    False,
                "action_required": True,
            }

            missing_metadata_rows.append(
                {
                    "raw_station_name":
                        raw_name,
                    "station_uid":
                        station_uid,
                    "required_information":
                        (
                            "Official BMD station ID; "
                            "latitude; longitude; "
                            "elevation; start date"
                        ),
                }
            )

        crosswalk_rows.append(row)

    master_v2_rows: list[
        dict[str, Any]
    ] = []

    for record in existing_master:
        station_id = clean_station_id(
            record["station_id"]
        )

        has_old_data = parse_bool(
            record.get(
                "has_temperature_data",
                False,
            )
        )

        has_new_data = (
            station_id
            in matched_station_ids
        )

        new_raw_names = "; ".join(
            sorted(
                new_raw_names_by_station_id.get(
                    station_id,
                    [],
                ),
                key=str.casefold,
            )
        )

        master_v2_rows.append(
            {
                "station_uid":
                    f"BMD_{station_id}",
                "station_id": station_id,
                "official_station_name":
                    record[
                        "official_station_name"
                    ].strip(),
                "old_xlsx_station_name":
                    record.get(
                        "raw_temperature_name",
                        "",
                    ).strip(),
                "new_tmin_station_name":
                    new_raw_names,
                "latitude":
                    record["latitude"],
                "longitude":
                    record["longitude"],
                "elevation_m":
                    record.get(
                        "elevation_m",
                        "",
                    ),
                "metadata_status":
                    "official_metadata_available",
                "has_old_xlsx_data":
                    has_old_data,
                "has_new_tmin_data":
                    has_new_data,
                "has_any_temperature_data":
                    has_old_data or has_new_data,
                "usable_for_spatial_analysis":
                    True,
                "metadata_notes":
                    record.get(
                        "metadata_notes",
                        "",
                    ),
            }
        )

    for row in missing_metadata_rows:
        master_v2_rows.append(
            {
                "station_uid":
                    row["station_uid"],
                "station_id": "",
                "official_station_name": "",
                "old_xlsx_station_name": "",
                "new_tmin_station_name":
                    row["raw_station_name"],
                "latitude": "",
                "longitude": "",
                "elevation_m": "",
                "metadata_status":
                    "official_metadata_missing",
                "has_old_xlsx_data": False,
                "has_new_tmin_data": True,
                "has_any_temperature_data":
                    True,
                "usable_for_spatial_analysis":
                    False,
                "metadata_notes":
                    (
                        "Provisional internal record; "
                        "do not use for area weighting "
                        "until official metadata is obtained."
                    ),
            }
        )

    crosswalk_rows.sort(
        key=lambda row:
            row["raw_station_name"].casefold()
    )

    master_v2_rows.sort(
        key=lambda row:
            row["station_uid"]
    )

    missing_metadata_rows.sort(
        key=lambda row:
            row["raw_station_name"].casefold()
    )

    official_metadata_only = [
        row
        for row in master_v2_rows
        if (
            row["metadata_status"]
            == "official_metadata_available"
            and not row[
                "has_any_temperature_data"
            ]
        )
    ]

    stations_with_any_data = [
        row
        for row in master_v2_rows
        if row["has_any_temperature_data"]
    ]

    write_csv(
        OUTPUT_DIRECTORY
        / "new_tmin_station_crosswalk.csv",
        crosswalk_rows,
        [
            "source_dataset",
            "raw_station_name",
            "raw_name_key",
            "station_uid",
            "station_id",
            "official_station_name",
            "latitude",
            "longitude",
            "metadata_status",
            "match_method",
            "usable_for_spatial_analysis",
            "action_required",
        ],
    )

    write_csv(
        OUTPUT_DIRECTORY
        / "station_master_v2.csv",
        master_v2_rows,
        [
            "station_uid",
            "station_id",
            "official_station_name",
            "old_xlsx_station_name",
            "new_tmin_station_name",
            "latitude",
            "longitude",
            "elevation_m",
            "metadata_status",
            "has_old_xlsx_data",
            "has_new_tmin_data",
            "has_any_temperature_data",
            "usable_for_spatial_analysis",
            "metadata_notes",
        ],
    )

    write_csv(
        OUTPUT_DIRECTORY
        / "missing_metadata_stations.csv",
        missing_metadata_rows,
        [
            "raw_station_name",
            "station_uid",
            "required_information",
        ],
    )

    write_csv(
        OUTPUT_DIRECTORY
        / "official_metadata_only_stations_v2.csv",
        official_metadata_only,
        [
            "station_uid",
            "station_id",
            "official_station_name",
            "old_xlsx_station_name",
            "new_tmin_station_name",
            "latitude",
            "longitude",
            "elevation_m",
            "metadata_status",
            "has_old_xlsx_data",
            "has_new_tmin_data",
            "has_any_temperature_data",
            "usable_for_spatial_analysis",
            "metadata_notes",
        ],
    )

    summary = {
        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "new_tmin_unique_stations":
            len(new_station_names),
        "new_tmin_matched_to_metadata":
            len(matched_station_ids),
        "new_tmin_missing_metadata":
            len(missing_metadata_rows),
        "missing_metadata_names":
            [
                row["raw_station_name"]
                for row in missing_metadata_rows
            ],
        "station_master_v2_rows":
            len(master_v2_rows),
        "stations_with_any_temperature_data":
            len(stations_with_any_data),
        "official_metadata_only_stations":
            len(official_metadata_only),
        "official_metadata_only_names":
            [
                row["official_station_name"]
                for row
                in official_metadata_only
            ],
    }

    (
        REPORT_DIRECTORY
        / "step2a_station_extension_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "STEP 2A: NEW TMIN STATION EXTENSION",
        "=" * 39,
        (
            "Unique stations in new Tmin file: "
            f"{len(new_station_names)}"
        ),
        (
            "Matched to official metadata: "
            f"{len(matched_station_ids)}"
        ),
        (
            "Missing official metadata: "
            f"{len(missing_metadata_rows)}"
        ),
        (
            "Extended station-master rows: "
            f"{len(master_v2_rows)}"
        ),
        (
            "Stations with any temperature data: "
            f"{len(stations_with_any_data)}"
        ),
        (
            "Official metadata-only stations: "
            f"{len(official_metadata_only)}"
        ),
        "",
        "Stations missing official metadata:",
        *[
            f"- {row['raw_station_name']}"
            for row in missing_metadata_rows
        ],
        "",
        "Official metadata stations with no data:",
        *[
            f"- {row['official_station_name']}"
            for row
            in official_metadata_only
        ],
    ]

    (
        REPORT_DIRECTORY
        / "step2a_station_extension_report.txt"
    ).write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(report_lines))

    failures: list[str] = []

    if (
        len(new_station_names)
        != EXPECTED_NEW_TMIN_STATIONS
    ):
        failures.append(
            "Expected "
            f"{EXPECTED_NEW_TMIN_STATIONS} "
            "new Tmin stations, found "
            f"{len(new_station_names)}."
        )

    if (
        len(matched_station_ids)
        != EXPECTED_MATCHED_TO_METADATA
    ):
        failures.append(
            "Expected "
            f"{EXPECTED_MATCHED_TO_METADATA} "
            "metadata matches, found "
            f"{len(matched_station_ids)}."
        )

    if (
        len(missing_metadata_rows)
        != EXPECTED_MISSING_METADATA
    ):
        failures.append(
            "Expected "
            f"{EXPECTED_MISSING_METADATA} "
            "stations missing metadata, found "
            f"{len(missing_metadata_rows)}."
        )

    actual_missing_names = {
        row["raw_station_name"]
        for row in missing_metadata_rows
    }

    if (
        actual_missing_names
        != EXPECTED_MISSING_NAMES
    ):
        failures.append(
            "Unexpected missing-metadata station "
            f"set: {sorted(actual_missing_names)}"
        )

    if (
        len(master_v2_rows)
        != EXPECTED_EXTENDED_MASTER_ROWS
    ):
        failures.append(
            "Expected "
            f"{EXPECTED_EXTENDED_MASTER_ROWS} "
            "extended master rows, found "
            f"{len(master_v2_rows)}."
        )

    if (
        len(stations_with_any_data)
        != EXPECTED_STATIONS_WITH_ANY_DATA
    ):
        failures.append(
            "Expected "
            f"{EXPECTED_STATIONS_WITH_ANY_DATA} "
            "stations with temperature data, "
            f"found {len(stations_with_any_data)}."
        )

    if (
        len(official_metadata_only)
        != EXPECTED_OFFICIAL_METADATA_ONLY
    ):
        failures.append(
            "Expected "
            f"{EXPECTED_OFFICIAL_METADATA_ONLY} "
            "official metadata-only stations, "
            f"found {len(official_metadata_only)}."
        )

    if failures:
        raise SystemExit(
            "\nSTEP 2A FAILED:\n- "
            + "\n- ".join(failures)
        )

    print("\nSTEP 2A PASSED.")


if __name__ == "__main__":
    main()
