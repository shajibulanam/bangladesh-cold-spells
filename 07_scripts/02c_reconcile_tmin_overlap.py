from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OLD_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "old_xlsx_daily_standardized.parquet"
)

NEW_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "new_tmin_2022_2025_daily_standardized.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "tmin_overlap_2022_2024_reconciliation.parquet"
)

REPORT_DIRECTORY = PROJECT_ROOT / "05_qc_reports"
ADMIN_DIRECTORY = PROJECT_ROOT / "00_admin"

OVERLAP_START = pd.Timestamp("2022-01-01")
OVERLAP_END = pd.Timestamp("2024-12-31")

EXPECTED_COMMON_STATIONS = 42
MIN_EXPECTED_KEYS = 45_000
MAX_EXPECTED_KEYS = 47_000

AGREEMENT_TOLERANCE = 0.05


def join_unique_text(
    values: pd.Series,
) -> str:
    """Join distinct nonblank values in source order."""
    output: list[str] = []

    for value in values:
        if pd.isna(value):
            continue

        text = str(value).strip()

        if text and text not in output:
            output.append(text)

    return "; ".join(output)


def aggregate_old_group(
    group: pd.DataFrame,
) -> pd.Series:
    """
    Collapse old-XLSX duplicate station-date rows without
    silently choosing between conflicting values.
    """
    ordered = group.sort_values("source_row")

    source_rows = [
        int(value)
        for value in ordered["source_row"]
    ]

    signatures: list[tuple] = []

    for row in ordered.itertuples(index=False):
        numeric_value = (
            None
            if pd.isna(row.tmin_old_xlsx)
            else float(row.tmin_old_xlsx)
        )

        signatures.append(
            (
                numeric_value,
                str(row.tmin_parse_status),
                str(row.tmin_raw),
            )
        )

    unique_signatures = set(signatures)

    if len(ordered) == 1:
        duplicate_status = "unique"

    elif len(unique_signatures) == 1:
        duplicate_status = (
            "exact_duplicate_same_value"
        )

    else:
        duplicate_status = (
            "conflicting_duplicate"
        )

    first = ordered.iloc[0]

    if duplicate_status == "conflicting_duplicate":
        old_compare_value = np.nan
        old_compare_parse_status = (
            "duplicate_conflict"
        )

    else:
        old_compare_value = first[
            "tmin_old_xlsx"
        ]

        old_compare_parse_status = first[
            "tmin_parse_status"
        ]

    return pd.Series(
        {
            "station_id_old":
                first["station_id"],
            "station_name_official_old":
                first["station_name_official"],
            "latitude_old":
                first["latitude"],
            "longitude_old":
                first["longitude"],
            "old_source_row_count":
                len(ordered),
            "old_source_rows":
                ";".join(
                    str(value)
                    for value in source_rows
                ),
            "old_tmin_raw_values":
                join_unique_text(
                    ordered["tmin_raw"]
                ),
            "old_tmin_parse_statuses":
                join_unique_text(
                    ordered[
                        "tmin_parse_status"
                    ]
                ),
            "old_duplicate_status":
                duplicate_status,
            "tmin_old_xlsx_compare":
                old_compare_value,
            "old_compare_parse_status":
                old_compare_parse_status,
        }
    )


def classify_row(
    row: pd.Series,
) -> pd.Series:
    """Assign source-selection and review statuses."""
    old_present = bool(row["old_record_present"])
    new_present = bool(row["new_record_present"])

    old_numeric = pd.notna(
        row["tmin_old_xlsx_compare"]
    )

    new_numeric = pd.notna(
        row["tmin_new_csv"]
    )

    old_conflicting_duplicate = (
        row["old_duplicate_status"]
        == "conflicting_duplicate"
    )

    new_zero = (
        new_numeric
        and math.isclose(
            float(row["tmin_new_csv"]),
            0.0,
            abs_tol=1e-12,
        )
    )

    difference = np.nan

    if old_numeric and new_numeric:
        difference = (
            float(row["tmin_new_csv"])
            - float(
                row["tmin_old_xlsx_compare"]
            )
        )

    selected_value = np.nan
    selected_source = "none"
    review_required = False
    status = "unclassified"

    if new_zero:
        selected_value = float(
            row["tmin_new_csv"]
        )
        selected_source = "new_csv"
        review_required = True
        status = "new_zero_requires_review"

    elif old_conflicting_duplicate:
        review_required = True

        if new_numeric:
            selected_value = float(
                row["tmin_new_csv"]
            )
            selected_source = "new_csv"
            status = (
                "old_conflicting_duplicate_"
                "new_selected"
            )

        else:
            status = (
                "old_conflicting_duplicate_"
                "unresolved"
            )

    elif old_present and new_present:
        if old_numeric and new_numeric:
            if math.isclose(
                float(
                    row[
                        "tmin_old_xlsx_compare"
                    ]
                ),
                float(row["tmin_new_csv"]),
                abs_tol=AGREEMENT_TOLERANCE,
                rel_tol=0.0,
            ):
                selected_value = float(
                    row["tmin_new_csv"]
                )
                selected_source = "new_csv"
                status = "exact_agreement"

            else:
                selected_value = float(
                    row["tmin_new_csv"]
                )
                selected_source = "new_csv"
                review_required = True
                status = (
                    "numeric_conflict_"
                    "new_selected"
                )

        elif new_numeric and not old_numeric:
            selected_value = float(
                row["tmin_new_csv"]
            )
            selected_source = "new_csv"
            status = "new_fills_old_missing"

        elif old_numeric and not new_numeric:
            selected_value = float(
                row[
                    "tmin_old_xlsx_compare"
                ]
            )
            selected_source = "old_xlsx"
            status = "old_fills_new_missing"

        else:
            status = "both_missing"

    elif new_present and not old_present:
        review_required = True

        if new_numeric:
            selected_value = float(
                row["tmin_new_csv"]
            )
            selected_source = "new_csv"
            status = "new_only_record"
        else:
            status = "new_only_missing"

    elif old_present and not new_present:
        review_required = True

        if old_numeric:
            selected_value = float(
                row[
                    "tmin_old_xlsx_compare"
                ]
            )
            selected_source = "old_xlsx"
            status = "old_only_record"
        else:
            status = "old_only_missing"

    return pd.Series(
        {
            "tmin_difference_new_minus_old":
                difference,
            "absolute_tmin_difference":
                (
                    abs(difference)
                    if pd.notna(difference)
                    else np.nan
                ),
            "reconciliation_status":
                status,
            "tmin_selected_candidate":
                selected_value,
            "tmin_selected_source":
                selected_source,
            "reconciliation_review_required":
                review_required,
        }
    )


def main() -> None:
    for required_file in [
        OLD_FILE,
        NEW_FILE,
    ]:
        if not required_file.exists():
            raise FileNotFoundError(
                f"Required file not found: "
                f"{required_file}"
            )

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    ADMIN_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    old_columns = [
        "source_row",
        "station_uid",
        "station_id",
        "station_name_official",
        "latitude",
        "longitude",
        "date",
        "date_parse_status",
        "tmin_raw",
        "tmin_old_xlsx",
        "tmin_parse_status",
    ]

    new_columns = [
        "source_row",
        "source_day_column",
        "source_cell",
        "station_uid",
        "station_id",
        "station_name_raw",
        "station_name_official",
        "latitude",
        "longitude",
        "date",
        "metadata_status",
        "tmin_raw",
        "tmin_new_csv",
        "tmin_parse_status",
        "tmin_zero_flag",
    ]

    old = pd.read_parquet(
        OLD_FILE,
        columns=old_columns,
    )

    new = pd.read_parquet(
        NEW_FILE,
        columns=new_columns,
    )

    old["date"] = pd.to_datetime(
        old["date"]
    )

    new["date"] = pd.to_datetime(
        new["date"]
    )

    old = old.loc[
        old["date_parse_status"].eq("valid")
        & old["date"].between(
            OVERLAP_START,
            OVERLAP_END,
        )
    ].copy()

    new = new.loc[
        new["date"].between(
            OVERLAP_START,
            OVERLAP_END,
        )
    ].copy()

    old_station_uids = set(
        old["station_uid"].dropna()
    )

    new_station_uids = set(
        new["station_uid"].dropna()
    )

    common_station_uids = (
        old_station_uids
        & new_station_uids
    )

    old = old.loc[
        old["station_uid"].isin(
            common_station_uids
        )
    ].copy()

    new = new.loc[
        new["station_uid"].isin(
            common_station_uids
        )
    ].copy()

    new_duplicate_mask = new.duplicated(
        subset=["station_uid", "date"],
        keep=False,
    )

    new_duplicate_keys = new.loc[
        new_duplicate_mask
    ].copy()

    new_duplicate_keys.to_csv(
        REPORT_DIRECTORY
        / "step3c_new_duplicate_keys.csv",
        index=False,
    )

    if not new_duplicate_keys.empty:
        raise ValueError(
            "The new Tmin dataset contains "
            "duplicate station-date keys."
        )

    old_duplicate_mask = old.duplicated(
        subset=["station_uid", "date"],
        keep=False,
    )

    old_duplicate_source_rows = old.loc[
        old_duplicate_mask
    ].sort_values(
        ["station_uid", "date", "source_row"]
    )

    old_duplicate_source_rows.to_csv(
        REPORT_DIRECTORY
        / "step3c_old_duplicate_source_rows.csv",
        index=False,
    )

    old_aggregated = (
        old.groupby(
            ["station_uid", "date"],
            as_index=False,
            sort=True,
        )
        .apply(
            aggregate_old_group,
            include_groups=False,
        )
        .reset_index()
    )

    if "level_2" in old_aggregated.columns:
        old_aggregated = (
            old_aggregated.drop(
                columns=["level_2"]
            )
        )

    new_daily = new.rename(
        columns={
            "source_row":
                "new_source_row",
            "source_day_column":
                "new_source_day_column",
            "source_cell":
                "new_source_cell",
            "station_id":
                "station_id_new",
            "station_name_raw":
                "station_name_raw_new",
            "station_name_official":
                "station_name_official_new",
            "latitude":
                "latitude_new",
            "longitude":
                "longitude_new",
            "metadata_status":
                "new_metadata_status",
            "tmin_raw":
                "new_tmin_raw",
            "tmin_parse_status":
                "new_tmin_parse_status",
        }
    )

    reconciliation = old_aggregated.merge(
        new_daily,
        on=["station_uid", "date"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    reconciliation[
        "old_record_present"
    ] = reconciliation["_merge"].isin(
        ["both", "left_only"]
    )

    reconciliation[
        "new_record_present"
    ] = reconciliation["_merge"].isin(
        ["both", "right_only"]
    )

    reconciliation["station_id"] = (
        reconciliation["station_id_new"]
        .combine_first(
            reconciliation["station_id_old"]
        )
    )

    reconciliation[
        "station_name_official"
    ] = (
        reconciliation[
            "station_name_official_new"
        ]
        .combine_first(
            reconciliation[
                "station_name_official_old"
            ]
        )
    )

    reconciliation["latitude"] = (
        reconciliation["latitude_new"]
        .combine_first(
            reconciliation["latitude_old"]
        )
    )

    reconciliation["longitude"] = (
        reconciliation["longitude_new"]
        .combine_first(
            reconciliation["longitude_old"]
        )
    )

    classification = reconciliation.apply(
        classify_row,
        axis=1,
    )

    reconciliation = pd.concat(
        [
            reconciliation,
            classification,
        ],
        axis=1,
    )

    reconciliation[
        "year"
    ] = reconciliation["date"].dt.year.astype(
        "int16"
    )

    reconciliation[
        "month"
    ] = reconciliation["date"].dt.month.astype(
        "int8"
    )

    reconciliation[
        "day"
    ] = reconciliation["date"].dt.day.astype(
        "int8"
    )

    final_columns = [
        "station_uid",
        "station_id",
        "station_name_official",
        "latitude",
        "longitude",
        "date",
        "year",
        "month",
        "day",

        "old_record_present",
        "old_source_row_count",
        "old_source_rows",
        "old_tmin_raw_values",
        "old_tmin_parse_statuses",
        "old_duplicate_status",
        "tmin_old_xlsx_compare",
        "old_compare_parse_status",

        "new_record_present",
        "new_source_row",
        "new_source_day_column",
        "new_source_cell",
        "station_name_raw_new",
        "new_tmin_raw",
        "tmin_new_csv",
        "new_tmin_parse_status",
        "tmin_zero_flag",

        "tmin_difference_new_minus_old",
        "absolute_tmin_difference",
        "reconciliation_status",
        "tmin_selected_candidate",
        "tmin_selected_source",
        "reconciliation_review_required",
    ]

    reconciliation = (
        reconciliation[final_columns]
        .sort_values(
            ["station_uid", "date"]
        )
        .reset_index(drop=True)
    )

    duplicate_final_keys = (
        reconciliation.duplicated(
            subset=["station_uid", "date"],
            keep=False,
        )
    )

    if duplicate_final_keys.any():
        raise ValueError(
            "Final reconciliation table contains "
            "duplicate station-date keys."
        )

    reconciliation.to_parquet(
        OUTPUT_FILE,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    old_duplicate_keys = (
        reconciliation.loc[
            reconciliation[
                "old_source_row_count"
            ].fillna(0).gt(1)
        ]
        .copy()
    )

    old_duplicate_keys.to_csv(
        REPORT_DIRECTORY
        / "step3c_old_duplicate_keys.csv",
        index=False,
    )

    numeric_conflicts = (
        reconciliation.loc[
            reconciliation[
                "reconciliation_status"
            ].eq(
                "numeric_conflict_"
                "new_selected"
            )
        ]
        .copy()
    )

    numeric_conflicts.to_csv(
        REPORT_DIRECTORY
        / "step3c_numeric_conflicts.csv",
        index=False,
    )

    zero_records = (
        reconciliation.loc[
            reconciliation[
                "reconciliation_status"
            ].eq(
                "new_zero_requires_review"
            )
        ]
        .copy()
    )

    zero_records.to_csv(
        REPORT_DIRECTORY
        / "step3c_new_zero_records.csv",
        index=False,
    )

    source_gaps = (
        reconciliation.loc[
            reconciliation[
                "reconciliation_status"
            ].isin(
                [
                    "new_only_record",
                    "new_only_missing",
                    "old_only_record",
                    "old_only_missing",
                ]
            )
        ]
        .copy()
    )

    source_gaps.to_csv(
        REPORT_DIRECTORY
        / "step3c_source_coverage_gaps.csv",
        index=False,
    )

    review_records = (
        reconciliation.loc[
            reconciliation[
                "reconciliation_review_required"
            ]
        ]
        .copy()
    )

    review_records.to_csv(
        REPORT_DIRECTORY
        / "step3c_review_required_records.csv",
        index=False,
    )

    status_counts = (
        reconciliation[
            "reconciliation_status"
        ]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )

    old_duplicate_status_counts = (
        reconciliation[
            "old_duplicate_status"
        ]
        .fillna("old_record_absent")
        .value_counts()
        .sort_index()
        .to_dict()
    )

    minimum_date = reconciliation[
        "date"
    ].min()

    maximum_date = reconciliation[
        "date"
    ].max()

    summary = {
        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "overlap_start":
            OVERLAP_START.date().isoformat(),
        "overlap_end":
            OVERLAP_END.date().isoformat(),
        "common_stations":
            len(common_station_uids),
        "old_filtered_source_rows":
            len(old),
        "new_filtered_daily_rows":
            len(new),
        "old_unique_station_date_keys":
            len(old_aggregated),
        "new_unique_station_date_keys":
            len(new_daily),
        "reconciliation_rows":
            len(reconciliation),
        "minimum_reconciled_date":
            minimum_date.date().isoformat(),
        "maximum_reconciled_date":
            maximum_date.date().isoformat(),
        "old_duplicate_source_rows":
            len(old_duplicate_source_rows),
        "old_duplicate_keys":
            len(old_duplicate_keys),
        "new_duplicate_keys":
            len(new_duplicate_keys),
        "numeric_conflicts":
            len(numeric_conflicts),
        "new_zero_records":
            len(zero_records),
        "source_coverage_gaps":
            len(source_gaps),
        "review_required_records":
            len(review_records),
        "reconciliation_status_counts":
            {
                str(key): int(value)
                for key, value
                in status_counts.items()
            },
        "old_duplicate_status_counts":
            {
                str(key): int(value)
                for key, value
                in old_duplicate_status_counts.items()
            },
        "output_file":
            str(OUTPUT_FILE),
        "output_size_bytes":
            OUTPUT_FILE.stat().st_size,
    }

    (
        REPORT_DIRECTORY
        / "step3c_reconciliation_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    parquet_schema = pq.ParquetFile(
        OUTPUT_FILE
    ).schema_arrow

    (
        ADMIN_DIRECTORY
        / "step3c_parquet_schema.json"
    ).write_text(
        json.dumps(
            {
                field.name: str(field.type)
                for field in parquet_schema
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "STEP 3C: TMIN OVERLAP RECONCILIATION",
        "=" * 42,
        (
            "Overlap period: "
            f"{OVERLAP_START.date()} to "
            f"{OVERLAP_END.date()}"
        ),
        (
            "Common stations: "
            f"{len(common_station_uids)}"
        ),
        (
            "Old filtered source rows: "
            f"{len(old):,}"
        ),
        (
            "Old unique station-date keys: "
            f"{len(old_aggregated):,}"
        ),
        (
            "New station-date keys: "
            f"{len(new_daily):,}"
        ),
        (
            "Final reconciliation rows: "
            f"{len(reconciliation):,}"
        ),
        (
            "Old duplicate source rows: "
            f"{len(old_duplicate_source_rows):,}"
        ),
        (
            "Old duplicate station-date keys: "
            f"{len(old_duplicate_keys):,}"
        ),
        (
            "New duplicate station-date keys: "
            f"{len(new_duplicate_keys):,}"
        ),
        (
            "Numeric conflicts: "
            f"{len(numeric_conflicts):,}"
        ),
        (
            "New zero values requiring review: "
            f"{len(zero_records):,}"
        ),
        (
            "Source-coverage gaps: "
            f"{len(source_gaps):,}"
        ),
        (
            "Total records requiring review: "
            f"{len(review_records):,}"
        ),
        "",
        "Reconciliation statuses:",
        *[
            f"- {key}: {value:,}"
            for key, value
            in status_counts.items()
        ],
        "",
        f"Output: {OUTPUT_FILE}",
    ]

    (
        REPORT_DIRECTORY
        / "step3c_reconciliation_report.txt"
    ).write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(report_lines))

    failures: list[str] = []

    if (
        len(common_station_uids)
        != EXPECTED_COMMON_STATIONS
    ):
        failures.append(
            f"Expected {EXPECTED_COMMON_STATIONS} "
            "common stations but found "
            f"{len(common_station_uids)}."
        )

    if not new_duplicate_keys.empty:
        failures.append(
            "The new dataset contains duplicate "
            "station-date keys."
        )

    if not (
        MIN_EXPECTED_KEYS
        <= len(reconciliation)
        <= MAX_EXPECTED_KEYS
    ):
        failures.append(
            "Unexpected number of overlap keys: "
            f"{len(reconciliation):,}."
        )

    if minimum_date != OVERLAP_START:
        failures.append(
            "Unexpected minimum overlap date: "
            f"{minimum_date}."
        )

    if maximum_date != OVERLAP_END:
        failures.append(
            "Unexpected maximum overlap date: "
            f"{maximum_date}."
        )

    if duplicate_final_keys.any():
        failures.append(
            "Duplicate keys exist in the final "
            "reconciliation table."
        )

    if failures:
        raise SystemExit(
            "\nSTEP 3C FAILED:\n- "
            + "\n- ".join(failures)
        )

    print("\nSTEP 3C PASSED.")


if __name__ == "__main__":
    main()
