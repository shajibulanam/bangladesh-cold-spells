from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "temperature_daily_unified_preliminary.parquet"
)

INVALID_DATE_FILE = (
    PROJECT_ROOT
    / "05_qc_reports"
    / "step3d_excluded_invalid_old_dates.csv"
)

RULE_FILE = (
    PROJECT_ROOT
    / "00_admin"
    / "step4a_qc_rules.yaml"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "temperature_daily_qc_stage1.parquet"
)

REPORT_DIRECTORY = PROJECT_ROOT / "05_qc_reports"
ADMIN_DIRECTORY = PROJECT_ROOT / "00_admin"

EXPECTED_STATIONS = 48
EXPECTED_INVALID_DATE_ROWS = 54
EXPECTED_START_DATE = pd.Timestamp("1981-01-01")
EXPECTED_END_DATE = pd.Timestamp("2025-12-31")


def load_rules() -> dict[str, Any]:
    """Load the documented QC screening rules."""
    if not RULE_FILE.exists():
        raise FileNotFoundError(
            f"QC rule file not found: {RULE_FILE}"
        )

    with RULE_FILE.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        rules = yaml.safe_load(file_handle)

    required_paths = [
        ("physical_screening", "tmin_minimum"),
        ("physical_screening", "tmin_maximum"),
        ("physical_screening", "tmax_minimum"),
        ("physical_screening", "tmax_maximum"),
        (
            "diurnal_temperature_range",
            "zero_tolerance",
        ),
        (
            "diurnal_temperature_range",
            "high_dtr_threshold",
        ),
        (
            "zero_temperature",
            "priority_winter_months",
        ),
    ]

    for first_key, second_key in required_paths:
        if (
            first_key not in rules
            or second_key not in rules[first_key]
        ):
            raise ValueError(
                "Missing QC rule: "
                f"{first_key}.{second_key}"
            )

    return rules


def make_flag_codes(
    data: pd.DataFrame,
    code_columns: list[tuple[str, str]],
) -> tuple[pd.Series, pd.Series]:
    """
    Build semicolon-delimited flag codes and a flag count
    without changing any observation.
    """
    codes = np.full(
        len(data),
        "",
        dtype=object,
    )

    counts = np.zeros(
        len(data),
        dtype=np.int16,
    )

    for code, column in code_columns:
        mask = (
            data[column]
            .fillna(False)
            .astype(bool)
            .to_numpy()
        )

        counts[mask] += 1

        codes[mask] = np.where(
            codes[mask] == "",
            code,
            codes[mask] + ";" + code,
        )

    return (
        pd.Series(
            codes,
            index=data.index,
            dtype="string",
        ),
        pd.Series(
            counts,
            index=data.index,
            dtype="int16",
        ),
    )


def add_review_fields(
    data: pd.DataFrame,
    issue_type: str,
    recommended_action: str,
) -> pd.DataFrame:
    """Add empty manual-review fields to a review table."""
    output = data.copy()

    output["review_issue_type"] = issue_type
    output["recommended_action"] = recommended_action
    output["review_status"] = "pending"

    output["decision"] = ""
    output["corrected_tmin"] = ""
    output["corrected_tmax"] = ""
    output["decision_reason"] = ""
    output["verification_source"] = ""
    output["reviewer"] = ""
    output["review_date"] = ""

    return output


def write_review_table(
    data: pd.DataFrame,
    path: Path,
    issue_type: str,
    recommended_action: str,
    columns: list[str],
) -> None:
    """Write a consistently structured review table."""
    reviewed = add_review_fields(
        data,
        issue_type,
        recommended_action,
    )

    final_columns = [
        column
        for column in columns
        if column in reviewed.columns
    ] + [
        "review_issue_type",
        "recommended_action",
        "review_status",
        "decision",
        "corrected_tmin",
        "corrected_tmax",
        "decision_reason",
        "verification_source",
        "reviewer",
        "review_date",
    ]

    reviewed[final_columns].to_csv(
        path,
        index=False,
    )


def main() -> None:
    for required_path in [
        INPUT_FILE,
        INVALID_DATE_FILE,
        RULE_FILE,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {required_path}"
            )

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    ADMIN_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    rules = load_rules()

    tmin_min = float(
        rules["physical_screening"][
            "tmin_minimum"
        ]
    )

    tmin_max = float(
        rules["physical_screening"][
            "tmin_maximum"
        ]
    )

    tmax_min = float(
        rules["physical_screening"][
            "tmax_minimum"
        ]
    )

    tmax_max = float(
        rules["physical_screening"][
            "tmax_maximum"
        ]
    )

    zero_tolerance = float(
        rules[
            "diurnal_temperature_range"
        ]["zero_tolerance"]
    )

    high_dtr_threshold = float(
        rules[
            "diurnal_temperature_range"
        ]["high_dtr_threshold"]
    )

    winter_months = {
        int(value)
        for value in rules[
            "zero_temperature"
        ]["priority_winter_months"]
    }

    data = pd.read_parquet(INPUT_FILE)

    input_row_count = len(data)

    data["date"] = pd.to_datetime(
        data["date"]
    )

    duplicate_input_keys = data.duplicated(
        subset=[
            "station_uid",
            "date",
        ],
        keep=False,
    )

    if duplicate_input_keys.any():
        raise ValueError(
            "The Step 3D input contains duplicate "
            "station-date keys."
        )

    if data["station_uid"].nunique() != EXPECTED_STATIONS:
        raise ValueError(
            f"Expected {EXPECTED_STATIONS} stations; "
            f"found {data['station_uid'].nunique()}."
        )

    if data["date"].min() != EXPECTED_START_DATE:
        raise ValueError(
            "Unexpected input start date: "
            f"{data['date'].min()}."
        )

    if data["date"].max() != EXPECTED_END_DATE:
        raise ValueError(
            "Unexpected input end date: "
            f"{data['date'].max()}."
        )

    # Preserve untouched working copies.
    data["tmin_qc_stage1"] = data[
        "tmin_preliminary"
    ].astype("float32")

    data["tmax_qc_stage1"] = data[
        "tmax_preliminary"
    ].astype("float32")

    # Missingness flags.
    data["qc_tmin_missing"] = (
        data["tmin_qc_stage1"].isna()
    )

    data["qc_tmax_missing"] = (
        data["tmax_qc_stage1"].isna()
    )

    data[
        "qc_tmax_missing_expected_2025"
    ] = (
        data["year"].eq(2025)
        & data["qc_tmax_missing"]
    )

    data[
        "qc_tmax_missing_unexpected"
    ] = (
        data["year"].le(2024)
        & data["qc_tmax_missing"]
    )

    # Zero Tmin screening.
    data["qc_tmin_zero"] = (
        data["tmin_qc_stage1"].notna()
        & np.isclose(
            data["tmin_qc_stage1"],
            0.0,
            atol=zero_tolerance,
            rtol=0.0,
        )
    )

    data[
        "qc_tmin_zero_winter_priority"
    ] = (
        data["qc_tmin_zero"]
        & data["month"].isin(
            winter_months
        )
    )

    # Broad physical-range screening.
    data[
        "qc_tmin_physical_range"
    ] = (
        data["tmin_qc_stage1"].notna()
        & (
            data["tmin_qc_stage1"].lt(
                tmin_min
            )
            | data["tmin_qc_stage1"].gt(
                tmin_max
            )
        )
    )

    data[
        "qc_tmax_physical_range"
    ] = (
        data["tmax_qc_stage1"].notna()
        & (
            data["tmax_qc_stage1"].lt(
                tmax_min
            )
            | data["tmax_qc_stage1"].gt(
                tmax_max
            )
        )
    )

    # Internal Tmax-Tmin consistency.
    both_temperatures = (
        data["tmin_qc_stage1"].notna()
        & data["tmax_qc_stage1"].notna()
    )

    data[
        "diurnal_temperature_range"
    ] = (
        data["tmax_qc_stage1"]
        - data["tmin_qc_stage1"]
    ).astype("float32")

    data["qc_tmax_lt_tmin"] = (
        both_temperatures
        & data[
            "diurnal_temperature_range"
        ].lt(0.0)
    )

    data["qc_dtr_zero"] = (
        both_temperatures
        & np.isclose(
            data[
                "diurnal_temperature_range"
            ],
            0.0,
            atol=zero_tolerance,
            rtol=0.0,
        )
    )

    data["qc_dtr_high"] = (
        both_temperatures
        & data[
            "diurnal_temperature_range"
        ].gt(high_dtr_threshold)
    )

    # Provenance and inherited-review flags.
    data["qc_metadata_pending"] = (
        data[
            "metadata_pending_flag"
        ].fillna(False).astype(bool)
    )

    data["qc_old_duplicate"] = (
        pd.to_numeric(
            data[
                "old_source_row_count"
            ],
            errors="coerce",
        )
        .fillna(0)
        .gt(1)
    )

    duplicate_text = (
        data[
            "old_record_duplicate_status"
        ].fillna("").astype(str)
        + ";"
        + data[
            "old_tmin_duplicate_status"
        ].fillna("").astype(str)
        + ";"
        + data[
            "old_tmax_duplicate_status"
        ].fillna("").astype(str)
    )

    data[
        "qc_old_duplicate_conflict"
    ] = duplicate_text.str.contains(
        "conflict",
        case=False,
        na=False,
    )

    reconciliation_text = data[
        "reconciliation_status"
    ].fillna("").astype(str)

    data[
        "qc_source_numeric_conflict"
    ] = reconciliation_text.str.contains(
        "numeric_conflict",
        case=False,
        na=False,
    )

    data[
        "qc_source_duplicate_conflict"
    ] = reconciliation_text.str.contains(
        "conflicting_duplicate",
        case=False,
        na=False,
    )

    data[
        "qc_inherited_value_review"
    ] = data[
        "record_review_required"
    ].fillna(False).astype(bool)

    # Record-level review decisions for this stage.
    critical_flags = (
        data[
            "qc_tmin_physical_range"
        ]
        | data[
            "qc_tmax_physical_range"
        ]
        | data["qc_tmax_lt_tmin"]
        | data[
            "qc_old_duplicate_conflict"
        ]
    )

    high_priority_flags = (
        data[
            "qc_tmin_zero_winter_priority"
        ]
        | data[
            "qc_source_numeric_conflict"
        ]
        | data[
            "qc_source_duplicate_conflict"
        ]
    )

    review_flags = (
        data["qc_tmin_zero"]
        | data["qc_dtr_zero"]
        | data["qc_dtr_high"]
        | data[
            "qc_inherited_value_review"
        ]
    )

    data[
        "qc_stage1_value_review_required"
    ] = (
        critical_flags
        | high_priority_flags
        | review_flags
    )

    data["qc_stage1_priority"] = np.select(
        [
            critical_flags,
            high_priority_flags,
            review_flags,
            data["qc_metadata_pending"],
        ],
        [
            "critical",
            "high",
            "review",
            "metadata_pending",
        ],
        default="none",
    )

    any_missing = (
        data["qc_tmin_missing"]
        | data["qc_tmax_missing"]
    )

    data["qc_stage1_status"] = np.select(
        [
            data[
                "qc_stage1_value_review_required"
            ],
            data["qc_metadata_pending"],
            any_missing,
        ],
        [
            "review_required",
            "metadata_pending",
            "pass_with_missing_data",
        ],
        default="pass_stage1",
    )

    code_columns = [
        (
            "TMIN_ZERO",
            "qc_tmin_zero",
        ),
        (
            "TMIN_ZERO_WINTER",
            "qc_tmin_zero_winter_priority",
        ),
        (
            "TMIN_PHYSICAL_RANGE",
            "qc_tmin_physical_range",
        ),
        (
            "TMAX_PHYSICAL_RANGE",
            "qc_tmax_physical_range",
        ),
        (
            "TMAX_LT_TMIN",
            "qc_tmax_lt_tmin",
        ),
        (
            "DTR_ZERO",
            "qc_dtr_zero",
        ),
        (
            "DTR_HIGH",
            "qc_dtr_high",
        ),
        (
            "OLD_DUPLICATE",
            "qc_old_duplicate",
        ),
        (
            "OLD_DUPLICATE_CONFLICT",
            "qc_old_duplicate_conflict",
        ),
        (
            "SOURCE_NUMERIC_CONFLICT",
            "qc_source_numeric_conflict",
        ),
        (
            "SOURCE_DUPLICATE_CONFLICT",
            "qc_source_duplicate_conflict",
        ),
        (
            "TMAX_MISSING_UNEXPECTED",
            "qc_tmax_missing_unexpected",
        ),
        (
            "METADATA_PENDING",
            "qc_metadata_pending",
        ),
    ]

    (
        data["qc_flag_codes"],
        data["qc_flag_count"],
    ) = make_flag_codes(
        data,
        code_columns,
    )

    # Validate that Step 4A did not change values.
    if not data[
        "tmin_qc_stage1"
    ].equals(
        data[
            "tmin_preliminary"
        ].astype("float32")
    ):
        raise ValueError(
            "Tmin values changed during Step 4A."
        )

    if not data[
        "tmax_qc_stage1"
    ].equals(
        data[
            "tmax_preliminary"
        ].astype("float32")
    ):
        raise ValueError(
            "Tmax values changed during Step 4A."
        )

    duplicate_output_keys = data.duplicated(
        subset=[
            "station_uid",
            "date",
        ],
        keep=False,
    )

    if duplicate_output_keys.any():
        raise ValueError(
            "Duplicate keys were introduced in Step 4A."
        )

    # Review-table columns.
    review_columns = [
        "station_uid",
        "station_id",
        "station_name_display",
        "date",
        "year",
        "month",
        "day",
        "record_origin",
        "metadata_status",
        "tmin_preliminary",
        "tmax_preliminary",
        "diurnal_temperature_range",
        "tmin_source",
        "tmax_source",
        "tmin_selection_status",
        "tmax_selection_status",
        "reconciliation_status",
        "old_source_rows",
        "new_source_cell",
        "qc_flag_codes",
        "qc_flag_count",
        "qc_stage1_priority",
        "qc_stage1_status",
    ]

    all_review = data.loc[
        data[
            "qc_stage1_value_review_required"
        ]
    ].copy()

    all_review[review_columns].to_csv(
        REPORT_DIRECTORY
        / "step4a_all_value_review_records.csv",
        index=False,
    )

    write_review_table(
        data.loc[
            data["qc_tmin_zero"]
        ],
        REPORT_DIRECTORY
        / "step4a_zero_tmin_review.csv",
        "zero_tmin",
        (
            "Verify the original BMD record and "
            "compare with adjacent days and nearby stations."
        ),
        review_columns,
    )

    write_review_table(
        data.loc[
            data["qc_tmax_lt_tmin"]
        ],
        REPORT_DIRECTORY
        / "step4a_tmax_below_tmin_review.csv",
        "tmax_below_tmin",
        (
            "Verify both Tmax and Tmin against the "
            "original BMD record; check for swapped "
            "or mistyped values."
        ),
        review_columns,
    )

    write_review_table(
        data.loc[
            data[
                "qc_tmin_physical_range"
            ]
            | data[
                "qc_tmax_physical_range"
            ]
        ],
        REPORT_DIRECTORY
        / "step4a_physical_range_review.csv",
        "physical_range",
        (
            "Verify the source value. Do not discard "
            "an extreme observation without source evidence."
        ),
        review_columns,
    )

    write_review_table(
        data.loc[
            data["qc_dtr_zero"]
            | data["qc_dtr_high"]
        ],
        REPORT_DIRECTORY
        / "step4a_diurnal_range_review.csv",
        "diurnal_temperature_range",
        (
            "Compare Tmax and Tmin with adjacent dates, "
            "station climatology and neighbouring stations."
        ),
        review_columns,
    )

    write_review_table(
        data.loc[
            data[
                "qc_source_numeric_conflict"
            ]
            | data[
                "qc_source_duplicate_conflict"
            ]
        ],
        REPORT_DIRECTORY
        / "step4a_source_conflict_review.csv",
        "source_conflict",
        (
            "Retain both sources. Verify against the "
            "most recent official BMD extract or source sheet."
        ),
        review_columns,
    )

    write_review_table(
        data.loc[
            data[
                "qc_old_duplicate_conflict"
            ]
        ],
        REPORT_DIRECTORY
        / "step4a_duplicate_conflict_review.csv",
        "old_source_duplicate_conflict",
        (
            "Verify the conflicting duplicate source rows. "
            "Do not average conflicting observations."
        ),
        review_columns,
    )

    # Invalid dates exist outside the unified daily table.
    invalid_dates = pd.read_csv(
        INVALID_DATE_FILE,
        dtype=str,
        keep_default_na=False,
    )

    if len(invalid_dates) != EXPECTED_INVALID_DATE_ROWS:
        raise ValueError(
            "Expected "
            f"{EXPECTED_INVALID_DATE_ROWS} invalid-date rows; "
            f"found {len(invalid_dates)}."
        )

    invalid_dates[
        "review_issue_type"
    ] = "invalid_calendar_date"

    invalid_dates[
        "recommended_action"
    ] = (
        "Check the original BMD source. Correct the date "
        "only when the intended date is verifiable; "
        "otherwise reject the source row."
    )

    invalid_dates["review_status"] = "pending"
    invalid_dates["corrected_date"] = ""
    invalid_dates["corrected_tmin"] = ""
    invalid_dates["corrected_tmax"] = ""
    invalid_dates["decision"] = ""
    invalid_dates["decision_reason"] = ""
    invalid_dates["verification_source"] = ""
    invalid_dates["reviewer"] = ""
    invalid_dates["review_date"] = ""

    invalid_dates.to_csv(
        REPORT_DIRECTORY
        / "step4a_invalid_date_review_template.csv",
        index=False,
    )

    # Duplicate review template including consistent duplicates.
    duplicate_rows = data.loc[
        data["qc_old_duplicate"]
    ].copy()

    write_review_table(
        duplicate_rows,
        REPORT_DIRECTORY
        / "step4a_all_old_duplicate_review.csv",
        "old_source_duplicate",
        (
            "Confirm whether duplicate source rows are exact, "
            "consistent or conflicting before final cleaning."
        ),
        review_columns,
    )

    # Station-level summary.
    station_summary = (
        data.groupby(
            [
                "station_uid",
                "station_id",
                "station_name_display",
                "metadata_status",
            ],
            dropna=False,
        )
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            total_records=("date", "size"),
            numeric_tmin=(
                "tmin_qc_stage1",
                "count",
            ),
            missing_tmin=(
                "qc_tmin_missing",
                "sum",
            ),
            numeric_tmax=(
                "tmax_qc_stage1",
                "count",
            ),
            missing_tmax=(
                "qc_tmax_missing",
                "sum",
            ),
            zero_tmin=(
                "qc_tmin_zero",
                "sum",
            ),
            physical_tmin_flags=(
                "qc_tmin_physical_range",
                "sum",
            ),
            physical_tmax_flags=(
                "qc_tmax_physical_range",
                "sum",
            ),
            tmax_below_tmin=(
                "qc_tmax_lt_tmin",
                "sum",
            ),
            zero_dtr=(
                "qc_dtr_zero",
                "sum",
            ),
            high_dtr=(
                "qc_dtr_high",
                "sum",
            ),
            source_conflicts=(
                "qc_source_numeric_conflict",
                "sum",
            ),
            duplicate_conflicts=(
                "qc_old_duplicate_conflict",
                "sum",
            ),
            value_review_records=(
                "qc_stage1_value_review_required",
                "sum",
            ),
        )
        .reset_index()
        .sort_values("station_uid")
    )

    station_summary.to_csv(
        REPORT_DIRECTORY
        / "step4a_station_qc_summary.csv",
        index=False,
    )

    # Station-year missingness summary.
    station_year_summary = (
        data.groupby(
            [
                "station_uid",
                "station_name_display",
                "year",
            ],
            dropna=False,
        )
        .agg(
            total_records=("date", "size"),
            numeric_tmin=(
                "tmin_qc_stage1",
                "count",
            ),
            missing_tmin=(
                "qc_tmin_missing",
                "sum",
            ),
            numeric_tmax=(
                "tmax_qc_stage1",
                "count",
            ),
            missing_tmax=(
                "qc_tmax_missing",
                "sum",
            ),
            value_review_records=(
                "qc_stage1_value_review_required",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "station_uid",
                "year",
            ]
        )
    )

    station_year_summary[
        "tmin_completeness_percent"
    ] = (
        100.0
        * station_year_summary[
            "numeric_tmin"
        ]
        / station_year_summary[
            "total_records"
        ]
    )

    station_year_summary[
        "tmax_completeness_percent"
    ] = (
        100.0
        * station_year_summary[
            "numeric_tmax"
        ]
        / station_year_summary[
            "total_records"
        ]
    )

    station_year_summary.to_csv(
        REPORT_DIRECTORY
        / "step4a_station_year_missingness.csv",
        index=False,
    )

    priority_counts = (
        data["qc_stage1_priority"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )

    status_counts = (
        data["qc_stage1_status"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )

    flag_counts = {
        column: int(
            data[column].sum()
        )
        for _, column in code_columns
    }

    # Save QC-layer dataset only after validation.
    data.to_parquet(
        OUTPUT_FILE,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    output_parquet = pq.ParquetFile(
        OUTPUT_FILE
    )

    output_row_count = (
        output_parquet.metadata.num_rows
    )

    summary = {
        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "input_file":
            str(INPUT_FILE),
        "output_file":
            str(OUTPUT_FILE),
        "input_rows":
            input_row_count,
        "output_rows":
            output_row_count,
        "unique_stations":
            int(
                data["station_uid"].nunique()
            ),
        "minimum_date":
            data["date"].min().date().isoformat(),
        "maximum_date":
            data["date"].max().date().isoformat(),
        "invalid_date_rows_outside_daily_table":
            len(invalid_dates),
        "value_review_records":
            len(all_review),
        "priority_counts":
            {
                str(key): int(value)
                for key, value
                in priority_counts.items()
            },
        "status_counts":
            {
                str(key): int(value)
                for key, value
                in status_counts.items()
            },
        "flag_counts":
            flag_counts,
        "numeric_tmin":
            int(
                data[
                    "tmin_qc_stage1"
                ].notna().sum()
            ),
        "missing_tmin":
            int(
                data[
                    "qc_tmin_missing"
                ].sum()
            ),
        "numeric_tmax":
            int(
                data[
                    "tmax_qc_stage1"
                ].notna().sum()
            ),
        "missing_tmax":
            int(
                data[
                    "qc_tmax_missing"
                ].sum()
            ),
        "output_size_bytes":
            OUTPUT_FILE.stat().st_size,
        "values_changed":
            False,
    }

    (
        REPORT_DIRECTORY
        / "step4a_qc_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    schema = output_parquet.schema_arrow

    (
        ADMIN_DIRECTORY
        / "step4a_parquet_schema.json"
    ).write_text(
        json.dumps(
            {
                field.name: str(field.type)
                for field in schema
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "STEP 4A: FIRST-PASS QUALITY CONTROL",
        "=" * 41,
        (
            "Input rows: "
            f"{input_row_count:,}"
        ),
        (
            "Output rows: "
            f"{output_row_count:,}"
        ),
        (
            "Unique stations: "
            f"{data['station_uid'].nunique()}"
        ),
        (
            "Date range: "
            f"{data['date'].min().date()} to "
            f"{data['date'].max().date()}"
        ),
        (
            "Invalid-date source rows awaiting review: "
            f"{len(invalid_dates):,}"
        ),
        (
            "Records requiring value review: "
            f"{len(all_review):,}"
        ),
        (
            "Missing Tmin records: "
            f"{data['qc_tmin_missing'].sum():,}"
        ),
        (
            "Missing Tmax records: "
            f"{data['qc_tmax_missing'].sum():,}"
        ),
        (
            "Zero Tmin flags: "
            f"{data['qc_tmin_zero'].sum():,}"
        ),
        (
            "Winter zero Tmin priority flags: "
            f"{data['qc_tmin_zero_winter_priority'].sum():,}"
        ),
        (
            "Tmin physical-range flags: "
            f"{data['qc_tmin_physical_range'].sum():,}"
        ),
        (
            "Tmax physical-range flags: "
            f"{data['qc_tmax_physical_range'].sum():,}"
        ),
        (
            "Tmax below Tmin flags: "
            f"{data['qc_tmax_lt_tmin'].sum():,}"
        ),
        (
            "Zero DTR flags: "
            f"{data['qc_dtr_zero'].sum():,}"
        ),
        (
            "High DTR flags: "
            f"{data['qc_dtr_high'].sum():,}"
        ),
        (
            "Old-source duplicate records: "
            f"{data['qc_old_duplicate'].sum():,}"
        ),
        (
            "Old-source duplicate conflicts: "
            f"{data['qc_old_duplicate_conflict'].sum():,}"
        ),
        (
            "Source numeric conflicts: "
            f"{data['qc_source_numeric_conflict'].sum():,}"
        ),
        (
            "Metadata-pending records: "
            f"{data['qc_metadata_pending'].sum():,}"
        ),
        "",
        "QC priorities:",
        *[
            f"- {key}: {value:,}"
            for key, value
            in priority_counts.items()
        ],
        "",
        "QC statuses:",
        *[
            f"- {key}: {value:,}"
            for key, value
            in status_counts.items()
        ],
        "",
        "No temperature value was changed.",
        "",
        f"Output: {OUTPUT_FILE}",
    ]

    (
        REPORT_DIRECTORY
        / "step4a_qc_report.txt"
    ).write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(report_lines))

    failures: list[str] = []

    if output_row_count != input_row_count:
        failures.append(
            "Output row count differs from input."
        )

    if data[
        "station_uid"
    ].nunique() != EXPECTED_STATIONS:
        failures.append(
            "Unexpected station count."
        )

    if duplicate_output_keys.any():
        failures.append(
            "Duplicate station-date keys exist."
        )

    if len(invalid_dates) != EXPECTED_INVALID_DATE_ROWS:
        failures.append(
            "Unexpected invalid-date review count."
        )

    if data[
        "date"
    ].min() != EXPECTED_START_DATE:
        failures.append(
            "Unexpected minimum date."
        )

    if data[
        "date"
    ].max() != EXPECTED_END_DATE:
        failures.append(
            "Unexpected maximum date."
        )

    if failures:
        raise SystemExit(
            "\nSTEP 4A FAILED:\n- "
            + "\n- ".join(failures)
        )

    print("\nSTEP 4A PASSED.")


if __name__ == "__main__":
    main()
