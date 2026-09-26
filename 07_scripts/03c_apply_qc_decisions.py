from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "temperature_daily_qc_stage1.parquet"
)

VALUE_LEDGER_FILE = (
    PROJECT_ROOT
    / "04_clean_data"
    / "qc_decisions"
    / "value_decisions.csv"
)

INVALID_DATE_LEDGER_FILE = (
    PROJECT_ROOT
    / "04_clean_data"
    / "qc_decisions"
    / "invalid_date_decisions.csv"
)

RULE_FILE = (
    PROJECT_ROOT
    / "00_admin"
    / "step4a_qc_rules.yaml"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "04_clean_data"
    / "temperature_daily_cleaned_stage1.parquet"
)

REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "05_qc_reports"
)

ADMIN_DIRECTORY = (
    PROJECT_ROOT
    / "00_admin"
)

EXPECTED_INVALID_DATE_DECISIONS = 54

VALUE_DECISIONS = {
    "keep",
    "set_tmin_missing",
    "set_tmax_missing",
    "set_both_missing",
    "correct_tmin",
    "correct_tmax",
    "correct_both",
    "select_old_tmin",
    "select_new_tmin",
}

INVALID_DATE_DECISIONS = {
    "reject_source_row",
    "insert_corrected_date",
    "replace_existing_date",
}

EVIDENCE_FIELDS = [
    "decision_reason",
    "verification_source",
    "reviewer",
    "review_date",
]


def parse_optional_float(
    value: Any,
) -> float | None:
    """Return a finite float or None."""
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()

    if text == "":
        return None

    number = float(text)

    if not math.isfinite(number):
        raise ValueError(
            f"Non-finite corrected value: {value!r}"
        )

    return number


def values_equal(
    first: Any,
    second: Any,
) -> bool:
    """Compare values while treating two missing values as equal."""
    if pd.isna(first) and pd.isna(second):
        return True

    if pd.isna(first) or pd.isna(second):
        return False

    return math.isclose(
        float(first),
        float(second),
        abs_tol=1e-7,
        rel_tol=0.0,
    )


def default_for_dtype(dtype: Any) -> Any:
    """Create a safe default for an inserted row."""
    if is_bool_dtype(dtype):
        return False

    if is_datetime64_any_dtype(dtype):
        return pd.NaT

    if is_numeric_dtype(dtype):
        return np.nan

    return ""


def validate_ledger(
    ledger: pd.DataFrame,
    allowed_decisions: set[str],
    ledger_name: str,
) -> None:
    """Validate a completed decision ledger."""
    required_columns = {
        "review_id",
        "review_status",
        "decision",
        *EVIDENCE_FIELDS,
    }

    missing_columns = required_columns.difference(
        ledger.columns
    )

    if missing_columns:
        raise ValueError(
            f"{ledger_name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if ledger["review_id"].duplicated().any():
        raise ValueError(
            f"{ledger_name} contains duplicate review IDs."
        )

    pending = ledger.loc[
        ledger["review_status"]
        .str.strip()
        .str.lower()
        .ne("completed")
    ]

    if not pending.empty:
        raise ValueError(
            f"{ledger_name} contains "
            f"{len(pending)} pending decisions."
        )

    normalized_decisions = (
        ledger["decision"]
        .str.strip()
        .str.lower()
    )

    invalid_decisions = sorted(
        set(normalized_decisions)
        - allowed_decisions
    )

    if invalid_decisions:
        raise ValueError(
            f"{ledger_name} contains invalid decisions: "
            f"{invalid_decisions}"
        )

    for field in EVIDENCE_FIELDS:
        missing_evidence = ledger.loc[
            ledger[field]
            .str.strip()
            .eq("")
        ]

        if not missing_evidence.empty:
            raise ValueError(
                f"{ledger_name} contains "
                f"{len(missing_evidence)} rows "
                f"without {field}."
            )


def main() -> None:
    required_files = [
        INPUT_FILE,
        VALUE_LEDGER_FILE,
        INVALID_DATE_LEDGER_FILE,
        RULE_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
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

    with RULE_FILE.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        rules = yaml.safe_load(file_handle)

    tmin_minimum = float(
        rules["physical_screening"][
            "tmin_minimum"
        ]
    )

    tmin_maximum = float(
        rules["physical_screening"][
            "tmin_maximum"
        ]
    )

    tmax_minimum = float(
        rules["physical_screening"][
            "tmax_minimum"
        ]
    )

    tmax_maximum = float(
        rules["physical_screening"][
            "tmax_maximum"
        ]
    )

    high_dtr_threshold = float(
        rules[
            "diurnal_temperature_range"
        ]["high_dtr_threshold"]
    )

    data = pd.read_parquet(INPUT_FILE)
    data["date"] = pd.to_datetime(
        data["date"]
    )

    input_rows = len(data)

    if data.duplicated(
        subset=[
            "station_uid",
            "date",
        ]
    ).any():
        raise ValueError(
            "The Step 4A dataset contains duplicate keys."
        )

    # Start from the unchanged Step 4A values.
    data["tmin_cleaned_stage1"] = (
        data["tmin_qc_stage1"]
        .astype("float32")
    )

    data["tmax_cleaned_stage1"] = (
        data["tmax_qc_stage1"]
        .astype("float32")
    )

    data["tmin_cleaned_source"] = (
        data["tmin_source"]
        .fillna("")
        .astype(str)
    )

    data["tmax_cleaned_source"] = (
        data["tmax_source"]
        .fillna("")
        .astype(str)
    )

    # Audit fields.
    data["step4b_review_id"] = ""
    data["step4b_review_status"] = "not_required"
    data["step4b_decision"] = "not_required"
    data["step4b_decision_reason"] = ""
    data["step4b_verification_source"] = ""
    data["step4b_reviewer"] = ""
    data["step4b_review_date"] = ""
    data["step4b_value_changed"] = False
    data["step4b_invalid_date_inserted"] = False
    data["step4b_requires_rescreening"] = False

    # ---------------------------------------------------------
    # Apply daily-value decisions
    # ---------------------------------------------------------

    value_ledger = pd.read_csv(
        VALUE_LEDGER_FILE,
        dtype=str,
        keep_default_na=False,
    )

    validate_ledger(
        value_ledger,
        VALUE_DECISIONS,
        "Value decision ledger",
    )

    expected_review = data.loc[
        data[
            "qc_stage1_value_review_required"
        ].fillna(False),
        [
            "station_uid",
            "date",
        ],
    ].copy()

    expected_review["review_id"] = (
        "VALUE|"
        + expected_review[
            "station_uid"
        ].astype(str)
        + "|"
        + expected_review[
            "date"
        ].dt.strftime("%Y-%m-%d")
    )

    expected_ids = set(
        expected_review["review_id"]
    )

    ledger_ids = set(
        value_ledger["review_id"]
    )

    if expected_ids != ledger_ids:
        raise ValueError(
            "The value ledger does not match the "
            "current Step 4A review records.\n"
            f"Missing IDs: {len(expected_ids - ledger_ids)}\n"
            f"Unexpected IDs: {len(ledger_ids - expected_ids)}"
        )

    key_to_index = {
        (
            row.station_uid,
            row.date.strftime("%Y-%m-%d"),
        ): index
        for index, row in data[
            [
                "station_uid",
                "date",
            ]
        ].iterrows()
    }

    value_application_log: list[
        dict[str, Any]
    ] = []

    for ledger_row in value_ledger.itertuples(
        index=False
    ):
        decision = (
            ledger_row.decision
            .strip()
            .lower()
        )

        key = (
            ledger_row.station_uid,
            str(ledger_row.date)[:10],
        )

        if key not in key_to_index:
            raise ValueError(
                f"Decision key not found: {key}"
            )

        index = key_to_index[key]

        before_tmin = data.at[
            index,
            "tmin_cleaned_stage1",
        ]

        before_tmax = data.at[
            index,
            "tmax_cleaned_stage1",
        ]

        after_tmin = before_tmin
        after_tmax = before_tmax

        corrected_tmin = parse_optional_float(
            ledger_row.corrected_tmin
        )

        corrected_tmax = parse_optional_float(
            ledger_row.corrected_tmax
        )

        if decision == "keep":
            pass

        elif decision == "set_tmin_missing":
            after_tmin = np.nan

        elif decision == "set_tmax_missing":
            after_tmax = np.nan

        elif decision == "set_both_missing":
            after_tmin = np.nan
            after_tmax = np.nan

        elif decision == "correct_tmin":
            if corrected_tmin is None:
                raise ValueError(
                    f"{ledger_row.review_id} "
                    "requires corrected_tmin."
                )

            after_tmin = corrected_tmin

        elif decision == "correct_tmax":
            if corrected_tmax is None:
                raise ValueError(
                    f"{ledger_row.review_id} "
                    "requires corrected_tmax."
                )

            after_tmax = corrected_tmax

        elif decision == "correct_both":
            if (
                corrected_tmin is None
                or corrected_tmax is None
            ):
                raise ValueError(
                    f"{ledger_row.review_id} requires "
                    "both corrected temperatures."
                )

            after_tmin = corrected_tmin
            after_tmax = corrected_tmax

        elif decision == "select_old_tmin":
            old_tmin = data.at[
                index,
                "tmin_old_xlsx",
            ]

            if pd.isna(old_tmin):
                raise ValueError(
                    f"{ledger_row.review_id} has no "
                    "old-XLSX Tmin value."
                )

            after_tmin = float(old_tmin)

        elif decision == "select_new_tmin":
            new_tmin = data.at[
                index,
                "tmin_new_csv",
            ]

            if pd.isna(new_tmin):
                raise ValueError(
                    f"{ledger_row.review_id} has no "
                    "new-CSV Tmin value."
                )

            after_tmin = float(new_tmin)

        data.at[
            index,
            "tmin_cleaned_stage1",
        ] = after_tmin

        data.at[
            index,
            "tmax_cleaned_stage1",
        ] = after_tmax

        if decision in {
            "correct_tmin",
            "correct_both",
        }:
            data.at[
                index,
                "tmin_cleaned_source",
            ] = "manual_correction"

        elif decision == "select_old_tmin":
            data.at[
                index,
                "tmin_cleaned_source",
            ] = "old_xlsx"

        elif decision == "select_new_tmin":
            data.at[
                index,
                "tmin_cleaned_source",
            ] = "new_csv"

        elif decision in {
            "set_tmin_missing",
            "set_both_missing",
        }:
            data.at[
                index,
                "tmin_cleaned_source",
            ] = "qc_set_missing"

        if decision in {
            "correct_tmax",
            "correct_both",
        }:
            data.at[
                index,
                "tmax_cleaned_source",
            ] = "manual_correction"

        elif decision in {
            "set_tmax_missing",
            "set_both_missing",
        }:
            data.at[
                index,
                "tmax_cleaned_source",
            ] = "qc_set_missing"

        value_changed = (
            not values_equal(
                before_tmin,
                after_tmin,
            )
            or not values_equal(
                before_tmax,
                after_tmax,
            )
        )

        data.at[
            index,
            "step4b_review_id",
        ] = ledger_row.review_id

        data.at[
            index,
            "step4b_review_status",
        ] = "completed"

        data.at[
            index,
            "step4b_decision",
        ] = decision

        data.at[
            index,
            "step4b_decision_reason",
        ] = ledger_row.decision_reason

        data.at[
            index,
            "step4b_verification_source",
        ] = ledger_row.verification_source

        data.at[
            index,
            "step4b_reviewer",
        ] = ledger_row.reviewer

        data.at[
            index,
            "step4b_review_date",
        ] = ledger_row.review_date

        data.at[
            index,
            "step4b_value_changed",
        ] = value_changed

        data.at[
            index,
            "step4b_requires_rescreening",
        ] = value_changed

        value_application_log.append(
            {
                "review_id":
                    ledger_row.review_id,
                "station_uid":
                    ledger_row.station_uid,
                "date":
                    str(ledger_row.date)[:10],
                "decision":
                    decision,
                "before_tmin":
                    before_tmin,
                "after_tmin":
                    after_tmin,
                "before_tmax":
                    before_tmax,
                "after_tmax":
                    after_tmax,
                "value_changed":
                    value_changed,
                "decision_reason":
                    ledger_row.decision_reason,
                "verification_source":
                    ledger_row.verification_source,
                "reviewer":
                    ledger_row.reviewer,
                "review_date":
                    ledger_row.review_date,
            }
        )

    # ---------------------------------------------------------
    # Apply invalid-date decisions
    # ---------------------------------------------------------

    invalid_ledger = pd.read_csv(
        INVALID_DATE_LEDGER_FILE,
        dtype=str,
        keep_default_na=False,
    )

    if (
        len(invalid_ledger)
        != EXPECTED_INVALID_DATE_DECISIONS
    ):
        raise ValueError(
            "Expected "
            f"{EXPECTED_INVALID_DATE_DECISIONS} "
            "invalid-date decisions; found "
            f"{len(invalid_ledger)}."
        )

    validate_ledger(
        invalid_ledger,
        INVALID_DATE_DECISIONS,
        "Invalid-date decision ledger",
    )

    metadata_columns = [
        "station_uid",
        "station_id",
        "official_station_name",
        "station_name_display",
        "latitude",
        "longitude",
        "metadata_status",
        "metadata_pending_flag",
        "usable_for_spatial_analysis",
    ]

    station_metadata = (
        data[metadata_columns]
        .drop_duplicates(
            subset=["station_uid"]
        )
        .set_index("station_uid")
    )

    existing_keys = {
        (
            row.station_uid,
            row.date.strftime("%Y-%m-%d"),
        )
        for row in data[
            [
                "station_uid",
                "date",
            ]
        ].itertuples(index=False)
    }

    inserted_records: list[
        dict[str, Any]
    ] = []

    invalid_application_log: list[
        dict[str, Any]
    ] = []

    for ledger_row in invalid_ledger.itertuples(
        index=False
    ):
        decision = (
            ledger_row.decision
            .strip()
            .lower()
        )

        source_tmin = parse_optional_float(
            getattr(
                ledger_row,
                "tmin_old_xlsx",
                "",
            )
        )

        source_tmax = parse_optional_float(
            getattr(
                ledger_row,
                "tmax_old_xlsx",
                "",
            )
        )

        corrected_tmin = parse_optional_float(
            ledger_row.corrected_tmin
        )

        corrected_tmax = parse_optional_float(
            ledger_row.corrected_tmax
        )

        incoming_tmin = (
            corrected_tmin
            if corrected_tmin is not None
            else source_tmin
        )

        incoming_tmax = (
            corrected_tmax
            if corrected_tmax is not None
            else source_tmax
        )

        log_record = {
            "review_id":
                ledger_row.review_id,
            "source_row":
                ledger_row.source_row,
            "station_uid":
                ledger_row.station_uid,
            "decision":
                decision,
            "corrected_date":
                ledger_row.corrected_date,
            "incoming_tmin":
                incoming_tmin,
            "incoming_tmax":
                incoming_tmax,
            "decision_reason":
                ledger_row.decision_reason,
            "verification_source":
                ledger_row.verification_source,
            "reviewer":
                ledger_row.reviewer,
            "review_date":
                ledger_row.review_date,
            "application_result":
                "",
        }

        if decision == "reject_source_row":
            log_record[
                "application_result"
            ] = "rejected_not_added"

            invalid_application_log.append(
                log_record
            )

            continue

        corrected_date = pd.to_datetime(
            ledger_row.corrected_date,
            format="%Y-%m-%d",
            errors="raise",
        )

        corrected_date_text = (
            corrected_date.strftime(
                "%Y-%m-%d"
            )
        )

        key = (
            ledger_row.station_uid,
            corrected_date_text,
        )

        if (
            ledger_row.station_uid
            not in station_metadata.index
        ):
            raise ValueError(
                "Unknown station UID in invalid-date "
                f"ledger: {ledger_row.station_uid}"
            )

        if decision == "insert_corrected_date":
            if key in existing_keys:
                raise ValueError(
                    f"{ledger_row.review_id}: corrected "
                    f"station-date {key} already exists."
                )

            new_record = {
                column: default_for_dtype(dtype)
                for column, dtype
                in data.dtypes.items()
            }

            metadata = station_metadata.loc[
                ledger_row.station_uid
            ]

            for column in metadata_columns:
                new_record[column] = metadata[column]

            new_record["date"] = corrected_date
            new_record["year"] = corrected_date.year
            new_record["month"] = corrected_date.month
            new_record["day"] = corrected_date.day

            new_record["record_origin"] = (
                "corrected_invalid_old_date"
            )

            new_record["integration_status"] = (
                "reviewed_corrected_invalid_date"
            )

            new_record["tmin_preliminary"] = (
                incoming_tmin
            )

            new_record["tmin_qc_stage1"] = (
                incoming_tmin
            )

            new_record["tmin_cleaned_stage1"] = (
                incoming_tmin
            )

            new_record["tmax_preliminary"] = (
                incoming_tmax
            )

            new_record["tmax_qc_stage1"] = (
                incoming_tmax
            )

            new_record["tmax_cleaned_stage1"] = (
                incoming_tmax
            )

            new_record["tmin_source"] = (
                "old_xlsx_corrected_date"
                if incoming_tmin is not None
                else "none"
            )

            new_record["tmax_source"] = (
                "old_xlsx_corrected_date"
                if incoming_tmax is not None
                else "none"
            )

            new_record["tmin_cleaned_source"] = (
                new_record["tmin_source"]
            )

            new_record["tmax_cleaned_source"] = (
                new_record["tmax_source"]
            )

            new_record["tmin_selection_status"] = (
                "automatic_invalid_date_correction"
            )

            new_record["tmax_selection_status"] = (
                "automatic_invalid_date_correction"
            )

            new_record["old_record_present"] = True
            new_record["old_source_row_count"] = 1
            new_record["old_source_rows"] = str(
                ledger_row.source_row
            )

            new_record["old_tmin_raw_values"] = (
                getattr(
                    ledger_row,
                    "tmin_raw",
                    "",
                )
            )

            new_record["old_tmax_raw_values"] = (
                getattr(
                    ledger_row,
                    "tmax_raw",
                    "",
                )
            )

            new_record["step4b_review_id"] = (
                ledger_row.review_id
            )

            new_record["step4b_review_status"] = (
                "completed"
            )

            new_record["step4b_decision"] = decision

            new_record[
                "step4b_decision_reason"
            ] = ledger_row.decision_reason

            new_record[
                "step4b_verification_source"
            ] = ledger_row.verification_source

            new_record["step4b_reviewer"] = (
                ledger_row.reviewer
            )

            new_record["step4b_review_date"] = (
                ledger_row.review_date
            )

            new_record[
                "step4b_invalid_date_inserted"
            ] = True

            new_record[
                "step4b_value_changed"
            ] = True

            new_record[
                "step4b_requires_rescreening"
            ] = True

            new_record["qc_stage1_status"] = (
                "not_screened_corrected_date"
            )

            new_record["qc_stage1_priority"] = (
                "not_screened"
            )

            new_record["qc_flag_codes"] = (
                "CORRECTED_INVALID_DATE"
            )

            inserted_records.append(
                new_record
            )

            existing_keys.add(key)

            log_record[
                "application_result"
            ] = "inserted_new_station_date"

        elif decision == "replace_existing_date":
            if key not in existing_keys:
                raise ValueError(
                    f"{ledger_row.review_id}: corrected "
                    f"station-date {key} does not exist."
                )

            target_indices = data.index[
                data["station_uid"].eq(
                    ledger_row.station_uid
                )
                & data["date"].eq(
                    corrected_date
                )
            ]

            if len(target_indices) != 1:
                raise ValueError(
                    f"Expected one existing record for "
                    f"{key}; found {len(target_indices)}."
                )

            target_index = target_indices[0]

            if incoming_tmin is not None:
                data.at[
                    target_index,
                    "tmin_cleaned_stage1",
                ] = incoming_tmin

                data.at[
                    target_index,
                    "tmin_cleaned_source",
                ] = "old_xlsx_corrected_date"

            if incoming_tmax is not None:
                data.at[
                    target_index,
                    "tmax_cleaned_stage1",
                ] = incoming_tmax

                data.at[
                    target_index,
                    "tmax_cleaned_source",
                ] = "old_xlsx_corrected_date"

            data.at[
                target_index,
                "step4b_review_id",
            ] = ledger_row.review_id

            data.at[
                target_index,
                "step4b_review_status",
            ] = "completed"

            data.at[
                target_index,
                "step4b_decision",
            ] = decision

            data.at[
                target_index,
                "step4b_decision_reason",
            ] = ledger_row.decision_reason

            data.at[
                target_index,
                "step4b_verification_source",
            ] = ledger_row.verification_source

            data.at[
                target_index,
                "step4b_reviewer",
            ] = ledger_row.reviewer

            data.at[
                target_index,
                "step4b_review_date",
            ] = ledger_row.review_date

            data.at[
                target_index,
                "step4b_value_changed",
            ] = True

            data.at[
                target_index,
                "step4b_requires_rescreening",
            ] = True

            log_record[
                "application_result"
            ] = "replaced_existing_station_date"

        invalid_application_log.append(
            log_record
        )

    if inserted_records:
        inserted_data = pd.DataFrame(
            inserted_records,
            columns=data.columns,
        )

        data = pd.concat(
            [
                data,
                inserted_data,
            ],
            ignore_index=True,
        )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    if data.duplicated(
        subset=[
            "station_uid",
            "date",
        ]
    ).any():
        raise ValueError(
            "Duplicate station-date keys exist "
            "after applying decisions."
        )

    # ---------------------------------------------------------
    # Re-screen cleaned values
    # ---------------------------------------------------------

    both_temperatures = (
        data[
            "tmin_cleaned_stage1"
        ].notna()
        & data[
            "tmax_cleaned_stage1"
        ].notna()
    )

    data["step4b_post_dtr"] = (
        data["tmax_cleaned_stage1"]
        - data["tmin_cleaned_stage1"]
    ).astype("float32")

    data["step4b_post_tmin_zero"] = (
        data["tmin_cleaned_stage1"]
        .eq(0.0)
    )

    data[
        "step4b_post_tmin_physical_range"
    ] = (
        data[
            "tmin_cleaned_stage1"
        ].notna()
        & (
            data[
                "tmin_cleaned_stage1"
            ].lt(tmin_minimum)
            | data[
                "tmin_cleaned_stage1"
            ].gt(tmin_maximum)
        )
    )

    data[
        "step4b_post_tmax_physical_range"
    ] = (
        data[
            "tmax_cleaned_stage1"
        ].notna()
        & (
            data[
                "tmax_cleaned_stage1"
            ].lt(tmax_minimum)
            | data[
                "tmax_cleaned_stage1"
            ].gt(tmax_maximum)
        )
    )

    data["step4b_post_tmax_lt_tmin"] = (
        both_temperatures
        & data[
            "step4b_post_dtr"
        ].lt(0.0)
    )

    data["step4b_post_high_dtr"] = (
        both_temperatures
        & data[
            "step4b_post_dtr"
        ].gt(high_dtr_threshold)
    )

    data["step4b_post_screen_flag"] = (
        data[
            "step4b_post_tmin_physical_range"
        ]
        | data[
            "step4b_post_tmax_physical_range"
        ]
        | data[
            "step4b_post_tmax_lt_tmin"
        ]
        | data[
            "step4b_post_high_dtr"
        ]
    )

    data = data.sort_values(
        [
            "station_uid",
            "date",
        ]
    ).reset_index(drop=True)

    data.to_parquet(
        OUTPUT_FILE,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    value_log = pd.DataFrame(
        value_application_log
    )

    invalid_log = pd.DataFrame(
        invalid_application_log
    )

    value_log.to_csv(
        REPORT_DIRECTORY
        / "step4b_value_decision_application_log.csv",
        index=False,
    )

    invalid_log.to_csv(
        REPORT_DIRECTORY
        / "step4b_invalid_date_application_log.csv",
        index=False,
    )

    remaining_flags = data.loc[
        data[
            "step4b_post_screen_flag"
        ]
    ].copy()

    remaining_flags.to_csv(
        REPORT_DIRECTORY
        / "step4b_post_decision_screen_flags.csv",
        index=False,
    )

    value_decision_counts = (
        value_ledger["decision"]
        .str.lower()
        .value_counts()
        .sort_index()
        .to_dict()
    )

    invalid_decision_counts = (
        invalid_ledger["decision"]
        .str.lower()
        .value_counts()
        .sort_index()
        .to_dict()
    )

    output_rows = len(data)
    inserted_count = len(inserted_records)

    duplicate_count = int(
        data.duplicated(
            subset=[
                "station_uid",
                "date",
            ]
        ).sum()
    )

    summary = {
        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "input_rows":
            input_rows,
        "output_rows":
            output_rows,
        "inserted_corrected_date_rows":
            inserted_count,
        "value_decisions_applied":
            len(value_ledger),
        "invalid_date_decisions_applied":
            len(invalid_ledger),
        "rows_with_changed_values":
            int(
                data[
                    "step4b_value_changed"
                ].sum()
            ),
        "remaining_post_decision_screen_flags":
            len(remaining_flags),
        "duplicate_final_keys":
            duplicate_count,
        "minimum_date":
            data["date"].min().date().isoformat(),
        "maximum_date":
            data["date"].max().date().isoformat(),
        "value_decision_counts":
            {
                str(key): int(value)
                for key, value
                in value_decision_counts.items()
            },
        "invalid_date_decision_counts":
            {
                str(key): int(value)
                for key, value
                in invalid_decision_counts.items()
            },
        "output_file":
            str(OUTPUT_FILE),
        "output_size_bytes":
            OUTPUT_FILE.stat().st_size,
    }

    (
        REPORT_DIRECTORY
        / "step4b_cleaning_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    schema = pq.ParquetFile(
        OUTPUT_FILE
    ).schema_arrow

    (
        ADMIN_DIRECTORY
        / "step4b_parquet_schema.json"
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
        "STEP 4B: QC DECISION APPLICATION",
        "=" * 39,
        f"Input daily rows: {input_rows:,}",
        f"Output daily rows: {output_rows:,}",
        (
            "Value decisions applied: "
            f"{len(value_ledger):,}"
        ),
        (
            "Invalid-date decisions applied: "
            f"{len(invalid_ledger):,}"
        ),
        (
            "Corrected-date rows inserted: "
            f"{inserted_count:,}"
        ),
        (
            "Rows with changed values: "
            f"{data['step4b_value_changed'].sum():,}"
        ),
        (
            "Remaining post-decision screen flags: "
            f"{len(remaining_flags):,}"
        ),
        (
            "Duplicate final keys: "
            f"{duplicate_count:,}"
        ),
        (
            "Date range: "
            f"{data['date'].min().date()} to "
            f"{data['date'].max().date()}"
        ),
        "",
        "Value decisions:",
        *[
            f"- {key}: {value:,}"
            for key, value
            in value_decision_counts.items()
        ],
        "",
        "Invalid-date decisions:",
        *[
            f"- {key}: {value:,}"
            for key, value
            in invalid_decision_counts.items()
        ],
        "",
        f"Output: {OUTPUT_FILE}",
    ]

    (
        REPORT_DIRECTORY
        / "step4b_cleaning_report.txt"
    ).write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(report_lines))

    if duplicate_count != 0:
        raise SystemExit(
            "STEP 4B FAILED: duplicate final keys."
        )

    if output_rows != input_rows + inserted_count:
        raise SystemExit(
            "STEP 4B FAILED: unexpected row count."
        )

    print("\nSTEP 4B PASSED.")


if __name__ == "__main__":
    main()
