from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QC_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "temperature_daily_qc_stage1.parquet"
)

INVALID_DATE_FILE = (
    PROJECT_ROOT
    / "05_qc_reports"
    / "step4a_invalid_date_review_template.csv"
)

VALUE_TEMPLATE = (
    PROJECT_ROOT
    / "05_qc_reports"
    / "step4b_value_decision_template.csv"
)

INVALID_TEMPLATE = (
    PROJECT_ROOT
    / "05_qc_reports"
    / "step4b_invalid_date_decision_template.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "05_qc_reports"
    / "step4b_ledger_preparation_summary.json"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "05_qc_reports"
    / "step4b_ledger_preparation_report.txt"
)

EXPECTED_INVALID_DATES = 54

VALUE_DECISIONS = (
    "keep|set_tmin_missing|set_tmax_missing|"
    "set_both_missing|correct_tmin|correct_tmax|"
    "correct_both|select_old_tmin|select_new_tmin"
)

INVALID_DATE_DECISIONS = (
    "reject_source_row|insert_corrected_date|"
    "replace_existing_date"
)


def main() -> None:
    for path in [QC_FILE, INVALID_DATE_FILE]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    VALUE_TEMPLATE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_parquet(QC_FILE)
    data["date"] = pd.to_datetime(data["date"])

    review = data.loc[
        data[
            "qc_stage1_value_review_required"
        ].fillna(False)
    ].copy()

    duplicate_keys = review.duplicated(
        subset=["station_uid", "date"],
        keep=False,
    )

    if duplicate_keys.any():
        raise ValueError(
            "Duplicate station-date keys exist in "
            "the Step 4A review records."
        )

    review["review_id"] = (
        "VALUE|"
        + review["station_uid"].astype(str)
        + "|"
        + review["date"].dt.strftime("%Y-%m-%d")
    )

    value_columns = [
        "review_id",
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
        "tmin_qc_stage1",
        "tmax_qc_stage1",
        "diurnal_temperature_range",

        "tmin_source",
        "tmax_source",
        "tmin_selection_status",
        "tmax_selection_status",
        "reconciliation_status",

        "tmin_old_xlsx",
        "tmin_new_csv",
        "old_tmin_raw_values",
        "old_tmax_raw_values",
        "old_source_rows",
        "new_source_cell",

        "qc_flag_codes",
        "qc_flag_count",
        "qc_stage1_priority",
        "qc_stage1_status",
    ]

    value_ledger = review[
        value_columns
    ].copy()

    value_ledger["allowed_decisions"] = (
        VALUE_DECISIONS
    )

    value_ledger["review_status"] = "pending"
    value_ledger["decision"] = ""
    value_ledger["corrected_tmin"] = ""
    value_ledger["corrected_tmax"] = ""
    value_ledger["decision_reason"] = ""
    value_ledger["verification_source"] = ""
    value_ledger["reviewer"] = ""
    value_ledger["review_date"] = ""

    value_ledger.to_csv(
        VALUE_TEMPLATE,
        index=False,
    )

    invalid = pd.read_csv(
        INVALID_DATE_FILE,
        dtype=str,
        keep_default_na=False,
    )

    if len(invalid) != EXPECTED_INVALID_DATES:
        raise ValueError(
            f"Expected {EXPECTED_INVALID_DATES} invalid-date "
            f"rows; found {len(invalid)}."
        )

    station_lookup = (
        data[
            [
                "station_uid",
                "station_id",
                "station_name_display",
            ]
        ]
        .drop_duplicates(
            subset=["station_uid"]
        )
    )

    invalid = invalid.drop(
        columns=[
            "station_id",
            "station_name_display",
        ],
        errors="ignore",
    )

    invalid = invalid.merge(
        station_lookup,
        on="station_uid",
        how="left",
        validate="many_to_one",
    )

    invalid["review_id"] = (
        "DATE|"
        + invalid["station_uid"].astype(str)
        + "|ROW"
        + invalid["source_row"].astype(str)
    )

    invalid["allowed_decisions"] = (
        INVALID_DATE_DECISIONS
    )

    invalid["review_status"] = "pending"
    invalid["decision"] = ""
    invalid["corrected_date"] = ""
    invalid["corrected_tmin"] = ""
    invalid["corrected_tmax"] = ""
    invalid["decision_reason"] = ""
    invalid["verification_source"] = ""
    invalid["reviewer"] = ""
    invalid["review_date"] = ""

    preferred_columns = [
        "review_id",
        "source_row",
        "station_uid",
        "station_id",
        "station_name_display",
        "station_name_raw",
        "year_raw",
        "month_raw",
        "day_raw",
        "date_parse_status",
        "tmin_raw",
        "tmin_old_xlsx",
        "tmin_parse_status",
        "tmax_raw",
        "tmax_old_xlsx",
        "tmax_parse_status",
        "allowed_decisions",
        "review_status",
        "decision",
        "corrected_date",
        "corrected_tmin",
        "corrected_tmax",
        "decision_reason",
        "verification_source",
        "reviewer",
        "review_date",
    ]

    final_invalid_columns = [
        column
        for column in preferred_columns
        if column in invalid.columns
    ]

    invalid[
        final_invalid_columns
    ].to_csv(
        INVALID_TEMPLATE,
        index=False,
    )

    summary = {
        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "value_review_rows":
            len(value_ledger),
        "invalid_date_review_rows":
            len(invalid),
        "value_template":
            str(VALUE_TEMPLATE),
        "invalid_date_template":
            str(INVALID_TEMPLATE),
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "STEP 4B: DECISION LEDGER PREPARATION",
        "=" * 42,
        (
            "Daily value-review rows: "
            f"{len(value_ledger):,}"
        ),
        (
            "Invalid-date review rows: "
            f"{len(invalid):,}"
        ),
        "",
        f"Value template: {VALUE_TEMPLATE}",
        f"Invalid-date template: {INVALID_TEMPLATE}",
        "",
        "Templates contain pending decisions only.",
    ]

    REPORT_FILE.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(report_lines))
    print("\nSTEP 4B LEDGER PREPARATION PASSED.")


if __name__ == "__main__":
    main()
