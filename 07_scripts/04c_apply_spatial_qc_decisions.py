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
    / "temperature_daily_qc_stage3_spatial.parquet"
)

POLICY_FILE = (
    PROJECT_ROOT
    / "00_admin"
    / "step5c_spatial_decision_policy.yaml"
)

STEP4A_RULE_FILE = (
    PROJECT_ROOT
    / "00_admin"
    / "step4a_qc_rules.yaml"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "04_clean_data"
    / "temperature_daily_cleaned_stage2.parquet"
)

REPORT_DIRECTORY = PROJECT_ROOT / "05_qc_reports"
ADMIN_DIRECTORY = PROJECT_ROOT / "00_admin"

EXPECTED_STATIONS = 48


def changed_mask(
    before: pd.Series,
    after: pd.Series,
) -> pd.Series:
    """Return True where values differ, treating two NaNs as equal."""
    same_numeric = (
        before.notna()
        & after.notna()
        & np.isclose(
            before.astype(float),
            after.astype(float),
            atol=1e-7,
            rtol=0.0,
        )
    )

    both_missing = (
        before.isna()
        & after.isna()
    )

    return ~(same_numeric | both_missing)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        return yaml.safe_load(file_handle)


def main() -> None:
    required_files = [
        INPUT_FILE,
        POLICY_FILE,
        STEP4A_RULE_FILE,
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

    policy = load_yaml(POLICY_FILE)
    step4a_rules = load_yaml(
        STEP4A_RULE_FILE
    )

    if policy["primary_dataset"][
        "interpolation_allowed"
    ]:
        raise ValueError(
            "Step 5C policy must not permit interpolation."
        )

    physical_rules = step4a_rules[
        "physical_screening"
    ]

    dtr_rules = step4a_rules[
        "diurnal_temperature_range"
    ]

    tmin_minimum = float(
        physical_rules["tmin_minimum"]
    )

    tmin_maximum = float(
        physical_rules["tmin_maximum"]
    )

    tmax_minimum = float(
        physical_rules["tmax_minimum"]
    )

    tmax_maximum = float(
        physical_rules["tmax_maximum"]
    )

    high_dtr_threshold = float(
        dtr_rules["high_dtr_threshold"]
    )

    zero_tolerance = float(
        dtr_rules["zero_tolerance"]
    )

    data = pd.read_parquet(INPUT_FILE)

    data["date"] = pd.to_datetime(
        data["date"]
    )

    data = data.sort_values(
        [
            "station_uid",
            "date",
        ]
    ).reset_index(drop=True)

    input_rows = len(data)

    if data.duplicated(
        subset=[
            "station_uid",
            "date",
        ]
    ).any():
        raise ValueError(
            "Step 5B input contains duplicate keys."
        )

    station_count = int(
        data["station_uid"].nunique()
    )

    if station_count != EXPECTED_STATIONS:
        raise ValueError(
            f"Expected {EXPECTED_STATIONS} stations; "
            f"found {station_count}."
        )

    original_minimum_date = data[
        "date"
    ].min()

    original_maximum_date = data[
        "date"
    ].max()

    original_tmin = data[
        "tmin_cleaned_stage1"
    ].copy()

    original_tmax = data[
        "tmax_cleaned_stage1"
    ].copy()

    # -----------------------------------------------------
    # Normalize required Step 5B flags
    # -----------------------------------------------------

    boolean_columns = [
        "qc_tmin_spatial_outlier",
        "qc_tmax_spatial_outlier",
        "qc_tmin_regional_support",
        "qc_tmax_regional_support",
        "qc_tmin_regional_cold_support",
        "step5b_regional_signal_supported",
        "step5b_regional_cold_signal_supported",
        "step5b_insufficient_same_day_neighbours",
        "step5b_spatial_metadata_unavailable",
        "step5b_spatial_baseline_unavailable",
        "step5a_candidate_review_required",
    ]

    for column in boolean_columns:
        if column not in data.columns:
            raise ValueError(
                f"Required Step 5B column missing: {column}"
            )

        data[column] = (
            data[column]
            .fillna(False)
            .astype(bool)
        )

    tmin_outlier = data[
        "qc_tmin_spatial_outlier"
    ]

    tmax_outlier = data[
        "qc_tmax_spatial_outlier"
    ]

    tmin_regional_support = data[
        "qc_tmin_regional_support"
    ]

    tmax_regional_support = data[
        "qc_tmax_regional_support"
    ]

    # The same variable must not be simultaneously classified
    # as an isolated outlier and regionally supported.
    contradictory_tmin = (
        tmin_outlier
        & tmin_regional_support
    )

    contradictory_tmax = (
        tmax_outlier
        & tmax_regional_support
    )

    if contradictory_tmin.any():
        raise ValueError(
            "Some Tmin observations are both isolated "
            "spatial outliers and regionally supported."
        )

    if contradictory_tmax.any():
        raise ValueError(
            "Some Tmax observations are both isolated "
            "spatial outliers and regionally supported."
        )

    # -----------------------------------------------------
    # Create stage-2 cleaned values
    # -----------------------------------------------------

    data["tmin_cleaned_stage2"] = (
        data["tmin_cleaned_stage1"]
        .astype("float32")
    )

    data["tmax_cleaned_stage2"] = (
        data["tmax_cleaned_stage1"]
        .astype("float32")
    )

    data[
        "tmin_cleaned_stage2_source"
    ] = (
        data["tmin_cleaned_source"]
        .fillna("")
        .astype(str)
    )

    data[
        "tmax_cleaned_stage2_source"
    ] = (
        data["tmax_cleaned_source"]
        .fillna("")
        .astype(str)
    )

    tmin_remove = (
        tmin_outlier
        & data[
            "tmin_cleaned_stage2"
        ].notna()
    )

    tmax_remove = (
        tmax_outlier
        & data[
            "tmax_cleaned_stage2"
        ].notna()
    )

    data.loc[
        tmin_remove,
        "tmin_cleaned_stage2",
    ] = np.nan

    data.loc[
        tmax_remove,
        "tmax_cleaned_stage2",
    ] = np.nan

    data.loc[
        tmin_remove,
        "tmin_cleaned_stage2_source",
    ] = "qc_set_missing_spatial_outlier"

    data.loc[
        tmax_remove,
        "tmax_cleaned_stage2_source",
    ] = "qc_set_missing_spatial_outlier"

    data[
        "step5c_tmin_set_missing"
    ] = tmin_remove

    data[
        "step5c_tmax_set_missing"
    ] = tmax_remove

    data[
        "step5c_any_value_changed"
    ] = (
        tmin_remove
        | tmax_remove
    )

    both_remove = (
        tmin_remove
        & tmax_remove
    )

    regional_cold = data[
        "step5b_regional_cold_signal_supported"
    ]

    regional_signal = data[
        "step5b_regional_signal_supported"
    ]

    insufficient = data[
        "step5b_insufficient_same_day_neighbours"
    ]

    unavailable = (
        data[
            "step5b_spatial_metadata_unavailable"
        ]
        | data[
            "step5b_spatial_baseline_unavailable"
        ]
    )

    step5a_candidate = data[
        "step5a_candidate_review_required"
    ]

    data["step5c_decision"] = np.select(
        [
            both_remove,
            tmin_remove,
            tmax_remove,
            regional_cold,
            regional_signal,
            insufficient,
            unavailable,
            step5a_candidate,
        ],
        [
            "set_both_missing_spatial_outlier",
            "set_tmin_missing_spatial_outlier",
            "set_tmax_missing_spatial_outlier",
            "keep_regional_cold_signal",
            "keep_regional_signal",
            "keep_insufficient_neighbours",
            "keep_spatial_evidence_unavailable",
            "keep_spatially_inconclusive",
        ],
        default="no_spatial_action_required",
    )

    data["step5c_decision_reason"] = np.select(
        [
            both_remove,
            tmin_remove,
            tmax_remove,
            regional_cold,
            regional_signal,
            insufficient,
            unavailable,
            step5a_candidate,
        ],
        [
            (
                "Both Tmin and Tmax were isolated spatial "
                "outliers relative to same-day neighbouring "
                "station anomalies; both were set missing."
            ),
            (
                "Tmin was an isolated spatial outlier relative "
                "to same-day neighbouring station anomalies; "
                "Tmin was set missing."
            ),
            (
                "Tmax was an isolated spatial outlier relative "
                "to same-day neighbouring station anomalies; "
                "Tmax was set missing."
            ),
            (
                "The Tmin anomaly was supported by neighbouring "
                "stations as a regional cold signal; the "
                "official observation was retained."
            ),
            (
                "The anomaly was supported by neighbouring "
                "stations; the official observation was retained."
            ),
            (
                "Fewer than the required same-day neighbouring "
                "anomalies were available; the observation was "
                "retained because evidence was insufficient."
            ),
            (
                "Station coordinates or reference climatology "
                "were unavailable; the observation was retained."
            ),
            (
                "The Step 5A candidate was not classified as an "
                "isolated spatial outlier; the observation was "
                "retained."
            ),
        ],
        default=(
            "The record did not require a Step 5C "
            "spatial-cleaning action."
        ),
    )

    data[
        "step5c_decision_method"
    ] = (
        "deterministic_spatial_policy_v1"
    )

    data[
        "step5c_interpolation_applied"
    ] = False

    data["step5c_status"] = np.select(
        [
            data[
                "step5c_any_value_changed"
            ],
            regional_cold,
            regional_signal,
            insufficient | unavailable,
            step5a_candidate,
        ],
        [
            "isolated_spatial_outlier_removed",
            "regional_cold_extreme_retained",
            "regional_signal_retained",
            "retained_insufficient_spatial_evidence",
            "candidate_retained_after_spatial_qc",
        ],
        default="qc_complete_no_spatial_action",
    )

    # -----------------------------------------------------
    # Validate exactly which values changed
    # -----------------------------------------------------

    actual_tmin_change = changed_mask(
        original_tmin,
        data["tmin_cleaned_stage2"],
    )

    actual_tmax_change = changed_mask(
        original_tmax,
        data["tmax_cleaned_stage2"],
    )

    if not actual_tmin_change.equals(
        tmin_remove
    ):
        raise ValueError(
            "Unexpected Tmin changes occurred."
        )

    if not actual_tmax_change.equals(
        tmax_remove
    ):
        raise ValueError(
            "Unexpected Tmax changes occurred."
        )

    # No new numeric value may be manufactured.
    new_tmin_created = (
        original_tmin.isna()
        & data[
            "tmin_cleaned_stage2"
        ].notna()
    )

    new_tmax_created = (
        original_tmax.isna()
        & data[
            "tmax_cleaned_stage2"
        ].notna()
    )

    if new_tmin_created.any():
        raise ValueError(
            "Step 5C created new numeric Tmin values."
        )

    if new_tmax_created.any():
        raise ValueError(
            "Step 5C created new numeric Tmax values."
        )

    # Regional cold-support Tmin values must remain unchanged.
    regional_cold_removed = (
        regional_cold
        & actual_tmin_change
    )

    if regional_cold_removed.any():
        raise ValueError(
            "A regionally supported cold Tmin was changed."
        )

    # Every isolated spatial outlier must now be missing.
    unresolved_tmin_outliers = (
        tmin_outlier
        & data[
            "tmin_cleaned_stage2"
        ].notna()
    )

    unresolved_tmax_outliers = (
        tmax_outlier
        & data[
            "tmax_cleaned_stage2"
        ].notna()
    )

    if unresolved_tmin_outliers.any():
        raise ValueError(
            "Some Tmin spatial outliers remain numeric."
        )

    if unresolved_tmax_outliers.any():
        raise ValueError(
            "Some Tmax spatial outliers remain numeric."
        )

    # -----------------------------------------------------
    # Re-screen the stage-2 values
    # -----------------------------------------------------

    both_values = (
        data[
            "tmin_cleaned_stage2"
        ].notna()
        & data[
            "tmax_cleaned_stage2"
        ].notna()
    )

    data["step5c_post_dtr"] = (
        data["tmax_cleaned_stage2"]
        - data["tmin_cleaned_stage2"]
    ).astype("float32")

    data[
        "step5c_post_tmin_zero"
    ] = (
        data[
            "tmin_cleaned_stage2"
        ].notna()
        & np.isclose(
            data[
                "tmin_cleaned_stage2"
            ],
            0.0,
            atol=zero_tolerance,
            rtol=0.0,
        )
    )

    data[
        "step5c_post_tmin_physical_range"
    ] = (
        data[
            "tmin_cleaned_stage2"
        ].notna()
        & (
            data[
                "tmin_cleaned_stage2"
            ].lt(tmin_minimum)
            | data[
                "tmin_cleaned_stage2"
            ].gt(tmin_maximum)
        )
    )

    data[
        "step5c_post_tmax_physical_range"
    ] = (
        data[
            "tmax_cleaned_stage2"
        ].notna()
        & (
            data[
                "tmax_cleaned_stage2"
            ].lt(tmax_minimum)
            | data[
                "tmax_cleaned_stage2"
            ].gt(tmax_maximum)
        )
    )

    data[
        "step5c_post_tmax_lt_tmin"
    ] = (
        both_values
        & data[
            "step5c_post_dtr"
        ].lt(0.0)
    )

    data[
        "step5c_post_high_dtr"
    ] = (
        both_values
        & data[
            "step5c_post_dtr"
        ].gt(high_dtr_threshold)
    )

    data[
        "step5c_remaining_hard_flag"
    ] = (
        data[
            "step5c_post_tmin_physical_range"
        ]
        | data[
            "step5c_post_tmax_physical_range"
        ]
        | data[
            "step5c_post_tmax_lt_tmin"
        ]
    )

    data[
        "step5c_remaining_soft_flag"
    ] = (
        data[
            "step5c_post_tmin_zero"
        ]
        | data[
            "step5c_post_high_dtr"
        ]
    )

    # -----------------------------------------------------
    # Final structural validation
    # -----------------------------------------------------

    if len(data) != input_rows:
        raise ValueError(
            "Step 5C changed the row count."
        )

    if data.duplicated(
        subset=[
            "station_uid",
            "date",
        ]
    ).any():
        raise ValueError(
            "Step 5C introduced duplicate keys."
        )

    if (
        data["station_uid"].nunique()
        != EXPECTED_STATIONS
    ):
        raise ValueError(
            "Step 5C changed the station count."
        )

    if (
        data["date"].min()
        != original_minimum_date
    ):
        raise ValueError(
            "Step 5C changed the minimum date."
        )

    if (
        data["date"].max()
        != original_maximum_date
    ):
        raise ValueError(
            "Step 5C changed the maximum date."
        )

    # -----------------------------------------------------
    # Reports
    # -----------------------------------------------------

    report_columns = [
        "station_uid",
        "station_id",
        "station_name_display",
        "date",
        "year",
        "month",
        "day",

        "tmin_cleaned_stage1",
        "tmax_cleaned_stage1",
        "tmin_cleaned_stage2",
        "tmax_cleaned_stage2",

        "tmin_cleaned_source",
        "tmax_cleaned_source",
        "tmin_cleaned_stage2_source",
        "tmax_cleaned_stage2_source",

        "tmin_clim_anomaly",
        "tmax_clim_anomaly",

        "tmin_spatial_neighbour_count",
        "tmin_spatial_neighbour_median_anomaly",
        "tmin_spatial_residual",
        "tmin_spatial_robust_z",

        "tmax_spatial_neighbour_count",
        "tmax_spatial_neighbour_median_anomaly",
        "tmax_spatial_residual",
        "tmax_spatial_robust_z",

        "step5a_flag_codes",
        "step5b_flag_codes",
        "step5b_status",

        "step5c_tmin_set_missing",
        "step5c_tmax_set_missing",
        "step5c_decision",
        "step5c_decision_reason",
        "step5c_status",

        "old_source_rows",
        "new_source_cell",
    ]

    removed_records = data.loc[
        data[
            "step5c_any_value_changed"
        ]
    ].copy()

    removed_records[
        report_columns
    ].to_csv(
        REPORT_DIRECTORY
        / "step5c_removed_spatial_outliers.csv",
        index=False,
    )

    retained_regional = data.loc[
        regional_signal
        | regional_cold
    ].copy()

    retained_regional[
        report_columns
    ].to_csv(
        REPORT_DIRECTORY
        / "step5c_retained_regional_signals.csv",
        index=False,
    )

    retained_insufficient = data.loc[
        insufficient
        | unavailable
    ].copy()

    retained_insufficient[
        report_columns
    ].to_csv(
        REPORT_DIRECTORY
        / "step5c_retained_insufficient_evidence.csv",
        index=False,
    )

    remaining_flags = data.loc[
        data[
            "step5c_remaining_hard_flag"
        ]
        | data[
            "step5c_remaining_soft_flag"
        ]
    ].copy()

    remaining_flags[
        report_columns
        + [
            "step5c_post_dtr",
            "step5c_post_tmin_zero",
            "step5c_post_tmin_physical_range",
            "step5c_post_tmax_physical_range",
            "step5c_post_tmax_lt_tmin",
            "step5c_post_high_dtr",
            "step5c_remaining_hard_flag",
            "step5c_remaining_soft_flag",
        ]
    ].to_csv(
        REPORT_DIRECTORY
        / "step5c_remaining_post_cleaning_flags.csv",
        index=False,
    )

    decision_log = data.loc[
        step5a_candidate
    ].copy()

    decision_log[
        report_columns
    ].to_csv(
        REPORT_DIRECTORY
        / "step5c_spatial_decision_log.csv",
        index=False,
    )

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
            total_records=("date", "size"),
            numeric_tmin_stage1=(
                "tmin_cleaned_stage1",
                "count",
            ),
            numeric_tmin_stage2=(
                "tmin_cleaned_stage2",
                "count",
            ),
            numeric_tmax_stage1=(
                "tmax_cleaned_stage1",
                "count",
            ),
            numeric_tmax_stage2=(
                "tmax_cleaned_stage2",
                "count",
            ),
            tmin_spatial_values_removed=(
                "step5c_tmin_set_missing",
                "sum",
            ),
            tmax_spatial_values_removed=(
                "step5c_tmax_set_missing",
                "sum",
            ),
            regional_cold_records_retained=(
                "step5b_regional_cold_signal_supported",
                "sum",
            ),
            regional_signal_records_retained=(
                "step5b_regional_signal_supported",
                "sum",
            ),
            insufficient_evidence_records_retained=(
                "step5b_insufficient_same_day_neighbours",
                "sum",
            ),
            remaining_hard_flags=(
                "step5c_remaining_hard_flag",
                "sum",
            ),
            remaining_soft_flags=(
                "step5c_remaining_soft_flag",
                "sum",
            ),
        )
        .reset_index()
        .sort_values("station_uid")
    )

    station_summary.to_csv(
        REPORT_DIRECTORY
        / "step5c_station_cleaning_summary.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Save output
    # -----------------------------------------------------

    data.to_parquet(
        OUTPUT_FILE,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    output_parquet = pq.ParquetFile(
        OUTPUT_FILE
    )

    decision_counts = (
        data["step5c_decision"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    status_counts = (
        data["step5c_status"]
        .value_counts()
        .sort_index()
        .to_dict()
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
            input_rows,
        "output_rows":
            output_parquet.metadata.num_rows,
        "unique_stations":
            station_count,
        "minimum_date":
            data["date"].min().date().isoformat(),
        "maximum_date":
            data["date"].max().date().isoformat(),
        "tmin_spatial_outliers_set_missing":
            int(tmin_remove.sum()),
        "tmax_spatial_outliers_set_missing":
            int(tmax_remove.sum()),
        "records_with_any_value_changed":
            int(
                data[
                    "step5c_any_value_changed"
                ].sum()
            ),
        "regional_signal_records_retained":
            int(regional_signal.sum()),
        "regional_cold_records_retained":
            int(regional_cold.sum()),
        "insufficient_or_unavailable_records_retained":
            int(
                (
                    insufficient
                    | unavailable
                ).sum()
            ),
        "remaining_hard_flags":
            int(
                data[
                    "step5c_remaining_hard_flag"
                ].sum()
            ),
        "remaining_soft_flags":
            int(
                data[
                    "step5c_remaining_soft_flag"
                ].sum()
            ),
        "interpolation_applied":
            False,
        "decision_counts":
            {
                str(key): int(value)
                for key, value
                in decision_counts.items()
            },
        "status_counts":
            {
                str(key): int(value)
                for key, value
                in status_counts.items()
            },
        "output_size_bytes":
            OUTPUT_FILE.stat().st_size,
    }

    (
        REPORT_DIRECTORY
        / "step5c_cleaning_summary.json"
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
        / "step5c_parquet_schema.json"
    ).write_text(
        json.dumps(
            {
                field.name: str(field.type)
                for field
                in output_parquet.schema_arrow
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "STEP 5C: CONSERVATIVE SPATIAL CLEANING",
        "=" * 43,
        f"Input rows: {input_rows:,}",
        (
            "Output rows: "
            f"{output_parquet.metadata.num_rows:,}"
        ),
        f"Unique stations: {station_count}",
        (
            "Tmin spatial outliers set missing: "
            f"{tmin_remove.sum():,}"
        ),
        (
            "Tmax spatial outliers set missing: "
            f"{tmax_remove.sum():,}"
        ),
        (
            "Records with any value changed: "
            f"{data['step5c_any_value_changed'].sum():,}"
        ),
        (
            "Regional signals retained: "
            f"{regional_signal.sum():,}"
        ),
        (
            "Regional cold signals retained: "
            f"{regional_cold.sum():,}"
        ),
        (
            "Insufficient/unavailable cases retained: "
            f"{(insufficient | unavailable).sum():,}"
        ),
        (
            "Remaining hard flags: "
            f"{data['step5c_remaining_hard_flag'].sum():,}"
        ),
        (
            "Remaining soft flags: "
            f"{data['step5c_remaining_soft_flag'].sum():,}"
        ),
        "",
        "Interpolation applied: No",
        "",
        "Decision counts:",
        *[
            f"- {key}: {value:,}"
            for key, value
            in decision_counts.items()
        ],
        "",
        f"Output: {OUTPUT_FILE}",
    ]

    (
        REPORT_DIRECTORY
        / "step5c_cleaning_report.txt"
    ).write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(report_lines))

    failures: list[str] = []

    if (
        output_parquet.metadata.num_rows
        != input_rows
    ):
        failures.append(
            "Output row count differs from input."
        )

    if (
        data["station_uid"].nunique()
        != EXPECTED_STATIONS
    ):
        failures.append(
            "Unexpected station count."
        )

    if data.duplicated(
        subset=[
            "station_uid",
            "date",
        ]
    ).any():
        failures.append(
            "Duplicate station-date keys exist."
        )

    if new_tmin_created.any():
        failures.append(
            "A new numeric Tmin was created."
        )

    if new_tmax_created.any():
        failures.append(
            "A new numeric Tmax was created."
        )

    if regional_cold_removed.any():
        failures.append(
            "A regionally supported cold value was removed."
        )

    if unresolved_tmin_outliers.any():
        failures.append(
            "Some Tmin spatial outliers remain numeric."
        )

    if unresolved_tmax_outliers.any():
        failures.append(
            "Some Tmax spatial outliers remain numeric."
        )

    if failures:
        raise SystemExit(
            "\nSTEP 5C FAILED:\n- "
            + "\n- ".join(failures)
        )

    print("\nSTEP 5C PASSED.")


if __name__ == "__main__":
    main()
