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


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "04_clean_data"
    / "temperature_daily_cleaned_stage2.parquet"
)

POLICY_FILE = (
    PROJECT_ROOT
    / "00_admin"
    / "step6a_network_policy.yaml"
)

COMPLETENESS_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "step6a_djf_station_winter_completeness.parquet"
)

FIXED_NETWORK_FILE = (
    PROJECT_ROOT
    / "02_metadata"
    / "step6a_fixed_station_network.csv"
)

REPORT_DIRECTORY = PROJECT_ROOT / "05_qc_reports"
ADMIN_DIRECTORY = PROJECT_ROOT / "00_admin"

EXPECTED_TOTAL_STATIONS = 48
MINIMUM_REASONABLE_FIXED_STATIONS = 15


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        return yaml.safe_load(file_handle)


def winter_label(start_year: int) -> str:
    return (
        f"{start_year}/"
        f"{str(start_year + 1)[-2:]}"
    )


def winter_dates(
    start_year: int,
) -> pd.DatetimeIndex:
    start_date = pd.Timestamp(
        year=start_year,
        month=12,
        day=1,
    )

    end_date = (
        pd.Timestamp(
            year=start_year + 1,
            month=3,
            day=1,
        )
        - pd.Timedelta(days=1)
    )

    return pd.date_range(
        start=start_date,
        end=end_date,
        freq="D",
    )


def longest_true_run(
    values: list[bool],
) -> int:
    longest = 0
    current = 0

    for value in values:
        if value:
            current += 1
            longest = max(
                longest,
                current,
            )
        else:
            current = 0

    return longest


def main() -> None:
    for path in [
        INPUT_FILE,
        POLICY_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    COMPLETENESS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIXED_NETWORK_FILE.parent.mkdir(
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

    study_rules = policy["study_period"]
    completeness_rules = policy[
        "station_winter_completeness"
    ]
    network_rules = policy["fixed_network"]
    winter_network_rules = policy[
        "individual_winter_network"
    ]

    first_winter = int(
        study_rules[
            "first_winter_start_year"
        ]
    )

    last_winter = int(
        study_rules[
            "last_winter_start_year"
        ]
    )

    winter_months = {
        int(value)
        for value in study_rules[
            "winter_months"
        ]
    }

    completeness_threshold = float(
        completeness_rules[
            "minimum_percent"
        ]
    )

    temperature_column = str(
        completeness_rules[
            "temperature_variable"
        ]
    )

    interpolation_allowed = bool(
        completeness_rules[
            "interpolation_allowed"
        ]
    )

    minimum_complete_winters = int(
        network_rules[
            "minimum_complete_winters"
        ]
    )

    minimum_complete_fraction = float(
        network_rules[
            "minimum_complete_winter_fraction"
        ]
    )

    official_metadata_required = bool(
        network_rules[
            "official_metadata_required"
        ]
    )

    coordinates_required = bool(
        network_rules[
            "coordinates_required"
        ]
    )

    minimum_winter_station_fraction = float(
        winter_network_rules[
            "minimum_fixed_station_fraction"
        ]
    )

    minimum_winter_station_count = int(
        winter_network_rules[
            "minimum_fixed_station_count"
        ]
    )

    if interpolation_allowed:
        raise ValueError(
            "Step 6A policy must not permit interpolation."
        )

    winter_years = list(
        range(
            first_winter,
            last_winter + 1,
        )
    )

    expected_winter_count = (
        last_winter
        - first_winter
        + 1
    )

    if expected_winter_count != 40:
        raise ValueError(
            "Expected 40 study winters; "
            f"found {expected_winter_count}."
        )

    data = pd.read_parquet(
        INPUT_FILE
    )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    if temperature_column not in data.columns:
        raise ValueError(
            f"Required temperature column missing: "
            f"{temperature_column}"
        )

    data[temperature_column] = (
        pd.to_numeric(
            data[temperature_column],
            errors="coerce",
        )
    )

    if data.duplicated(
        subset=[
            "station_uid",
            "date",
        ]
    ).any():
        raise ValueError(
            "Input contains duplicate station-date keys."
        )

    station_metadata_columns = [
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

    missing_metadata_columns = [
        column
        for column in station_metadata_columns
        if column not in data.columns
    ]

    if missing_metadata_columns:
        raise ValueError(
            "Missing station metadata columns: "
            f"{missing_metadata_columns}"
        )

    station_metadata = (
        data[
            station_metadata_columns
        ]
        .drop_duplicates(
            subset=["station_uid"]
        )
        .sort_values("station_uid")
        .reset_index(drop=True)
    )

    station_count = len(station_metadata)

    if station_count != EXPECTED_TOTAL_STATIONS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_STATIONS} stations; "
            f"found {station_count}."
        )

    # -----------------------------------------------------
    # Extract DJF records and derive winter labels
    # -----------------------------------------------------

    djf = data.loc[
        data["month"].isin(
            winter_months
        ),
        [
            "station_uid",
            "date",
            "year",
            "month",
            "day",
            temperature_column,
        ],
    ].copy()

    djf["winter_start_year"] = np.where(
        djf["month"].eq(12),
        djf["year"],
        djf["year"] - 1,
    ).astype("int16")

    djf = djf.loc[
        djf["winter_start_year"].between(
            first_winter,
            last_winter,
        )
    ].copy()

    if djf.duplicated(
        subset=[
            "station_uid",
            "date",
        ]
    ).any():
        raise ValueError(
            "Duplicate DJF station-date keys exist."
        )

    observed_summary = (
        djf.groupby(
            [
                "station_uid",
                "winter_start_year",
            ],
            as_index=False,
        )
        .agg(
            represented_days=(
                "date",
                "nunique",
            ),
            numeric_tmin_days=(
                temperature_column,
                "count",
            ),
            first_represented_date=(
                "date",
                "min",
            ),
            last_represented_date=(
                "date",
                "max",
            ),
        )
    )

    # -----------------------------------------------------
    # Create all station-winter combinations
    # -----------------------------------------------------

    station_winter_grid = (
        pd.MultiIndex.from_product(
            [
                station_metadata[
                    "station_uid"
                ].tolist(),
                winter_years,
            ],
            names=[
                "station_uid",
                "winter_start_year",
            ],
        )
        .to_frame(index=False)
    )

    completeness = (
        station_winter_grid
        .merge(
            station_metadata,
            on="station_uid",
            how="left",
            validate="many_to_one",
        )
        .merge(
            observed_summary,
            on=[
                "station_uid",
                "winter_start_year",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    for column in [
        "represented_days",
        "numeric_tmin_days",
    ]:
        completeness[column] = (
            completeness[column]
            .fillna(0)
            .astype("int16")
        )

    completeness["winter_label"] = (
        completeness[
            "winter_start_year"
        ].map(winter_label)
    )

    expected_days_map = {
        year: len(winter_dates(year))
        for year in winter_years
    }

    completeness["expected_days"] = (
        completeness[
            "winter_start_year"
        ].map(expected_days_map)
        .astype("int16")
    )

    completeness[
        "minimum_required_numeric_days"
    ] = np.ceil(
        completeness["expected_days"]
        * completeness_threshold
        / 100.0
    ).astype("int16")

    completeness[
        "missing_station_date_rows"
    ] = (
        completeness["expected_days"]
        - completeness["represented_days"]
    ).clip(lower=0).astype("int16")

    completeness[
        "represented_rows_missing_tmin"
    ] = (
        completeness["represented_days"]
        - completeness["numeric_tmin_days"]
    ).clip(lower=0).astype("int16")

    completeness[
        "total_missing_tmin_days"
    ] = (
        completeness["expected_days"]
        - completeness["numeric_tmin_days"]
    ).clip(lower=0).astype("int16")

    completeness[
        "tmin_completeness_percent"
    ] = (
        100.0
        * completeness[
            "numeric_tmin_days"
        ]
        / completeness[
            "expected_days"
        ]
    ).astype("float32")

    completeness[
        "winter_complete"
    ] = (
        completeness[
            "numeric_tmin_days"
        ]
        .ge(
            completeness[
                "minimum_required_numeric_days"
            ]
        )
    )

    completeness[
        "station_winter_status"
    ] = np.select(
        [
            completeness[
                "winter_complete"
            ],
            completeness[
                "represented_days"
            ].eq(0),
            completeness[
                "numeric_tmin_days"
            ].eq(0),
        ],
        [
            "complete",
            "no_station_date_rows",
            "represented_without_numeric_tmin",
        ],
        default="incomplete",
    )

    expected_completeness_rows = (
        station_count
        * expected_winter_count
    )

    if (
        len(completeness)
        != expected_completeness_rows
    ):
        raise ValueError(
            "Unexpected station-winter row count: "
            f"{len(completeness):,}; expected "
            f"{expected_completeness_rows:,}."
        )

    if completeness.duplicated(
        subset=[
            "station_uid",
            "winter_start_year",
        ]
    ).any():
        raise ValueError(
            "Duplicate station-winter keys exist."
        )

    if not set(
        completeness[
            "expected_days"
        ].unique()
    ).issubset({90, 91}):
        raise ValueError(
            "Unexpected DJF expected-day count."
        )

    # -----------------------------------------------------
    # Build station-level fixed-network summary
    # -----------------------------------------------------

    station_rows: list[
        dict[str, Any]
    ] = []

    for metadata_row in (
        station_metadata.itertuples(
            index=False
        )
    ):
        station_uid = (
            metadata_row.station_uid
        )

        station_data = (
            completeness.loc[
                completeness[
                    "station_uid"
                ].eq(station_uid)
            ]
            .sort_values(
                "winter_start_year"
            )
        )

        complete_values = (
            station_data[
                "winter_complete"
            ]
            .astype(bool)
            .tolist()
        )

        complete_count = int(
            sum(complete_values)
        )

        complete_fraction = (
            complete_count
            / expected_winter_count
        )

        numeric_winters = station_data.loc[
            station_data[
                "numeric_tmin_days"
            ].gt(0)
        ]

        complete_winters = station_data.loc[
            station_data[
                "winter_complete"
            ]
        ]

        official_metadata_available = (
            str(
                metadata_row.metadata_status
            )
            == "official_metadata_available"
        )

        coordinates_available = (
            pd.notna(
                metadata_row.latitude
            )
            and pd.notna(
                metadata_row.longitude
            )
            and bool(
                metadata_row.usable_for_spatial_analysis
            )
        )

        metadata_rule_pass = (
            official_metadata_available
            if official_metadata_required
            else True
        )

        coordinate_rule_pass = (
            coordinates_available
            if coordinates_required
            else True
        )

        completeness_rule_pass = (
            complete_count
            >= minimum_complete_winters
            and complete_fraction
            >= minimum_complete_fraction
        )

        fixed_network_eligible = (
            metadata_rule_pass
            and coordinate_rule_pass
            and completeness_rule_pass
        )

        exclusion_reasons: list[str] = []

        if not metadata_rule_pass:
            exclusion_reasons.append(
                "official_metadata_unavailable"
            )

        if not coordinate_rule_pass:
            exclusion_reasons.append(
                "coordinates_unavailable"
            )

        if not completeness_rule_pass:
            exclusion_reasons.append(
                "insufficient_complete_winters"
            )

        station_rows.append(
            {
                "station_uid":
                    station_uid,
                "station_id":
                    metadata_row.station_id,
                "official_station_name":
                    metadata_row.official_station_name,
                "station_name_display":
                    metadata_row.station_name_display,
                "latitude":
                    metadata_row.latitude,
                "longitude":
                    metadata_row.longitude,
                "metadata_status":
                    metadata_row.metadata_status,
                "usable_for_spatial_analysis":
                    bool(
                        metadata_row.usable_for_spatial_analysis
                    ),

                "study_winter_count":
                    expected_winter_count,
                "complete_winter_count":
                    complete_count,
                "incomplete_winter_count":
                    (
                        expected_winter_count
                        - complete_count
                    ),
                "complete_winter_fraction":
                    complete_fraction,
                "complete_winter_percent":
                    100.0 * complete_fraction,
                "longest_consecutive_complete_winters":
                    longest_true_run(
                        complete_values
                    ),

                "first_winter_with_numeric_tmin":
                    (
                        numeric_winters.iloc[0][
                            "winter_label"
                        ]
                        if not numeric_winters.empty
                        else ""
                    ),
                "last_winter_with_numeric_tmin":
                    (
                        numeric_winters.iloc[-1][
                            "winter_label"
                        ]
                        if not numeric_winters.empty
                        else ""
                    ),
                "first_complete_winter":
                    (
                        complete_winters.iloc[0][
                            "winter_label"
                        ]
                        if not complete_winters.empty
                        else ""
                    ),
                "last_complete_winter":
                    (
                        complete_winters.iloc[-1][
                            "winter_label"
                        ]
                        if not complete_winters.empty
                        else ""
                    ),

                "mean_winter_completeness_percent":
                    float(
                        station_data[
                            "tmin_completeness_percent"
                        ].mean()
                    ),
                "minimum_winter_completeness_percent":
                    float(
                        station_data[
                            "tmin_completeness_percent"
                        ].min()
                    ),

                "official_metadata_rule_pass":
                    metadata_rule_pass,
                "coordinate_rule_pass":
                    coordinate_rule_pass,
                "completeness_rule_pass":
                    completeness_rule_pass,
                "fixed_network_eligible":
                    fixed_network_eligible,
                "fixed_network_exclusion_reason":
                    (
                        ";".join(
                            exclusion_reasons
                        )
                        if exclusion_reasons
                        else ""
                    ),
            }
        )

    station_summary = pd.DataFrame(
        station_rows
    ).sort_values(
        [
            "fixed_network_eligible",
            "station_uid",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    fixed_network = station_summary.loc[
        station_summary[
            "fixed_network_eligible"
        ]
    ].copy()

    fixed_station_count = len(
        fixed_network
    )

    if (
        fixed_station_count
        < MINIMUM_REASONABLE_FIXED_STATIONS
    ):
        raise ValueError(
            "Too few fixed-network stations were selected: "
            f"{fixed_station_count}. Expected at least "
            f"{MINIMUM_REASONABLE_FIXED_STATIONS}."
        )

    if fixed_network[
        "metadata_status"
    ].ne(
        "official_metadata_available"
    ).any():
        raise ValueError(
            "A metadata-pending station entered the "
            "fixed network."
        )

    fixed_station_uids = set(
        fixed_network[
            "station_uid"
        ]
    )

    completeness[
        "in_fixed_network"
    ] = completeness[
        "station_uid"
    ].isin(
        fixed_station_uids
    )

    completeness[
        "usable_for_winter_analysis"
    ] = (
        completeness[
            "in_fixed_network"
        ]
        & completeness[
            "winter_complete"
        ]
    )

    # -----------------------------------------------------
    # Build winter-level network summary
    # -----------------------------------------------------

    minimum_required_fixed_stations = max(
        minimum_winter_station_count,
        math.ceil(
            fixed_station_count
            * minimum_winter_station_fraction
        ),
    )

    winter_summary = (
        completeness.loc[
            completeness[
                "in_fixed_network"
            ]
        ]
        .groupby(
            [
                "winter_start_year",
                "winter_label",
                "expected_days",
            ],
            as_index=False,
        )
        .agg(
            fixed_network_station_count=(
                "station_uid",
                "nunique",
            ),
            complete_fixed_stations=(
                "winter_complete",
                "sum",
            ),
            mean_fixed_network_completeness_percent=(
                "tmin_completeness_percent",
                "mean",
            ),
            minimum_fixed_station_completeness_percent=(
                "tmin_completeness_percent",
                "min",
            ),
            total_numeric_station_days=(
                "numeric_tmin_days",
                "sum",
            ),
            total_expected_station_days=(
                "expected_days",
                "sum",
            ),
        )
        .sort_values(
            "winter_start_year"
        )
        .reset_index(drop=True)
    )

    winter_summary[
        "complete_fixed_station_fraction"
    ] = (
        winter_summary[
            "complete_fixed_stations"
        ]
        / winter_summary[
            "fixed_network_station_count"
        ]
    )

    winter_summary[
        "complete_fixed_station_percent"
    ] = (
        100.0
        * winter_summary[
            "complete_fixed_station_fraction"
        ]
    )

    winter_summary[
        "minimum_required_fixed_stations"
    ] = minimum_required_fixed_stations

    winter_summary[
        "winter_network_eligible"
    ] = (
        winter_summary[
            "complete_fixed_stations"
        ]
        .ge(
            minimum_required_fixed_stations
        )
    )

    if len(winter_summary) != expected_winter_count:
        raise ValueError(
            "Expected 40 winter-network rows; "
            f"found {len(winter_summary)}."
        )

    # -----------------------------------------------------
    # Explicit missing-date report for the fixed network
    # -----------------------------------------------------

    expected_date_rows: list[
        dict[str, Any]
    ] = []

    for station_uid in sorted(
        fixed_station_uids
    ):
        for winter_start_year in winter_years:
            for observation_date in winter_dates(
                winter_start_year
            ):
                expected_date_rows.append(
                    {
                        "station_uid":
                            station_uid,
                        "winter_start_year":
                            winter_start_year,
                        "winter_label":
                            winter_label(
                                winter_start_year
                            ),
                        "date":
                            observation_date,
                    }
                )

    expected_dates = pd.DataFrame(
        expected_date_rows
    )

    actual_fixed_djf = djf.loc[
        djf["station_uid"].isin(
            fixed_station_uids
        ),
        [
            "station_uid",
            "date",
            temperature_column,
        ],
    ].copy()

    missing_dates = expected_dates.merge(
        actual_fixed_djf,
        on=[
            "station_uid",
            "date",
        ],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing_dates["missing_reason"] = np.select(
        [
            missing_dates[
                "_merge"
            ].eq("left_only"),
            missing_dates[
                temperature_column
            ].isna(),
        ],
        [
            "missing_station_date_row",
            "represented_row_missing_tmin",
        ],
        default="available",
    )

    missing_dates = missing_dates.loc[
        missing_dates[
            "missing_reason"
        ].ne("available")
    ].drop(
        columns=["_merge"]
    )

    missing_dates = missing_dates.merge(
        station_metadata[
            [
                "station_uid",
                "station_name_display",
            ]
        ],
        on="station_uid",
        how="left",
        validate="many_to_one",
    )

    missing_dates = missing_dates[
        [
            "station_uid",
            "station_name_display",
            "winter_start_year",
            "winter_label",
            "date",
            "missing_reason",
        ]
    ].sort_values(
        [
            "station_uid",
            "date",
        ]
    )

    # -----------------------------------------------------
    # Save outputs
    # -----------------------------------------------------

    completeness = completeness.sort_values(
        [
            "station_uid",
            "winter_start_year",
        ]
    ).reset_index(drop=True)

    completeness.to_parquet(
        COMPLETENESS_FILE,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    fixed_network.to_csv(
        FIXED_NETWORK_FILE,
        index=False,
    )

    station_summary.to_csv(
        REPORT_DIRECTORY
        / "step6a_station_network_summary.csv",
        index=False,
    )

    winter_summary.to_csv(
        REPORT_DIRECTORY
        / "step6a_winter_network_summary.csv",
        index=False,
    )

    missing_dates.to_csv(
        REPORT_DIRECTORY
        / "step6a_fixed_network_missing_djf_dates.csv",
        index=False,
    )

    excluded_stations = station_summary.loc[
        ~station_summary[
            "fixed_network_eligible"
        ]
    ].copy()

    excluded_stations.to_csv(
        REPORT_DIRECTORY
        / "step6a_excluded_station_list.csv",
        index=False,
    )

    completeness_matrix = (
        completeness.pivot(
            index=[
                "station_uid",
                "station_name_display",
            ],
            columns="winter_label",
            values="tmin_completeness_percent",
        )
        .reset_index()
    )

    completeness_matrix.to_csv(
        REPORT_DIRECTORY
        / "step6a_djf_completeness_matrix.csv",
        index=False,
    )

    binary_matrix = (
        completeness.pivot(
            index=[
                "station_uid",
                "station_name_display",
            ],
            columns="winter_label",
            values="winter_complete",
        )
        .reset_index()
    )

    binary_matrix.to_csv(
        REPORT_DIRECTORY
        / "step6a_djf_complete_winter_matrix.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Summary and report
    # -----------------------------------------------------

    eligible_winter_count = int(
        winter_summary[
            "winter_network_eligible"
        ].sum()
    )

    incomplete_fixed_station_winters = int(
        (
            completeness[
                "in_fixed_network"
            ]
            & ~completeness[
                "winter_complete"
            ]
        ).sum()
    )

    summary = {
        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "input_file":
            str(INPUT_FILE),
        "temperature_column":
            temperature_column,
        "first_winter":
            winter_label(first_winter),
        "last_winter":
            winter_label(last_winter),
        "study_winter_count":
            expected_winter_count,
        "total_station_count":
            station_count,
        "station_winter_rows":
            len(completeness),
        "winter_completeness_threshold_percent":
            completeness_threshold,
        "minimum_complete_winters":
            minimum_complete_winters,
        "minimum_complete_winter_fraction":
            minimum_complete_fraction,
        "fixed_network_station_count":
            fixed_station_count,
        "excluded_station_count":
            len(excluded_stations),
        "minimum_complete_fixed_stations_per_winter":
            minimum_required_fixed_stations,
        "eligible_study_winters":
            eligible_winter_count,
        "ineligible_study_winters":
            (
                expected_winter_count
                - eligible_winter_count
            ),
        "incomplete_fixed_station_winters":
            incomplete_fixed_station_winters,
        "fixed_network_missing_djf_dates":
            len(missing_dates),
        "interpolation_applied":
            False,
        "completeness_file":
            str(COMPLETENESS_FILE),
        "fixed_network_file":
            str(FIXED_NETWORK_FILE),
    }

    (
        REPORT_DIRECTORY
        / "step6a_network_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    parquet_schema = pq.ParquetFile(
        COMPLETENESS_FILE
    ).schema_arrow

    (
        ADMIN_DIRECTORY
        / "step6a_parquet_schema.json"
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
        "STEP 6A: DJF COMPLETENESS AND FIXED NETWORK",
        "=" * 49,
        (
            "Study winters: "
            f"{winter_label(first_winter)} to "
            f"{winter_label(last_winter)}"
        ),
        (
            "Number of study winters: "
            f"{expected_winter_count}"
        ),
        (
            "Total stations assessed: "
            f"{station_count}"
        ),
        (
            "Station-winter records: "
            f"{len(completeness):,}"
        ),
        (
            "Complete-winter threshold: "
            f"{completeness_threshold:.1f}%"
        ),
        (
            "Minimum complete winters for fixed network: "
            f"{minimum_complete_winters}"
        ),
        (
            "Fixed-network stations selected: "
            f"{fixed_station_count}"
        ),
        (
            "Stations excluded: "
            f"{len(excluded_stations)}"
        ),
        (
            "Minimum complete fixed stations per winter: "
            f"{minimum_required_fixed_stations}"
        ),
        (
            "Study winters meeting network rule: "
            f"{eligible_winter_count}"
        ),
        (
            "Study winters failing network rule: "
            f"{expected_winter_count - eligible_winter_count}"
        ),
        (
            "Incomplete fixed station-winters: "
            f"{incomplete_fixed_station_winters}"
        ),
        (
            "Missing DJF dates in fixed network: "
            f"{len(missing_dates):,}"
        ),
        "",
        "Interpolation applied: No",
        "",
        f"Completeness output: {COMPLETENESS_FILE}",
        f"Fixed network: {FIXED_NETWORK_FILE}",
    ]

    (
        REPORT_DIRECTORY
        / "step6a_network_report.txt"
    ).write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(report_lines))

    # -----------------------------------------------------
    # Final validation
    # -----------------------------------------------------

    failures: list[str] = []

    if len(completeness) != expected_completeness_rows:
        failures.append(
            "Unexpected station-winter row count."
        )

    if completeness.duplicated(
        subset=[
            "station_uid",
            "winter_start_year",
        ]
    ).any():
        failures.append(
            "Duplicate station-winter keys exist."
        )

    if station_count != EXPECTED_TOTAL_STATIONS:
        failures.append(
            "Unexpected station count."
        )

    if fixed_station_count < (
        MINIMUM_REASONABLE_FIXED_STATIONS
    ):
        failures.append(
            "Too few fixed-network stations."
        )

    if fixed_network[
        "metadata_status"
    ].ne(
        "official_metadata_available"
    ).any():
        failures.append(
            "A metadata-pending station entered "
            "the fixed network."
        )

    invalid_usable_rows = completeness.loc[
        completeness[
            "usable_for_winter_analysis"
        ]
        & ~completeness[
            "winter_complete"
        ]
    ]

    if not invalid_usable_rows.empty:
        failures.append(
            "An incomplete station-winter is marked usable."
        )

    if failures:
        raise SystemExit(
            "\nSTEP 6A FAILED:\n- "
            + "\n- ".join(failures)
        )

    print("\nSTEP 6A PASSED.")


if __name__ == "__main__":
    main()
