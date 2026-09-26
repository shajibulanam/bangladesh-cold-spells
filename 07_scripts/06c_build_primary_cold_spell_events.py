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
    / "temperature_daily_djf_with_compound_cold_indicators.parquet"
)

POLICY_FILE = (
    PROJECT_ROOT
    / "00_admin"
    / "step7c_primary_event_policy.yaml"
)

DAILY_OUTPUT_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "step7c_primary_daily_event_diagnostics.parquet"
)

EVENT_READY_OUTPUT_FILE = (
    PROJECT_ROOT
    / "04_clean_data"
    / "temperature_daily_djf_with_primary_event_membership.parquet"
)

EVENT_DIRECTORY = PROJECT_ROOT / "06_events"

EVENT_CATALOGUE_FILE = (
    EVENT_DIRECTORY
    / "step7c_primary_cold_spell_event_catalogue.csv"
)

EVENT_DAY_FILE = (
    EVENT_DIRECTORY
    / "step7c_primary_cold_spell_event_days.csv"
)

STATION_PARTICIPATION_FILE = (
    EVENT_DIRECTORY
    / "step7c_primary_event_station_participation.csv"
)

REPORT_DIRECTORY = PROJECT_ROOT / "05_qc_reports"
ADMIN_DIRECTORY = PROJECT_ROOT / "00_admin"

TEMPERATURE_COLUMN = "analysis_tmin_unadjusted"

SEVERITY_LABELS = {
    -1: "unavailable",
    0: "no_bmd_cold_wave",
    1: "mild",
    2: "moderate",
    3: "severe",
    4: "very_severe",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        return yaml.safe_load(file_handle)


def parse_boolean(
    values: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)

    normalized = (
        values.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
        "": False,
    }

    parsed = normalized.map(mapping)

    unknown = (
        parsed.isna()
        & normalized.ne("")
    )

    if unknown.any():
        examples = sorted(
            normalized.loc[
                unknown
            ].unique().tolist()
        )

        raise ValueError(
            "Unrecognized Boolean values: "
            f"{examples}"
        )

    return parsed.fillna(False).astype(bool)


def values_equal_with_nan(
    first: pd.Series,
    second: pd.Series,
) -> bool:
    first_values = pd.to_numeric(
        first,
        errors="coerce",
    ).to_numpy(dtype=float)

    second_values = pd.to_numeric(
        second,
        errors="coerce",
    ).to_numpy(dtype=float)

    return bool(
        np.allclose(
            first_values,
            second_values,
            atol=1e-7,
            rtol=0.0,
            equal_nan=True,
        )
    )


def safe_fraction(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    result = np.where(
        denominator.gt(0),
        numerator / denominator,
        np.nan,
    )

    return pd.Series(
        result,
        index=numerator.index,
        dtype="float64",
    )


def assign_events(
    daily: pd.DataFrame,
    minimum_duration: int,
) -> pd.DataFrame:
    """
    Convert consecutive national candidate cold days
    into separate event IDs.

    Runs are calculated independently inside each winter.
    """
    result = daily.sort_values(
        [
            "winter_start_year",
            "date",
        ]
    ).copy()

    result["candidate_run_length"] = 0
    result["primary_event_active"] = False

    result["primary_event_id"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )

    result[
        "primary_event_sequence"
    ] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int16",
    )

    result[
        "primary_event_day_index"
    ] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int16",
    )

    for winter_start_year, winter in result.groupby(
        "winter_start_year",
        sort=True,
    ):
        winter = winter.sort_values("date")

        candidate = winter[
            "national_candidate_cold_day"
        ].astype(bool)

        previous_candidate = candidate.shift(
            fill_value=False
        )

        consecutive_date = (
            winter["date"]
            .diff()
            .dt.days
            .eq(1)
        )

        new_run = (
            candidate
            & (
                ~previous_candidate
                | ~consecutive_date.fillna(False)
            )
        )

        run_number = new_run.cumsum()

        event_sequence = 0

        candidate_rows = winter.loc[
            candidate
        ]

        if candidate_rows.empty:
            continue

        for _, run in candidate_rows.groupby(
            run_number.loc[candidate]
        ):
            run_indices = run.index
            run_length = len(run_indices)

            result.loc[
                run_indices,
                "candidate_run_length",
            ] = run_length

            if run_length < minimum_duration:
                continue

            event_sequence += 1

            event_id = (
                f"BDCS_{int(winter_start_year)}_"
                f"{event_sequence:02d}"
            )

            result.loc[
                run_indices,
                "primary_event_active",
            ] = True

            result.loc[
                run_indices,
                "primary_event_id",
            ] = event_id

            result.loc[
                run_indices,
                "primary_event_sequence",
            ] = event_sequence

            result.loc[
                run_indices,
                "primary_event_day_index",
            ] = np.arange(
                1,
                run_length + 1,
            )

    result[
        "candidate_run_length"
    ] = result[
        "candidate_run_length"
    ].astype("int16")

    return result


def build_event_catalogue(
    event_days: pd.DataFrame,
) -> pd.DataFrame:
    event_columns = [
        "event_id",
        "winter_start_year",
        "winter_label",
        "start_date",
        "end_date",
        "duration_days",
        "start_winter_day_index",
        "end_winter_day_index",
        "network_station_count",
        "minimum_station_winter_eligible_count",
        "minimum_observed_eligible_station_count",
        "minimum_cold_station_count",
        "maximum_cold_station_count",
        "mean_cold_station_count",
        "peak_cold_spatial_fraction",
        "mean_cold_spatial_fraction",
        "peak_coverage_date",
        "cumulative_cold_station_days",
        "spatial_duration_index",
        "national_minimum_tmin",
        "cold_station_minimum_tmin",
        "maximum_bmd_severity_rank",
        "most_severe_bmd_category",
        "bmd_mild_station_days",
        "bmd_moderate_station_days",
        "bmd_severe_station_days",
        "bmd_very_severe_station_days",
        "total_compound_bmd_deficit",
        "total_compound_p10_deficit",
    ]

    if event_days.empty:
        return pd.DataFrame(
            columns=event_columns
        )

    rows: list[dict[str, Any]] = []

    for event_id, event in event_days.groupby(
        "primary_event_id",
        sort=True,
    ):
        event = event.sort_values("date")

        peak_row = (
            event.sort_values(
                [
                    "compound_cold_station_count",
                    "compound_cold_spatial_fraction",
                    "date",
                ],
                ascending=[
                    False,
                    False,
                    True,
                ],
            )
            .iloc[0]
        )

        maximum_rank = int(
            event[
                "maximum_bmd_severity_rank"
            ].max()
        )

        rows.append(
            {
                "event_id": event_id,

                "winter_start_year": int(
                    event[
                        "winter_start_year"
                    ].iloc[0]
                ),

                "winter_label": event[
                    "winter_label"
                ].iloc[0],

                "start_date": event["date"].min(),
                "end_date": event["date"].max(),
                "duration_days": len(event),

                "start_winter_day_index": int(
                    event[
                        "winter_day_index"
                    ].min()
                ),

                "end_winter_day_index": int(
                    event[
                        "winter_day_index"
                    ].max()
                ),

                "network_station_count": int(
                    event[
                        "network_station_count"
                    ].iloc[0]
                ),

                "minimum_station_winter_eligible_count": int(
                    event[
                        "station_winter_eligible_count"
                    ].min()
                ),

                "minimum_observed_eligible_station_count": int(
                    event[
                        "observed_eligible_station_count"
                    ].min()
                ),

                "minimum_cold_station_count": int(
                    event[
                        "compound_cold_station_count"
                    ].min()
                ),

                "maximum_cold_station_count": int(
                    event[
                        "compound_cold_station_count"
                    ].max()
                ),

                "mean_cold_station_count": float(
                    event[
                        "compound_cold_station_count"
                    ].mean()
                ),

                "peak_cold_spatial_fraction": float(
                    event[
                        "compound_cold_spatial_fraction"
                    ].max()
                ),

                "mean_cold_spatial_fraction": float(
                    event[
                        "compound_cold_spatial_fraction"
                    ].mean()
                ),

                "peak_coverage_date": peak_row[
                    "date"
                ],

                "cumulative_cold_station_days": int(
                    event[
                        "compound_cold_station_count"
                    ].sum()
                ),

                "spatial_duration_index": float(
                    event[
                        "compound_cold_spatial_fraction"
                    ].sum()
                ),

                "national_minimum_tmin": float(
                    event[
                        "national_minimum_tmin"
                    ].min()
                ),

                "cold_station_minimum_tmin": float(
                    event[
                        "cold_station_minimum_tmin"
                    ].min()
                ),

                "maximum_bmd_severity_rank":
                    maximum_rank,

                "most_severe_bmd_category":
                    SEVERITY_LABELS.get(
                        maximum_rank,
                        "unavailable",
                    ),

                "bmd_mild_station_days": int(
                    event[
                        "bmd_mild_station_count"
                    ].sum()
                ),

                "bmd_moderate_station_days": int(
                    event[
                        "bmd_moderate_station_count"
                    ].sum()
                ),

                "bmd_severe_station_days": int(
                    event[
                        "bmd_severe_station_count"
                    ].sum()
                ),

                "bmd_very_severe_station_days": int(
                    event[
                        "bmd_very_severe_station_count"
                    ].sum()
                ),

                "total_compound_bmd_deficit": float(
                    event[
                        "total_compound_bmd_deficit"
                    ].sum()
                ),

                "total_compound_p10_deficit": float(
                    event[
                        "total_compound_p10_deficit"
                    ].sum()
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "start_date",
                "event_id",
            ]
        )
        .reset_index(drop=True)
    )


def build_station_participation(
    data: pd.DataFrame,
    catalogue: pd.DataFrame,
    member_column: str,
    usable_column: str,
    cold_column: str,
) -> pd.DataFrame:
    output_columns = [
        "event_id",
        "winter_label",
        "event_start_date",
        "event_end_date",
        "event_duration_days",
        "station_uid",
        "station_id",
        "station_name_display",
        "event_observed_day_count",
        "event_cold_day_count",
        "event_cold_day_fraction",
        "first_cold_date",
        "last_cold_date",
        "minimum_tmin",
        "maximum_bmd_severity_rank",
        "most_severe_bmd_category",
        "mild_day_count",
        "moderate_day_count",
        "severe_day_count",
        "very_severe_day_count",
        "total_bmd_deficit",
        "total_p10_deficit",
    ]

    if catalogue.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    frames: list[pd.DataFrame] = []

    for event in catalogue.itertuples(
        index=False
    ):
        selected = data.loc[
            data["date"].between(
                event.start_date,
                event.end_date,
            )
            & data[member_column]
        ].copy()

        selected["_observed"] = (
            selected[usable_column]
            & selected[
                TEMPERATURE_COLUMN
            ].notna()
        )

        selected["_cold"] = (
            selected["_observed"]
            & selected[cold_column]
        )

        observed_counts = (
            selected.loc[
                selected["_observed"]
            ]
            .groupby(
                "station_uid",
                as_index=False,
            )
            .agg(
                event_observed_day_count=(
                    "date",
                    "count",
                )
            )
        )

        cold = selected.loc[
            selected["_cold"]
        ].copy()

        if cold.empty:
            continue

        if "station_id" not in cold.columns:
            cold["station_id"] = ""

        cold["_mild"] = (
            cold[
                "bmd_absolute_category_raw"
            ].eq("mild")
        )

        cold["_moderate"] = (
            cold[
                "bmd_absolute_category_raw"
            ].eq("moderate")
        )

        cold["_severe"] = (
            cold[
                "bmd_absolute_category_raw"
            ].eq("severe")
        )

        cold["_very_severe"] = (
            cold[
                "bmd_absolute_category_raw"
            ].eq("very_severe")
        )

        participation = (
            cold.groupby(
                [
                    "station_uid",
                    "station_id",
                    "station_name_display",
                ],
                as_index=False,
                dropna=False,
            )
            .agg(
                event_cold_day_count=(
                    "date",
                    "count",
                ),

                first_cold_date=(
                    "date",
                    "min",
                ),

                last_cold_date=(
                    "date",
                    "max",
                ),

                minimum_tmin=(
                    TEMPERATURE_COLUMN,
                    "min",
                ),

                maximum_bmd_severity_rank=(
                    "bmd_absolute_severity_rank_raw",
                    "max",
                ),

                mild_day_count=(
                    "_mild",
                    "sum",
                ),

                moderate_day_count=(
                    "_moderate",
                    "sum",
                ),

                severe_day_count=(
                    "_severe",
                    "sum",
                ),

                very_severe_day_count=(
                    "_very_severe",
                    "sum",
                ),

                total_bmd_deficit=(
                    "tmin_below_bmd_10c_deficit",
                    "sum",
                ),

                total_p10_deficit=(
                    "tmin_below_station_p10_deficit",
                    "sum",
                ),
            )
        )

        participation = participation.merge(
            observed_counts,
            on="station_uid",
            how="left",
            validate="one_to_one",
        )

        participation[
            "event_observed_day_count"
        ] = (
            participation[
                "event_observed_day_count"
            ]
            .fillna(0)
            .astype("int16")
        )

        participation[
            "event_cold_day_fraction"
        ] = np.where(
            participation[
                "event_observed_day_count"
            ].gt(0),
            participation[
                "event_cold_day_count"
            ]
            / participation[
                "event_observed_day_count"
            ],
            np.nan,
        )

        participation[
            "most_severe_bmd_category"
        ] = (
            participation[
                "maximum_bmd_severity_rank"
            ]
            .fillna(-1)
            .astype(int)
            .map(SEVERITY_LABELS)
            .fillna("unavailable")
        )

        participation["event_id"] = (
            event.event_id
        )

        participation["winter_label"] = (
            event.winter_label
        )

        participation[
            "event_start_date"
        ] = event.start_date

        participation[
            "event_end_date"
        ] = event.end_date

        participation[
            "event_duration_days"
        ] = int(
            event.duration_days
        )

        frames.append(
            participation[
                output_columns
            ]
        )

    if not frames:
        return pd.DataFrame(
            columns=output_columns
        )

    return (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "event_start_date",
                "event_id",
                "station_name_display",
            ]
        )
        .reset_index(drop=True)
    )


def main() -> None:
    for path in [
        INPUT_FILE,
        POLICY_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    DAILY_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    EVENT_READY_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    EVENT_DIRECTORY.mkdir(
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

    study_rules = policy[
        "study_period"
    ]

    network_rules = policy[
        "network"
    ]

    availability_rules = policy[
        "daily_data_availability"
    ]

    cold_rules = policy[
        "national_cold_day"
    ]

    event_rules = policy[
        "cold_spell_event"
    ]

    member_column = str(
        network_rules[
            "membership_column"
        ]
    )

    usable_column = str(
        network_rules[
            "station_winter_usable_column"
        ]
    )

    cold_column = str(
        policy[
            "cold_station_condition"
        ]["column"]
    )

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

    expected_winter_count = int(
        study_rules[
            "expected_winter_count"
        ]
    )

    minimum_complete_fraction = float(
        availability_rules[
            "minimum_complete_station_fraction"
        ]
    )

    minimum_complete_count = int(
        availability_rules[
            "minimum_complete_station_count"
        ]
    )

    minimum_observed_fraction = float(
        availability_rules[
            "minimum_observed_station_fraction"
        ]
    )

    minimum_observed_count = int(
        availability_rules[
            "minimum_observed_station_count"
        ]
    )

    minimum_cold_fraction = float(
        cold_rules[
            "minimum_cold_station_fraction"
        ]
    )

    minimum_cold_count = int(
        cold_rules[
            "minimum_cold_station_count"
        ]
    )

    minimum_duration = int(
        event_rules[
            "minimum_consecutive_days"
        ]
    )

    if bool(
        event_rules[
            "merge_one_day_gaps"
        ]
    ):
        raise ValueError(
            "Step 7C must not merge one-day gaps."
        )

    if bool(
        event_rules[
            "allow_event_across_winter_boundary"
        ]
    ):
        raise ValueError(
            "Step 7C events must not cross winters."
        )

    data = pd.read_parquet(
        INPUT_FILE
    )

    data["station_uid"] = (
        data["station_uid"]
        .astype(str)
    )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    data = data.sort_values(
        [
            "station_uid",
            "date",
        ]
    ).reset_index(drop=True)

    required_columns = [
        "station_uid",
        "station_name_display",
        "date",
        "year",
        "month",
        "day",
        "winter_start_year",
        "winter_label",
        "winter_day_index",
        "winter_expected_days",
        TEMPERATURE_COLUMN,
        member_column,
        usable_column,
        cold_column,
        "bmd_absolute_category_raw",
        "bmd_absolute_severity_rank_raw",
        "tmin_below_bmd_10c_deficit",
        "tmin_below_station_p10_deficit",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing: "
            f"{missing_columns}"
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

    if (
        data[
            "winter_start_year"
        ].nunique()
        != expected_winter_count
    ):
        raise ValueError(
            "Expected 40 winters in the input."
        )

    if not data[
        "winter_start_year"
    ].between(
        first_winter,
        last_winter,
    ).all():
        raise ValueError(
            "Unexpected winter exists in the input."
        )

    data[member_column] = parse_boolean(
        data[member_column]
    )

    data[usable_column] = parse_boolean(
        data[usable_column]
    )

    data[cold_column] = parse_boolean(
        data[cold_column]
    )

    data[
        TEMPERATURE_COLUMN
    ] = pd.to_numeric(
        data[
            TEMPERATURE_COLUMN
        ],
        errors="coerce",
    )

    original_rows = len(data)

    original_keys = data[
        [
            "station_uid",
            "date",
        ]
    ].copy()

    original_temperature = data[
        TEMPERATURE_COLUMN
    ].copy()

    primary = data.loc[
        data[member_column]
    ].copy()

    network_station_count = int(
        primary[
            "station_uid"
        ].nunique()
    )

    if network_station_count < 15:
        raise ValueError(
            "Primary network contains fewer than 15 stations."
        )

    primary["_station_winter_eligible"] = (
        primary[usable_column]
    )

    primary["_observed_eligible"] = (
        primary[
            "_station_winter_eligible"
        ]
        & primary[
            TEMPERATURE_COLUMN
        ].notna()
    )

    primary["_compound_cold"] = (
        primary[
            "_observed_eligible"
        ]
        & primary[cold_column]
    )

    primary["_observed_tmin"] = (
        primary[
            TEMPERATURE_COLUMN
        ].where(
            primary[
                "_observed_eligible"
            ]
        )
    )

    primary["_cold_tmin"] = (
        primary[
            TEMPERATURE_COLUMN
        ].where(
            primary["_compound_cold"]
        )
    )

    primary["_observed_rank"] = (
        pd.to_numeric(
            primary[
                "bmd_absolute_severity_rank_raw"
            ],
            errors="coerce",
        )
        .where(
            primary[
                "_observed_eligible"
            ]
        )
    )

    raw_category = (
        primary[
            "bmd_absolute_category_raw"
        ]
        .fillna("missing")
        .astype(str)
    )

    primary["_mild"] = (
        primary[
            "_observed_eligible"
        ]
        & raw_category.eq("mild")
    )

    primary["_moderate"] = (
        primary[
            "_observed_eligible"
        ]
        & raw_category.eq("moderate")
    )

    primary["_severe"] = (
        primary[
            "_observed_eligible"
        ]
        & raw_category.eq("severe")
    )

    primary["_very_severe"] = (
        primary[
            "_observed_eligible"
        ]
        & raw_category.eq("very_severe")
    )

    primary["_compound_bmd_deficit"] = (
        pd.to_numeric(
            primary[
                "tmin_below_bmd_10c_deficit"
            ],
            errors="coerce",
        )
        .where(
            primary["_compound_cold"]
        )
    )

    primary["_compound_p10_deficit"] = (
        pd.to_numeric(
            primary[
                "tmin_below_station_p10_deficit"
            ],
            errors="coerce",
        )
        .where(
            primary["_compound_cold"]
        )
    )

    daily = (
        primary.groupby(
            [
                "date",
                "year",
                "month",
                "day",
                "winter_start_year",
                "winter_label",
                "winter_day_index",
                "winter_expected_days",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            station_winter_eligible_count=(
                "_station_winter_eligible",
                "sum",
            ),

            observed_eligible_station_count=(
                "_observed_eligible",
                "sum",
            ),

            compound_cold_station_count=(
                "_compound_cold",
                "sum",
            ),

            bmd_mild_station_count=(
                "_mild",
                "sum",
            ),

            bmd_moderate_station_count=(
                "_moderate",
                "sum",
            ),

            bmd_severe_station_count=(
                "_severe",
                "sum",
            ),

            bmd_very_severe_station_count=(
                "_very_severe",
                "sum",
            ),

            national_minimum_tmin=(
                "_observed_tmin",
                "min",
            ),

            national_median_tmin=(
                "_observed_tmin",
                "median",
            ),

            cold_station_minimum_tmin=(
                "_cold_tmin",
                "min",
            ),

            maximum_bmd_severity_rank=(
                "_observed_rank",
                "max",
            ),

            total_compound_bmd_deficit=(
                "_compound_bmd_deficit",
                "sum",
            ),

            total_compound_p10_deficit=(
                "_compound_p10_deficit",
                "sum",
            ),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    daily[
        "network_station_count"
    ] = network_station_count

    minimum_required_complete = max(
        minimum_complete_count,
        math.ceil(
            network_station_count
            * minimum_complete_fraction
        ),
    )

    daily[
        "minimum_required_complete_stations"
    ] = minimum_required_complete

    daily[
        "winter_network_eligible"
    ] = (
        daily[
            "station_winter_eligible_count"
        ].ge(
            minimum_required_complete
        )
    )

    daily[
        "minimum_required_observed_stations"
    ] = np.maximum(
        minimum_observed_count,
        np.ceil(
            daily[
                "station_winter_eligible_count"
            ]
            * minimum_observed_fraction
        ),
    ).astype("int16")

    daily[
        "daily_observation_eligible"
    ] = (
        daily[
            "observed_eligible_station_count"
        ].ge(
            daily[
                "minimum_required_observed_stations"
            ]
        )
    )

    daily[
        "national_network_day_usable"
    ] = (
        daily[
            "winter_network_eligible"
        ]
        & daily[
            "daily_observation_eligible"
        ]
    )

    daily[
        "compound_cold_spatial_fraction"
    ] = safe_fraction(
        daily[
            "compound_cold_station_count"
        ],
        daily[
            "observed_eligible_station_count"
        ],
    )

    daily[
        "compound_cold_spatial_percent"
    ] = (
        100.0
        * daily[
            "compound_cold_spatial_fraction"
        ]
    )

    daily[
        "minimum_required_cold_stations"
    ] = np.maximum(
        minimum_cold_count,
        np.ceil(
            daily[
                "observed_eligible_station_count"
            ]
            * minimum_cold_fraction
        ),
    ).astype("int16")

    daily[
        "national_candidate_cold_day"
    ] = (
        daily[
            "national_network_day_usable"
        ]
        & daily[
            "compound_cold_station_count"
        ].ge(
            daily[
                "minimum_required_cold_stations"
            ]
        )
    )

    daily[
        "maximum_bmd_severity_rank"
    ] = (
        pd.to_numeric(
            daily[
                "maximum_bmd_severity_rank"
            ],
            errors="coerce",
        )
        .fillna(-1)
        .astype("int8")
    )

    daily[
        "most_severe_bmd_category"
    ] = (
        daily[
            "maximum_bmd_severity_rank"
        ]
        .map(SEVERITY_LABELS)
        .fillna("unavailable")
    )

    daily = assign_events(
        daily=daily,
        minimum_duration=minimum_duration,
    )

    event_days = daily.loc[
        daily[
            "primary_event_active"
        ]
    ].copy()

    catalogue = build_event_catalogue(
        event_days
    )

    participation = (
        build_station_participation(
            data=data,
            catalogue=catalogue,
            member_column=member_column,
            usable_column=usable_column,
            cold_column=cold_column,
        )
    )

    if not catalogue.empty:
        participant_counts = (
            participation.groupby(
                "event_id",
                as_index=False,
            )
            .agg(
                participating_station_count=(
                    "station_uid",
                    "nunique",
                )
            )
        )

        catalogue = catalogue.merge(
            participant_counts,
            on="event_id",
            how="left",
            validate="one_to_one",
        )

        catalogue[
            "participating_station_count"
        ] = (
            catalogue[
                "participating_station_count"
            ]
            .fillna(0)
            .astype("int16")
        )
    else:
        catalogue[
            "participating_station_count"
        ] = pd.Series(
            dtype="int16"
        )

    # -----------------------------------------------------
    # Validate events
    # -----------------------------------------------------

    validation_issues: list[
        dict[str, Any]
    ] = []

    for event_id, event in event_days.groupby(
        "primary_event_id"
    ):
        event = event.sort_values("date")

        date_differences = (
            event["date"]
            .diff()
            .dropna()
            .dt.days
        )

        if not date_differences.eq(1).all():
            validation_issues.append(
                {
                    "event_id": event_id,
                    "issue":
                        "nonconsecutive_dates",
                }
            )

        if (
            event[
                "winter_start_year"
            ].nunique()
            != 1
        ):
            validation_issues.append(
                {
                    "event_id": event_id,
                    "issue":
                        "crosses_winter_boundary",
                }
            )

        if len(event) < minimum_duration:
            validation_issues.append(
                {
                    "event_id": event_id,
                    "issue":
                        "shorter_than_minimum_duration",
                }
            )

        if not event[
            "national_network_day_usable"
        ].all():
            validation_issues.append(
                {
                    "event_id": event_id,
                    "issue":
                        "contains_unusable_network_day",
                }
            )

        if not event[
            "national_candidate_cold_day"
        ].all():
            validation_issues.append(
                {
                    "event_id": event_id,
                    "issue":
                        "contains_non_candidate_day",
                }
            )

        spatial_failure = (
            event[
                "compound_cold_station_count"
            ].lt(
                event[
                    "minimum_required_cold_stations"
                ]
            )
        )

        if spatial_failure.any():
            validation_issues.append(
                {
                    "event_id": event_id,
                    "issue":
                        "fails_spatial_requirement",
                }
            )

    validation_report = pd.DataFrame(
        validation_issues,
        columns=[
            "event_id",
            "issue",
        ],
    )

    validation_report.to_csv(
        REPORT_DIRECTORY
        / "step7c_event_validation_issues.csv",
        index=False,
    )

    if not validation_report.empty:
        raise ValueError(
            "Step 7C event validation failed."
        )

    # -----------------------------------------------------
    # Add national event membership to station-day data
    # -----------------------------------------------------

    daily_membership = daily[
        [
            "date",
            "national_network_day_usable",
            "national_candidate_cold_day",
            "compound_cold_station_count",
            "compound_cold_spatial_fraction",
            "minimum_required_cold_stations",
            "candidate_run_length",
            "primary_event_active",
            "primary_event_id",
            "primary_event_day_index",
        ]
    ].copy()

    event_ready = data.merge(
        daily_membership,
        on="date",
        how="left",
        validate="many_to_one",
    )

    event_ready[
        "primary_station_participates_on_event_day"
    ] = (
        event_ready[
            "primary_event_active"
        ].fillna(False).astype(bool)
        & event_ready[member_column]
        & event_ready[usable_column]
        & event_ready[
            TEMPERATURE_COLUMN
        ].notna()
        & event_ready[cold_column]
    )

    event_ready[
        "step7c_temperature_adjusted"
    ] = False

    event_ready[
        "step7c_interpolation_applied"
    ] = False

    event_ready = event_ready.sort_values(
        [
            "station_uid",
            "date",
        ]
    ).reset_index(drop=True)

    if len(event_ready) != original_rows:
        raise ValueError(
            "Step 7C changed the row count."
        )

    if event_ready.duplicated(
        subset=[
            "station_uid",
            "date",
        ]
    ).any():
        raise ValueError(
            "Step 7C introduced duplicate keys."
        )

    if not original_keys.equals(
        event_ready[
            [
                "station_uid",
                "date",
            ]
        ]
    ):
        raise ValueError(
            "Step 7C changed station-date keys "
            "or ordering."
        )

    if not values_equal_with_nan(
        original_temperature,
        event_ready[
            TEMPERATURE_COLUMN
        ],
    ):
        raise ValueError(
            "Step 7C changed Tmin values."
        )

    # -----------------------------------------------------
    # Winter summary
    # -----------------------------------------------------

    winter_rows: list[
        dict[str, Any]
    ] = []

    for winter_start_year, winter in daily.groupby(
        "winter_start_year",
        sort=True,
    ):
        winter_events = catalogue.loc[
            catalogue[
                "winter_start_year"
            ].eq(
                winter_start_year
            )
        ]

        winter_rows.append(
            {
                "winter_start_year":
                    int(winter_start_year),

                "winter_label":
                    winter[
                        "winter_label"
                    ].iloc[0],

                "network_station_count":
                    network_station_count,

                "usable_network_days":
                    int(
                        winter[
                            "national_network_day_usable"
                        ].sum()
                    ),

                "candidate_cold_days":
                    int(
                        winter[
                            "national_candidate_cold_day"
                        ].sum()
                    ),

                "event_days":
                    int(
                        winter[
                            "primary_event_active"
                        ].sum()
                    ),

                "event_count":
                    len(winter_events),

                "total_event_duration_days":
                    (
                        int(
                            winter_events[
                                "duration_days"
                            ].sum()
                        )
                        if not winter_events.empty
                        else 0
                    ),

                "maximum_event_duration_days":
                    (
                        int(
                            winter_events[
                                "duration_days"
                            ].max()
                        )
                        if not winter_events.empty
                        else 0
                    ),

                "maximum_daily_cold_station_count":
                    int(
                        winter[
                            "compound_cold_station_count"
                        ].max()
                    ),

                "maximum_daily_cold_spatial_fraction":
                    float(
                        winter[
                            "compound_cold_spatial_fraction"
                        ].max()
                    ),

                "winter_minimum_tmin":
                    float(
                        winter[
                            "national_minimum_tmin"
                        ].min()
                    ),
            }
        )

    winter_summary = pd.DataFrame(
        winter_rows
    )

    # -----------------------------------------------------
    # Save outputs
    # -----------------------------------------------------

    daily.to_parquet(
        DAILY_OUTPUT_FILE,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    event_ready.to_parquet(
        EVENT_READY_OUTPUT_FILE,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    catalogue.to_csv(
        EVENT_CATALOGUE_FILE,
        index=False,
    )

    event_days.to_csv(
        EVENT_DAY_FILE,
        index=False,
    )

    participation.to_csv(
        STATION_PARTICIPATION_FILE,
        index=False,
    )

    winter_summary.to_csv(
        REPORT_DIRECTORY
        / "step7c_primary_winter_event_summary.csv",
        index=False,
    )

    daily_parquet = pq.ParquetFile(
        DAILY_OUTPUT_FILE
    )

    event_ready_parquet = pq.ParquetFile(
        EVENT_READY_OUTPUT_FILE
    )

    event_count = len(catalogue)
    event_day_count = len(event_days)

    longest_event = (
        int(
            catalogue[
                "duration_days"
            ].max()
        )
        if not catalogue.empty
        else 0
    )

    winters_with_events = (
        int(
            catalogue[
                "winter_start_year"
            ].nunique()
        )
        if not catalogue.empty
        else 0
    )

    maximum_spatial_fraction = (
        float(
            catalogue[
                "peak_cold_spatial_fraction"
            ].max()
        )
        if not catalogue.empty
        else np.nan
    )

    summary = {
        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "input_file":
            str(INPUT_FILE),

        "network_station_count":
            network_station_count,

        "daily_rows":
            daily_parquet.metadata.num_rows,

        "station_day_rows":
            event_ready_parquet.metadata.num_rows,

        "primary_event_count":
            event_count,

        "primary_event_day_count":
            event_day_count,

        "winters_with_events":
            winters_with_events,

        "longest_event_days":
            longest_event,

        "maximum_event_spatial_fraction":
            maximum_spatial_fraction,

        "minimum_cold_station_fraction":
            minimum_cold_fraction,

        "minimum_cold_station_count":
            minimum_cold_count,

        "minimum_consecutive_days":
            minimum_duration,

        "validation_issue_count":
            len(validation_report),

        "temperature_adjusted":
            False,

        "interpolation_applied":
            False,

        "event_catalogue_file":
            str(EVENT_CATALOGUE_FILE),
    }

    (
        REPORT_DIRECTORY
        / "step7c_primary_event_summary.json"
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
        / "step7c_parquet_schema.json"
    ).write_text(
        json.dumps(
            {
                "daily_diagnostic_schema": {
                    field.name: str(
                        field.type
                    )
                    for field
                    in daily_parquet.schema_arrow
                },

                "event_ready_schema": {
                    field.name: str(
                        field.type
                    )
                    for field
                    in event_ready_parquet.schema_arrow
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "STEP 7C: PRIMARY BANGLADESH COLD-SPELL EVENTS",
        "=" * 52,

        (
            "Primary network stations: "
            f"{network_station_count}"
        ),

        (
            "Daily national records: "
            f"{len(daily):,}"
        ),

        (
            "National candidate cold days: "
            f"{daily['national_candidate_cold_day'].sum():,}"
        ),

        (
            "Primary event count: "
            f"{event_count}"
        ),

        (
            "Primary event-day count: "
            f"{event_day_count}"
        ),

        (
            "Winters with events: "
            f"{winters_with_events}"
        ),

        (
            "Longest event: "
            f"{longest_event} days"
        ),

        "",
        "Primary event rule:",

        "- Compound p10 plus BMD station condition",
        "- At least 20 percent of observed stations",
        "- At least 5 cold stations",
        "- At least 3 consecutive days",
        "- No one-day gap merging",
        "- No event crossing winter boundary",

        "",
        (
            "Event validation issues: "
            f"{len(validation_report)}"
        ),

        "Temperature adjusted: No",
        "Interpolation applied: No",

        "",
        f"Event catalogue: {EVENT_CATALOGUE_FILE}",
        f"Event days: {EVENT_DAY_FILE}",
        (
            "Station participation: "
            f"{STATION_PARTICIPATION_FILE}"
        ),
    ]

    (
        REPORT_DIRECTORY
        / "step7c_primary_event_report.txt"
    ).write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(report_lines))
    print("\nSTEP 7C PASSED.")


if __name__ == "__main__":
    main()
