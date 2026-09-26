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

DAILY_INPUT_FILE = (
    PROJECT_ROOT
    / "04_clean_data"
    / "temperature_daily_cleaned_stage2.parquet"
)

FIXED_NETWORK_FILE = (
    PROJECT_ROOT
    / "02_metadata"
    / "step6a_fixed_station_network.csv"
)

COMPLETENESS_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "step6a_djf_station_winter_completeness.parquet"
)

HOMOGENEITY_FILE = (
    PROJECT_ROOT
    / "02_metadata"
    / "step6b_station_homogeneity_status.csv"
)

POLICY_FILE = (
    PROJECT_ROOT
    / "00_admin"
    / "step6c_network_freeze_policy.yaml"
)

STATION_NETWORK_OUTPUT = (
    PROJECT_ROOT
    / "02_metadata"
    / "step6c_analytical_station_network.csv"
)

MEMBERSHIP_OUTPUT = (
    PROJECT_ROOT
    / "02_metadata"
    / "step6c_sensitivity_network_membership.csv"
)

ANALYSIS_OUTPUT = (
    PROJECT_ROOT
    / "04_clean_data"
    / "temperature_daily_djf_analysis_unadjusted.parquet"
)

REPORT_DIRECTORY = PROJECT_ROOT / "05_qc_reports"
ADMIN_DIRECTORY = PROJECT_ROOT / "00_admin"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        return yaml.safe_load(file_handle)


def boolean_series(
    values: pd.Series,
) -> pd.Series:
    """Safely parse Boolean or text Boolean fields."""
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
            normalized.loc[unknown]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unrecognized Boolean values: "
            f"{examples}"
        )

    return parsed.fillna(False).astype(bool)


def winter_label(
    winter_start_year: int,
) -> str:
    return (
        f"{winter_start_year}/"
        f"{str(winter_start_year + 1)[-2:]}"
    )


def build_djf_calendar(
    first_winter: int,
    last_winter: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for start_year in range(
        first_winter,
        last_winter + 1,
    ):
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

        dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq="D",
        )

        expected_days = len(dates)

        for winter_day_index, date in enumerate(
            dates,
            start=1,
        ):
            rows.append(
                {
                    "date": date,
                    "year": int(date.year),
                    "month": int(date.month),
                    "day": int(date.day),
                    "winter_start_year":
                        start_year,
                    "winter_label":
                        winter_label(
                            start_year
                        ),
                    "winter_day_index":
                        winter_day_index,
                    "winter_expected_days":
                        expected_days,
                    "winter_month_order":
                        (
                            1
                            if date.month == 12
                            else 2
                            if date.month == 1
                            else 3
                        ),
                }
            )

    calendar = pd.DataFrame(rows)

    calendar["year"] = (
        calendar["year"]
        .astype("int16")
    )

    calendar["month"] = (
        calendar["month"]
        .astype("int8")
    )

    calendar["day"] = (
        calendar["day"]
        .astype("int8")
    )

    calendar[
        "winter_start_year"
    ] = (
        calendar[
            "winter_start_year"
        ]
        .astype("int16")
    )

    calendar[
        "winter_day_index"
    ] = (
        calendar[
            "winter_day_index"
        ]
        .astype("int16")
    )

    calendar[
        "winter_expected_days"
    ] = (
        calendar[
            "winter_expected_days"
        ]
        .astype("int16")
    )

    calendar[
        "winter_month_order"
    ] = (
        calendar[
            "winter_month_order"
        ]
        .astype("int8")
    )

    return calendar


def values_equal_with_nan(
    first: pd.Series,
    second: pd.Series,
) -> bool:
    first_numeric = pd.to_numeric(
        first,
        errors="coerce",
    ).to_numpy(dtype=float)

    second_numeric = pd.to_numeric(
        second,
        errors="coerce",
    ).to_numpy(dtype=float)

    return bool(
        np.allclose(
            first_numeric,
            second_numeric,
            atol=1e-7,
            rtol=0.0,
            equal_nan=True,
        )
    )


def main() -> None:
    required_files = [
        DAILY_INPUT_FILE,
        FIXED_NETWORK_FILE,
        COMPLETENESS_FILE,
        HOMOGENEITY_FILE,
        POLICY_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    STATION_NETWORK_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ANALYSIS_OUTPUT.parent.mkdir(
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
    temperature_rules = policy["temperature"]
    network_definitions = policy["networks"]
    winter_rules = policy[
        "winter_network_eligibility"
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

    study_winter_count = (
        last_winter
        - first_winter
        + 1
    )

    if study_winter_count != 40:
        raise ValueError(
            "Step 6C expects exactly 40 winters."
        )

    input_temperature_column = str(
        temperature_rules[
            "input_column"
        ]
    )

    output_temperature_column = str(
        temperature_rules[
            "output_column"
        ]
    )

    if bool(
        temperature_rules[
            "homogenization_allowed"
        ]
    ):
        raise ValueError(
            "Step 6C must not permit homogenization."
        )

    if bool(
        temperature_rules[
            "interpolation_allowed"
        ]
    ):
        raise ValueError(
            "Step 6C must not permit interpolation."
        )

    if bool(
        temperature_rules[
            "statistical_adjustment_allowed"
        ]
    ):
        raise ValueError(
            "Step 6C must not permit statistical adjustment."
        )

    primary_network_id = str(
        policy["primary_network_id"]
    )

    if (
        primary_network_id
        not in network_definitions
    ):
        raise ValueError(
            "Primary network ID is not defined."
        )

    minimum_station_fraction = float(
        winter_rules[
            "minimum_station_fraction"
        ]
    )

    minimum_station_count = int(
        winter_rules[
            "minimum_station_count"
        ]
    )

    freeze_timestamp = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    # -----------------------------------------------------
    # Load and validate fixed station network
    # -----------------------------------------------------

    fixed = pd.read_csv(
        FIXED_NETWORK_FILE
    )

    fixed["station_uid"] = (
        fixed["station_uid"]
        .astype(str)
    )

    if fixed.empty:
        raise ValueError(
            "The Step 6A fixed network is empty."
        )

    if fixed[
        "station_uid"
    ].duplicated().any():
        raise ValueError(
            "Duplicate stations exist in "
            "the Step 6A fixed network."
        )

    if "fixed_network_eligible" in fixed.columns:
        fixed[
            "fixed_network_eligible"
        ] = boolean_series(
            fixed[
                "fixed_network_eligible"
            ]
        )

        if not fixed[
            "fixed_network_eligible"
        ].all():
            raise ValueError(
                "The Step 6A fixed network contains "
                "a noneligible station."
            )

    if fixed[
        "metadata_status"
    ].ne(
        "official_metadata_available"
    ).any():
        raise ValueError(
            "A station without official metadata "
            "exists in the fixed network."
        )

    # -----------------------------------------------------
    # Load Step 6B homogeneity classifications
    # -----------------------------------------------------

    homogeneity = pd.read_csv(
        HOMOGENEITY_FILE
    )

    homogeneity[
        "station_uid"
    ] = (
        homogeneity[
            "station_uid"
        ].astype(str)
    )

    if homogeneity[
        "station_uid"
    ].duplicated().any():
        raise ValueError(
            "Duplicate station classifications "
            "exist in Step 6B."
        )

    fixed_ids = set(
        fixed["station_uid"]
    )

    homogeneity_ids = set(
        homogeneity["station_uid"]
    )

    if fixed_ids != homogeneity_ids:
        raise ValueError(
            "Step 6A and Step 6B station sets differ.\n"
            f"Missing from Step 6B: "
            f"{sorted(fixed_ids - homogeneity_ids)}\n"
            f"Unexpected in Step 6B: "
            f"{sorted(homogeneity_ids - fixed_ids)}"
        )

    for column in [
        "automatic_adjustment_applied",
        "automatic_station_exclusion_applied",
    ]:
        if column in homogeneity.columns:
            homogeneity[column] = (
                boolean_series(
                    homogeneity[column]
                )
            )

            if homogeneity[column].any():
                raise ValueError(
                    f"Step 6B unexpectedly marked "
                    f"{column}=True."
                )

    homogeneity_columns = [
        "station_uid",
        "source_transition_status",
        "all_month_median_shift_post_minus_pre",
        "all_month_standardized_shift",
        "djf_median_shift_post_minus_pre",
        "djf_standardized_shift",
        "variance_shift_flag",
        "best_break_after_year",
        "break_shift_right_minus_left",
        "standardized_break_shift",
        "break_classification",
        "combined_homogeneity_status",
        "recommended_step6c_action",
        "automatic_adjustment_applied",
        "automatic_station_exclusion_applied",
    ]

    homogeneity_columns = [
        column
        for column in homogeneity_columns
        if column in homogeneity.columns
    ]

    stations = fixed.merge(
        homogeneity[
            homogeneity_columns
        ],
        on="station_uid",
        how="left",
        validate="one_to_one",
    )

    required_status_columns = [
        "source_transition_status",
        "combined_homogeneity_status",
    ]

    for column in required_status_columns:
        if stations[column].isna().any():
            raise ValueError(
                f"Missing station status in {column}."
            )

    stations[
        "primary_temperature_treatment"
    ] = "retain_unadjusted"

    stations[
        "homogeneity_adjustment_applied"
    ] = False

    stations[
        "network_freeze_timestamp_utc"
    ] = freeze_timestamp

    # -----------------------------------------------------
    # Define network memberships
    # -----------------------------------------------------

    membership_rows: list[
        dict[str, Any]
    ] = []

    member_columns: list[str] = []

    for (
        network_id,
        definition,
    ) in network_definitions.items():
        excluded_combined = set(
            definition.get(
                "exclude_combined_homogeneity_statuses",
                [],
            )
        )

        excluded_source = set(
            definition.get(
                "exclude_source_transition_statuses",
                [],
            )
        )

        combined_excluded = stations[
            "combined_homogeneity_status"
        ].isin(
            excluded_combined
        )

        source_excluded = stations[
            "source_transition_status"
        ].isin(
            excluded_source
        )

        included = ~(
            combined_excluded
            | source_excluded
        )

        member_column = (
            f"member_{network_id}"
        )

        stations[member_column] = (
            included.astype(bool)
        )

        member_columns.append(
            member_column
        )

        for station_row in (
            stations.itertuples(
                index=False
            )
        ):
            is_included = bool(
                getattr(
                    station_row,
                    member_column,
                )
            )

            exclusion_reasons: list[str] = []

            if (
                station_row.combined_homogeneity_status
                in excluded_combined
            ):
                exclusion_reasons.append(
                    "combined_homogeneity_status="
                    + str(
                        station_row.combined_homogeneity_status
                    )
                )

            if (
                station_row.source_transition_status
                in excluded_source
            ):
                exclusion_reasons.append(
                    "source_transition_status="
                    + str(
                        station_row.source_transition_status
                    )
                )

            membership_rows.append(
                {
                    "network_id":
                        network_id,
                    "network_description":
                        definition[
                            "description"
                        ],
                    "station_uid":
                        station_row.station_uid,
                    "station_id":
                        station_row.station_id,
                    "station_name_display":
                        station_row.station_name_display,
                    "combined_homogeneity_status":
                        station_row.combined_homogeneity_status,
                    "source_transition_status":
                        station_row.source_transition_status,
                    "included":
                        is_included,
                    "exclusion_reason":
                        (
                            ""
                            if is_included
                            else ";".join(
                                exclusion_reasons
                            )
                        ),
                    "temperature_treatment":
                        "retain_unadjusted",
                    "network_freeze_timestamp_utc":
                        freeze_timestamp,
                }
            )

    primary_member_column = (
        f"member_{primary_network_id}"
    )

    if not stations[
        primary_member_column
    ].all():
        raise ValueError(
            "The primary network must retain every "
            "Step 6A fixed station."
        )

    membership = pd.DataFrame(
        membership_rows
    )

    membership["included"] = (
        membership["included"]
        .astype(bool)
    )

    # -----------------------------------------------------
    # Read station-winter completeness
    # -----------------------------------------------------

    completeness = pd.read_parquet(
        COMPLETENESS_FILE
    )

    completeness[
        "station_uid"
    ] = (
        completeness[
            "station_uid"
        ].astype(str)
    )

    completeness = completeness.loc[
        completeness[
            "station_uid"
        ].isin(
            fixed_ids
        )
        & completeness[
            "winter_start_year"
        ].between(
            first_winter,
            last_winter,
        )
    ].copy()

    if completeness.duplicated(
        subset=[
            "station_uid",
            "winter_start_year",
        ]
    ).any():
        raise ValueError(
            "Duplicate fixed station-winter rows exist."
        )

    expected_completeness_rows = (
        len(fixed)
        * study_winter_count
    )

    if (
        len(completeness)
        != expected_completeness_rows
    ):
        raise ValueError(
            "Unexpected number of fixed "
            "station-winter records: "
            f"{len(completeness):,}; expected "
            f"{expected_completeness_rows:,}."
        )

    completeness[
        "winter_complete"
    ] = boolean_series(
        completeness[
            "winter_complete"
        ]
    )

    # -----------------------------------------------------
    # Build a complete DJF station-date calendar
    # -----------------------------------------------------

    calendar = build_djf_calendar(
        first_winter,
        last_winter,
    )

    station_grid = (
        stations[
            ["station_uid"]
        ]
        .merge(
            calendar,
            how="cross",
        )
    )

    expected_analysis_rows = (
        len(fixed)
        * len(calendar)
    )

    if (
        len(station_grid)
        != expected_analysis_rows
    ):
        raise ValueError(
            "Unexpected station-calendar row count."
        )

    daily = pd.read_parquet(
        DAILY_INPUT_FILE
    )

    daily["station_uid"] = (
        daily["station_uid"]
        .astype(str)
    )

    daily["date"] = pd.to_datetime(
        daily["date"]
    )

    if input_temperature_column not in daily.columns:
        raise ValueError(
            "Required temperature column missing: "
            f"{input_temperature_column}"
        )

    if daily.duplicated(
        subset=[
            "station_uid",
            "date",
        ]
    ).any():
        raise ValueError(
            "The Step 5C daily dataset contains "
            "duplicate station-date keys."
        )

    daily = daily.loc[
        daily[
            "station_uid"
        ].isin(
            fixed_ids
        )
        & daily[
            "date"
        ].isin(
            set(calendar["date"])
        )
    ].copy()

    optional_daily_columns = [
        "record_origin",
        "integration_status",

        "tmin_preliminary",
        "tmin_qc_stage1",
        "tmin_cleaned_stage1",
        "tmin_cleaned_stage2",

        "tmin_source",
        "tmin_cleaned_source",
        "tmin_cleaned_stage2_source",
        "tmin_selection_status",
        "reconciliation_status",

        "step4b_decision",
        "step4b_decision_reason",

        "step5a_flag_codes",
        "step5a_priority",
        "step5a_status",

        "step5b_flag_codes",
        "step5b_priority",
        "step5b_status",

        "step5c_decision",
        "step5c_decision_reason",
        "step5c_status",
        "step5c_remaining_hard_flag",
        "step5c_remaining_soft_flag",

        "old_source_rows",
        "new_source_cell",
    ]

    daily_columns = [
        "station_uid",
        "date",
    ] + [
        column
        for column in optional_daily_columns
        if column in daily.columns
    ]

    daily_subset = daily[
        daily_columns
    ].copy()

    analysis = station_grid.merge(
        daily_subset,
        on=[
            "station_uid",
            "date",
        ],
        how="left",
        validate="one_to_one",
        indicator="_daily_merge",
    )

    analysis[
        "source_station_date_row_present"
    ] = (
        analysis[
            "_daily_merge"
        ].eq("both")
    )

    analysis = analysis.drop(
        columns=["_daily_merge"]
    )

    # Merge station metadata and memberships.
    station_output_columns = [
        "station_uid",
        "station_id",
        "official_station_name",
        "station_name_display",
        "latitude",
        "longitude",
        "metadata_status",
        "usable_for_spatial_analysis",
        "complete_winter_count",
        "complete_winter_percent",
        "longest_consecutive_complete_winters",
        "source_transition_status",
        "combined_homogeneity_status",
        "break_classification",
        "best_break_after_year",
        "recommended_step6c_action",
        "primary_temperature_treatment",
        "homogeneity_adjustment_applied",
        "network_freeze_timestamp_utc",
        *member_columns,
    ]

    station_output_columns = [
        column
        for column in station_output_columns
        if column in stations.columns
    ]

    analysis = analysis.merge(
        stations[
            station_output_columns
        ],
        on="station_uid",
        how="left",
        validate="many_to_one",
    )

    completeness_columns = [
        "station_uid",
        "winter_start_year",
        "expected_days",
        "minimum_required_numeric_days",
        "represented_days",
        "numeric_tmin_days",
        "total_missing_tmin_days",
        "tmin_completeness_percent",
        "winter_complete",
        "station_winter_status",
    ]

    available_completeness_columns = [
        column
        for column in completeness_columns
        if column in completeness.columns
    ]

    completeness_merge = completeness[
        available_completeness_columns
    ].rename(
        columns={
            "expected_days":
                "station_winter_expected_days",
        }
    )

    analysis = analysis.merge(
        completeness_merge,
        on=[
            "station_uid",
            "winter_start_year",
        ],
        how="left",
        validate="many_to_one",
    )

    if analysis[
        "winter_complete"
    ].isna().any():
        raise ValueError(
            "Some analysis rows lack station-winter "
            "completeness information."
        )

    analysis[
        "winter_complete"
    ] = boolean_series(
        analysis[
            "winter_complete"
        ]
    )

    if (
        "station_winter_expected_days"
        in analysis.columns
    ):
        mismatch = (
            analysis[
                "winter_expected_days"
            ]
            .ne(
                analysis[
                    "station_winter_expected_days"
                ]
            )
        )

        if mismatch.any():
            raise ValueError(
                "Calendar expected days differ from "
                "Step 6A expected days."
            )

    # -----------------------------------------------------
    # Freeze the unadjusted analysis temperature
    # -----------------------------------------------------

    analysis[
        output_temperature_column
    ] = pd.to_numeric(
        analysis[
            input_temperature_column
        ],
        errors="coerce",
    ).astype("float32")

    source_column = (
        "tmin_cleaned_stage2_source"
    )

    if source_column in analysis.columns:
        analysis[
            "analysis_tmin_source"
        ] = (
            analysis[source_column]
            .fillna("")
            .astype(str)
        )
    else:
        analysis[
            "analysis_tmin_source"
        ] = ""

    analysis.loc[
        analysis[
            output_temperature_column
        ].isna()
        & analysis[
            "analysis_tmin_source"
        ].eq(""),
        "analysis_tmin_source",
    ] = "missing"

    analysis[
        "analysis_temperature_adjusted"
    ] = False

    analysis[
        "analysis_interpolation_applied"
    ] = False

    analysis[
        "analysis_missing_reason"
    ] = np.select(
        [
            analysis[
                output_temperature_column
            ].notna(),

            ~analysis[
                "source_station_date_row_present"
            ],
        ],
        [
            "available",
            "missing_station_date_row",
        ],
        default="represented_row_missing_tmin",
    )

    # -----------------------------------------------------
    # Add network-specific row usability
    # -----------------------------------------------------

    for network_id in (
        network_definitions.keys()
    ):
        member_column = (
            f"member_{network_id}"
        )

        station_winter_column = (
            f"station_winter_usable_{network_id}"
        )

        observation_column = (
            f"observation_usable_{network_id}"
        )

        analysis[
            station_winter_column
        ] = (
            analysis[
                member_column
            ].astype(bool)
            & analysis[
                "winter_complete"
            ]
        )

        analysis[
            observation_column
        ] = (
            analysis[
                station_winter_column
            ]
            & analysis[
                output_temperature_column
            ].notna()
        )

    analysis = analysis.sort_values(
        [
            "station_uid",
            "date",
        ]
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Build network-size and winter-availability reports
    # -----------------------------------------------------

    completeness_with_membership = (
        completeness.merge(
            stations[
                [
                    "station_uid",
                    *member_columns,
                ]
            ],
            on="station_uid",
            how="left",
            validate="many_to_one",
        )
    )

    network_size_rows: list[
        dict[str, Any]
    ] = []

    winter_availability_rows: list[
        dict[str, Any]
    ] = []

    for (
        network_id,
        definition,
    ) in network_definitions.items():
        member_column = (
            f"member_{network_id}"
        )

        station_members = stations.loc[
            stations[
                member_column
            ]
        ].copy()

        network_size = len(
            station_members
        )

        minimum_required_stations = max(
            minimum_station_count,
            math.ceil(
                network_size
                * minimum_station_fraction
            ),
        )

        network_completeness = (
            completeness_with_membership.loc[
                completeness_with_membership[
                    member_column
                ]
            ]
            .copy()
        )

        eligible_winter_count = 0

        for winter_start_year in range(
            first_winter,
            last_winter + 1,
        ):
            winter_data = (
                network_completeness.loc[
                    network_completeness[
                        "winter_start_year"
                    ].eq(
                        winter_start_year
                    )
                ]
            )

            complete_station_count = int(
                winter_data[
                    "winter_complete"
                ].sum()
            )

            complete_fraction = (
                complete_station_count
                / network_size
                if network_size > 0
                else np.nan
            )

            network_eligible = (
                network_size > 0
                and complete_station_count
                >= minimum_required_stations
            )

            if network_eligible:
                eligible_winter_count += 1

            usable_numeric_station_days = int(
                winter_data.loc[
                    winter_data[
                        "winter_complete"
                    ],
                    "numeric_tmin_days",
                ].sum()
            )

            total_numeric_station_days = int(
                winter_data[
                    "numeric_tmin_days"
                ].sum()
            )

            winter_availability_rows.append(
                {
                    "network_id":
                        network_id,
                    "network_description":
                        definition[
                            "description"
                        ],
                    "winter_start_year":
                        winter_start_year,
                    "winter_label":
                        winter_label(
                            winter_start_year
                        ),
                    "network_station_count":
                        network_size,
                    "complete_station_count":
                        complete_station_count,
                    "complete_station_fraction":
                        complete_fraction,
                    "complete_station_percent":
                        (
                            100.0
                            * complete_fraction
                            if math.isfinite(
                                complete_fraction
                            )
                            else np.nan
                        ),
                    "minimum_required_stations":
                        minimum_required_stations,
                    "winter_network_eligible":
                        network_eligible,
                    "total_numeric_station_days":
                        total_numeric_station_days,
                    "usable_numeric_station_days":
                        usable_numeric_station_days,
                }
            )

        network_size_rows.append(
            {
                "network_id":
                    network_id,
                "network_description":
                    definition[
                        "description"
                    ],
                "station_count":
                    network_size,
                "excluded_from_primary_count":
                    len(fixed)
                    - network_size,
                "minimum_required_stations_per_winter":
                    minimum_required_stations,
                "eligible_winter_count":
                    eligible_winter_count,
                "ineligible_winter_count":
                    study_winter_count
                    - eligible_winter_count,
                "high_priority_station_count":
                    int(
                        station_members[
                            "combined_homogeneity_status"
                        ].eq(
                            "high_priority_homogeneity_review"
                        ).sum()
                    ),
                "moderate_priority_station_count":
                    int(
                        station_members[
                            "combined_homogeneity_status"
                        ].eq(
                            "moderate_priority_homogeneity_review"
                        ).sum()
                    ),
                "temperature_treatment":
                    "retain_unadjusted",
            }
        )

    network_size_summary = pd.DataFrame(
        network_size_rows
    )

    winter_availability = pd.DataFrame(
        winter_availability_rows
    ).sort_values(
        [
            "network_id",
            "winter_start_year",
        ]
    ).reset_index(drop=True)

    sensitivity_exclusions = (
        membership.loc[
            ~membership["included"]
        ]
        .sort_values(
            [
                "network_id",
                "station_uid",
            ]
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Validate the frozen analytical dataset
    # -----------------------------------------------------

    failures: list[str] = []

    if len(analysis) != expected_analysis_rows:
        failures.append(
            "Unexpected DJF analysis row count."
        )

    if analysis.duplicated(
        subset=[
            "station_uid",
            "date",
        ]
    ).any():
        failures.append(
            "Duplicate station-date keys exist."
        )

    if (
        analysis[
            "station_uid"
        ].nunique()
        != len(fixed)
    ):
        failures.append(
            "The DJF analysis station count differs "
            "from the Step 6A fixed network."
        )

    if (
        analysis[
            "winter_start_year"
        ].nunique()
        != study_winter_count
    ):
        failures.append(
            "The DJF analysis dataset does not "
            "contain all 40 winters."
        )

    if analysis[
        "analysis_temperature_adjusted"
    ].any():
        failures.append(
            "An analysis temperature was marked adjusted."
        )

    if analysis[
        "analysis_interpolation_applied"
    ].any():
        failures.append(
            "Interpolation was incorrectly applied."
        )

    represented_rows = analysis.loc[
        analysis[
            "source_station_date_row_present"
        ]
    ]

    if not values_equal_with_nan(
        represented_rows[
            input_temperature_column
        ],
        represented_rows[
            output_temperature_column
        ],
    ):
        failures.append(
            "The unadjusted analysis temperature "
            "differs from cleaned stage-2 Tmin."
        )

    missing_grid_rows = analysis.loc[
        ~analysis[
            "source_station_date_row_present"
        ]
    ]

    if missing_grid_rows[
        output_temperature_column
    ].notna().any():
        failures.append(
            "An absent station-date row received "
            "a numeric temperature."
        )

    if not stations[
        primary_member_column
    ].all():
        failures.append(
            "The primary network excludes a "
            "Step 6A fixed station."
        )

    for member_column in member_columns:
        if (
            stations[member_column]
            & ~stations[
                primary_member_column
            ]
        ).any():
            failures.append(
                f"{member_column} is not a subset "
                "of the primary network."
            )

    primary_size = int(
        stations[
            primary_member_column
        ].sum()
    )

    if primary_size < minimum_station_count:
        failures.append(
            "Primary network contains fewer than "
            "the required minimum station count."
        )

    if failures:
        raise SystemExit(
            "\nSTEP 6C FAILED:\n- "
            + "\n- ".join(failures)
        )

    # -----------------------------------------------------
    # Save outputs
    # -----------------------------------------------------

    stations = stations.sort_values(
        "station_uid"
    ).reset_index(drop=True)

    stations.to_csv(
        STATION_NETWORK_OUTPUT,
        index=False,
    )

    membership = membership.sort_values(
        [
            "network_id",
            "station_uid",
        ]
    ).reset_index(drop=True)

    membership.to_csv(
        MEMBERSHIP_OUTPUT,
        index=False,
    )

    analysis.to_parquet(
        ANALYSIS_OUTPUT,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    network_size_summary.to_csv(
        REPORT_DIRECTORY
        / "step6c_network_size_summary.csv",
        index=False,
    )

    winter_availability.to_csv(
        REPORT_DIRECTORY
        / "step6c_winter_availability_by_network.csv",
        index=False,
    )

    sensitivity_exclusions.to_csv(
        REPORT_DIRECTORY
        / "step6c_sensitivity_excluded_stations.csv",
        index=False,
    )

    station_status_counts = (
        stations[
            "combined_homogeneity_status"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    primary_availability = (
        winter_availability.loc[
            winter_availability[
                "network_id"
            ].eq(
                primary_network_id
            )
        ]
    )

    summary = {
        "created_utc":
            freeze_timestamp,
        "input_daily_file":
            str(DAILY_INPUT_FILE),
        "fixed_network_file":
            str(FIXED_NETWORK_FILE),
        "homogeneity_file":
            str(HOMOGENEITY_FILE),
        "analysis_output":
            str(ANALYSIS_OUTPUT),
        "first_winter":
            winter_label(
                first_winter
            ),
        "last_winter":
            winter_label(
                last_winter
            ),
        "study_winter_count":
            study_winter_count,
        "fixed_station_count":
            len(fixed),
        "djf_calendar_day_count":
            len(calendar),
        "analysis_row_count":
            len(analysis),
        "primary_network_id":
            primary_network_id,
        "primary_station_count":
            primary_size,
        "primary_eligible_winter_count":
            int(
                primary_availability[
                    "winter_network_eligible"
                ].sum()
            ),
        "homogeneity_status_counts":
            {
                str(key): int(value)
                for key, value
                in station_status_counts.items()
            },
        "network_sizes":
            {
                row.network_id:
                    int(row.station_count)
                for row in (
                    network_size_summary.itertuples(
                        index=False
                    )
                )
            },
        "temperature_column":
            output_temperature_column,
        "temperature_adjustment_applied":
            False,
        "interpolation_applied":
            False,
    }

    (
        REPORT_DIRECTORY
        / "step6c_network_freeze_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output_schema = pq.ParquetFile(
        ANALYSIS_OUTPUT
    ).schema_arrow

    (
        ADMIN_DIRECTORY
        / "step6c_parquet_schema.json"
    ).write_text(
        json.dumps(
            {
                field.name: str(field.type)
                for field in output_schema
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "STEP 6C: ANALYTICAL NETWORK FREEZE",
        "=" * 42,
        (
            "Study period: "
            f"{winter_label(first_winter)} to "
            f"{winter_label(last_winter)}"
        ),
        (
            "Study winters: "
            f"{study_winter_count}"
        ),
        (
            "Step 6A fixed stations: "
            f"{len(fixed)}"
        ),
        (
            "Primary stations retained unadjusted: "
            f"{primary_size}"
        ),
        (
            "DJF calendar dates: "
            f"{len(calendar):,}"
        ),
        (
            "Final DJF station-date rows: "
            f"{len(analysis):,}"
        ),
        (
            "Primary network-eligible winters: "
            f"{summary['primary_eligible_winter_count']}"
        ),
        "",
        "Network sizes:",
        *[
            (
                f"- {row.network_id}: "
                f"{row.station_count} stations; "
                f"{row.eligible_winter_count} "
                "eligible winters"
            )
            for row in (
                network_size_summary.itertuples(
                    index=False
                )
            )
        ],
        "",
        "Homogeneity adjustment applied: No",
        "Interpolation applied: No",
        "",
        f"Station network: {STATION_NETWORK_OUTPUT}",
        f"Membership table: {MEMBERSHIP_OUTPUT}",
        f"Analysis dataset: {ANALYSIS_OUTPUT}",
    ]

    (
        REPORT_DIRECTORY
        / "step6c_network_freeze_report.txt"
    ).write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(report_lines))
    print("\nSTEP 6C PASSED.")


if __name__ == "__main__":
    main()
