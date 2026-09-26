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

FULL_DAILY_FILE = (
    PROJECT_ROOT
    / "04_clean_data"
    / "temperature_daily_cleaned_stage2.parquet"
)

DJF_ANALYSIS_FILE = (
    PROJECT_ROOT
    / "04_clean_data"
    / "temperature_daily_djf_analysis_unadjusted.parquet"
)

STATION_NETWORK_FILE = (
    PROJECT_ROOT
    / "02_metadata"
    / "step6c_analytical_station_network.csv"
)

COMPLETENESS_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "step6a_djf_station_winter_completeness.parquet"
)

POLICY_FILE = (
    PROJECT_ROOT
    / "00_admin"
    / "step7a_station_threshold_policy.yaml"
)

FULL_THRESHOLD_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "step7a_station_calendar_thresholds.parquet"
)

LEAVE_ONE_OUT_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "step7a_leave_one_year_out_p10_thresholds.parquet"
)

ANALYSIS_OUTPUT_FILE = (
    PROJECT_ROOT
    / "04_clean_data"
    / "temperature_daily_djf_with_station_thresholds.parquet"
)

REPORT_DIRECTORY = PROJECT_ROOT / "05_qc_reports"
ADMIN_DIRECTORY = PROJECT_ROOT / "00_admin"

PRIMARY_MEMBER_COLUMN = (
    "member_primary_all_fixed_unadjusted"
)

PRIMARY_USABLE_COLUMN = (
    "station_winter_usable_primary_all_fixed_unadjusted"
)


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

    unknown = parsed.isna() & normalized.ne("")

    if unknown.any():
        raise ValueError(
            "Unrecognized Boolean values: "
            f"{sorted(normalized.loc[unknown].unique())}"
        )

    return parsed.fillna(False).astype(bool)


def calendar_day_index(
    month: pd.Series,
    day: pd.Series,
) -> pd.Series:
    """
    Map month/day to the leap-year calendar of 2000.

    This creates a stable 1-366 calendar index.
    """
    dates = pd.to_datetime(
        {
            "year": np.full(
                len(month),
                2000,
                dtype=np.int16,
            ),
            "month": month.astype(int),
            "day": day.astype(int),
        }
    )

    return dates.dt.dayofyear.astype("int16")


def build_djf_threshold_calendar() -> pd.DataFrame:
    """
    Create a 91-day leap-winter calendar from
    1 December through 29 February.
    """
    dates = pd.date_range(
        start="1999-12-01",
        end="2000-02-29",
        freq="D",
    )

    calendar = pd.DataFrame(
        {
            "template_date": dates,
            "month": dates.month.astype(
                "int8"
            ),
            "day": dates.day.astype(
                "int8"
            ),
            "threshold_season_day":
                np.arange(
                    1,
                    len(dates) + 1,
                    dtype=np.int16,
                ),
        }
    )

    calendar["clim_day"] = (
        calendar_day_index(
            calendar["month"],
            calendar["day"],
        )
    )

    return calendar


def circular_day_distance(
    day_values: np.ndarray,
    target_day: int,
) -> np.ndarray:
    direct = np.abs(
        day_values.astype(int)
        - int(target_day)
    )

    return np.minimum(
        direct,
        366 - direct,
    )


def calculate_quantile(
    values: np.ndarray,
    probability: float,
    method: str,
) -> float:
    return float(
        np.quantile(
            values,
            probability,
            method=method,
        )
    )


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


def main() -> None:
    required_files = [
        FULL_DAILY_FILE,
        DJF_ANALYSIS_FILE,
        STATION_NETWORK_FILE,
        COMPLETENESS_FILE,
        POLICY_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    FULL_THRESHOLD_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ANALYSIS_OUTPUT_FILE.parent.mkdir(
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

    reference_rules = policy[
        "reference_period"
    ]

    temperature_rules = policy[
        "temperature"
    ]

    percentile_rules = policy[
        "calendar_percentiles"
    ]

    baseline_rules = policy[
        "baseline_selection"
    ]

    reference_start = pd.Timestamp(
        reference_rules["start_date"]
    )

    reference_end = pd.Timestamp(
        reference_rules["end_date"]
    )

    expected_reference_years = int(
        reference_rules[
            "expected_calendar_years"
        ]
    )

    source_temperature_column = str(
        temperature_rules[
            "source_column"
        ]
    )

    analysis_temperature_column = str(
        temperature_rules[
            "analysis_column"
        ]
    )

    if bool(
        temperature_rules[
            "interpolation_allowed"
        ]
    ):
        raise ValueError(
            "Step 7A must not permit interpolation."
        )

    if bool(
        temperature_rules[
            "adjustment_allowed"
        ]
    ):
        raise ValueError(
            "Step 7A must not permit adjustment."
        )

    primary_percentile = float(
        percentile_rules[
            "primary_percentile"
        ]
    )

    additional_percentiles = [
        float(value)
        for value in percentile_rules[
            "additional_percentiles"
        ]
    ]

    if primary_percentile != 0.10:
        raise ValueError(
            "Step 7A primary percentile must be 0.10."
        )

    if sorted(
        additional_percentiles
    ) != [0.05, 0.20]:
        raise ValueError(
            "Expected additional percentiles 0.05 and 0.20."
        )

    window_half_width = int(
        percentile_rules[
            "window_half_width_days"
        ]
    )

    quantile_method = str(
        percentile_rules[
            "quantile_method"
        ]
    )

    minimum_full_observations = int(
        baseline_rules[
            "minimum_full_sample_observations"
        ]
    )

    minimum_full_years = int(
        baseline_rules[
            "minimum_full_sample_years"
        ]
    )

    minimum_loo_observations = int(
        baseline_rules[
            "minimum_leave_one_out_observations"
        ]
    )

    minimum_loo_years = int(
        baseline_rules[
            "minimum_leave_one_out_years"
        ]
    )

    # -----------------------------------------------------
    # Primary station network
    # -----------------------------------------------------

    stations = pd.read_csv(
        STATION_NETWORK_FILE
    )

    stations["station_uid"] = (
        stations["station_uid"]
        .astype(str)
    )

    if PRIMARY_MEMBER_COLUMN not in stations.columns:
        raise ValueError(
            "Primary network membership column missing."
        )

    stations[
        PRIMARY_MEMBER_COLUMN
    ] = parse_boolean(
        stations[
            PRIMARY_MEMBER_COLUMN
        ]
    )

    primary_stations = stations.loc[
        stations[
            PRIMARY_MEMBER_COLUMN
        ]
    ].copy()

    if primary_stations.empty:
        raise ValueError(
            "Primary analytical network is empty."
        )

    if primary_stations[
        "station_uid"
    ].duplicated().any():
        raise ValueError(
            "Duplicate primary-network stations exist."
        )

    station_ids = set(
        primary_stations[
            "station_uid"
        ]
    )

    station_count = len(
        primary_stations
    )

    # -----------------------------------------------------
    # Station-winter completeness
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
        ].isin(station_ids)
    ].copy()

    completeness[
        "winter_complete"
    ] = parse_boolean(
        completeness[
            "winter_complete"
        ]
    )

    completeness_lookup = completeness[
        [
            "station_uid",
            "winter_start_year",
            "winter_complete",
        ]
    ].copy()

    if completeness_lookup.duplicated(
        subset=[
            "station_uid",
            "winter_start_year",
        ]
    ).any():
        raise ValueError(
            "Duplicate station-winter completeness rows exist."
        )

    # -----------------------------------------------------
    # Build the 1991-2020 baseline pool
    # -----------------------------------------------------

    full_daily = pd.read_parquet(
        FULL_DAILY_FILE
    )

    full_daily[
        "station_uid"
    ] = (
        full_daily[
            "station_uid"
        ].astype(str)
    )

    full_daily["date"] = pd.to_datetime(
        full_daily["date"]
    )

    if source_temperature_column not in full_daily.columns:
        raise ValueError(
            "Required source temperature column missing: "
            f"{source_temperature_column}"
        )

    if full_daily.duplicated(
        subset=[
            "station_uid",
            "date",
        ]
    ).any():
        raise ValueError(
            "The full daily dataset contains duplicate keys."
        )

    baseline = full_daily.loc[
        full_daily[
            "station_uid"
        ].isin(station_ids)
        & full_daily["date"].between(
            reference_start,
            reference_end,
        )
        & full_daily["date"].dt.month.isin(
            [11, 12, 1, 2, 3]
        ),
        [
            "station_uid",
            "date",
            source_temperature_column,
        ],
    ].copy()

    baseline[
        source_temperature_column
    ] = pd.to_numeric(
        baseline[
            source_temperature_column
        ],
        errors="coerce",
    )

    baseline["year"] = (
        baseline["date"]
        .dt.year
        .astype("int16")
    )

    baseline["month"] = (
        baseline["date"]
        .dt.month
        .astype("int8")
    )

    baseline["day"] = (
        baseline["date"]
        .dt.day
        .astype("int8")
    )

    baseline[
        "baseline_calendar_year"
    ] = baseline["year"]

    # November and December belong to the upcoming/current
    # DJF winter. January-March belong to the winter that
    # began in the previous calendar year.
    baseline[
        "threshold_winter_start_year"
    ] = np.where(
        baseline["month"].isin(
            [11, 12]
        ),
        baseline["year"],
        baseline["year"] - 1,
    ).astype("int16")

    baseline["clim_day"] = (
        calendar_day_index(
            baseline["month"],
            baseline["day"],
        )
    )

    baseline = baseline.merge(
        completeness_lookup.rename(
            columns={
                "winter_start_year":
                    "threshold_winter_start_year",
            }
        ),
        on=[
            "station_uid",
            "threshold_winter_start_year",
        ],
        how="left",
        validate="many_to_one",
    )

    if baseline[
        "winter_complete"
    ].isna().any():
        missing_examples = (
            baseline.loc[
                baseline[
                    "winter_complete"
                ].isna(),
                [
                    "station_uid",
                    "date",
                    "threshold_winter_start_year",
                ],
            ]
            .head(10)
            .to_dict("records")
        )

        raise ValueError(
            "Some baseline dates lack winter-completeness "
            f"information: {missing_examples}"
        )

    baseline[
        "winter_complete"
    ] = parse_boolean(
        baseline[
            "winter_complete"
        ]
    )

    baseline_pool = baseline.loc[
        baseline[
            "winter_complete"
        ]
        & baseline[
            source_temperature_column
        ].notna()
    ].copy()

    if baseline_pool.empty:
        raise ValueError(
            "The Step 7A baseline pool is empty."
        )

    # -----------------------------------------------------
    # Calculate full-reference thresholds
    # -----------------------------------------------------

    target_calendar = (
        build_djf_threshold_calendar()
    )

    if len(target_calendar) != 91:
        raise ValueError(
            "Expected 91 DJF threshold calendar days."
        )

    full_threshold_rows: list[
        dict[str, Any]
    ] = []

    leave_one_out_rows: list[
        dict[str, Any]
    ] = []

    reference_years = list(
        range(
            reference_start.year,
            reference_end.year + 1,
        )
    )

    if (
        len(reference_years)
        != expected_reference_years
    ):
        raise ValueError(
            "Unexpected reference-period year count."
        )

    station_lookup = (
        primary_stations
        .set_index("station_uid")
    )

    for station_uid in sorted(
        station_ids
    ):
        station_pool = baseline_pool.loc[
            baseline_pool[
                "station_uid"
            ].eq(station_uid)
        ].copy()

        day_values = station_pool[
            "clim_day"
        ].to_numpy(dtype=int)

        temperature_values = (
            station_pool[
                source_temperature_column
            ]
            .to_numpy(dtype=float)
        )

        year_values = station_pool[
            "baseline_calendar_year"
        ].to_numpy(dtype=int)

        station_metadata = (
            station_lookup.loc[
                station_uid
            ]
        )

        for target in (
            target_calendar.itertuples(
                index=False
            )
        ):
            distances = (
                circular_day_distance(
                    day_values,
                    int(target.clim_day),
                )
            )

            window_mask = (
                distances <= window_half_width
            )

            sample_values = (
                temperature_values[
                    window_mask
                ]
            )

            sample_years = (
                year_values[
                    window_mask
                ]
            )

            finite_mask = np.isfinite(
                sample_values
            )

            sample_values = (
                sample_values[
                    finite_mask
                ]
            )

            sample_years = (
                sample_years[
                    finite_mask
                ]
            )

            full_observation_count = int(
                len(sample_values)
            )

            full_year_count = int(
                len(
                    np.unique(
                        sample_years
                    )
                )
            )

            full_available = (
                full_observation_count
                >= minimum_full_observations
                and full_year_count
                >= minimum_full_years
            )

            if full_available:
                p05 = calculate_quantile(
                    sample_values,
                    0.05,
                    quantile_method,
                )

                p10 = calculate_quantile(
                    sample_values,
                    0.10,
                    quantile_method,
                )

                p20 = calculate_quantile(
                    sample_values,
                    0.20,
                    quantile_method,
                )

                sample_minimum = float(
                    np.min(sample_values)
                )

                sample_maximum = float(
                    np.max(sample_values)
                )
            else:
                p05 = np.nan
                p10 = np.nan
                p20 = np.nan
                sample_minimum = np.nan
                sample_maximum = np.nan

            full_threshold_rows.append(
                {
                    "station_uid":
                        station_uid,
                    "station_id":
                        station_metadata.get(
                            "station_id",
                            "",
                        ),
                    "station_name_display":
                        station_metadata.get(
                            "station_name_display",
                            "",
                        ),
                    "month":
                        int(target.month),
                    "day":
                        int(target.day),
                    "clim_day":
                        int(target.clim_day),
                    "threshold_season_day":
                        int(
                            target.threshold_season_day
                        ),
                    "window_half_width_days":
                        window_half_width,
                    "window_total_days":
                        (
                            2
                            * window_half_width
                            + 1
                        ),
                    "baseline_observation_count":
                        full_observation_count,
                    "baseline_calendar_year_count":
                        full_year_count,
                    "baseline_sample_minimum":
                        sample_minimum,
                    "baseline_sample_maximum":
                        sample_maximum,
                    "station_p05_threshold_full":
                        p05,
                    "station_p10_threshold_full":
                        p10,
                    "station_p20_threshold_full":
                        p20,
                    "full_threshold_available":
                        full_available,
                    "reference_start":
                        reference_start,
                    "reference_end":
                        reference_end,
                    "quantile_method":
                        quantile_method,
                }
            )

            for excluded_year in (
                reference_years
            ):
                leave_one_out_mask = (
                    sample_years
                    != excluded_year
                )

                loo_values = (
                    sample_values[
                        leave_one_out_mask
                    ]
                )

                loo_years = (
                    sample_years[
                        leave_one_out_mask
                    ]
                )

                loo_observation_count = int(
                    len(loo_values)
                )

                loo_year_count = int(
                    len(
                        np.unique(
                            loo_years
                        )
                    )
                )

                loo_available = (
                    loo_observation_count
                    >= minimum_loo_observations
                    and loo_year_count
                    >= minimum_loo_years
                )

                if loo_available:
                    loo_p10 = (
                        calculate_quantile(
                            loo_values,
                            primary_percentile,
                            quantile_method,
                        )
                    )
                else:
                    loo_p10 = np.nan

                leave_one_out_rows.append(
                    {
                        "station_uid":
                            station_uid,
                        "month":
                            int(target.month),
                        "day":
                            int(target.day),
                        "clim_day":
                            int(target.clim_day),
                        "threshold_season_day":
                            int(
                                target.threshold_season_day
                            ),
                        "excluded_calendar_year":
                            int(excluded_year),
                        "loo_observation_count":
                            loo_observation_count,
                        "loo_calendar_year_count":
                            loo_year_count,
                        "station_p10_threshold_loo":
                            loo_p10,
                        "loo_threshold_available":
                            loo_available,
                    }
                )

    thresholds = pd.DataFrame(
        full_threshold_rows
    )

    leave_one_out = pd.DataFrame(
        leave_one_out_rows
    )

    expected_full_rows = (
        station_count
        * len(target_calendar)
    )

    expected_loo_rows = (
        station_count
        * len(target_calendar)
        * expected_reference_years
    )

    if len(thresholds) != expected_full_rows:
        raise ValueError(
            "Unexpected full-threshold row count: "
            f"{len(thresholds):,}; expected "
            f"{expected_full_rows:,}."
        )

    if len(leave_one_out) != expected_loo_rows:
        raise ValueError(
            "Unexpected leave-one-out row count: "
            f"{len(leave_one_out):,}; expected "
            f"{expected_loo_rows:,}."
        )

    if thresholds.duplicated(
        subset=[
            "station_uid",
            "clim_day",
        ]
    ).any():
        raise ValueError(
            "Duplicate full-threshold keys exist."
        )

    if leave_one_out.duplicated(
        subset=[
            "station_uid",
            "clim_day",
            "excluded_calendar_year",
        ]
    ).any():
        raise ValueError(
            "Duplicate leave-one-out threshold keys exist."
        )

    unavailable_full = thresholds.loc[
        ~thresholds[
            "full_threshold_available"
        ]
    ].copy()

    unavailable_loo = leave_one_out.loc[
        ~leave_one_out[
            "loo_threshold_available"
        ]
    ].copy()

    if not unavailable_full.empty:
        raise ValueError(
            "Some full thresholds are unavailable: "
            f"{len(unavailable_full)} rows."
        )

    if not unavailable_loo.empty:
        raise ValueError(
            "Some leave-one-out thresholds are unavailable: "
            f"{len(unavailable_loo)} rows."
        )

    monotonic_failure = thresholds.loc[
        ~(
            thresholds[
                "station_p05_threshold_full"
            ].le(
                thresholds[
                    "station_p10_threshold_full"
                ]
            )
            & thresholds[
                "station_p10_threshold_full"
            ].le(
                thresholds[
                    "station_p20_threshold_full"
                ]
            )
        )
    ]

    if not monotonic_failure.empty:
        raise ValueError(
            "Percentile thresholds are not monotonic."
        )

    thresholds = thresholds.sort_values(
        [
            "station_uid",
            "threshold_season_day",
        ]
    ).reset_index(drop=True)

    leave_one_out = (
        leave_one_out.sort_values(
            [
                "station_uid",
                "threshold_season_day",
                "excluded_calendar_year",
            ]
        )
        .reset_index(drop=True)
    )

    thresholds.to_parquet(
        FULL_THRESHOLD_FILE,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    leave_one_out.to_parquet(
        LEAVE_ONE_OUT_FILE,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    # -----------------------------------------------------
    # Attach thresholds to the complete DJF analysis table
    # -----------------------------------------------------

    analysis = pd.read_parquet(
        DJF_ANALYSIS_FILE
    )

    analysis[
        "station_uid"
    ] = (
        analysis[
            "station_uid"
        ].astype(str)
    )

    analysis["date"] = pd.to_datetime(
        analysis["date"]
    )

    if analysis_temperature_column not in analysis.columns:
        raise ValueError(
            "Analysis Tmin column missing: "
            f"{analysis_temperature_column}"
        )

    if PRIMARY_USABLE_COLUMN not in analysis.columns:
        raise ValueError(
            "Primary station-winter usability "
            "column is missing."
        )

    analysis[
        PRIMARY_USABLE_COLUMN
    ] = parse_boolean(
        analysis[
            PRIMARY_USABLE_COLUMN
        ]
    )

    if analysis.duplicated(
        subset=[
            "station_uid",
            "date",
        ]
    ).any():
        raise ValueError(
            "The Step 6C analysis dataset contains "
            "duplicate station-date keys."
        )

    original_row_count = len(
        analysis
    )

    original_analysis_tmin = (
        analysis[
            analysis_temperature_column
        ].copy()
    )

    analysis[
        "analysis_calendar_year"
    ] = (
        analysis["date"]
        .dt.year
        .astype("int16")
    )

    analysis["clim_day"] = (
        calendar_day_index(
            analysis["month"],
            analysis["day"],
        )
    )

    threshold_merge_columns = [
        "station_uid",
        "month",
        "day",
        "clim_day",
        "threshold_season_day",
        "baseline_observation_count",
        "baseline_calendar_year_count",
        "station_p05_threshold_full",
        "station_p10_threshold_full",
        "station_p20_threshold_full",
        "full_threshold_available",
    ]

    analysis = analysis.merge(
        thresholds[
            threshold_merge_columns
        ],
        on=[
            "station_uid",
            "month",
            "day",
            "clim_day",
        ],
        how="left",
        validate="many_to_one",
    )

    analysis = analysis.merge(
        leave_one_out[
            [
                "station_uid",
                "clim_day",
                "excluded_calendar_year",
                "loo_observation_count",
                "loo_calendar_year_count",
                "station_p10_threshold_loo",
                "loo_threshold_available",
            ]
        ],
        left_on=[
            "station_uid",
            "clim_day",
            "analysis_calendar_year",
        ],
        right_on=[
            "station_uid",
            "clim_day",
            "excluded_calendar_year",
        ],
        how="left",
        validate="many_to_one",
    )

    analysis[
        "date_within_reference_period"
    ] = analysis["date"].between(
        reference_start,
        reference_end,
    )

    missing_baseline_loo = analysis.loc[
        analysis[
            "date_within_reference_period"
        ]
        & analysis[
            "station_p10_threshold_loo"
        ].isna()
    ]

    if not missing_baseline_loo.empty:
        raise ValueError(
            "Some baseline dates lack leave-one-year-out "
            f"thresholds: {len(missing_baseline_loo)} rows."
        )

    analysis[
        "station_p10_threshold_analysis"
    ] = np.where(
        analysis[
            "date_within_reference_period"
        ],
        analysis[
            "station_p10_threshold_loo"
        ],
        analysis[
            "station_p10_threshold_full"
        ],
    ).astype("float32")

    analysis[
        "threshold_assignment_method"
    ] = np.where(
        analysis[
            "date_within_reference_period"
        ],
        "leave_one_calendar_year_out",
        "full_1991_2020_reference",
    )

    analysis[
        "station_percentile_threshold_available"
    ] = (
        analysis[
            "station_p10_threshold_analysis"
        ].notna()
    )

    analysis[
        "eligible_for_station_percentile_test_primary"
    ] = (
        analysis[
            PRIMARY_USABLE_COLUMN
        ]
        & analysis[
            analysis_temperature_column
        ].notna()
        & analysis[
            "station_percentile_threshold_available"
        ]
    )

    analysis[
        "station_percentile_cold_day_p10"
    ] = (
        analysis[
            "eligible_for_station_percentile_test_primary"
        ]
        & analysis[
            analysis_temperature_column
        ].lt(
            analysis[
                "station_p10_threshold_analysis"
            ]
        )
    )

    analysis[
        "tmin_minus_station_p10"
    ] = (
        analysis[
            analysis_temperature_column
        ]
        - analysis[
            "station_p10_threshold_analysis"
        ]
    ).astype("float32")

    analysis[
        "station_p10_cold_deficit"
    ] = (
        analysis[
            "station_p10_threshold_analysis"
        ]
        - analysis[
            analysis_temperature_column
        ]
    ).where(
        analysis[
            "station_percentile_cold_day_p10"
        ]
    ).astype("float32")

    analysis[
        "station_p10_threshold_reference_start"
    ] = reference_start

    analysis[
        "station_p10_threshold_reference_end"
    ] = reference_end

    analysis[
        "station_p10_threshold_percentile"
    ] = np.float32(
        primary_percentile
    )

    analysis[
        "station_p10_threshold_window_days"
    ] = np.int8(
        2
        * window_half_width
        + 1
    )

    analysis[
        "station_threshold_interpolation_applied"
    ] = False

    analysis[
        "station_threshold_temperature_adjusted"
    ] = False

    analysis = analysis.drop(
        columns=[
            "excluded_calendar_year",
        ],
        errors="ignore",
    )

    analysis = analysis.sort_values(
        [
            "station_uid",
            "date",
        ]
    ).reset_index(drop=True)

    if len(analysis) != original_row_count:
        raise ValueError(
            "Threshold assignment changed the analysis row count."
        )

    if analysis.duplicated(
        subset=[
            "station_uid",
            "date",
        ]
    ).any():
        raise ValueError(
            "Threshold assignment introduced duplicate keys."
        )

    if not values_equal_with_nan(
        original_analysis_tmin,
        analysis[
            analysis_temperature_column
        ],
    ):
        raise ValueError(
            "Step 7A changed analysis Tmin values."
        )

    if analysis[
        "station_p10_threshold_analysis"
    ].isna().any():
        raise ValueError(
            "Some DJF rows lack a station p10 threshold."
        )

    if analysis[
        "station_threshold_interpolation_applied"
    ].any():
        raise ValueError(
            "Interpolation was incorrectly applied."
        )

    if analysis[
        "station_threshold_temperature_adjusted"
    ].any():
        raise ValueError(
            "A temperature adjustment was incorrectly applied."
        )

    analysis.to_parquet(
        ANALYSIS_OUTPUT_FILE,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    # -----------------------------------------------------
    # Diagnostic reports
    # -----------------------------------------------------

    baseline_boundary_rows: list[
        dict[str, Any]
    ] = []

    for station_uid in sorted(
        station_ids
    ):
        station_baseline = (
            baseline_pool.loc[
                baseline_pool[
                    "station_uid"
                ].eq(station_uid)
            ]
            .sort_values("date")
        )

        metadata = (
            station_lookup.loc[
                station_uid
            ]
        )

        baseline_boundary_rows.append(
            {
                "station_uid":
                    station_uid,
                "station_name_display":
                    metadata.get(
                        "station_name_display",
                        "",
                    ),
                "first_eligible_baseline_date":
                    (
                        station_baseline[
                            "date"
                        ].min()
                    ),
                "last_eligible_baseline_date":
                    (
                        station_baseline[
                            "date"
                        ].max()
                    ),
                "eligible_baseline_observations":
                    len(
                        station_baseline
                    ),
                "eligible_baseline_calendar_years":
                    station_baseline[
                        "baseline_calendar_year"
                    ].nunique(),
                "jan_feb_1991_observations":
                    int(
                        (
                            station_baseline[
                                "date"
                            ].between(
                                "1991-01-01",
                                "1991-02-28",
                            )
                        ).sum()
                    ),
                "december_2020_observations":
                    int(
                        (
                            station_baseline[
                                "date"
                            ].between(
                                "2020-12-01",
                                "2020-12-31",
                            )
                        ).sum()
                    ),
                "november_buffer_observations":
                    int(
                        station_baseline[
                            "month"
                        ].eq(11).sum()
                    ),
                "march_buffer_observations":
                    int(
                        station_baseline[
                            "month"
                        ].eq(3).sum()
                    ),
            }
        )

    boundary_audit = pd.DataFrame(
        baseline_boundary_rows
    )

    loo_comparison = (
        leave_one_out.merge(
            thresholds[
                [
                    "station_uid",
                    "clim_day",
                    "station_p10_threshold_full",
                ]
            ],
            on=[
                "station_uid",
                "clim_day",
            ],
            how="left",
            validate="many_to_one",
        )
    )

    loo_comparison[
        "absolute_loo_full_difference"
    ] = (
        loo_comparison[
            "station_p10_threshold_loo"
        ]
        - loo_comparison[
            "station_p10_threshold_full"
        ]
    ).abs()

    threshold_station_summary = (
        thresholds.groupby(
            [
                "station_uid",
                "station_id",
                "station_name_display",
            ],
            as_index=False,
        )
        .agg(
            threshold_day_count=(
                "clim_day",
                "count",
            ),
            minimum_baseline_observations=(
                "baseline_observation_count",
                "min",
            ),
            maximum_baseline_observations=(
                "baseline_observation_count",
                "max",
            ),
            minimum_baseline_year_count=(
                "baseline_calendar_year_count",
                "min",
            ),
            minimum_p10_threshold=(
                "station_p10_threshold_full",
                "min",
            ),
            maximum_p10_threshold=(
                "station_p10_threshold_full",
                "max",
            ),
            mean_p10_threshold=(
                "station_p10_threshold_full",
                "mean",
            ),
        )
    )

    maximum_loo_difference = (
        loo_comparison.groupby(
            "station_uid",
            as_index=False,
        )
        .agg(
            maximum_absolute_loo_full_difference=(
                "absolute_loo_full_difference",
                "max",
            ),
            mean_absolute_loo_full_difference=(
                "absolute_loo_full_difference",
                "mean",
            ),
        )
    )

    threshold_station_summary = (
        threshold_station_summary.merge(
            maximum_loo_difference,
            on="station_uid",
            how="left",
            validate="one_to_one",
        )
    )

    threshold_calendar_summary = (
        thresholds.groupby(
            [
                "threshold_season_day",
                "month",
                "day",
                "clim_day",
            ],
            as_index=False,
        )
        .agg(
            station_count=(
                "station_uid",
                "nunique",
            ),
            median_station_p05=(
                "station_p05_threshold_full",
                "median",
            ),
            median_station_p10=(
                "station_p10_threshold_full",
                "median",
            ),
            median_station_p20=(
                "station_p20_threshold_full",
                "median",
            ),
            minimum_station_p10=(
                "station_p10_threshold_full",
                "min",
            ),
            maximum_station_p10=(
                "station_p10_threshold_full",
                "max",
            ),
            minimum_sample_observations=(
                "baseline_observation_count",
                "min",
            ),
        )
        .sort_values(
            "threshold_season_day"
        )
    )

    analysis[
        "threshold_period_group"
    ] = np.where(
        analysis[
            "date_within_reference_period"
        ],
        "reference_1991_2020",
        "outside_reference_period",
    )

    cold_day_summary = (
        analysis.groupby(
            [
                "station_uid",
                "station_name_display",
                "threshold_period_group",
            ],
            as_index=False,
        )
        .agg(
            eligible_station_days=(
                "eligible_for_station_percentile_test_primary",
                "sum",
            ),
            station_p10_cold_days=(
                "station_percentile_cold_day_p10",
                "sum",
            ),
            mean_analysis_threshold=(
                "station_p10_threshold_analysis",
                "mean",
            ),
            minimum_analysis_tmin=(
                analysis_temperature_column,
                "min",
            ),
        )
    )

    cold_day_summary[
        "station_p10_cold_day_percent"
    ] = np.where(
        cold_day_summary[
            "eligible_station_days"
        ].gt(0),
        100.0
        * cold_day_summary[
            "station_p10_cold_days"
        ]
        / cold_day_summary[
            "eligible_station_days"
        ],
        np.nan,
    )

    issue_columns = [
        "issue_type",
        "station_uid",
        "month",
        "day",
        "clim_day",
        "excluded_calendar_year",
        "observation_count",
        "calendar_year_count",
    ]

    issue_rows: list[
        dict[str, Any]
    ] = []

    for row in (
        unavailable_full.itertuples(
            index=False
        )
    ):
        issue_rows.append(
            {
                "issue_type":
                    "full_threshold_unavailable",
                "station_uid":
                    row.station_uid,
                "month":
                    row.month,
                "day":
                    row.day,
                "clim_day":
                    row.clim_day,
                "excluded_calendar_year":
                    "",
                "observation_count":
                    row.baseline_observation_count,
                "calendar_year_count":
                    row.baseline_calendar_year_count,
            }
        )

    for row in (
        unavailable_loo.itertuples(
            index=False
        )
    ):
        issue_rows.append(
            {
                "issue_type":
                    "loo_threshold_unavailable",
                "station_uid":
                    row.station_uid,
                "month":
                    row.month,
                "day":
                    row.day,
                "clim_day":
                    row.clim_day,
                "excluded_calendar_year":
                    row.excluded_calendar_year,
                "observation_count":
                    row.loo_observation_count,
                "calendar_year_count":
                    row.loo_calendar_year_count,
            }
        )

    issue_report = pd.DataFrame(
        issue_rows,
        columns=issue_columns,
    )

    boundary_audit.to_csv(
        REPORT_DIRECTORY
        / "step7a_baseline_boundary_audit.csv",
        index=False,
    )

    threshold_station_summary.to_csv(
        REPORT_DIRECTORY
        / "step7a_threshold_station_summary.csv",
        index=False,
    )

    threshold_calendar_summary.to_csv(
        REPORT_DIRECTORY
        / "step7a_threshold_calendar_summary.csv",
        index=False,
    )

    cold_day_summary.to_csv(
        REPORT_DIRECTORY
        / "step7a_station_cold_day_summary.csv",
        index=False,
    )

    issue_report.to_csv(
        REPORT_DIRECTORY
        / "step7a_threshold_sample_issues.csv",
        index=False,
    )

    full_parquet = pq.ParquetFile(
        FULL_THRESHOLD_FILE
    )

    loo_parquet = pq.ParquetFile(
        LEAVE_ONE_OUT_FILE
    )

    analysis_parquet = pq.ParquetFile(
        ANALYSIS_OUTPUT_FILE
    )

    reference_analysis = analysis.loc[
        analysis[
            "date_within_reference_period"
        ]
        & analysis[
            "eligible_for_station_percentile_test_primary"
        ]
    ]

    outside_analysis = analysis.loc[
        ~analysis[
            "date_within_reference_period"
        ]
        & analysis[
            "eligible_for_station_percentile_test_primary"
        ]
    ]

    summary = {
        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "reference_start":
            reference_start.date().isoformat(),
        "reference_end":
            reference_end.date().isoformat(),
        "reference_calendar_years":
            expected_reference_years,
        "primary_station_count":
            station_count,
        "threshold_calendar_days":
            len(target_calendar),
        "full_threshold_rows":
            full_parquet.metadata.num_rows,
        "leave_one_out_threshold_rows":
            loo_parquet.metadata.num_rows,
        "analysis_rows":
            analysis_parquet.metadata.num_rows,
        "window_half_width_days":
            window_half_width,
        "window_total_days":
            (
                2
                * window_half_width
                + 1
            ),
        "primary_percentile":
            primary_percentile,
        "quantile_method":
            quantile_method,
        "full_threshold_unavailable_rows":
            len(unavailable_full),
        "leave_one_out_unavailable_rows":
            len(unavailable_loo),
        "reference_period_eligible_station_days":
            len(reference_analysis),
        "reference_period_p10_cold_days":
            int(
                reference_analysis[
                    "station_percentile_cold_day_p10"
                ].sum()
            ),
        "outside_reference_eligible_station_days":
            len(outside_analysis),
        "outside_reference_p10_cold_days":
            int(
                outside_analysis[
                    "station_percentile_cold_day_p10"
                ].sum()
            ),
        "temperature_adjusted":
            False,
        "interpolation_applied":
            False,
        "full_threshold_file":
            str(FULL_THRESHOLD_FILE),
        "leave_one_out_file":
            str(LEAVE_ONE_OUT_FILE),
        "analysis_output_file":
            str(ANALYSIS_OUTPUT_FILE),
    }

    (
        REPORT_DIRECTORY
        / "step7a_threshold_summary.json"
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
        / "step7a_parquet_schema.json"
    ).write_text(
        json.dumps(
            {
                "full_threshold_schema": {
                    field.name: str(
                        field.type
                    )
                    for field
                    in full_parquet.schema_arrow
                },
                "leave_one_out_schema": {
                    field.name: str(
                        field.type
                    )
                    for field
                    in loo_parquet.schema_arrow
                },
                "analysis_schema": {
                    field.name: str(
                        field.type
                    )
                    for field
                    in analysis_parquet.schema_arrow
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "STEP 7A: STATION-SPECIFIC COLD THRESHOLDS",
        "=" * 48,
        (
            "Reference period: "
            f"{reference_start.date()} to "
            f"{reference_end.date()}"
        ),
        (
            "Primary stations: "
            f"{station_count}"
        ),
        (
            "DJF threshold calendar days: "
            f"{len(target_calendar)}"
        ),
        (
            "Full threshold rows: "
            f"{full_parquet.metadata.num_rows:,}"
        ),
        (
            "Leave-one-year-out rows: "
            f"{loo_parquet.metadata.num_rows:,}"
        ),
        (
            "Analysis station-date rows: "
            f"{analysis_parquet.metadata.num_rows:,}"
        ),
        (
            "Primary percentile: "
            f"{primary_percentile:.2f}"
        ),
        (
            "Calendar window: "
            f"{2 * window_half_width + 1} days"
        ),
        (
            "Quantile method: "
            f"{quantile_method}"
        ),
        (
            "Unavailable full thresholds: "
            f"{len(unavailable_full)}"
        ),
        (
            "Unavailable leave-one-out thresholds: "
            f"{len(unavailable_loo)}"
        ),
        (
            "Reference-period eligible station-days: "
            f"{len(reference_analysis):,}"
        ),
        (
            "Reference-period p10 cold days: "
            f"{reference_analysis['station_percentile_cold_day_p10'].sum():,}"
        ),
        "",
        "Temperature adjusted: No",
        "Interpolation applied: No",
        "",
        f"Full thresholds: {FULL_THRESHOLD_FILE}",
        f"Leave-one-out thresholds: {LEAVE_ONE_OUT_FILE}",
        f"Analysis output: {ANALYSIS_OUTPUT_FILE}",
    ]

    (
        REPORT_DIRECTORY
        / "step7a_threshold_report.txt"
    ).write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(report_lines))
    print("\nSTEP 7A PASSED.")


if __name__ == "__main__":
    main()
