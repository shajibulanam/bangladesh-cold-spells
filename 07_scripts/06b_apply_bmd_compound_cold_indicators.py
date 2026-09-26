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
    / "04_clean_data"
    / "temperature_daily_djf_with_station_thresholds.parquet"
)

POLICY_FILE = (
    PROJECT_ROOT
    / "00_admin"
    / "step7b_compound_cold_policy.yaml"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "04_clean_data"
    / "temperature_daily_djf_with_compound_cold_indicators.parquet"
)

REPORT_DIRECTORY = PROJECT_ROOT / "05_qc_reports"
ADMIN_DIRECTORY = PROJECT_ROOT / "00_admin"

PRIMARY_USABLE_COLUMN = (
    "station_winter_usable_primary_all_fixed_unadjusted"
)

TEMPERATURE_COLUMN = "analysis_tmin_unadjusted"
P10_THRESHOLD_COLUMN = "station_p10_threshold_analysis"


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
        raise ValueError(
            "Unrecognized Boolean values: "
            f"{sorted(normalized.loc[unknown].unique())}"
        )

    return parsed.fillna(False).astype(bool)


def classify_bmd_values(
    values: pd.Series,
    mild_upper: float,
    moderate_upper: float,
    severe_upper: float,
    very_severe_upper: float,
) -> tuple[pd.Series, pd.Series]:
    """
    Classify numeric Tmin using continuous BMD boundaries.

    Rank:
    -1 = missing
     0 = above BMD cold-wave threshold
     1 = mild
     2 = moderate
     3 = severe
     4 = very severe
    """
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    category = np.select(
        [
            numeric.isna(),
            numeric.le(
                very_severe_upper
            ),
            numeric.gt(
                very_severe_upper
            )
            & numeric.le(
                severe_upper
            ),
            numeric.gt(
                severe_upper
            )
            & numeric.le(
                moderate_upper
            ),
            numeric.gt(
                moderate_upper
            )
            & numeric.le(
                mild_upper
            ),
            numeric.gt(
                mild_upper
            ),
        ],
        [
            "missing",
            "very_severe",
            "severe",
            "moderate",
            "mild",
            "above_bmd_cold_wave_threshold",
        ],
        default="unclassified",
    )

    rank = np.select(
        [
            numeric.isna(),
            numeric.le(
                very_severe_upper
            ),
            numeric.gt(
                very_severe_upper
            )
            & numeric.le(
                severe_upper
            ),
            numeric.gt(
                severe_upper
            )
            & numeric.le(
                moderate_upper
            ),
            numeric.gt(
                moderate_upper
            )
            & numeric.le(
                mild_upper
            ),
            numeric.gt(
                mild_upper
            ),
        ],
        [
            -1,
            4,
            3,
            2,
            1,
            0,
        ],
        default=-1,
    )

    return (
        pd.Series(
            category,
            index=values.index,
            dtype="string",
        ),
        pd.Series(
            rank,
            index=values.index,
            dtype="int8",
        ),
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


def main() -> None:
    for required_file in [
        INPUT_FILE,
        POLICY_FILE,
    ]:
        if not required_file.exists():
            raise FileNotFoundError(
                f"Required file not found: "
                f"{required_file}"
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

    temperature_rules = policy[
        "temperature"
    ]

    category_rules = policy[
        "bmd_absolute_categories"
    ]

    if bool(
        temperature_rules[
            "interpolation_allowed"
        ]
    ):
        raise ValueError(
            "Step 7B must not permit interpolation."
        )

    if bool(
        temperature_rules[
            "adjustment_allowed"
        ]
    ):
        raise ValueError(
            "Step 7B must not permit temperature adjustment."
        )

    temperature_column = str(
        temperature_rules["column"]
    )

    mild_upper = float(
        category_rules[
            "mild"
        ]["upper_inclusive_celsius"]
    )

    moderate_upper = float(
        category_rules[
            "moderate"
        ]["upper_inclusive_celsius"]
    )

    severe_upper = float(
        category_rules[
            "severe"
        ]["upper_inclusive_celsius"]
    )

    very_severe_upper = float(
        category_rules[
            "very_severe"
        ]["upper_inclusive_celsius"]
    )

    if not (
        very_severe_upper
        < severe_upper
        < moderate_upper
        < mild_upper
    ):
        raise ValueError(
            "BMD category boundaries are not ordered."
        )

    data = pd.read_parquet(
        INPUT_FILE
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
        "winter_start_year",
        "winter_label",
        temperature_column,
        P10_THRESHOLD_COLUMN,
        "station_percentile_cold_day_p10",
        "eligible_for_station_percentile_test_primary",
        "station_percentile_threshold_available",
        PRIMARY_USABLE_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required Step 7A columns are missing: "
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

    original_rows = len(data)

    original_temperature = data[
        temperature_column
    ].copy()

    original_keys = data[
        [
            "station_uid",
            "date",
        ]
    ].copy()

    data[
        temperature_column
    ] = pd.to_numeric(
        data[
            temperature_column
        ],
        errors="coerce",
    )

    data[
        P10_THRESHOLD_COLUMN
    ] = pd.to_numeric(
        data[
            P10_THRESHOLD_COLUMN
        ],
        errors="coerce",
    )

    for column in [
        "station_percentile_cold_day_p10",
        "eligible_for_station_percentile_test_primary",
        "station_percentile_threshold_available",
        PRIMARY_USABLE_COLUMN,
    ]:
        data[column] = parse_boolean(
            data[column]
        )

    # -----------------------------------------------------
    # Raw BMD absolute classification
    # -----------------------------------------------------

    (
        data["bmd_absolute_category_raw"],
        data["bmd_absolute_severity_rank_raw"],
    ) = classify_bmd_values(
        data[
            temperature_column
        ],
        mild_upper=mild_upper,
        moderate_upper=moderate_upper,
        severe_upper=severe_upper,
        very_severe_upper=(
            very_severe_upper
        ),
    )

    numeric_tmin = data[
        temperature_column
    ].notna()

    station_winter_usable = data[
        PRIMARY_USABLE_COLUMN
    ]

    threshold_available = data[
        "station_percentile_threshold_available"
    ]

    data[
        "eligible_for_bmd_absolute_test_primary"
    ] = (
        station_winter_usable
        & numeric_tmin
    )

    data[
        "eligible_for_compound_cold_test_primary"
    ] = (
        station_winter_usable
        & numeric_tmin
        & threshold_available
    )

    # Analytical BMD category distinguishes unusable and missing rows.
    data[
        "bmd_absolute_category_primary"
    ] = data[
        "bmd_absolute_category_raw"
    ].copy()

    data.loc[
        ~station_winter_usable,
        "bmd_absolute_category_primary",
    ] = "ineligible_station_winter"

    data.loc[
        station_winter_usable
        & ~numeric_tmin,
        "bmd_absolute_category_primary",
    ] = "missing_tmin"

    data[
        "bmd_absolute_severity_rank_primary"
    ] = np.where(
        data[
            "eligible_for_bmd_absolute_test_primary"
        ],
        data[
            "bmd_absolute_severity_rank_raw"
        ],
        -1,
    ).astype("int8")

    # -----------------------------------------------------
    # BMD category flags
    # -----------------------------------------------------

    eligible_bmd = data[
        "eligible_for_bmd_absolute_test_primary"
    ]

    data[
        "station_bmd_mild_day"
    ] = (
        eligible_bmd
        & data[
            "bmd_absolute_category_raw"
        ].eq("mild")
    )

    data[
        "station_bmd_moderate_day"
    ] = (
        eligible_bmd
        & data[
            "bmd_absolute_category_raw"
        ].eq("moderate")
    )

    data[
        "station_bmd_severe_day"
    ] = (
        eligible_bmd
        & data[
            "bmd_absolute_category_raw"
        ].eq("severe")
    )

    data[
        "station_bmd_very_severe_day"
    ] = (
        eligible_bmd
        & data[
            "bmd_absolute_category_raw"
        ].eq("very_severe")
    )

    data[
        "station_bmd_cold_wave_day"
    ] = (
        data[
            "station_bmd_mild_day"
        ]
        | data[
            "station_bmd_moderate_day"
        ]
        | data[
            "station_bmd_severe_day"
        ]
        | data[
            "station_bmd_very_severe_day"
        ]
    )

    # -----------------------------------------------------
    # Relative and compound indicators
    # -----------------------------------------------------

    percentile_cold = data[
        "station_percentile_cold_day_p10"
    ]

    bmd_cold = data[
        "station_bmd_cold_wave_day"
    ]

    invalid_percentile_flags = (
        percentile_cold
        & ~data[
            "eligible_for_compound_cold_test_primary"
        ]
    )

    if invalid_percentile_flags.any():
        raise ValueError(
            "A p10 cold-day flag occurs on an "
            "ineligible or threshold-missing row."
        )

    data[
        "station_compound_cold_day_primary"
    ] = (
        percentile_cold
        & bmd_cold
    )

    data[
        "station_percentile_only_cold_day"
    ] = (
        percentile_cold
        & ~bmd_cold
    )

    data[
        "station_bmd_only_cold_day"
    ] = (
        bmd_cold
        & ~percentile_cold
    )

    data[
        "station_cold_union_day"
    ] = (
        percentile_cold
        | bmd_cold
    )

    data[
        "station_neither_cold_condition"
    ] = (
        data[
            "eligible_for_compound_cold_test_primary"
        ]
        & ~percentile_cold
        & ~bmd_cold
    )

    # The primary event-candidate station-day is the compound condition.
    data[
        "station_cold_candidate_primary"
    ] = data[
        "station_compound_cold_day_primary"
    ]

    data[
        "station_cold_candidate_percentile_sensitivity"
    ] = percentile_cold

    data[
        "station_cold_candidate_bmd_sensitivity"
    ] = bmd_cold

    data[
        "station_cold_candidate_union_sensitivity"
    ] = data[
        "station_cold_union_day"
    ]

    data["station_day_cold_condition"] = np.select(
        [
            ~station_winter_usable,
            station_winter_usable
            & ~numeric_tmin,
            station_winter_usable
            & numeric_tmin
            & ~threshold_available,
            data[
                "station_compound_cold_day_primary"
            ],
            data[
                "station_percentile_only_cold_day"
            ],
            data[
                "station_bmd_only_cold_day"
            ],
            data[
                "station_neither_cold_condition"
            ],
        ],
        [
            "ineligible_station_winter",
            "missing_tmin",
            "station_threshold_unavailable",
            "compound_p10_and_bmd",
            "percentile_only",
            "bmd_only",
            "neither",
        ],
        default="unclassified",
    )

    # -----------------------------------------------------
    # Cold-deficit metrics
    # -----------------------------------------------------

    data[
        "tmin_below_bmd_10c_deficit"
    ] = (
        mild_upper
        - data[
            temperature_column
        ]
    ).where(
        bmd_cold
    ).astype("float32")

    data[
        "tmin_below_station_p10_deficit"
    ] = (
        data[
            P10_THRESHOLD_COLUMN
        ]
        - data[
            temperature_column
        ]
    ).where(
        percentile_cold
    ).astype("float32")

    data[
        "compound_bmd_deficit"
    ] = data[
        "tmin_below_bmd_10c_deficit"
    ].where(
        data[
            "station_compound_cold_day_primary"
        ]
    ).astype("float32")

    data[
        "compound_p10_deficit"
    ] = data[
        "tmin_below_station_p10_deficit"
    ].where(
        data[
            "station_compound_cold_day_primary"
        ]
    ).astype("float32")

    data[
        "step7b_temperature_adjusted"
    ] = False

    data[
        "step7b_interpolation_applied"
    ] = False

    # -----------------------------------------------------
    # Boundary validation using synthetic values
    # -----------------------------------------------------

    boundary_values = pd.Series(
        [
            np.nan,
            3.9,
            4.0,
            4.1,
            6.0,
            6.1,
            8.0,
            8.1,
            10.0,
            10.1,
        ],
        name="test_tmin",
    )

    (
        boundary_categories,
        boundary_ranks,
    ) = classify_bmd_values(
        boundary_values,
        mild_upper=mild_upper,
        moderate_upper=moderate_upper,
        severe_upper=severe_upper,
        very_severe_upper=(
            very_severe_upper
        ),
    )

    boundary_validation = pd.DataFrame(
        {
            "test_tmin": boundary_values,
            "assigned_category":
                boundary_categories,
            "assigned_rank":
                boundary_ranks,
            "expected_category": [
                "missing",
                "very_severe",
                "very_severe",
                "severe",
                "severe",
                "moderate",
                "moderate",
                "mild",
                "mild",
                "above_bmd_cold_wave_threshold",
            ],
            "expected_rank": [
                -1,
                4,
                4,
                3,
                3,
                2,
                2,
                1,
                1,
                0,
            ],
        }
    )

    boundary_validation[
        "category_pass"
    ] = (
        boundary_validation[
            "assigned_category"
        ].eq(
            boundary_validation[
                "expected_category"
            ]
        )
    )

    boundary_validation[
        "rank_pass"
    ] = (
        boundary_validation[
            "assigned_rank"
        ].eq(
            boundary_validation[
                "expected_rank"
            ]
        )
    )

    boundary_validation[
        "overall_pass"
    ] = (
        boundary_validation[
            "category_pass"
        ]
        & boundary_validation[
            "rank_pass"
        ]
    )

    if not boundary_validation[
        "overall_pass"
    ].all():
        raise ValueError(
            "BMD category boundary validation failed."
        )

    # -----------------------------------------------------
    # Integrity validation
    # -----------------------------------------------------

    if len(data) != original_rows:
        raise ValueError(
            "Step 7B changed the row count."
        )

    if not original_keys.equals(
        data[
            [
                "station_uid",
                "date",
            ]
        ]
    ):
        raise ValueError(
            "Step 7B changed station-date keys "
            "or row ordering."
        )

    if not values_equal_with_nan(
        original_temperature,
        data[
            temperature_column
        ],
    ):
        raise ValueError(
            "Step 7B changed Tmin values."
        )

    if data.duplicated(
        subset=[
            "station_uid",
            "date",
        ]
    ).any():
        raise ValueError(
            "Step 7B introduced duplicate keys."
        )

    if data[
        "step7b_temperature_adjusted"
    ].any():
        raise ValueError(
            "A temperature adjustment was applied."
        )

    if data[
        "step7b_interpolation_applied"
    ].any():
        raise ValueError(
            "Interpolation was applied."
        )

    # Every eligible numeric day must have exactly one raw category.
    eligible_categories = data.loc[
        eligible_bmd,
        "bmd_absolute_category_raw",
    ]

    allowed_numeric_categories = {
        "above_bmd_cold_wave_threshold",
        "mild",
        "moderate",
        "severe",
        "very_severe",
    }

    unexpected_categories = (
        set(
            eligible_categories.unique()
        )
        - allowed_numeric_categories
    )

    if unexpected_categories:
        raise ValueError(
            "Unexpected eligible BMD categories: "
            f"{sorted(unexpected_categories)}"
        )

    # Mutually exclusive condition classes.
    condition_columns = [
        "station_compound_cold_day_primary",
        "station_percentile_only_cold_day",
        "station_bmd_only_cold_day",
        "station_neither_cold_condition",
    ]

    eligible_condition_count = (
        data[
            condition_columns
        ]
        .sum(axis=1)
    )

    invalid_condition_count = (
        data[
            "eligible_for_compound_cold_test_primary"
        ]
        & eligible_condition_count.ne(1)
    )

    if invalid_condition_count.any():
        raise ValueError(
            "Eligible rows do not have exactly one "
            "compound-condition classification."
        )

    # -----------------------------------------------------
    # Reports
    # -----------------------------------------------------

    category_definition = pd.DataFrame(
        [
            {
                "category":
                    "above_bmd_cold_wave_threshold",
                "severity_rank": 0,
                "lower_boundary_celsius":
                    mild_upper,
                "lower_boundary_inclusive":
                    False,
                "upper_boundary_celsius":
                    np.nan,
                "upper_boundary_inclusive":
                    False,
            },
            {
                "category": "mild",
                "severity_rank": 1,
                "lower_boundary_celsius":
                    moderate_upper,
                "lower_boundary_inclusive":
                    False,
                "upper_boundary_celsius":
                    mild_upper,
                "upper_boundary_inclusive":
                    True,
            },
            {
                "category": "moderate",
                "severity_rank": 2,
                "lower_boundary_celsius":
                    severe_upper,
                "lower_boundary_inclusive":
                    False,
                "upper_boundary_celsius":
                    moderate_upper,
                "upper_boundary_inclusive":
                    True,
            },
            {
                "category": "severe",
                "severity_rank": 3,
                "lower_boundary_celsius":
                    very_severe_upper,
                "lower_boundary_inclusive":
                    False,
                "upper_boundary_celsius":
                    severe_upper,
                "upper_boundary_inclusive":
                    True,
            },
            {
                "category": "very_severe",
                "severity_rank": 4,
                "lower_boundary_celsius":
                    np.nan,
                "lower_boundary_inclusive":
                    False,
                "upper_boundary_celsius":
                    very_severe_upper,
                "upper_boundary_inclusive":
                    True,
            },
        ]
    )

    station_summary = (
        data.groupby(
            [
                "station_uid",
                "station_id",
                "station_name_display",
            ],
            dropna=False,
        )
        .agg(
            total_djf_rows=(
                "date",
                "size",
            ),
            complete_winter_station_days=(
                PRIMARY_USABLE_COLUMN,
                "sum",
            ),
            eligible_numeric_station_days=(
                "eligible_for_bmd_absolute_test_primary",
                "sum",
            ),
            percentile_p10_cold_days=(
                "station_percentile_cold_day_p10",
                "sum",
            ),
            bmd_cold_wave_days=(
                "station_bmd_cold_wave_day",
                "sum",
            ),
            compound_cold_days=(
                "station_compound_cold_day_primary",
                "sum",
            ),
            percentile_only_days=(
                "station_percentile_only_cold_day",
                "sum",
            ),
            bmd_only_days=(
                "station_bmd_only_cold_day",
                "sum",
            ),
            bmd_mild_days=(
                "station_bmd_mild_day",
                "sum",
            ),
            bmd_moderate_days=(
                "station_bmd_moderate_day",
                "sum",
            ),
            bmd_severe_days=(
                "station_bmd_severe_day",
                "sum",
            ),
            bmd_very_severe_days=(
                "station_bmd_very_severe_day",
                "sum",
            ),
            minimum_tmin=(
                temperature_column,
                "min",
            ),
            maximum_bmd_severity_rank=(
                "bmd_absolute_severity_rank_primary",
                "max",
            ),
        )
        .reset_index()
        .sort_values("station_uid")
    )

    station_summary[
        "compound_cold_day_percent"
    ] = (
        100.0
        * safe_fraction(
            station_summary[
                "compound_cold_days"
            ],
            station_summary[
                "eligible_numeric_station_days"
            ],
        )
    )

    station_winter_summary = (
        data.groupby(
            [
                "station_uid",
                "station_name_display",
                "winter_start_year",
                "winter_label",
            ],
            dropna=False,
        )
        .agg(
            winter_complete=(
                PRIMARY_USABLE_COLUMN,
                "max",
            ),
            expected_days=(
                "winter_expected_days",
                "max",
            ),
            numeric_tmin_days=(
                "eligible_for_bmd_absolute_test_primary",
                "sum",
            ),
            percentile_p10_cold_days=(
                "station_percentile_cold_day_p10",
                "sum",
            ),
            bmd_cold_wave_days=(
                "station_bmd_cold_wave_day",
                "sum",
            ),
            compound_cold_days=(
                "station_compound_cold_day_primary",
                "sum",
            ),
            bmd_mild_days=(
                "station_bmd_mild_day",
                "sum",
            ),
            bmd_moderate_days=(
                "station_bmd_moderate_day",
                "sum",
            ),
            bmd_severe_days=(
                "station_bmd_severe_day",
                "sum",
            ),
            bmd_very_severe_days=(
                "station_bmd_very_severe_day",
                "sum",
            ),
            minimum_tmin=(
                temperature_column,
                "min",
            ),
            maximum_bmd_severity_rank=(
                "bmd_absolute_severity_rank_primary",
                "max",
            ),
            total_compound_bmd_deficit=(
                "compound_bmd_deficit",
                "sum",
            ),
            total_compound_p10_deficit=(
                "compound_p10_deficit",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "station_uid",
                "winter_start_year",
            ]
        )
    )

    daily_summary = (
        data.groupby(
            [
                "date",
                "year",
                "month",
                "day",
                "winter_start_year",
                "winter_label",
                "winter_day_index",
            ],
            as_index=False,
        )
        .agg(
            primary_network_station_count=(
                "station_uid",
                "nunique",
            ),
            complete_station_count=(
                PRIMARY_USABLE_COLUMN,
                "sum",
            ),
            observed_eligible_station_count=(
                "eligible_for_bmd_absolute_test_primary",
                "sum",
            ),
            percentile_p10_cold_station_count=(
                "station_percentile_cold_day_p10",
                "sum",
            ),
            bmd_cold_station_count=(
                "station_bmd_cold_wave_day",
                "sum",
            ),
            compound_cold_station_count=(
                "station_compound_cold_day_primary",
                "sum",
            ),
            percentile_only_station_count=(
                "station_percentile_only_cold_day",
                "sum",
            ),
            bmd_only_station_count=(
                "station_bmd_only_cold_day",
                "sum",
            ),
            bmd_mild_station_count=(
                "station_bmd_mild_day",
                "sum",
            ),
            bmd_moderate_station_count=(
                "station_bmd_moderate_day",
                "sum",
            ),
            bmd_severe_station_count=(
                "station_bmd_severe_day",
                "sum",
            ),
            bmd_very_severe_station_count=(
                "station_bmd_very_severe_day",
                "sum",
            ),
            national_minimum_tmin=(
                temperature_column,
                "min",
            ),
            national_median_tmin=(
                temperature_column,
                "median",
            ),
            maximum_bmd_severity_rank=(
                "bmd_absolute_severity_rank_primary",
                "max",
            ),
            total_compound_bmd_deficit=(
                "compound_bmd_deficit",
                "sum",
            ),
            total_compound_p10_deficit=(
                "compound_p10_deficit",
                "sum",
            ),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    daily_summary[
        "percentile_p10_spatial_fraction"
    ] = safe_fraction(
        daily_summary[
            "percentile_p10_cold_station_count"
        ],
        daily_summary[
            "observed_eligible_station_count"
        ],
    )

    daily_summary[
        "bmd_cold_spatial_fraction"
    ] = safe_fraction(
        daily_summary[
            "bmd_cold_station_count"
        ],
        daily_summary[
            "observed_eligible_station_count"
        ],
    )

    daily_summary[
        "compound_cold_spatial_fraction"
    ] = safe_fraction(
        daily_summary[
            "compound_cold_station_count"
        ],
        daily_summary[
            "observed_eligible_station_count"
        ],
    )

    daily_summary[
        "percentile_p10_spatial_percent"
    ] = (
        100.0
        * daily_summary[
            "percentile_p10_spatial_fraction"
        ]
    )

    daily_summary[
        "bmd_cold_spatial_percent"
    ] = (
        100.0
        * daily_summary[
            "bmd_cold_spatial_fraction"
        ]
    )

    daily_summary[
        "compound_cold_spatial_percent"
    ] = (
        100.0
        * daily_summary[
            "compound_cold_spatial_fraction"
        ]
    )

    rank_to_category = {
        -1: "unavailable",
        0: "no_bmd_cold_wave",
        1: "mild",
        2: "moderate",
        3: "severe",
        4: "very_severe",
    }

    daily_summary[
        "most_severe_bmd_category"
    ] = (
        daily_summary[
            "maximum_bmd_severity_rank"
        ]
        .fillna(-1)
        .astype(int)
        .map(rank_to_category)
    )

    category_counts = (
        data[
            "station_day_cold_condition"
        ]
        .value_counts()
        .rename_axis(
            "station_day_cold_condition"
        )
        .reset_index(
            name="record_count"
        )
    )

    # -----------------------------------------------------
    # Save reports and output
    # -----------------------------------------------------

    category_definition.to_csv(
        REPORT_DIRECTORY
        / "step7b_bmd_category_definition.csv",
        index=False,
    )

    boundary_validation.to_csv(
        REPORT_DIRECTORY
        / "step7b_bmd_boundary_validation.csv",
        index=False,
    )

    station_summary.to_csv(
        REPORT_DIRECTORY
        / "step7b_station_category_summary.csv",
        index=False,
    )

    station_winter_summary.to_csv(
        REPORT_DIRECTORY
        / "step7b_station_winter_category_summary.csv",
        index=False,
    )

    daily_summary.to_csv(
        REPORT_DIRECTORY
        / "step7b_daily_spatial_cold_summary.csv",
        index=False,
    )

    category_counts.to_csv(
        REPORT_DIRECTORY
        / "step7b_station_day_condition_counts.csv",
        index=False,
    )

    data.to_parquet(
        OUTPUT_FILE,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    output_parquet = pq.ParquetFile(
        OUTPUT_FILE
    )

    total_eligible = int(
        data[
            "eligible_for_compound_cold_test_primary"
        ].sum()
    )

    total_p10 = int(
        data[
            "station_percentile_cold_day_p10"
        ].sum()
    )

    total_bmd = int(
        data[
            "station_bmd_cold_wave_day"
        ].sum()
    )

    total_compound = int(
        data[
            "station_compound_cold_day_primary"
        ].sum()
    )

    total_percentile_only = int(
        data[
            "station_percentile_only_cold_day"
        ].sum()
    )

    total_bmd_only = int(
        data[
            "station_bmd_only_cold_day"
        ].sum()
    )

    category_totals = {
        "mild": int(
            data[
                "station_bmd_mild_day"
            ].sum()
        ),
        "moderate": int(
            data[
                "station_bmd_moderate_day"
            ].sum()
        ),
        "severe": int(
            data[
                "station_bmd_severe_day"
            ].sum()
        ),
        "very_severe": int(
            data[
                "station_bmd_very_severe_day"
            ].sum()
        ),
    }

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
            original_rows,
        "output_rows":
            output_parquet.metadata.num_rows,
        "unique_stations":
            int(
                data[
                    "station_uid"
                ].nunique()
            ),
        "unique_winters":
            int(
                data[
                    "winter_start_year"
                ].nunique()
            ),
        "eligible_compound_station_days":
            total_eligible,
        "percentile_p10_cold_station_days":
            total_p10,
        "bmd_cold_wave_station_days":
            total_bmd,
        "compound_cold_station_days":
            total_compound,
        "percentile_only_station_days":
            total_percentile_only,
        "bmd_only_station_days":
            total_bmd_only,
        "bmd_category_totals":
            category_totals,
        "primary_station_day_definition":
            (
                "Tmin below station p10 "
                "AND Tmin at or below 10 C"
            ),
        "temperature_adjusted":
            False,
        "interpolation_applied":
            False,
    }

    (
        REPORT_DIRECTORY
        / "step7b_compound_cold_summary.json"
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
        / "step7b_parquet_schema.json"
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
        "STEP 7B: BMD AND COMPOUND COLD INDICATORS",
        "=" * 48,
        f"Input rows: {original_rows:,}",
        (
            "Output rows: "
            f"{output_parquet.metadata.num_rows:,}"
        ),
        (
            "Unique stations: "
            f"{data['station_uid'].nunique()}"
        ),
        (
            "Unique winters: "
            f"{data['winter_start_year'].nunique()}"
        ),
        (
            "Eligible compound station-days: "
            f"{total_eligible:,}"
        ),
        (
            "Station p10 cold days: "
            f"{total_p10:,}"
        ),
        (
            "BMD absolute cold-wave days: "
            f"{total_bmd:,}"
        ),
        (
            "Primary compound cold days: "
            f"{total_compound:,}"
        ),
        (
            "Percentile-only cold days: "
            f"{total_percentile_only:,}"
        ),
        (
            "BMD-only cold days: "
            f"{total_bmd_only:,}"
        ),
        "",
        "BMD category totals:",
        *[
            f"- {key}: {value:,}"
            for key, value
            in category_totals.items()
        ],
        "",
        "Primary station-day condition:",
        (
            "- Tmin below station p10 AND "
            "Tmin at or below 10 C"
        ),
        "",
        "Temperature adjusted: No",
        "Interpolation applied: No",
        "",
        f"Output: {OUTPUT_FILE}",
    ]

    (
        REPORT_DIRECTORY
        / "step7b_compound_cold_report.txt"
    ).write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(report_lines))
    print("\nSTEP 7B PASSED.")


if __name__ == "__main__":
    main()
