from __future__ import annotations

import json
import math
import zlib
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

FIXED_NETWORK_FILE = (
    PROJECT_ROOT
    / "02_metadata"
    / "step6a_fixed_station_network.csv"
)

POLICY_FILE = (
    PROJECT_ROOT
    / "00_admin"
    / "step6b_homogeneity_policy.yaml"
)

MONTHLY_OUTPUT_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "step6b_monthly_station_network_residuals.parquet"
)

ANNUAL_OUTPUT_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "step6b_annual_station_network_residuals.parquet"
)

HOMOGENEITY_STATUS_FILE = (
    PROJECT_ROOT
    / "02_metadata"
    / "step6b_station_homogeneity_status.csv"
)

REPORT_DIRECTORY = PROJECT_ROOT / "05_qc_reports"
ADMIN_DIRECTORY = PROJECT_ROOT / "00_admin"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        return yaml.safe_load(file_handle)


def robust_scale(
    values: np.ndarray,
    minimum_scale: float,
) -> float:
    clean = np.asarray(
        values,
        dtype=float,
    )

    clean = clean[
        np.isfinite(clean)
    ]

    if len(clean) == 0:
        return float("nan")

    median = float(
        np.median(clean)
    )

    mad = float(
        np.median(
            np.abs(clean - median)
        )
    )

    return max(
        1.4826 * mad,
        minimum_scale,
    )


def bootstrap_median_shift_ci(
    pre_values: np.ndarray,
    post_values: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    pre = np.asarray(
        pre_values,
        dtype=float,
    )

    post = np.asarray(
        post_values,
        dtype=float,
    )

    pre = pre[np.isfinite(pre)]
    post = post[np.isfinite(post)]

    if len(pre) == 0 or len(post) == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)

    pre_samples = rng.choice(
        pre,
        size=(
            repetitions,
            len(pre),
        ),
        replace=True,
    )

    post_samples = rng.choice(
        post,
        size=(
            repetitions,
            len(post),
        ),
        replace=True,
    )

    shifts = (
        np.median(
            post_samples,
            axis=1,
        )
        - np.median(
            pre_samples,
            axis=1,
        )
    )

    lower = float(
        np.quantile(
            shifts,
            0.025,
        )
    )

    upper = float(
        np.quantile(
            shifts,
            0.975,
        )
    )

    return lower, upper


def confidence_interval_excludes_zero(
    lower: float,
    upper: float,
) -> bool:
    if not (
        math.isfinite(lower)
        and math.isfinite(upper)
    ):
        return False

    return lower > 0.0 or upper < 0.0


def classify_shift(
    shift: float,
    standardized_shift: float,
    ci_lower: float,
    ci_upper: float,
    moderate_absolute: float,
    moderate_standardized: float,
    high_absolute: float,
    high_standardized: float,
) -> str:
    if not (
        math.isfinite(shift)
        and math.isfinite(
            standardized_shift
        )
    ):
        return "insufficient_data"

    significant = (
        confidence_interval_excludes_zero(
            ci_lower,
            ci_upper,
        )
    )

    if (
        significant
        and abs(shift) >= high_absolute
        and abs(
            standardized_shift
        ) >= high_standardized
    ):
        return "high_shift_signal"

    if (
        significant
        and abs(shift) >= moderate_absolute
        and abs(
            standardized_shift
        ) >= moderate_standardized
    ):
        return "moderate_shift_signal"

    return "no_major_shift_signal"


def transition_statistics(
    station_data: pd.DataFrame,
    pre_start: pd.Timestamp,
    pre_end: pd.Timestamp,
    post_start: pd.Timestamp,
    post_end: pd.Timestamp,
    months_filter: set[int] | None,
    minimum_months: int,
    minimum_scale: float,
    bootstrap_repetitions: int,
    seed: int,
    moderate_absolute: float,
    moderate_standardized: float,
    high_absolute: float,
    high_standardized: float,
) -> dict[str, Any]:
    selected = station_data.loc[
        station_data[
            "network_residual_available"
        ]
    ].copy()

    if months_filter is not None:
        selected = selected.loc[
            selected["month"].isin(
                months_filter
            )
        ]

    pre = selected.loc[
        selected[
            "month_start"
        ].between(
            pre_start,
            pre_end,
        ),
        "station_network_residual",
    ].dropna().astype(float).to_numpy()

    post = selected.loc[
        selected[
            "month_start"
        ].between(
            post_start,
            post_end,
        ),
        "station_network_residual",
    ].dropna().astype(float).to_numpy()

    result: dict[str, Any] = {
        "pre_month_count":
            len(pre),
        "post_month_count":
            len(post),
        "pre_median_residual":
            np.nan,
        "post_median_residual":
            np.nan,
        "median_shift_post_minus_pre":
            np.nan,
        "pooled_robust_scale":
            np.nan,
        "standardized_shift":
            np.nan,
        "bootstrap_ci_lower":
            np.nan,
        "bootstrap_ci_upper":
            np.nan,
        "pre_robust_scale":
            np.nan,
        "post_robust_scale":
            np.nan,
        "post_pre_scale_ratio":
            np.nan,
        "shift_classification":
            "insufficient_data",
    }

    if (
        len(pre) < minimum_months
        or len(post) < minimum_months
    ):
        return result

    pre_median = float(
        np.median(pre)
    )

    post_median = float(
        np.median(post)
    )

    shift = (
        post_median
        - pre_median
    )

    pooled = np.concatenate(
        [pre, post]
    )

    pooled_scale = robust_scale(
        pooled,
        minimum_scale,
    )

    standardized = (
        shift / pooled_scale
        if math.isfinite(
            pooled_scale
        )
        and pooled_scale > 0
        else np.nan
    )

    ci_lower, ci_upper = (
        bootstrap_median_shift_ci(
            pre,
            post,
            bootstrap_repetitions,
            seed,
        )
    )

    pre_scale = robust_scale(
        pre,
        minimum_scale,
    )

    post_scale = robust_scale(
        post,
        minimum_scale,
    )

    if (
        math.isfinite(pre_scale)
        and math.isfinite(post_scale)
        and min(
            pre_scale,
            post_scale,
        ) > 0
    ):
        scale_ratio = (
            max(
                pre_scale,
                post_scale,
            )
            / min(
                pre_scale,
                post_scale,
            )
        )
    else:
        scale_ratio = np.nan

    classification = classify_shift(
        shift=shift,
        standardized_shift=standardized,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        moderate_absolute=(
            moderate_absolute
        ),
        moderate_standardized=(
            moderate_standardized
        ),
        high_absolute=high_absolute,
        high_standardized=(
            high_standardized
        ),
    )

    result.update(
        {
            "pre_median_residual":
                pre_median,
            "post_median_residual":
                post_median,
            "median_shift_post_minus_pre":
                shift,
            "pooled_robust_scale":
                pooled_scale,
            "standardized_shift":
                standardized,
            "bootstrap_ci_lower":
                ci_lower,
            "bootstrap_ci_upper":
                ci_upper,
            "pre_robust_scale":
                pre_scale,
            "post_robust_scale":
                post_scale,
            "post_pre_scale_ratio":
                scale_ratio,
            "shift_classification":
                classification,
        }
    )

    return result


def scan_general_breakpoint(
    annual_data: pd.DataFrame,
    minimum_years_each_side: int,
    minimum_scale: float,
    bootstrap_repetitions: int,
    seed: int,
    moderate_absolute: float,
    moderate_standardized: float,
    high_absolute: float,
    high_standardized: float,
) -> dict[str, Any]:
    usable = annual_data.loc[
        annual_data[
            "annual_residual_valid"
        ]
    ].sort_values(
        "year"
    )

    result: dict[str, Any] = {
        "valid_year_count":
            len(usable),
        "candidate_break_count":
            0,
        "best_break_after_year":
            np.nan,
        "left_year_count":
            np.nan,
        "right_year_count":
            np.nan,
        "left_median_residual":
            np.nan,
        "right_median_residual":
            np.nan,
        "break_shift_right_minus_left":
            np.nan,
        "pooled_robust_scale":
            np.nan,
        "standardized_break_shift":
            np.nan,
        "bootstrap_ci_lower":
            np.nan,
        "bootstrap_ci_upper":
            np.nan,
        "break_classification":
            "insufficient_data",
    }

    if len(usable) < (
        2 * minimum_years_each_side
    ):
        return result

    values = usable[
        "annual_median_network_residual"
    ].astype(float).to_numpy()

    pooled_scale = robust_scale(
        values,
        minimum_scale,
    )

    candidates: list[
        dict[str, Any]
    ] = []

    years = usable[
        "year"
    ].astype(int).to_numpy()

    for split_index in range(
        minimum_years_each_side,
        len(usable)
        - minimum_years_each_side
        + 1,
    ):
        left = values[:split_index]
        right = values[split_index:]

        if (
            len(left)
            < minimum_years_each_side
            or len(right)
            < minimum_years_each_side
        ):
            continue

        left_median = float(
            np.median(left)
        )

        right_median = float(
            np.median(right)
        )

        shift = (
            right_median
            - left_median
        )

        standardized = (
            shift / pooled_scale
            if math.isfinite(
                pooled_scale
            )
            and pooled_scale > 0
            else np.nan
        )

        candidates.append(
            {
                "break_after_year":
                    int(
                        years[
                            split_index - 1
                        ]
                    ),
                "left_values":
                    left,
                "right_values":
                    right,
                "left_count":
                    len(left),
                "right_count":
                    len(right),
                "left_median":
                    left_median,
                "right_median":
                    right_median,
                "shift":
                    shift,
                "standardized":
                    standardized,
            }
        )

    result[
        "candidate_break_count"
    ] = len(candidates)

    if not candidates:
        return result

    best = max(
        candidates,
        key=lambda candidate: abs(
            candidate["standardized"]
        )
        if math.isfinite(
            candidate["standardized"]
        )
        else -np.inf,
    )

    ci_lower, ci_upper = (
        bootstrap_median_shift_ci(
            best["left_values"],
            best["right_values"],
            bootstrap_repetitions,
            seed,
        )
    )

    classification = classify_shift(
        shift=best["shift"],
        standardized_shift=(
            best["standardized"]
        ),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        moderate_absolute=(
            moderate_absolute
        ),
        moderate_standardized=(
            moderate_standardized
        ),
        high_absolute=high_absolute,
        high_standardized=(
            high_standardized
        ),
    )

    result.update(
        {
            "best_break_after_year":
                best[
                    "break_after_year"
                ],
            "left_year_count":
                best["left_count"],
            "right_year_count":
                best["right_count"],
            "left_median_residual":
                best["left_median"],
            "right_median_residual":
                best["right_median"],
            "break_shift_right_minus_left":
                best["shift"],
            "pooled_robust_scale":
                pooled_scale,
            "standardized_break_shift":
                best["standardized"],
            "bootstrap_ci_lower":
                ci_lower,
            "bootstrap_ci_upper":
                ci_upper,
            "break_classification":
                classification,
        }
    )

    return result


def main() -> None:
    for required_file in [
        INPUT_FILE,
        FIXED_NETWORK_FILE,
        POLICY_FILE,
    ]:
        if not required_file.exists():
            raise FileNotFoundError(
                f"Required file not found: "
                f"{required_file}"
            )

    MONTHLY_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    HOMOGENEITY_STATUS_FILE.parent.mkdir(
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
        "temperature_variable"
    ]

    monthly_rules = policy[
        "monthly_series"
    ]

    transition_rules = policy[
        "source_transition"
    ]

    breakpoint_rules = policy[
        "general_breakpoint_scan"
    ]

    value_column = str(
        temperature_rules[
            "value_column"
        ]
    )

    climatology_column = str(
        temperature_rules[
            "climatology_column"
        ]
    )

    monthly_completeness_threshold = float(
        monthly_rules[
            "minimum_daily_completeness_percent"
        ]
    )

    minimum_network_neighbours = int(
        monthly_rules[
            "minimum_network_neighbours"
        ]
    )

    transition_date = pd.Timestamp(
        transition_rules[
            "transition_date"
        ]
    )

    pre_start = pd.Timestamp(
        transition_rules[
            "pre_period"
        ]["start_date"]
    )

    pre_end = pd.Timestamp(
        transition_rules[
            "pre_period"
        ]["end_date"]
    ).to_period("M").to_timestamp()

    post_start = pd.Timestamp(
        transition_rules[
            "post_period"
        ]["start_date"]
    )

    post_end = pd.Timestamp(
        transition_rules[
            "post_period"
        ]["end_date"]
    ).to_period("M").to_timestamp()

    minimum_all_months = int(
        transition_rules[
            "minimum_all_months_per_period"
        ]
    )

    minimum_djf_months = int(
        transition_rules[
            "minimum_djf_months_per_period"
        ]
    )

    bootstrap_repetitions = int(
        transition_rules[
            "bootstrap_repetitions"
        ]
    )

    bootstrap_seed = int(
        transition_rules[
            "bootstrap_seed"
        ]
    )

    minimum_scale = float(
        transition_rules[
            "minimum_robust_scale_celsius"
        ]
    )

    moderate_shift_absolute = float(
        transition_rules[
            "moderate_shift"
        ][
            "minimum_absolute_shift_celsius"
        ]
    )

    moderate_shift_standardized = float(
        transition_rules[
            "moderate_shift"
        ][
            "minimum_standardized_shift"
        ]
    )

    high_shift_absolute = float(
        transition_rules[
            "high_shift"
        ][
            "minimum_absolute_shift_celsius"
        ]
    )

    high_shift_standardized = float(
        transition_rules[
            "high_shift"
        ][
            "minimum_standardized_shift"
        ]
    )

    variance_ratio_threshold = float(
        transition_rules[
            "variance_ratio_threshold"
        ]
    )

    minimum_valid_months_per_year = int(
        breakpoint_rules[
            "minimum_valid_months_per_year"
        ]
    )

    minimum_years_each_side = int(
        breakpoint_rules[
            "minimum_years_each_side"
        ]
    )

    moderate_break_absolute = float(
        breakpoint_rules[
            "moderate_break"
        ][
            "minimum_absolute_shift_celsius"
        ]
    )

    moderate_break_standardized = float(
        breakpoint_rules[
            "moderate_break"
        ][
            "minimum_standardized_shift"
        ]
    )

    high_break_absolute = float(
        breakpoint_rules[
            "high_break"
        ][
            "minimum_absolute_shift_celsius"
        ]
    )

    high_break_standardized = float(
        breakpoint_rules[
            "high_break"
        ][
            "minimum_standardized_shift"
        ]
    )

    fixed_network = pd.read_csv(
        FIXED_NETWORK_FILE
    )

    if fixed_network.empty:
        raise ValueError(
            "The Step 6A fixed network is empty."
        )

    if fixed_network[
        "station_uid"
    ].duplicated().any():
        raise ValueError(
            "Duplicate station IDs exist in "
            "the fixed-network table."
        )

    fixed_station_uids = set(
        fixed_network[
            "station_uid"
        ].astype(str)
    )

    fixed_station_count = len(
        fixed_station_uids
    )

    data = pd.read_parquet(
        INPUT_FILE
    )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    if value_column not in data.columns:
        raise ValueError(
            f"Required column missing: "
            f"{value_column}"
        )

    if climatology_column not in data.columns:
        raise ValueError(
            f"Required column missing: "
            f"{climatology_column}"
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

    fixed_daily = data.loc[
        data["station_uid"].isin(
            fixed_station_uids
        )
    ].copy()

    if (
        fixed_daily[
            "station_uid"
        ].nunique()
        != fixed_station_count
    ):
        raise ValueError(
            "Not all fixed-network stations occur "
            "in the daily dataset."
        )

    fixed_daily[value_column] = (
        pd.to_numeric(
            fixed_daily[value_column],
            errors="coerce",
        )
    )

    fixed_daily[
        climatology_column
    ] = pd.to_numeric(
        fixed_daily[
            climatology_column
        ],
        errors="coerce",
    )

    fixed_daily[
        "daily_tmin_climatological_anomaly"
    ] = (
        fixed_daily[value_column]
        - fixed_daily[
            climatology_column
        ]
    )

    fixed_daily["month_start"] = (
        fixed_daily["date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    fixed_daily["year"] = (
        fixed_daily[
            "month_start"
        ].dt.year.astype("int16")
    )

    fixed_daily["month"] = (
        fixed_daily[
            "month_start"
        ].dt.month.astype("int8")
    )

    # -----------------------------------------------------
    # Monthly station anomaly series
    # -----------------------------------------------------

    monthly = (
        fixed_daily.groupby(
            [
                "station_uid",
                "month_start",
                "year",
                "month",
            ],
            as_index=False,
        )
        .agg(
            represented_days=(
                "date",
                "nunique",
            ),
            numeric_tmin_days=(
                value_column,
                "count",
            ),
            anomaly_days=(
                "daily_tmin_climatological_anomaly",
                "count",
            ),
            station_month_anomaly=(
                "daily_tmin_climatological_anomaly",
                "median",
            ),
        )
    )

    monthly[
        "expected_calendar_days"
    ] = (
        monthly[
            "month_start"
        ].dt.days_in_month
        .astype("int16")
    )

    monthly[
        "minimum_required_daily_values"
    ] = np.ceil(
        monthly[
            "expected_calendar_days"
        ]
        * monthly_completeness_threshold
        / 100.0
    ).astype("int16")

    monthly[
        "monthly_tmin_completeness_percent"
    ] = (
        100.0
        * monthly[
            "numeric_tmin_days"
        ]
        / monthly[
            "expected_calendar_days"
        ]
    ).astype("float32")

    monthly[
        "monthly_anomaly_completeness_percent"
    ] = (
        100.0
        * monthly[
            "anomaly_days"
        ]
        / monthly[
            "expected_calendar_days"
        ]
    ).astype("float32")

    monthly["monthly_valid"] = (
        monthly[
            "numeric_tmin_days"
        ].ge(
            monthly[
                "minimum_required_daily_values"
            ]
        )
        & monthly[
            "anomaly_days"
        ].ge(
            monthly[
                "minimum_required_daily_values"
            ]
        )
        & monthly[
            "station_month_anomaly"
        ].notna()
    )

    # Dominant source for numeric observations.
    source_data = fixed_daily.loc[
        fixed_daily[
            value_column
        ].notna(),
        [
            "station_uid",
            "month_start",
            "tmin_cleaned_stage2_source",
        ],
    ].copy()

    source_counts = (
        source_data.groupby(
            [
                "station_uid",
                "month_start",
                "tmin_cleaned_stage2_source",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "source_day_count",
            }
        )
    )

    dominant_source = (
        source_counts.sort_values(
            [
                "station_uid",
                "month_start",
                "source_day_count",
                "tmin_cleaned_stage2_source",
            ],
            ascending=[
                True,
                True,
                False,
                True,
            ],
        )
        .drop_duplicates(
            subset=[
                "station_uid",
                "month_start",
            ]
        )
        .rename(
            columns={
                "tmin_cleaned_stage2_source":
                    "dominant_tmin_source",
                "source_day_count":
                    "dominant_source_day_count",
            }
        )
    )

    monthly = monthly.merge(
        dominant_source[
            [
                "station_uid",
                "month_start",
                "dominant_tmin_source",
                "dominant_source_day_count",
            ]
        ],
        on=[
            "station_uid",
            "month_start",
        ],
        how="left",
        validate="one_to_one",
    )

    monthly[
        "dominant_tmin_source"
    ] = (
        monthly[
            "dominant_tmin_source"
        ]
        .fillna("none")
        .astype(str)
    )

    monthly[
        "dominant_source_day_count"
    ] = (
        monthly[
            "dominant_source_day_count"
        ]
        .fillna(0)
        .astype("int16")
    )

    # -----------------------------------------------------
    # Leave-one-station-out network reference
    # -----------------------------------------------------

    valid_monthly = monthly.loc[
        monthly["monthly_valid"],
        [
            "station_uid",
            "month_start",
            "station_month_anomaly",
        ],
    ].copy()

    targets = valid_monthly.rename(
        columns={
            "station_uid":
                "target_station_uid",
            "station_month_anomaly":
                "target_station_month_anomaly",
        }
    )

    neighbours = valid_monthly.rename(
        columns={
            "station_uid":
                "neighbour_station_uid",
            "station_month_anomaly":
                "neighbour_station_month_anomaly",
        }
    )

    pairs = targets.merge(
        neighbours,
        on="month_start",
        how="inner",
        validate="many_to_many",
    )

    pairs = pairs.loc[
        pairs[
            "target_station_uid"
        ].ne(
            pairs[
                "neighbour_station_uid"
            ]
        )
    ].copy()

    network_reference = (
        pairs.groupby(
            [
                "target_station_uid",
                "month_start",
            ],
            as_index=False,
        )
        .agg(
            network_neighbour_count=(
                "neighbour_station_uid",
                "nunique",
            ),
            leave_one_out_network_median_anomaly=(
                "neighbour_station_month_anomaly",
                "median",
            ),
            leave_one_out_network_mean_anomaly=(
                "neighbour_station_month_anomaly",
                "mean",
            ),
            network_minimum_anomaly=(
                "neighbour_station_month_anomaly",
                "min",
            ),
            network_maximum_anomaly=(
                "neighbour_station_month_anomaly",
                "max",
            ),
        )
        .rename(
            columns={
                "target_station_uid":
                    "station_uid",
            }
        )
    )

    monthly = monthly.merge(
        network_reference,
        on=[
            "station_uid",
            "month_start",
        ],
        how="left",
        validate="one_to_one",
    )

    monthly[
        "network_neighbour_count"
    ] = (
        monthly[
            "network_neighbour_count"
        ]
        .fillna(0)
        .astype("int16")
    )

    monthly[
        "network_residual_available"
    ] = (
        monthly["monthly_valid"]
        & monthly[
            "network_neighbour_count"
        ].ge(
            minimum_network_neighbours
        )
        & monthly[
            "leave_one_out_network_median_anomaly"
        ].notna()
    )

    monthly[
        "station_network_residual"
    ] = (
        monthly[
            "station_month_anomaly"
        ]
        - monthly[
            "leave_one_out_network_median_anomaly"
        ]
    ).where(
        monthly[
            "network_residual_available"
        ]
    ).astype("float32")

    monthly[
        "period_relative_to_source_transition"
    ] = np.select(
        [
            monthly[
                "month_start"
            ].lt(transition_date),
            monthly[
                "month_start"
            ].ge(transition_date),
        ],
        [
            "pre_2022_source",
            "post_2022_source",
        ],
        default="unknown",
    )

    monthly[
        "is_djf_month"
    ] = monthly[
        "month"
    ].isin(
        [12, 1, 2]
    )

    if monthly.duplicated(
        subset=[
            "station_uid",
            "month_start",
        ]
    ).any():
        raise ValueError(
            "Duplicate station-month rows exist."
        )

    # -----------------------------------------------------
    # Monthly network coverage
    # -----------------------------------------------------

    monthly_network_coverage = (
        monthly.groupby(
            "month_start",
            as_index=False,
        )
        .agg(
            fixed_station_count=(
                "station_uid",
                "nunique",
            ),
            valid_station_months=(
                "monthly_valid",
                "sum",
            ),
            residual_available_station_months=(
                "network_residual_available",
                "sum",
            ),
            network_median_station_anomaly=(
                "station_month_anomaly",
                "median",
            ),
        )
        .sort_values("month_start")
    )

    monthly_network_coverage[
        "valid_station_fraction"
    ] = (
        monthly_network_coverage[
            "valid_station_months"
        ]
        / monthly_network_coverage[
            "fixed_station_count"
        ]
    )

    monthly_network_coverage.to_csv(
        REPORT_DIRECTORY
        / "step6b_monthly_network_coverage.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Source composition report
    # -----------------------------------------------------

    numeric_source = fixed_daily.loc[
        fixed_daily[
            value_column
        ].notna(),
        [
            "station_uid",
            "date",
            "tmin_cleaned_stage2_source",
        ],
    ].copy()

    numeric_source[
        "source_period"
    ] = np.where(
        numeric_source["date"].lt(
            transition_date
        ),
        "pre_2022",
        "post_2022",
    )

    source_composition = (
        numeric_source.groupby(
            [
                "station_uid",
                "source_period",
                "tmin_cleaned_stage2_source",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size":
                    "numeric_day_count",
            }
        )
    )

    source_composition.to_csv(
        REPORT_DIRECTORY
        / "step6b_source_composition.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Source-transition tests
    # -----------------------------------------------------

    transition_rows: list[
        dict[str, Any]
    ] = []

    fixed_lookup = (
        fixed_network.set_index(
            "station_uid"
        )
    )

    for station_uid in sorted(
        fixed_station_uids
    ):
        station_months = monthly.loc[
            monthly[
                "station_uid"
            ].eq(station_uid)
        ].copy()

        seed_component = (
            zlib.crc32(
                station_uid.encode(
                    "utf-8"
                )
            )
            & 0xFFFFFFFF
        )

        all_month_stats = (
            transition_statistics(
                station_data=station_months,
                pre_start=pre_start,
                pre_end=pre_end,
                post_start=post_start,
                post_end=post_end,
                months_filter=None,
                minimum_months=(
                    minimum_all_months
                ),
                minimum_scale=minimum_scale,
                bootstrap_repetitions=(
                    bootstrap_repetitions
                ),
                seed=(
                    bootstrap_seed
                    + seed_component
                ),
                moderate_absolute=(
                    moderate_shift_absolute
                ),
                moderate_standardized=(
                    moderate_shift_standardized
                ),
                high_absolute=(
                    high_shift_absolute
                ),
                high_standardized=(
                    high_shift_standardized
                ),
            )
        )

        djf_stats = transition_statistics(
            station_data=station_months,
            pre_start=pre_start,
            pre_end=pre_end,
            post_start=post_start,
            post_end=post_end,
            months_filter={12, 1, 2},
            minimum_months=(
                minimum_djf_months
            ),
            minimum_scale=minimum_scale,
            bootstrap_repetitions=(
                bootstrap_repetitions
            ),
            seed=(
                bootstrap_seed
                + seed_component
                + 1
            ),
            moderate_absolute=(
                moderate_shift_absolute
            ),
            moderate_standardized=(
                moderate_shift_standardized
            ),
            high_absolute=(
                high_shift_absolute
            ),
            high_standardized=(
                high_shift_standardized
            ),
        )

        classifications = {
            all_month_stats[
                "shift_classification"
            ],
            djf_stats[
                "shift_classification"
            ],
        }

        variance_values = [
            all_month_stats[
                "post_pre_scale_ratio"
            ],
            djf_stats[
                "post_pre_scale_ratio"
            ],
        ]

        finite_variance_values = [
            value
            for value
            in variance_values
            if isinstance(
                value,
                (int, float, np.number),
            )
            and math.isfinite(
                float(value)
            )
        ]

        maximum_scale_ratio = (
            max(finite_variance_values)
            if finite_variance_values
            else np.nan
        )

        variance_shift_flag = (
            math.isfinite(
                maximum_scale_ratio
            )
            and maximum_scale_ratio
            >= variance_ratio_threshold
        )

        if (
            "high_shift_signal"
            in classifications
        ):
            transition_status = (
                "high_source_transition_signal"
            )

        elif (
            "moderate_shift_signal"
            in classifications
            or variance_shift_flag
        ):
            transition_status = (
                "moderate_source_transition_signal"
            )

        elif classifications == {
            "insufficient_data"
        }:
            transition_status = (
                "insufficient_transition_data"
            )

        else:
            transition_status = (
                "no_major_source_transition_signal"
            )

        metadata = fixed_lookup.loc[
            station_uid
        ]

        row: dict[str, Any] = {
            "station_uid":
                station_uid,
            "station_id":
                metadata.get(
                    "station_id",
                    "",
                ),
            "station_name_display":
                metadata.get(
                    "station_name_display",
                    "",
                ),
            "latitude":
                metadata.get(
                    "latitude",
                    np.nan,
                ),
            "longitude":
                metadata.get(
                    "longitude",
                    np.nan,
                ),
        }

        for key, value in (
            all_month_stats.items()
        ):
            row[
                f"all_month_{key}"
            ] = value

        for key, value in (
            djf_stats.items()
        ):
            row[
                f"djf_{key}"
            ] = value

        row[
            "maximum_post_pre_scale_ratio"
        ] = maximum_scale_ratio

        row[
            "variance_shift_flag"
        ] = variance_shift_flag

        row[
            "source_transition_status"
        ] = transition_status

        transition_rows.append(row)

    transition_summary = pd.DataFrame(
        transition_rows
    ).sort_values(
        [
            "source_transition_status",
            "station_uid",
        ]
    ).reset_index(drop=True)

    transition_summary.to_csv(
        REPORT_DIRECTORY
        / "step6b_source_transition_summary.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Annual residual series
    # -----------------------------------------------------

    residual_months = monthly.loc[
        monthly[
            "network_residual_available"
        ],
        [
            "station_uid",
            "month_start",
            "year",
            "month",
            "station_network_residual",
        ],
    ].copy()

    annual = (
        residual_months.groupby(
            [
                "station_uid",
                "year",
            ],
            as_index=False,
        )
        .agg(
            valid_month_count=(
                "station_network_residual",
                "count",
            ),
            annual_median_network_residual=(
                "station_network_residual",
                "median",
            ),
            annual_mean_network_residual=(
                "station_network_residual",
                "mean",
            ),
            annual_minimum_network_residual=(
                "station_network_residual",
                "min",
            ),
            annual_maximum_network_residual=(
                "station_network_residual",
                "max",
            ),
        )
    )

    annual[
        "annual_residual_valid"
    ] = (
        annual[
            "valid_month_count"
        ].ge(
            minimum_valid_months_per_year
        )
    )

    annual[
        "annual_median_network_residual"
    ] = annual[
        "annual_median_network_residual"
    ].where(
        annual[
            "annual_residual_valid"
        ]
    )

    # Add all station-year combinations for visibility.
    year_minimum = int(
        monthly["year"].min()
    )

    year_maximum = int(
        monthly["year"].max()
    )

    station_year_grid = (
        pd.MultiIndex.from_product(
            [
                sorted(
                    fixed_station_uids
                ),
                range(
                    year_minimum,
                    year_maximum + 1,
                ),
            ],
            names=[
                "station_uid",
                "year",
            ],
        )
        .to_frame(index=False)
    )

    annual = station_year_grid.merge(
        annual,
        on=[
            "station_uid",
            "year",
        ],
        how="left",
        validate="one_to_one",
    )

    annual[
        "valid_month_count"
    ] = (
        annual[
            "valid_month_count"
        ]
        .fillna(0)
        .astype("int16")
    )

    annual[
        "annual_residual_valid"
    ] = (
        annual[
            "annual_residual_valid"
        ]
        .fillna(False)
        .astype(bool)
    )

    annual = annual.sort_values(
        [
            "station_uid",
            "year",
        ]
    ).reset_index(drop=True)

    annual.to_parquet(
        ANNUAL_OUTPUT_FILE,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    # -----------------------------------------------------
    # General breakpoint scan
    # -----------------------------------------------------

    breakpoint_rows: list[
        dict[str, Any]
    ] = []

    for station_uid in sorted(
        fixed_station_uids
    ):
        station_annual = annual.loc[
            annual[
                "station_uid"
            ].eq(station_uid)
        ].copy()

        seed_component = (
            zlib.crc32(
                station_uid.encode(
                    "utf-8"
                )
            )
            & 0xFFFFFFFF
        )

        break_stats = (
            scan_general_breakpoint(
                annual_data=station_annual,
                minimum_years_each_side=(
                    minimum_years_each_side
                ),
                minimum_scale=minimum_scale,
                bootstrap_repetitions=(
                    bootstrap_repetitions
                ),
                seed=(
                    bootstrap_seed
                    + seed_component
                    + 10
                ),
                moderate_absolute=(
                    moderate_break_absolute
                ),
                moderate_standardized=(
                    moderate_break_standardized
                ),
                high_absolute=(
                    high_break_absolute
                ),
                high_standardized=(
                    high_break_standardized
                ),
            )
        )

        metadata = fixed_lookup.loc[
            station_uid
        ]

        row = {
            "station_uid":
                station_uid,
            "station_id":
                metadata.get(
                    "station_id",
                    "",
                ),
            "station_name_display":
                metadata.get(
                    "station_name_display",
                    "",
                ),
        }

        row.update(break_stats)
        breakpoint_rows.append(row)

    breakpoint_summary = pd.DataFrame(
        breakpoint_rows
    ).sort_values(
        [
            "break_classification",
            "station_uid",
        ]
    ).reset_index(drop=True)

    breakpoint_summary.to_csv(
        REPORT_DIRECTORY
        / "step6b_general_breakpoint_summary.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Combined homogeneity classification
    # -----------------------------------------------------

    homogeneity = (
        fixed_network.merge(
            transition_summary,
            on=[
                "station_uid",
                "station_id",
                "station_name_display",
                "latitude",
                "longitude",
            ],
            how="left",
            validate="one_to_one",
        )
        .merge(
            breakpoint_summary[
                [
                    "station_uid",
                    "valid_year_count",
                    "best_break_after_year",
                    "break_shift_right_minus_left",
                    "standardized_break_shift",
                    "bootstrap_ci_lower",
                    "bootstrap_ci_upper",
                    "break_classification",
                ]
            ].rename(
                columns={
                    "bootstrap_ci_lower":
                        "break_bootstrap_ci_lower",
                    "bootstrap_ci_upper":
                        "break_bootstrap_ci_upper",
                }
            ),
            on="station_uid",
            how="left",
            validate="one_to_one",
        )
    )

    source_high = homogeneity[
        "source_transition_status"
    ].eq(
        "high_source_transition_signal"
    )

    source_moderate = homogeneity[
        "source_transition_status"
    ].eq(
        "moderate_source_transition_signal"
    )

    source_insufficient = homogeneity[
        "source_transition_status"
    ].eq(
        "insufficient_transition_data"
    )

    break_high = homogeneity[
        "break_classification"
    ].eq(
        "high_shift_signal"
    )

    break_moderate = homogeneity[
        "break_classification"
    ].eq(
        "moderate_shift_signal"
    )

    break_insufficient = homogeneity[
        "break_classification"
    ].eq(
        "insufficient_data"
    )

    homogeneity[
        "combined_homogeneity_status"
    ] = np.select(
        [
            source_high | break_high,
            source_moderate
            | break_moderate,
            source_insufficient
            & break_insufficient,
        ],
        [
            "high_priority_homogeneity_review",
            "moderate_priority_homogeneity_review",
            "insufficient_homogeneity_data",
        ],
        default="no_major_inhomogeneity_signal",
    )

    homogeneity[
        "automatic_adjustment_applied"
    ] = False

    homogeneity[
        "automatic_station_exclusion_applied"
    ] = False

    homogeneity[
        "recommended_step6c_action"
    ] = np.select(
        [
            homogeneity[
                "combined_homogeneity_status"
            ].eq(
                "high_priority_homogeneity_review"
            ),
            homogeneity[
                "combined_homogeneity_status"
            ].eq(
                "moderate_priority_homogeneity_review"
            ),
            homogeneity[
                "combined_homogeneity_status"
            ].eq(
                "insufficient_homogeneity_data"
            ),
        ],
        [
            (
                "retain_unadjusted; perform sensitivity "
                "test with station excluded"
            ),
            (
                "retain_unadjusted; inspect residual "
                "series and test sensitivity"
            ),
            (
                "retain_unadjusted; document "
                "insufficient homogeneity evidence"
            ),
        ],
        default=(
            "retain_unadjusted in analytical network"
        ),
    )

    homogeneity.to_csv(
        HOMOGENEITY_STATUS_FILE,
        index=False,
    )

    flagged_stations = homogeneity.loc[
        ~homogeneity[
            "combined_homogeneity_status"
        ].eq(
            "no_major_inhomogeneity_signal"
        )
    ].copy()

    flagged_stations.to_csv(
        REPORT_DIRECTORY
        / "step6b_flagged_stations.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Save monthly output
    # -----------------------------------------------------

    monthly = monthly.sort_values(
        [
            "station_uid",
            "month_start",
        ]
    ).reset_index(drop=True)

    monthly.to_parquet(
        MONTHLY_OUTPUT_FILE,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    # -----------------------------------------------------
    # Reports and validation
    # -----------------------------------------------------

    transition_status_counts = (
        transition_summary[
            "source_transition_status"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    break_status_counts = (
        breakpoint_summary[
            "break_classification"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    combined_status_counts = (
        homogeneity[
            "combined_homogeneity_status"
        ]
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
        "fixed_network_file":
            str(FIXED_NETWORK_FILE),
        "fixed_station_count":
            fixed_station_count,
        "monthly_station_rows":
            len(monthly),
        "valid_monthly_station_rows":
            int(
                monthly[
                    "monthly_valid"
                ].sum()
            ),
        "network_residual_available_rows":
            int(
                monthly[
                    "network_residual_available"
                ].sum()
            ),
        "annual_station_rows":
            len(annual),
        "source_transition_date":
            transition_date.date().isoformat(),
        "pre_period_start":
            pre_start.date().isoformat(),
        "pre_period_end":
            pre_end.date().isoformat(),
        "post_period_start":
            post_start.date().isoformat(),
        "post_period_end":
            post_end.date().isoformat(),
        "transition_status_counts":
            {
                str(key): int(value)
                for key, value
                in transition_status_counts.items()
            },
        "breakpoint_status_counts":
            {
                str(key): int(value)
                for key, value
                in break_status_counts.items()
            },
        "combined_status_counts":
            {
                str(key): int(value)
                for key, value
                in combined_status_counts.items()
            },
        "flagged_station_count":
            len(flagged_stations),
        "automatic_adjustment_applied":
            False,
        "automatic_station_exclusion_applied":
            False,
        "monthly_output":
            str(MONTHLY_OUTPUT_FILE),
        "annual_output":
            str(ANNUAL_OUTPUT_FILE),
        "homogeneity_status_file":
            str(HOMOGENEITY_STATUS_FILE),
    }

    (
        REPORT_DIRECTORY
        / "step6b_homogeneity_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monthly_schema = pq.ParquetFile(
        MONTHLY_OUTPUT_FILE
    ).schema_arrow

    annual_schema = pq.ParquetFile(
        ANNUAL_OUTPUT_FILE
    ).schema_arrow

    (
        ADMIN_DIRECTORY
        / "step6b_parquet_schema.json"
    ).write_text(
        json.dumps(
            {
                "monthly_schema": {
                    field.name: str(
                        field.type
                    )
                    for field
                    in monthly_schema
                },
                "annual_schema": {
                    field.name: str(
                        field.type
                    )
                    for field
                    in annual_schema
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "STEP 6B: SOURCE TRANSITION AND HOMOGENEITY",
        "=" * 50,
        (
            "Fixed-network stations assessed: "
            f"{fixed_station_count}"
        ),
        (
            "Monthly station records: "
            f"{len(monthly):,}"
        ),
        (
            "Valid monthly records: "
            f"{monthly['monthly_valid'].sum():,}"
        ),
        (
            "Network residual records: "
            f"{monthly['network_residual_available'].sum():,}"
        ),
        (
            "Source transition tested at: "
            f"{transition_date.date()}"
        ),
        (
            "Pre-transition window: "
            f"{pre_start.date()} to "
            f"{pre_end.date()}"
        ),
        (
            "Post-transition window: "
            f"{post_start.date()} to "
            f"{post_end.date()}"
        ),
        "",
        "Source-transition classifications:",
        *[
            f"- {key}: {value:,}"
            for key, value
            in transition_status_counts.items()
        ],
        "",
        "General-break classifications:",
        *[
            f"- {key}: {value:,}"
            for key, value
            in break_status_counts.items()
        ],
        "",
        "Combined homogeneity classifications:",
        *[
            f"- {key}: {value:,}"
            for key, value
            in combined_status_counts.items()
        ],
        "",
        (
            "Stations requiring additional attention: "
            f"{len(flagged_stations)}"
        ),
        "",
        "Automatic temperature adjustment applied: No",
        "Automatic station exclusion applied: No",
        "",
        f"Monthly output: {MONTHLY_OUTPUT_FILE}",
        f"Annual output: {ANNUAL_OUTPUT_FILE}",
        (
            "Station status output: "
            f"{HOMOGENEITY_STATUS_FILE}"
        ),
    ]

    (
        REPORT_DIRECTORY
        / "step6b_homogeneity_report.txt"
    ).write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(report_lines))

    failures: list[str] = []

    if (
        transition_summary[
            "station_uid"
        ].nunique()
        != fixed_station_count
    ):
        failures.append(
            "Transition summary does not contain "
            "every fixed station."
        )

    if (
        breakpoint_summary[
            "station_uid"
        ].nunique()
        != fixed_station_count
    ):
        failures.append(
            "Breakpoint summary does not contain "
            "every fixed station."
        )

    if (
        homogeneity[
            "station_uid"
        ].nunique()
        != fixed_station_count
    ):
        failures.append(
            "Homogeneity status table does not "
            "contain every fixed station."
        )

    if monthly.duplicated(
        subset=[
            "station_uid",
            "month_start",
        ]
    ).any():
        failures.append(
            "Duplicate station-month rows exist."
        )

    if annual.duplicated(
        subset=[
            "station_uid",
            "year",
        ]
    ).any():
        failures.append(
            "Duplicate station-year rows exist."
        )

    if homogeneity[
        "automatic_adjustment_applied"
    ].any():
        failures.append(
            "An automatic homogenization adjustment "
            "was incorrectly applied."
        )

    if homogeneity[
        "automatic_station_exclusion_applied"
    ].any():
        failures.append(
            "A station was incorrectly excluded."
        )

    if failures:
        raise SystemExit(
            "\nSTEP 6B FAILED:\n- "
            + "\n- ".join(failures)
        )

    print("\nSTEP 6B PASSED.")


if __name__ == "__main__":
    main()
