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
    / "temperature_daily_cleaned_stage1.parquet"
)

RULE_FILE = (
    PROJECT_ROOT
    / "00_admin"
    / "step5a_qc_rules.yaml"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "temperature_daily_qc_stage2_temporal.parquet"
)

REFERENCE_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "step5a_station_calendar_climatology.parquet"
)

REPORT_DIRECTORY = PROJECT_ROOT / "05_qc_reports"
ADMIN_DIRECTORY = PROJECT_ROOT / "00_admin"

EXPECTED_STATIONS = 48


def load_rules() -> dict[str, Any]:
    if not RULE_FILE.exists():
        raise FileNotFoundError(
            f"Rule file not found: {RULE_FILE}"
        )

    with RULE_FILE.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        rules = yaml.safe_load(file_handle)

    return rules


def calendar_day_index(
    month: pd.Series,
    day: pd.Series,
) -> pd.Series:
    """
    Map every month-day to the leap-year calendar of 2000.

    This produces day indices 1-366 and allows February 29
    to be handled explicitly.
    """
    reference_dates = pd.to_datetime(
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

    return reference_dates.dt.dayofyear.astype(
        "int16"
    )


def baseline_coverage(
    data: pd.DataFrame,
    value_column: str,
    prefix: str,
    baseline_start: pd.Timestamp,
    baseline_end: pd.Timestamp,
    minimum_years: int,
) -> pd.DataFrame:
    baseline = data.loc[
        data["date"].between(
            baseline_start,
            baseline_end,
        )
        & data[value_column].notna(),
        [
            "station_uid",
            "date",
            "year",
            value_column,
        ],
    ].copy()

    if baseline.empty:
        return pd.DataFrame(
            columns=[
                "station_uid",
                f"{prefix}_baseline_first_date",
                f"{prefix}_baseline_last_date",
                f"{prefix}_baseline_observations",
                f"{prefix}_baseline_years",
                f"{prefix}_baseline_eligible",
            ]
        )

    coverage = (
        baseline.groupby(
            "station_uid",
            as_index=False,
        )
        .agg(
            baseline_first_date=("date", "min"),
            baseline_last_date=("date", "max"),
            baseline_observations=(
                value_column,
                "count",
            ),
            baseline_years=("year", "nunique"),
        )
    )

    coverage[
        "baseline_eligible"
    ] = coverage[
        "baseline_years"
    ].ge(minimum_years)

    coverage = coverage.rename(
        columns={
            "baseline_first_date":
                f"{prefix}_baseline_first_date",
            "baseline_last_date":
                f"{prefix}_baseline_last_date",
            "baseline_observations":
                f"{prefix}_baseline_observations",
            "baseline_years":
                f"{prefix}_baseline_years",
            "baseline_eligible":
                f"{prefix}_baseline_eligible",
        }
    )

    return coverage


def build_calendar_climatology(
    data: pd.DataFrame,
    value_column: str,
    prefix: str,
    baseline_start: pd.Timestamp,
    baseline_end: pd.Timestamp,
    window_half_width: int,
    minimum_years: int,
    minimum_observations: int,
    minimum_scale: float,
) -> pd.DataFrame:
    """
    Calculate robust station-calendar-day statistics using
    a circular calendar window around each target day.
    """
    baseline = data.loc[
        data["date"].between(
            baseline_start,
            baseline_end,
        )
        & data[value_column].notna(),
        [
            "station_uid",
            "year",
            "clim_day",
            value_column,
        ],
    ].copy()

    rows: list[dict[str, Any]] = []

    for station_uid, group in baseline.groupby(
        "station_uid",
        sort=True,
    ):
        baseline_years = int(
            group["year"].nunique()
        )

        if baseline_years < minimum_years:
            continue

        values_by_day: dict[int, np.ndarray] = {}

        for clim_day, day_group in group.groupby(
            "clim_day"
        ):
            values_by_day[int(clim_day)] = (
                day_group[value_column]
                .dropna()
                .astype(float)
                .to_numpy()
            )

        for target_day in range(1, 367):
            window_arrays: list[np.ndarray] = []

            for offset in range(
                -window_half_width,
                window_half_width + 1,
            ):
                window_day = (
                    (
                        target_day
                        - 1
                        + offset
                    )
                    % 366
                ) + 1

                values = values_by_day.get(
                    window_day
                )

                if (
                    values is not None
                    and len(values) > 0
                ):
                    window_arrays.append(values)

            if not window_arrays:
                continue

            window_values = np.concatenate(
                window_arrays
            )

            window_values = window_values[
                np.isfinite(window_values)
            ]

            sample_size = len(window_values)

            if sample_size < minimum_observations:
                continue

            median = float(
                np.median(window_values)
            )

            mad = float(
                np.median(
                    np.abs(
                        window_values - median
                    )
                )
            )

            robust_scale = max(
                1.4826 * mad,
                minimum_scale,
            )

            rows.append(
                {
                    "station_uid":
                        station_uid,
                    "clim_day":
                        target_day,
                    f"{prefix}_clim_n":
                        sample_size,
                    f"{prefix}_clim_years":
                        baseline_years,
                    f"{prefix}_clim_median":
                        median,
                    f"{prefix}_clim_mad":
                        mad,
                    f"{prefix}_clim_scale":
                        robust_scale,
                    f"{prefix}_clim_p01":
                        float(
                            np.quantile(
                                window_values,
                                0.01,
                            )
                        ),
                    f"{prefix}_clim_p99":
                        float(
                            np.quantile(
                                window_values,
                                0.99,
                            )
                        ),
                }
            )

    return pd.DataFrame(rows)


def persistence_flags(
    data: pd.DataFrame,
    value_column: str,
    minimum_run_days: int,
    rounding_decimals: int,
) -> tuple[pd.Series, pd.Series]:
    """
    Mark consecutive runs containing the same rounded value.
    Missing values and date gaps break the run.
    """
    flags = pd.Series(
        False,
        index=data.index,
        dtype=bool,
    )

    run_lengths = pd.Series(
        0,
        index=data.index,
        dtype="int16",
    )

    for _, group in data.groupby(
        "station_uid",
        sort=False,
    ):
        ordered = group.sort_values("date")

        values = (
            ordered[value_column]
            .round(rounding_decimals)
        )

        date_gap = (
            ordered["date"]
            .diff()
            .dt.days
            .ne(1)
        )

        value_change = (
            values.ne(values.shift())
            | values.isna()
            | values.shift().isna()
        )

        new_run = date_gap | value_change

        run_id = new_run.cumsum()

        sizes = (
            ordered.groupby(run_id)[
                value_column
            ]
            .transform("size")
            .astype("int16")
        )

        valid = values.notna()

        ordered_flags = (
            valid
            & sizes.ge(minimum_run_days)
        )

        flags.loc[
            ordered.index
        ] = ordered_flags.to_numpy()

        run_lengths.loc[
            ordered.index
        ] = np.where(
            valid,
            sizes,
            0,
        ).astype("int16")

    return flags, run_lengths


def make_flag_codes(
    data: pd.DataFrame,
    flag_definitions: list[
        tuple[str, str]
    ],
) -> tuple[pd.Series, pd.Series]:
    codes = np.full(
        len(data),
        "",
        dtype=object,
    )

    counts = np.zeros(
        len(data),
        dtype=np.int16,
    )

    for code, column in flag_definitions:
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


def main() -> None:
    for required_file in [
        INPUT_FILE,
        RULE_FILE,
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

    rules = load_rules()

    reference_rules = rules[
        "reference_climatology"
    ]

    outlier_rules = rules[
        "climatological_outlier"
    ]

    temporal_rules = rules[
        "temporal_consistency"
    ]

    persistence_rules = rules[
        "persistence"
    ]

    baseline_start = pd.Timestamp(
        reference_rules["start_date"]
    )

    baseline_end = pd.Timestamp(
        reference_rules["end_date"]
    )

    window_half_width = int(
        reference_rules[
            "calendar_window_half_width_days"
        ]
    )

    minimum_baseline_years = int(
        reference_rules[
            "minimum_baseline_years"
        ]
    )

    minimum_window_observations = int(
        reference_rules[
            "minimum_window_observations"
        ]
    )

    minimum_scale = float(
        reference_rules[
            "minimum_robust_scale_celsius"
        ]
    )

    robust_z_threshold = float(
        outlier_rules[
            "robust_z_threshold"
        ]
    )

    tmin_absolute_anomaly_threshold = float(
        outlier_rules[
            "tmin_minimum_absolute_anomaly_celsius"
        ]
    )

    tmax_absolute_anomaly_threshold = float(
        outlier_rules[
            "tmax_minimum_absolute_anomaly_celsius"
        ]
    )

    tmin_large_change_threshold = float(
        temporal_rules[
            "tmin_large_daily_change_celsius"
        ]
    )

    tmax_large_change_threshold = float(
        temporal_rules[
            "tmax_large_daily_change_celsius"
        ]
    )

    tmin_spike_threshold = float(
        temporal_rules[
            "tmin_isolated_spike_difference_celsius"
        ]
    )

    tmax_spike_threshold = float(
        temporal_rules[
            "tmax_isolated_spike_difference_celsius"
        ]
    )

    neighbour_consistency_threshold = float(
        temporal_rules[
            "neighbour_day_consistency_celsius"
        ]
    )

    minimum_identical_run_days = int(
        persistence_rules[
            "minimum_identical_run_days"
        ]
    )

    rounding_decimals = int(
        persistence_rules[
            "rounding_decimals"
        ]
    )

    data = pd.read_parquet(INPUT_FILE)

    input_rows = len(data)

    data["date"] = pd.to_datetime(
        data["date"]
    )

    data = data.sort_values(
        [
            "station_uid",
            "date",
        ]
    ).reset_index(drop=True)

    if data.duplicated(
        subset=[
            "station_uid",
            "date",
        ]
    ).any():
        raise ValueError(
            "Input contains duplicate station-date keys."
        )

    station_count = int(
        data["station_uid"].nunique()
    )

    if station_count != EXPECTED_STATIONS:
        raise ValueError(
            f"Expected {EXPECTED_STATIONS} stations; "
            f"found {station_count}."
        )

    input_minimum_date = data["date"].min()
    input_maximum_date = data["date"].max()

    data["clim_day"] = calendar_day_index(
        data["month"],
        data["day"],
    )

    # -----------------------------------------------------
    # Baseline coverage and climatology
    # -----------------------------------------------------

    tmin_coverage = baseline_coverage(
        data,
        "tmin_cleaned_stage1",
        "tmin",
        baseline_start,
        baseline_end,
        minimum_baseline_years,
    )

    tmax_coverage = baseline_coverage(
        data,
        "tmax_cleaned_stage1",
        "tmax",
        baseline_start,
        baseline_end,
        minimum_baseline_years,
    )

    station_identity = (
        data[
            [
                "station_uid",
                "station_id",
                "station_name_display",
                "metadata_status",
            ]
        ]
        .drop_duplicates(
            subset=["station_uid"]
        )
    )

    coverage = (
        station_identity
        .merge(
            tmin_coverage,
            on="station_uid",
            how="left",
            validate="one_to_one",
        )
        .merge(
            tmax_coverage,
            on="station_uid",
            how="left",
            validate="one_to_one",
        )
    )

    for column in [
        "tmin_baseline_eligible",
        "tmax_baseline_eligible",
    ]:
        coverage[column] = (
            coverage[column]
            .fillna(False)
            .astype(bool)
        )

    coverage.to_csv(
        REPORT_DIRECTORY
        / "step5a_baseline_coverage.csv",
        index=False,
    )

    tmin_reference = (
        build_calendar_climatology(
            data,
            "tmin_cleaned_stage1",
            "tmin",
            baseline_start,
            baseline_end,
            window_half_width,
            minimum_baseline_years,
            minimum_window_observations,
            minimum_scale,
        )
    )

    tmax_reference = (
        build_calendar_climatology(
            data,
            "tmax_cleaned_stage1",
            "tmax",
            baseline_start,
            baseline_end,
            window_half_width,
            minimum_baseline_years,
            minimum_window_observations,
            minimum_scale,
        )
    )

    reference = tmin_reference.merge(
        tmax_reference,
        on=[
            "station_uid",
            "clim_day",
        ],
        how="outer",
        validate="one_to_one",
    )

    reference = reference.sort_values(
        [
            "station_uid",
            "clim_day",
        ]
    ).reset_index(drop=True)

    if reference.empty:
        raise ValueError(
            "No reference climatology was generated."
        )

    reference.to_parquet(
        REFERENCE_FILE,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    data = data.merge(
        reference,
        on=[
            "station_uid",
            "clim_day",
        ],
        how="left",
        validate="many_to_one",
    )

    # -----------------------------------------------------
    # Climatological anomaly checks
    # -----------------------------------------------------

    data["tmin_clim_anomaly"] = (
        data["tmin_cleaned_stage1"]
        - data["tmin_clim_median"]
    ).astype("float32")

    data["tmax_clim_anomaly"] = (
        data["tmax_cleaned_stage1"]
        - data["tmax_clim_median"]
    ).astype("float32")

    data["tmin_clim_robust_z"] = (
        data["tmin_clim_anomaly"]
        / data["tmin_clim_scale"]
    ).astype("float32")

    data["tmax_clim_robust_z"] = (
        data["tmax_clim_anomaly"]
        / data["tmax_clim_scale"]
    ).astype("float32")

    data[
        "qc_tmin_climatology_unavailable"
    ] = (
        data["tmin_cleaned_stage1"].notna()
        & data["tmin_clim_median"].isna()
    )

    data[
        "qc_tmax_climatology_unavailable"
    ] = (
        data["tmax_cleaned_stage1"].notna()
        & data["tmax_clim_median"].isna()
    )

    data[
        "qc_tmin_climatological_outlier"
    ] = (
        data["tmin_cleaned_stage1"].notna()
        & data["tmin_clim_median"].notna()
        & data[
            "tmin_clim_anomaly"
        ].abs().ge(
            tmin_absolute_anomaly_threshold
        )
        & data[
            "tmin_clim_robust_z"
        ].abs().ge(
            robust_z_threshold
        )
    )

    data[
        "qc_tmax_climatological_outlier"
    ] = (
        data["tmax_cleaned_stage1"].notna()
        & data["tmax_clim_median"].notna()
        & data[
            "tmax_clim_anomaly"
        ].abs().ge(
            tmax_absolute_anomaly_threshold
        )
        & data[
            "tmax_clim_robust_z"
        ].abs().ge(
            robust_z_threshold
        )
    )

    # -----------------------------------------------------
    # Temporal neighbours and daily changes
    # -----------------------------------------------------

    station_group = data.groupby(
        "station_uid",
        sort=False,
    )

    data["previous_date"] = (
        station_group["date"].shift(1)
    )

    data["next_date"] = (
        station_group["date"].shift(-1)
    )

    data["previous_tmin"] = (
        station_group[
            "tmin_cleaned_stage1"
        ].shift(1)
    )

    data["next_tmin"] = (
        station_group[
            "tmin_cleaned_stage1"
        ].shift(-1)
    )

    data["previous_tmax"] = (
        station_group[
            "tmax_cleaned_stage1"
        ].shift(1)
    )

    data["next_tmax"] = (
        station_group[
            "tmax_cleaned_stage1"
        ].shift(-1)
    )

    previous_consecutive = (
        data["date"]
        .sub(data["previous_date"])
        .dt.days
        .eq(1)
    )

    next_consecutive = (
        data["next_date"]
        .sub(data["date"])
        .dt.days
        .eq(1)
    )

    data["tmin_change_from_previous"] = (
        data["tmin_cleaned_stage1"]
        - data["previous_tmin"]
    ).astype("float32")

    data["tmax_change_from_previous"] = (
        data["tmax_cleaned_stage1"]
        - data["previous_tmax"]
    ).astype("float32")

    data["qc_tmin_large_daily_change"] = (
        previous_consecutive
        & data["tmin_cleaned_stage1"].notna()
        & data["previous_tmin"].notna()
        & data[
            "tmin_change_from_previous"
        ].abs().ge(
            tmin_large_change_threshold
        )
    )

    data["qc_tmax_large_daily_change"] = (
        previous_consecutive
        & data["tmax_cleaned_stage1"].notna()
        & data["previous_tmax"].notna()
        & data[
            "tmax_change_from_previous"
        ].abs().ge(
            tmax_large_change_threshold
        )
    )

    data["qc_tmin_isolated_spike"] = (
        previous_consecutive
        & next_consecutive
        & data["tmin_cleaned_stage1"].notna()
        & data["previous_tmin"].notna()
        & data["next_tmin"].notna()
        & (
            data["tmin_cleaned_stage1"]
            - data["previous_tmin"]
        ).abs().ge(
            tmin_spike_threshold
        )
        & (
            data["tmin_cleaned_stage1"]
            - data["next_tmin"]
        ).abs().ge(
            tmin_spike_threshold
        )
        & (
            data["previous_tmin"]
            - data["next_tmin"]
        ).abs().le(
            neighbour_consistency_threshold
        )
    )

    data["qc_tmax_isolated_spike"] = (
        previous_consecutive
        & next_consecutive
        & data["tmax_cleaned_stage1"].notna()
        & data["previous_tmax"].notna()
        & data["next_tmax"].notna()
        & (
            data["tmax_cleaned_stage1"]
            - data["previous_tmax"]
        ).abs().ge(
            tmax_spike_threshold
        )
        & (
            data["tmax_cleaned_stage1"]
            - data["next_tmax"]
        ).abs().ge(
            tmax_spike_threshold
        )
        & (
            data["previous_tmax"]
            - data["next_tmax"]
        ).abs().le(
            neighbour_consistency_threshold
        )
    )

    # -----------------------------------------------------
    # Persistence checks
    # -----------------------------------------------------

    (
        data["qc_tmin_persistence"],
        data["tmin_identical_run_length"],
    ) = persistence_flags(
        data,
        "tmin_cleaned_stage1",
        minimum_identical_run_days,
        rounding_decimals,
    )

    (
        data["qc_tmax_persistence"],
        data["tmax_identical_run_length"],
    ) = persistence_flags(
        data,
        "tmax_cleaned_stage1",
        minimum_identical_run_days,
        rounding_decimals,
    )

    data["qc_step4b_screen_carryover"] = (
        data[
            "step4b_post_screen_flag"
        ]
        .fillna(False)
        .astype(bool)
    )

    # -----------------------------------------------------
    # Combined Step 5A assessment
    # -----------------------------------------------------

    substantive_flags = [
        "qc_tmin_climatological_outlier",
        "qc_tmax_climatological_outlier",
        "qc_tmin_large_daily_change",
        "qc_tmax_large_daily_change",
        "qc_tmin_isolated_spike",
        "qc_tmax_isolated_spike",
        "qc_tmin_persistence",
        "qc_tmax_persistence",
        "qc_step4b_screen_carryover",
    ]

    data[
        "step5a_candidate_review_required"
    ] = data[
        substantive_flags
    ].any(axis=1)

    high_priority = (
        (
            data[
                "qc_tmin_climatological_outlier"
            ]
            & data[
                "qc_tmin_isolated_spike"
            ]
        )
        | (
            data[
                "qc_tmax_climatological_outlier"
            ]
            & data[
                "qc_tmax_isolated_spike"
            ]
        )
        | (
            data[
                "qc_step4b_screen_carryover"
            ]
            & (
                data[
                    "qc_tmin_climatological_outlier"
                ]
                | data[
                    "qc_tmax_climatological_outlier"
                ]
            )
        )
    )

    baseline_unavailable = (
        data[
            "qc_tmin_climatology_unavailable"
        ]
        | data[
            "qc_tmax_climatology_unavailable"
        ]
    )

    data["step5a_priority"] = np.select(
        [
            high_priority,
            data[
                "step5a_candidate_review_required"
            ],
            baseline_unavailable,
        ],
        [
            "high",
            "review",
            "baseline_unavailable",
        ],
        default="none",
    )

    data["step5a_status"] = np.select(
        [
            data[
                "step5a_candidate_review_required"
            ],
            baseline_unavailable,
        ],
        [
            "candidate_for_spatial_qc",
            "pass_temporal_baseline_unavailable",
        ],
        default="pass_step5a",
    )

    flag_definitions = [
        (
            "TMIN_CLIM_OUTLIER",
            "qc_tmin_climatological_outlier",
        ),
        (
            "TMAX_CLIM_OUTLIER",
            "qc_tmax_climatological_outlier",
        ),
        (
            "TMIN_LARGE_DAILY_CHANGE",
            "qc_tmin_large_daily_change",
        ),
        (
            "TMAX_LARGE_DAILY_CHANGE",
            "qc_tmax_large_daily_change",
        ),
        (
            "TMIN_ISOLATED_SPIKE",
            "qc_tmin_isolated_spike",
        ),
        (
            "TMAX_ISOLATED_SPIKE",
            "qc_tmax_isolated_spike",
        ),
        (
            "TMIN_PERSISTENCE",
            "qc_tmin_persistence",
        ),
        (
            "TMAX_PERSISTENCE",
            "qc_tmax_persistence",
        ),
        (
            "STEP4B_SCREEN_CARRYOVER",
            "qc_step4b_screen_carryover",
        ),
        (
            "TMIN_NO_BASELINE",
            "qc_tmin_climatology_unavailable",
        ),
        (
            "TMAX_NO_BASELINE",
            "qc_tmax_climatology_unavailable",
        ),
    ]

    (
        data["step5a_flag_codes"],
        data["step5a_flag_count"],
    ) = make_flag_codes(
        data,
        flag_definitions,
    )

    # -----------------------------------------------------
    # Validate that values have not changed
    # -----------------------------------------------------

    if not data[
        "tmin_cleaned_stage1"
    ].equals(
        pd.read_parquet(
            INPUT_FILE,
            columns=[
                "tmin_cleaned_stage1"
            ],
        )[
            "tmin_cleaned_stage1"
        ]
    ):
        raise ValueError(
            "Tmin values changed during Step 5A."
        )

    if not data[
        "tmax_cleaned_stage1"
    ].equals(
        pd.read_parquet(
            INPUT_FILE,
            columns=[
                "tmax_cleaned_stage1"
            ],
        )[
            "tmax_cleaned_stage1"
        ]
    ):
        raise ValueError(
            "Tmax values changed during Step 5A."
        )

    if data.duplicated(
        subset=[
            "station_uid",
            "date",
        ]
    ).any():
        raise ValueError(
            "Duplicate keys were introduced."
        )

    if len(data) != input_rows:
        raise ValueError(
            "Output row count differs from input."
        )

    if data["date"].min() != input_minimum_date:
        raise ValueError(
            "Minimum date changed."
        )

    if data["date"].max() != input_maximum_date:
        raise ValueError(
            "Maximum date changed."
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
        "tmin_clim_median",
        "tmax_clim_median",
        "tmin_clim_anomaly",
        "tmax_clim_anomaly",
        "tmin_clim_robust_z",
        "tmax_clim_robust_z",
        "previous_tmin",
        "next_tmin",
        "previous_tmax",
        "next_tmax",
        "tmin_change_from_previous",
        "tmax_change_from_previous",
        "tmin_identical_run_length",
        "tmax_identical_run_length",
        "step5a_flag_codes",
        "step5a_flag_count",
        "step5a_priority",
        "step5a_status",
        "old_source_rows",
        "new_source_cell",
        "step4b_decision",
    ]

    climatological_candidates = data.loc[
        data[
            "qc_tmin_climatological_outlier"
        ]
        | data[
            "qc_tmax_climatological_outlier"
        ]
    ].copy()

    climatological_candidates[
        report_columns
    ].to_csv(
        REPORT_DIRECTORY
        / "step5a_climatological_outlier_candidates.csv",
        index=False,
    )

    temporal_candidates = data.loc[
        data[
            "qc_tmin_large_daily_change"
        ]
        | data[
            "qc_tmax_large_daily_change"
        ]
        | data[
            "qc_tmin_isolated_spike"
        ]
        | data[
            "qc_tmax_isolated_spike"
        ]
    ].copy()

    temporal_candidates[
        report_columns
    ].to_csv(
        REPORT_DIRECTORY
        / "step5a_temporal_change_candidates.csv",
        index=False,
    )

    persistence_candidates = data.loc[
        data["qc_tmin_persistence"]
        | data["qc_tmax_persistence"]
    ].copy()

    persistence_candidates[
        report_columns
    ].to_csv(
        REPORT_DIRECTORY
        / "step5a_persistence_candidates.csv",
        index=False,
    )

    all_candidates = data.loc[
        data[
            "step5a_candidate_review_required"
        ]
    ].copy()

    all_candidates[
        report_columns
    ].to_csv(
        REPORT_DIRECTORY
        / "step5a_all_candidate_records.csv",
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
            first_date=("date", "min"),
            last_date=("date", "max"),
            total_records=("date", "size"),
            numeric_tmin=(
                "tmin_cleaned_stage1",
                "count",
            ),
            numeric_tmax=(
                "tmax_cleaned_stage1",
                "count",
            ),
            tmin_clim_outliers=(
                "qc_tmin_climatological_outlier",
                "sum",
            ),
            tmax_clim_outliers=(
                "qc_tmax_climatological_outlier",
                "sum",
            ),
            tmin_large_changes=(
                "qc_tmin_large_daily_change",
                "sum",
            ),
            tmax_large_changes=(
                "qc_tmax_large_daily_change",
                "sum",
            ),
            tmin_isolated_spikes=(
                "qc_tmin_isolated_spike",
                "sum",
            ),
            tmax_isolated_spikes=(
                "qc_tmax_isolated_spike",
                "sum",
            ),
            tmin_persistence_records=(
                "qc_tmin_persistence",
                "sum",
            ),
            tmax_persistence_records=(
                "qc_tmax_persistence",
                "sum",
            ),
            candidate_records=(
                "step5a_candidate_review_required",
                "sum",
            ),
        )
        .reset_index()
        .sort_values("station_uid")
    )

    station_summary.to_csv(
        REPORT_DIRECTORY
        / "step5a_station_qc_summary.csv",
        index=False,
    )

    flag_counts = {
        column: int(
            data[column].sum()
        )
        for _, column in flag_definitions
    }

    priority_counts = (
        data["step5a_priority"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    status_counts = (
        data["step5a_status"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    # Save after validation.
    data.to_parquet(
        OUTPUT_FILE,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    output_parquet = pq.ParquetFile(
        OUTPUT_FILE
    )

    output_rows = (
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
        "reference_file":
            str(REFERENCE_FILE),
        "input_rows":
            input_rows,
        "output_rows":
            output_rows,
        "unique_stations":
            station_count,
        "minimum_date":
            data["date"].min().date().isoformat(),
        "maximum_date":
            data["date"].max().date().isoformat(),
        "baseline_start":
            baseline_start.date().isoformat(),
        "baseline_end":
            baseline_end.date().isoformat(),
        "tmin_baseline_eligible_stations":
            int(
                coverage[
                    "tmin_baseline_eligible"
                ].sum()
            ),
        "tmax_baseline_eligible_stations":
            int(
                coverage[
                    "tmax_baseline_eligible"
                ].sum()
            ),
        "reference_rows":
            len(reference),
        "candidate_records":
            len(all_candidates),
        "climatological_candidate_records":
            len(climatological_candidates),
        "temporal_candidate_records":
            len(temporal_candidates),
        "persistence_candidate_records":
            len(persistence_candidates),
        "flag_counts":
            flag_counts,
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
        "values_changed":
            False,
        "output_size_bytes":
            OUTPUT_FILE.stat().st_size,
    }

    (
        REPORT_DIRECTORY
        / "step5a_qc_summary.json"
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
        / "step5a_parquet_schema.json"
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
        "STEP 5A: CLIMATOLOGICAL AND TEMPORAL QC",
        "=" * 47,
        f"Input rows: {input_rows:,}",
        f"Output rows: {output_rows:,}",
        f"Unique stations: {station_count}",
        (
            "Date range: "
            f"{data['date'].min().date()} to "
            f"{data['date'].max().date()}"
        ),
        (
            "Reference period: "
            f"{baseline_start.date()} to "
            f"{baseline_end.date()}"
        ),
        (
            "Tmin baseline-eligible stations: "
            f"{coverage['tmin_baseline_eligible'].sum():,}"
        ),
        (
            "Tmax baseline-eligible stations: "
            f"{coverage['tmax_baseline_eligible'].sum():,}"
        ),
        (
            "Calendar climatology rows: "
            f"{len(reference):,}"
        ),
        (
            "All Step 5A candidate records: "
            f"{len(all_candidates):,}"
        ),
        (
            "Climatological candidate records: "
            f"{len(climatological_candidates):,}"
        ),
        (
            "Temporal-change candidate records: "
            f"{len(temporal_candidates):,}"
        ),
        (
            "Persistence candidate records: "
            f"{len(persistence_candidates):,}"
        ),
        "",
        "Flag counts:",
        *[
            f"- {key}: {value:,}"
            for key, value
            in flag_counts.items()
        ],
        "",
        "Priorities:",
        *[
            f"- {key}: {value:,}"
            for key, value
            in priority_counts.items()
        ],
        "",
        "No temperature value was changed.",
        "",
        f"Output: {OUTPUT_FILE}",
    ]

    (
        REPORT_DIRECTORY
        / "step5a_qc_report.txt"
    ).write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(report_lines))

    failures: list[str] = []

    if output_rows != input_rows:
        failures.append(
            "Output row count differs from input."
        )

    if station_count != EXPECTED_STATIONS:
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

    if reference.empty:
        failures.append(
            "Reference climatology is empty."
        )

    if failures:
        raise SystemExit(
            "\nSTEP 5A FAILED:\n- "
            + "\n- ".join(failures)
        )

    print("\nSTEP 5A PASSED.")


if __name__ == "__main__":
    main()
