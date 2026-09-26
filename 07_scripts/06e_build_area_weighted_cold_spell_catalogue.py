from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
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
    / "step7d_area_weighted_event_policy.yaml"
)

WEIGHT_FILE = (
    PROJECT_ROOT
    / "02_metadata"
    / "step7d_primary_station_area_weights.csv"
)

VORONOI_FILE = (
    PROJECT_ROOT
    / "02_metadata"
    / "step7d_primary_station_voronoi_weights.gpkg"
)

BOUNDARY_FILE = (
    PROJECT_ROOT
    / "02_metadata"
    / "step7d_bangladesh_boundary.gpkg"
)

DAILY_OUTPUT_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "step7d_area_weighted_daily_cold_diagnostics.parquet"
)

EVENT_READY_FILE = (
    PROJECT_ROOT
    / "04_clean_data"
    / "temperature_daily_djf_with_area_weighted_event_membership.parquet"
)

EVENT_DIRECTORY = PROJECT_ROOT / "06_events"

EVENT_CATALOGUE_FILE = (
    EVENT_DIRECTORY
    / "step7d_primary_area_weighted_cold_spell_catalogue.csv"
)

EVENT_DAY_FILE = (
    EVENT_DIRECTORY
    / "step7d_primary_area_weighted_cold_spell_days.csv"
)

PARTICIPATION_FILE = (
    EVENT_DIRECTORY
    / "step7d_primary_area_weighted_station_participation.csv"
)

FOOTPRINT_FILE = (
    EVENT_DIRECTORY
    / "step7d_primary_area_weighted_event_footprints.gpkg"
)

REPORT_DIRECTORY = PROJECT_ROOT / "05_qc_reports"
ADMIN_DIRECTORY = PROJECT_ROOT / "00_admin"

WINTER_SUMMARY_FILE = (
    REPORT_DIRECTORY
    / "step7d_area_weighted_winter_summary.csv"
)

VALIDATION_FILE = (
    REPORT_DIRECTORY
    / "step7d_area_weighted_validation_issues.csv"
)

SUMMARY_FILE = (
    REPORT_DIRECTORY
    / "step7d_area_weighted_event_summary.json"
)

REPORT_FILE = (
    REPORT_DIRECTORY
    / "step7d_area_weighted_event_report.txt"
)

SCHEMA_FILE = (
    ADMIN_DIRECTORY
    / "step7d_parquet_schema.json"
)

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


def parse_boolean(values: pd.Series) -> pd.Series:
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

    if parsed.isna().any():
        unknown = sorted(
            normalized.loc[
                parsed.isna()
            ].unique().tolist()
        )

        raise ValueError(
            "Unrecognized Boolean values: "
            f"{unknown}"
        )

    return parsed.astype(bool)


def union_geometry(
    geometries: gpd.GeoSeries,
):
    if hasattr(
        geometries,
        "union_all",
    ):
        return geometries.union_all()

    return geometries.unary_union


def values_equal_with_nan(
    first: pd.Series,
    second: pd.Series,
) -> bool:
    return bool(
        np.allclose(
            pd.to_numeric(
                first,
                errors="coerce",
            ).to_numpy(dtype=float),
            pd.to_numeric(
                second,
                errors="coerce",
            ).to_numpy(dtype=float),
            atol=1e-7,
            rtol=0.0,
            equal_nan=True,
        )
    )


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    return pd.Series(
        np.where(
            denominator.gt(0),
            numerator / denominator,
            np.nan,
        ),
        index=numerator.index,
        dtype="float64",
    )


def assign_events(
    daily: pd.DataFrame,
    minimum_duration: int,
) -> pd.DataFrame:
    result = daily.sort_values(
        [
            "winter_start_year",
            "date",
        ]
    ).copy()

    result["candidate_run_length"] = 0
    result["area_weighted_event_active"] = False

    result[
        "area_weighted_event_id"
    ] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )

    result[
        "area_weighted_event_day_index"
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
            "regional_p10_cold_day"
        ].astype(bool)

        previous_candidate = candidate.shift(
            fill_value=False
        )

        consecutive_date = (
            winter["date"]
            .diff()
            .dt.days
            .eq(1)
            .fillna(False)
        )

        run_start = (
            candidate
            & (
                ~previous_candidate
                | ~consecutive_date
            )
        )

        run_number = run_start.cumsum()
        event_sequence = 0

        candidate_rows = winter.loc[
            candidate
        ]

        for _, run in candidate_rows.groupby(
            run_number.loc[candidate]
        ):
            indices = run.index
            run_length = len(indices)

            result.loc[
                indices,
                "candidate_run_length",
            ] = run_length

            if run_length < minimum_duration:
                continue

            event_sequence += 1

            event_id = (
                f"BDP10A_{int(winter_start_year)}_"
                f"{event_sequence:02d}"
            )

            result.loc[
                indices,
                "area_weighted_event_active",
            ] = True

            result.loc[
                indices,
                "area_weighted_event_id",
            ] = event_id

            result.loc[
                indices,
                "area_weighted_event_day_index",
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


def main() -> None:
    required_files = [
        INPUT_FILE,
        POLICY_FILE,
        WEIGHT_FILE,
        VORONOI_FILE,
        BOUNDARY_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    for directory in [
        DAILY_OUTPUT_FILE.parent,
        EVENT_READY_FILE.parent,
        EVENT_DIRECTORY,
        REPORT_DIRECTORY,
        ADMIN_DIRECTORY,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    policy = load_yaml(POLICY_FILE)

    network_rules = policy["network"]
    availability_rules = policy[
        "daily_data_availability"
    ]
    regional_rules = policy[
        "regional_cold_day"
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

    eligibility_column = str(
        network_rules[
            "percentile_eligibility_column"
        ]
    )

    cold_column = str(
        policy[
            "primary_station_condition"
        ]["column"]
    )

    p05_column = str(
        policy[
            "severe_station_condition"
        ]["threshold_column"]
    )

    minimum_observed_area = float(
        availability_rules[
            "minimum_observed_national_area_fraction"
        ]
    )

    minimum_observed_stations = int(
        availability_rules[
            "minimum_observed_station_count"
        ]
    )

    minimum_cold_area = float(
        regional_rules[
            "minimum_cold_national_area_fraction"
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
            "Primary events must not merge gaps."
        )

    if bool(
        event_rules[
            "allow_cross_winter_events"
        ]
    ):
        raise ValueError(
            "Primary events must not cross winters."
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
        TEMPERATURE_COLUMN,
        "station_p10_threshold_analysis",
        p05_column,
        member_column,
        usable_column,
        eligibility_column,
        cold_column,
        "bmd_absolute_category_raw",
        "bmd_absolute_severity_rank_raw",
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

    for column in [
        member_column,
        usable_column,
        eligibility_column,
        cold_column,
    ]:
        data[column] = parse_boolean(
            data[column]
        )

    for column in [
        TEMPERATURE_COLUMN,
        "station_p10_threshold_analysis",
        p05_column,
        "bmd_absolute_severity_rank_raw",
    ]:
        data[column] = pd.to_numeric(
            data[column],
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

    weights = pd.read_csv(
        WEIGHT_FILE
    )

    weights["station_uid"] = (
        weights["station_uid"]
        .astype(str)
    )

    if weights["station_uid"].duplicated().any():
        raise ValueError(
            "The area-weight table contains "
            "duplicate station IDs."
        )

    weight_sum = float(
        weights[
            "national_area_weight"
        ].sum()
    )

    if not np.isclose(
        weight_sum,
        1.0,
        atol=1e-8,
        rtol=0.0,
    ):
        raise ValueError(
            "Station area weights do not sum to one."
        )

    primary_ids = set(
        data.loc[
            data[member_column],
            "station_uid",
        ].unique()
    )

    weight_ids = set(
        weights["station_uid"].unique()
    )

    if primary_ids != weight_ids:
        raise ValueError(
            "Primary-network stations do not match "
            "the Voronoi weight table.\n"
            f"Missing weights: "
            f"{sorted(primary_ids - weight_ids)}\n"
            f"Extra weights: "
            f"{sorted(weight_ids - primary_ids)}"
        )

    primary = (
        data.loc[
            data[member_column]
        ]
        .merge(
            weights[
                [
                    "station_uid",
                    "cell_area_km2",
                    "national_area_weight",
                    "cell_centroid_longitude",
                    "cell_centroid_latitude",
                ]
            ],
            on="station_uid",
            how="left",
            validate="many_to_one",
        )
    )

    primary["_observed"] = (
        primary[eligibility_column]
        & primary[TEMPERATURE_COLUMN].notna()
        & primary[
            "station_p10_threshold_analysis"
        ].notna()
    )

    primary["_p10_cold"] = (
        primary["_observed"]
        & primary[cold_column]
    )

    primary["_p05_severe"] = (
        primary["_observed"]
        & primary[p05_column].notna()
        & primary[
            TEMPERATURE_COLUMN
        ].lt(
            primary[p05_column]
        )
    )

    primary["_observed_area"] = np.where(
        primary["_observed"],
        primary["national_area_weight"],
        0.0,
    )

    primary["_p10_cold_area"] = np.where(
        primary["_p10_cold"],
        primary["national_area_weight"],
        0.0,
    )

    primary["_p05_severe_area"] = np.where(
        primary["_p05_severe"],
        primary["national_area_weight"],
        0.0,
    )

    primary["_weighted_tmin"] = np.where(
        primary["_observed"],
        (
            primary["national_area_weight"]
            * primary[TEMPERATURE_COLUMN]
        ),
        0.0,
    )

    primary["_p10_deficit"] = np.where(
        primary["_p10_cold"],
        (
            primary[
                "station_p10_threshold_analysis"
            ]
            - primary[TEMPERATURE_COLUMN]
        ),
        0.0,
    )

    primary[
        "_area_weighted_p10_deficit"
    ] = (
        primary[
            "national_area_weight"
        ]
        * primary["_p10_deficit"]
    )

    primary["_cold_tmin"] = (
        primary[TEMPERATURE_COLUMN]
        .where(
            primary["_p10_cold"]
        )
    )

    category = (
        primary[
            "bmd_absolute_category_raw"
        ]
        .fillna("missing")
        .astype(str)
    )

    for category_name in [
        "mild",
        "moderate",
        "severe",
        "very_severe",
    ]:
        primary[
            f"_bmd_{category_name}_area"
        ] = np.where(
            primary["_observed"]
            & category.eq(
                category_name
            ),
            primary[
                "national_area_weight"
            ],
            0.0,
        )

    group_columns = [
        "date",
        "year",
        "month",
        "day",
        "winter_start_year",
        "winter_label",
        "winter_day_index",
    ]

    daily = (
        primary.groupby(
            group_columns,
            as_index=False,
            dropna=False,
        )
        .agg(
            observed_station_count=(
                "_observed",
                "sum",
            ),
            p10_cold_station_count=(
                "_p10_cold",
                "sum",
            ),
            p05_severe_station_count=(
                "_p05_severe",
                "sum",
            ),
            observed_national_area_fraction=(
                "_observed_area",
                "sum",
            ),
            p10_cold_national_area_fraction=(
                "_p10_cold_area",
                "sum",
            ),
            p05_severe_national_area_fraction=(
                "_p05_severe_area",
                "sum",
            ),
            weighted_tmin_numerator=(
                "_weighted_tmin",
                "sum",
            ),
            area_weighted_p10_deficit=(
                "_area_weighted_p10_deficit",
                "sum",
            ),
            cold_station_minimum_tmin=(
                "_cold_tmin",
                "min",
            ),
            bmd_mild_area_fraction=(
                "_bmd_mild_area",
                "sum",
            ),
            bmd_moderate_area_fraction=(
                "_bmd_moderate_area",
                "sum",
            ),
            bmd_severe_area_fraction=(
                "_bmd_severe_area",
                "sum",
            ),
            bmd_very_severe_area_fraction=(
                "_bmd_very_severe_area",
                "sum",
            ),
            maximum_bmd_severity_rank=(
                "bmd_absolute_severity_rank_raw",
                "max",
            ),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    daily[
        "area_weighted_mean_tmin"
    ] = safe_divide(
        daily[
            "weighted_tmin_numerator"
        ],
        daily[
            "observed_national_area_fraction"
        ],
    )

    daily[
        "cold_area_mean_p10_deficit"
    ] = safe_divide(
        daily[
            "area_weighted_p10_deficit"
        ],
        daily[
            "p10_cold_national_area_fraction"
        ],
    )

    daily[
        "p10_cold_fraction_of_observed_area"
    ] = safe_divide(
        daily[
            "p10_cold_national_area_fraction"
        ],
        daily[
            "observed_national_area_fraction"
        ],
    )

    daily[
        "daily_area_coverage_usable"
    ] = (
        daily[
            "observed_national_area_fraction"
        ].ge(
            minimum_observed_area
        )
        & daily[
            "observed_station_count"
        ].ge(
            minimum_observed_stations
        )
    )

    daily[
        "regional_p10_cold_day"
    ] = (
        daily[
            "daily_area_coverage_usable"
        ]
        & daily[
            "p10_cold_national_area_fraction"
        ].ge(
            minimum_cold_area
        )
    )

    daily[
        "maximum_bmd_severity_rank"
    ] = (
        daily[
            "maximum_bmd_severity_rank"
        ]
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
            "area_weighted_event_active"
        ]
    ].copy()

    if event_days.empty:
        raise ValueError(
            "No area-weighted primary events were "
            "identified. Review daily diagnostics "
            "before changing any threshold."
        )

    voronoi = gpd.read_file(
        VORONOI_FILE,
        layer="station_voronoi_weights",
    )

    voronoi["station_uid"] = (
        voronoi["station_uid"]
        .astype(str)
    )

    boundary = gpd.read_file(
        BOUNDARY_FILE,
        layer="bangladesh_boundary",
    )

    voronoi_projected = (
        voronoi.to_crs(
            policy[
                "spatial_projection"
            ]["proj_string"]
        )
    )

    boundary_projected = (
        boundary.to_crs(
            voronoi_projected.crs
        )
    )

    national_area_km2 = float(
        boundary_projected.geometry.iloc[0].area
        / 1_000_000.0
    )

    catalogue_rows = []
    participation_rows = []
    footprint_rows = []

    for event_id, event in event_days.groupby(
        "area_weighted_event_id",
        sort=True,
    ):
        event = event.sort_values("date")

        start_date = event["date"].min()
        end_date = event["date"].max()

        station_rows = primary.loc[
            primary["date"].between(
                start_date,
                end_date,
            )
            & primary["_p10_cold"]
        ].copy()

        affected_station_ids = sorted(
            station_rows[
                "station_uid"
            ].unique().tolist()
        )

        affected_polygons = (
            voronoi_projected.loc[
                voronoi_projected[
                    "station_uid"
                ].isin(
                    affected_station_ids
                )
            ]
        )

        footprint_geometry = union_geometry(
            affected_polygons.geometry
        )

        footprint_area_km2 = float(
            footprint_geometry.area
            / 1_000_000.0
        )

        footprint_centroid = (
            footprint_geometry.centroid
        )

        centroid_wgs84 = gpd.GeoSeries(
            [footprint_centroid],
            crs=voronoi_projected.crs,
        ).to_crs(
            "EPSG:4326"
        ).iloc[0]

        min_x, min_y, max_x, max_y = (
            footprint_geometry.bounds
        )

        peak_row = (
            event.sort_values(
                [
                    "p10_cold_national_area_fraction",
                    "p10_cold_station_count",
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

        total_cold_area_days = float(
            event[
                "p10_cold_national_area_fraction"
            ].sum()
        )

        total_deficit = float(
            event[
                "area_weighted_p10_deficit"
            ].sum()
        )

        catalogue_rows.append(
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
                "start_date": start_date,
                "end_date": end_date,
                "duration_days": len(event),
                "peak_coverage_date": (
                    peak_row["date"]
                ),
                "minimum_observed_area_fraction":
                    float(
                        event[
                            "observed_national_area_fraction"
                        ].min()
                    ),
                "minimum_p10_cold_area_fraction":
                    float(
                        event[
                            "p10_cold_national_area_fraction"
                        ].min()
                    ),
                "mean_p10_cold_area_fraction":
                    float(
                        event[
                            "p10_cold_national_area_fraction"
                        ].mean()
                    ),
                "peak_p10_cold_area_fraction":
                    float(
                        event[
                            "p10_cold_national_area_fraction"
                        ].max()
                    ),
                "cumulative_cold_area_days":
                    total_cold_area_days,
                "minimum_area_weighted_mean_tmin":
                    float(
                        event[
                            "area_weighted_mean_tmin"
                        ].min()
                    ),
                "minimum_station_tmin":
                    float(
                        event[
                            "cold_station_minimum_tmin"
                        ].min()
                    ),
                "total_area_weighted_p10_deficit":
                    total_deficit,
                "mean_affected_area_p10_deficit":
                    (
                        total_deficit
                        / total_cold_area_days
                        if total_cold_area_days > 0
                        else np.nan
                    ),
                "peak_p05_severe_area_fraction":
                    float(
                        event[
                            "p05_severe_national_area_fraction"
                        ].max()
                    ),
                "peak_bmd_mild_area_fraction":
                    float(
                        event[
                            "bmd_mild_area_fraction"
                        ].max()
                    ),
                "peak_bmd_moderate_area_fraction":
                    float(
                        event[
                            "bmd_moderate_area_fraction"
                        ].max()
                    ),
                "peak_bmd_severe_area_fraction":
                    float(
                        event[
                            "bmd_severe_area_fraction"
                        ].max()
                    ),
                "peak_bmd_very_severe_area_fraction":
                    float(
                        event[
                            "bmd_very_severe_area_fraction"
                        ].max()
                    ),
                "maximum_bmd_severity_rank":
                    maximum_rank,
                "most_severe_bmd_category":
                    SEVERITY_LABELS.get(
                        maximum_rank,
                        "unavailable",
                    ),
                "affected_station_count":
                    len(
                        affected_station_ids
                    ),
                "event_footprint_area_km2":
                    footprint_area_km2,
                "event_footprint_area_fraction":
                    (
                        footprint_area_km2
                        / national_area_km2
                    ),
                "event_centre_longitude":
                    float(
                        centroid_wgs84.x
                    ),
                "event_centre_latitude":
                    float(
                        centroid_wgs84.y
                    ),
                "event_east_west_extent_km":
                    float(
                        (max_x - min_x)
                        / 1000.0
                    ),
                "event_north_south_extent_km":
                    float(
                        (max_y - min_y)
                        / 1000.0
                    ),
            }
        )

        footprint_rows.append(
            {
                "event_id": event_id,
                "winter_label":
                    event[
                        "winter_label"
                    ].iloc[0],
                "start_date":
                    str(start_date.date()),
                "end_date":
                    str(end_date.date()),
                "duration_days":
                    len(event),
                "geometry":
                    footprint_geometry,
            }
        )

        grouped = (
            station_rows.groupby(
                [
                    "station_uid",
                    "station_name_display",
                ],
                as_index=False,
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
                mean_tmin=(
                    TEMPERATURE_COLUMN,
                    "mean",
                ),
                total_station_p10_deficit=(
                    "_p10_deficit",
                    "sum",
                ),
                national_area_weight=(
                    "national_area_weight",
                    "first",
                ),
                cell_area_km2=(
                    "cell_area_km2",
                    "first",
                ),
                maximum_bmd_severity_rank=(
                    "bmd_absolute_severity_rank_raw",
                    "max",
                ),
            )
        )

        grouped[
            "weighted_cold_area_days"
        ] = (
            grouped[
                "national_area_weight"
            ]
            * grouped[
                "event_cold_day_count"
            ]
        )

        grouped["event_id"] = event_id
        grouped[
            "winter_label"
        ] = event[
            "winter_label"
        ].iloc[0]
        grouped[
            "event_start_date"
        ] = start_date
        grouped[
            "event_end_date"
        ] = end_date
        grouped[
            "event_duration_days"
        ] = len(event)

        grouped[
            "most_severe_bmd_category"
        ] = (
            grouped[
                "maximum_bmd_severity_rank"
            ]
            .fillna(-1)
            .astype(int)
            .map(SEVERITY_LABELS)
            .fillna("unavailable")
        )

        participation_rows.append(
            grouped
        )

    catalogue = (
        pd.DataFrame(
            catalogue_rows
        )
        .sort_values(
            [
                "start_date",
                "event_id",
            ]
        )
        .reset_index(drop=True)
    )

    participation = (
        pd.concat(
            participation_rows,
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

    footprints = gpd.GeoDataFrame(
        footprint_rows,
        geometry="geometry",
        crs=voronoi_projected.crs,
    ).to_crs(
        "EPSG:4326"
    )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    issues = []

    if (
        daily[
            "observed_national_area_fraction"
        ].gt(1.0 + 1e-8).any()
    ):
        issues.append(
            {
                "issue":
                    "observed_area_exceeds_one"
            }
        )

    if (
        daily[
            "p10_cold_national_area_fraction"
        ].gt(
            daily[
                "observed_national_area_fraction"
            ]
            + 1e-8
        ).any()
    ):
        issues.append(
            {
                "issue":
                    "cold_area_exceeds_observed_area"
            }
        )

    for event_id, event in event_days.groupby(
        "area_weighted_event_id"
    ):
        event = event.sort_values("date")

        if len(event) < minimum_duration:
            issues.append(
                {
                    "event_id": event_id,
                    "issue":
                        "event_too_short",
                }
            )

        if (
            event["date"]
            .diff()
            .dropna()
            .dt.days
            .ne(1)
            .any()
        ):
            issues.append(
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
            issues.append(
                {
                    "event_id": event_id,
                    "issue":
                        "event_crosses_winter",
                }
            )

        if (
            event[
                "observed_national_area_fraction"
            ].lt(
                minimum_observed_area
            ).any()
        ):
            issues.append(
                {
                    "event_id": event_id,
                    "issue":
                        "insufficient_observed_area",
                }
            )

        if (
            event[
                "p10_cold_national_area_fraction"
            ].lt(
                minimum_cold_area
            ).any()
        ):
            issues.append(
                {
                    "event_id": event_id,
                    "issue":
                        "insufficient_cold_area",
                }
            )

    validation = pd.DataFrame(
        issues,
        columns=[
            "event_id",
            "issue",
        ],
    )

    validation.to_csv(
        VALIDATION_FILE,
        index=False,
    )

    if not validation.empty:
        raise ValueError(
            "Step 7D validation failed. Review: "
            f"{VALIDATION_FILE}"
        )

    # --------------------------------------------------
    # Attach daily event membership
    # --------------------------------------------------

    daily_membership = daily[
        [
            "date",
            "observed_national_area_fraction",
            "p10_cold_national_area_fraction",
            "p05_severe_national_area_fraction",
            "daily_area_coverage_usable",
            "regional_p10_cold_day",
            "candidate_run_length",
            "area_weighted_event_active",
            "area_weighted_event_id",
            "area_weighted_event_day_index",
        ]
    ]

    event_ready = (
        data.merge(
            weights[
                [
                    "station_uid",
                    "cell_area_km2",
                    "national_area_weight",
                ]
            ],
            on="station_uid",
            how="left",
            validate="many_to_one",
        )
        .merge(
            daily_membership,
            on="date",
            how="left",
            validate="many_to_one",
        )
    )

    event_ready[
        "station_participates_in_area_weighted_event"
    ] = (
        event_ready[
            "area_weighted_event_active"
        ].fillna(False).astype(bool)
        & event_ready[member_column]
        & event_ready[eligibility_column]
        & event_ready[cold_column]
    )

    event_ready[
        "step7d_temperature_adjusted"
    ] = False

    event_ready[
        "step7d_interpolation_applied"
    ] = False

    event_ready = event_ready.sort_values(
        [
            "station_uid",
            "date",
        ]
    ).reset_index(drop=True)

    if len(event_ready) != original_rows:
        raise ValueError(
            "Step 7D changed the row count."
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
            "Step 7D changed station-date keys."
        )

    if not values_equal_with_nan(
        original_temperature,
        event_ready[
            TEMPERATURE_COLUMN
        ],
    ):
        raise ValueError(
            "Step 7D changed Tmin values."
        )

    # --------------------------------------------------
    # Winter summary
    # --------------------------------------------------

    winter_summary = (
        daily.groupby(
            [
                "winter_start_year",
                "winter_label",
            ],
            as_index=False,
        )
        .agg(
            usable_area_coverage_days=(
                "daily_area_coverage_usable",
                "sum",
            ),
            regional_p10_cold_days=(
                "regional_p10_cold_day",
                "sum",
            ),
            event_days=(
                "area_weighted_event_active",
                "sum",
            ),
            maximum_p10_cold_area_fraction=(
                "p10_cold_national_area_fraction",
                "max",
            ),
            maximum_p05_severe_area_fraction=(
                "p05_severe_national_area_fraction",
                "max",
            ),
            minimum_area_weighted_mean_tmin=(
                "area_weighted_mean_tmin",
                "min",
            ),
        )
    )

    event_counts = (
        catalogue.groupby(
            "winter_start_year",
            as_index=False,
        )
        .agg(
            event_count=(
                "event_id",
                "count",
            ),
            total_event_duration_days=(
                "duration_days",
                "sum",
            ),
            maximum_event_duration_days=(
                "duration_days",
                "max",
            ),
        )
    )

    winter_summary = (
        winter_summary.merge(
            event_counts,
            on="winter_start_year",
            how="left",
            validate="one_to_one",
        )
        .fillna(
            {
                "event_count": 0,
                "total_event_duration_days": 0,
                "maximum_event_duration_days": 0,
            }
        )
        .sort_values(
            "winter_start_year"
        )
        .reset_index(drop=True)
    )

    for column in [
        "event_count",
        "total_event_duration_days",
        "maximum_event_duration_days",
    ]:
        winter_summary[column] = (
            winter_summary[column]
            .astype(int)
        )

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------

    daily.to_parquet(
        DAILY_OUTPUT_FILE,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    event_ready.to_parquet(
        EVENT_READY_FILE,
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
        PARTICIPATION_FILE,
        index=False,
    )

    footprints.to_file(
        FOOTPRINT_FILE,
        layer="area_weighted_event_footprints",
        driver="GPKG",
    )

    winter_summary.to_csv(
        WINTER_SUMMARY_FILE,
        index=False,
    )

    daily_parquet = pq.ParquetFile(
        DAILY_OUTPUT_FILE
    )

    event_ready_parquet = pq.ParquetFile(
        EVENT_READY_FILE
    )

    schema = {
        "daily_diagnostic_schema": {
            field.name: str(field.type)
            for field
            in daily_parquet.schema_arrow
        },
        "event_ready_schema": {
            field.name: str(field.type)
            for field
            in event_ready_parquet.schema_arrow
        },
    }

    SCHEMA_FILE.write_text(
        json.dumps(
            schema,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "primary_definition":
            (
                "Station p10, at least 20 percent "
                "of Bangladesh area, at least "
                "3 consecutive days"
            ),
        "event_count":
            len(catalogue),
        "event_day_count":
            len(event_days),
        "winters_with_events":
            int(
                catalogue[
                    "winter_start_year"
                ].nunique()
            ),
        "longest_event_days":
            int(
                catalogue[
                    "duration_days"
                ].max()
            ),
        "maximum_event_cold_area_fraction":
            float(
                catalogue[
                    "peak_p10_cold_area_fraction"
                ].max()
            ),
        "minimum_observed_area_fraction":
            minimum_observed_area,
        "minimum_cold_area_fraction":
            minimum_cold_area,
        "minimum_duration_days":
            minimum_duration,
        "validation_issue_count":
            len(validation),
        "temperature_adjusted":
            False,
        "interpolation_applied":
            False,
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
        "STEP 7D-B: AREA-WEIGHTED PRIMARY COLD SPELLS",
        "=" * 50,
        (
            "Primary definition: station p10, "
            "20% Bangladesh area, 3 consecutive days"
        ),
        (
            "Area-weighted events: "
            f"{len(catalogue)}"
        ),
        (
            "Area-weighted event days: "
            f"{len(event_days)}"
        ),
        (
            "Winters with events: "
            f"{catalogue['winter_start_year'].nunique()}"
        ),
        (
            "Longest event: "
            f"{catalogue['duration_days'].max()} days"
        ),
        (
            "Maximum affected area: "
            f"{100 * catalogue['peak_p10_cold_area_fraction'].max():.2f}%"
        ),
        (
            "Validation issues: "
            f"{len(validation)}"
        ),
        "",
        "Temperature adjusted: No",
        "Interpolation applied: No",
        "",
        f"Catalogue: {EVENT_CATALOGUE_FILE}",
        f"Event days: {EVENT_DAY_FILE}",
        f"Event footprints: {FOOTPRINT_FILE}",
    ]

    REPORT_FILE.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(report_lines))
    print("\nSTEP 7D-B PASSED.")


if __name__ == "__main__":
    main()
