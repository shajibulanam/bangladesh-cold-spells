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
    / "temperature_daily_qc_stage2_temporal.parquet"
)

RULE_FILE = (
    PROJECT_ROOT
    / "00_admin"
    / "step5b_spatial_qc_rules.yaml"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "temperature_daily_qc_stage3_spatial.parquet"
)

NEIGHBOUR_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "step5b_fixed_station_neighbours.parquet"
)

REPORT_DIRECTORY = PROJECT_ROOT / "05_qc_reports"
ADMIN_DIRECTORY = PROJECT_ROOT / "00_admin"

EXPECTED_STATIONS = 48
EXPECTED_SPATIALLY_ELIGIBLE_STATIONS = 43


TMIN_RELEVANT_COLUMNS = [
    "qc_tmin_climatological_outlier",
    "qc_tmin_large_daily_change",
    "qc_tmin_isolated_spike",
    "qc_tmin_persistence",
    "step4b_post_tmin_physical_range",
    "step4b_post_tmax_lt_tmin",
    "step4b_post_high_dtr",
]

TMAX_RELEVANT_COLUMNS = [
    "qc_tmax_climatological_outlier",
    "qc_tmax_large_daily_change",
    "qc_tmax_isolated_spike",
    "qc_tmax_persistence",
    "step4b_post_tmax_physical_range",
    "step4b_post_tmax_lt_tmin",
    "step4b_post_high_dtr",
]


def load_rules() -> dict[str, Any]:
    if not RULE_FILE.exists():
        raise FileNotFoundError(
            f"Rule file not found: {RULE_FILE}"
        )

    with RULE_FILE.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        return yaml.safe_load(file_handle)


def haversine_distance_km(
    latitude_1: np.ndarray,
    longitude_1: np.ndarray,
    latitude_2: np.ndarray,
    longitude_2: np.ndarray,
) -> np.ndarray:
    """Calculate great-circle distances in kilometres."""
    earth_radius_km = 6371.0088

    lat1 = np.radians(latitude_1)
    lon1 = np.radians(longitude_1)
    lat2 = np.radians(latitude_2)
    lon2 = np.radians(longitude_2)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    haversine = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_lon / 2.0) ** 2
    )

    angular_distance = (
        2.0
        * np.arcsin(
            np.sqrt(
                np.clip(
                    haversine,
                    0.0,
                    1.0,
                )
            )
        )
    )

    return earth_radius_km * angular_distance


def build_neighbour_network(
    stations: pd.DataFrame,
    maximum_neighbours: int,
    maximum_distance_km: float,
) -> pd.DataFrame:
    """Create a directed nearest-neighbour network."""
    rows: list[dict[str, Any]] = []

    records = stations.reset_index(
        drop=True
    )

    for _, target in records.iterrows():
        other = records.loc[
            records["station_uid"].ne(
                target["station_uid"]
            )
        ].copy()

        distances = haversine_distance_km(
            np.full(
                len(other),
                float(target["latitude"]),
            ),
            np.full(
                len(other),
                float(target["longitude"]),
            ),
            other["latitude"].astype(float).to_numpy(),
            other["longitude"].astype(float).to_numpy(),
        )

        other["distance_km"] = distances

        other = (
            other.loc[
                other["distance_km"].le(
                    maximum_distance_km
                )
            ]
            .sort_values(
                [
                    "distance_km",
                    "station_uid",
                ]
            )
            .head(maximum_neighbours)
            .reset_index(drop=True)
        )

        for neighbour_rank, neighbour in (
            other.iterrows()
        ):
            rows.append(
                {
                    "target_station_uid":
                        target["station_uid"],
                    "target_station_name":
                        target[
                            "station_name_display"
                        ],
                    "target_latitude":
                        float(target["latitude"]),
                    "target_longitude":
                        float(target["longitude"]),

                    "neighbour_rank":
                        int(neighbour_rank + 1),
                    "neighbour_station_uid":
                        neighbour["station_uid"],
                    "neighbour_station_name":
                        neighbour[
                            "station_name_display"
                        ],
                    "neighbour_latitude":
                        float(neighbour["latitude"]),
                    "neighbour_longitude":
                        float(neighbour["longitude"]),
                    "distance_km":
                        float(neighbour["distance_km"]),
                }
            )

    return pd.DataFrame(rows)


def any_existing_flags(
    data: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    existing_columns = [
        column
        for column in columns
        if column in data.columns
    ]

    if not existing_columns:
        return pd.Series(
            False,
            index=data.index,
            dtype=bool,
        )

    return (
        data[existing_columns]
        .fillna(False)
        .astype(bool)
        .any(axis=1)
    )


def calculate_spatial_metrics(
    data: pd.DataFrame,
    candidates: pd.DataFrame,
    network: pd.DataFrame,
    variable: str,
    relevant_column: str,
    minimum_same_day_neighbours: int,
    minimum_scale: float,
    residual_threshold: float,
    standardized_threshold: float,
    support_median_threshold: float,
    support_residual_threshold: float,
    minimum_same_sign_fraction: float,
    cold_target_threshold: float,
    cold_neighbour_threshold: float,
) -> pd.DataFrame:
    """
    Calculate neighbour-anomaly metrics for one variable.

    The function returns one row for every Step 5A candidate.
    """
    anomaly_column = (
        f"{variable}_clim_anomaly"
    )

    neighbour_anomaly_column = (
        f"neighbour_{variable}_anomaly"
    )

    output_columns = {
        "neighbour_count":
            f"{variable}_spatial_neighbour_count",
        "neighbour_median":
            f"{variable}_spatial_neighbour_median_anomaly",
        "neighbour_mad":
            f"{variable}_spatial_neighbour_mad",
        "neighbour_scale":
            f"{variable}_spatial_neighbour_scale",
        "neighbour_minimum":
            f"{variable}_spatial_neighbour_minimum_anomaly",
        "neighbour_maximum":
            f"{variable}_spatial_neighbour_maximum_anomaly",
        "same_sign_fraction":
            f"{variable}_spatial_same_sign_fraction",
        "residual":
            f"{variable}_spatial_residual",
        "robust_z":
            f"{variable}_spatial_robust_z",
        "outlier":
            f"qc_{variable}_spatial_outlier",
        "regional_support":
            f"qc_{variable}_regional_support",
        "cold_support":
            f"qc_{variable}_regional_cold_support",
        "insufficient":
            f"qc_{variable}_spatial_insufficient",
    }

    result = candidates[
        [
            "_row_id",
            anomaly_column,
            relevant_column,
        ]
    ].drop_duplicates(
        subset=["_row_id"]
    ).copy()

    result[
        output_columns["neighbour_count"]
    ] = 0

    for key in [
        "neighbour_median",
        "neighbour_mad",
        "neighbour_scale",
        "neighbour_minimum",
        "neighbour_maximum",
        "same_sign_fraction",
        "residual",
        "robust_z",
    ]:
        result[output_columns[key]] = np.nan

    for key in [
        "outlier",
        "regional_support",
        "cold_support",
        "insufficient",
    ]:
        result[output_columns[key]] = False

    if candidates.empty or network.empty:
        return result

    target_columns = [
        "_row_id",
        "station_uid",
        "date",
        anomaly_column,
        relevant_column,
    ]

    pairs = candidates[
        target_columns
    ].merge(
        network[
            [
                "target_station_uid",
                "neighbour_station_uid",
                "neighbour_rank",
                "distance_km",
            ]
        ],
        left_on="station_uid",
        right_on="target_station_uid",
        how="left",
        validate="many_to_many",
    )

    neighbour_daily = data[
        [
            "station_uid",
            "date",
            anomaly_column,
        ]
    ].rename(
        columns={
            "station_uid":
                "neighbour_station_uid",
            anomaly_column:
                neighbour_anomaly_column,
        }
    )

    pairs = pairs.merge(
        neighbour_daily,
        on=[
            "neighbour_station_uid",
            "date",
        ],
        how="left",
        validate="many_to_one",
    )

    valid = pairs.loc[
        pairs[anomaly_column].notna()
        & pairs[
            neighbour_anomaly_column
        ].notna()
    ].copy()

    if valid.empty:
        result[
            output_columns["insufficient"]
        ] = (
            result[relevant_column]
            .fillna(False)
            .astype(bool)
            & result[anomaly_column].notna()
        )

        return result

    grouped = valid.groupby(
        "_row_id",
        sort=False,
    )

    basic = grouped[
        neighbour_anomaly_column
    ].agg(
        neighbour_count="count",
        neighbour_median="median",
        neighbour_minimum="min",
        neighbour_maximum="max",
    )

    valid = valid.merge(
        basic[
            ["neighbour_median"]
        ],
        left_on="_row_id",
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    valid["absolute_deviation"] = (
        valid[
            neighbour_anomaly_column
        ]
        - valid["neighbour_median"]
    ).abs()

    neighbour_mad = (
        valid.groupby(
            "_row_id",
            sort=False,
        )["absolute_deviation"]
        .median()
        .rename("neighbour_mad")
    )

    target_sign = np.sign(
        valid[anomaly_column]
    )

    neighbour_sign = np.sign(
        valid[
            neighbour_anomaly_column
        ]
    )

    valid["same_sign"] = (
        target_sign.eq(neighbour_sign)
        & target_sign.ne(0)
    )

    same_sign_fraction = (
        valid.groupby(
            "_row_id",
            sort=False,
        )["same_sign"]
        .mean()
        .rename("same_sign_fraction")
    )

    metrics = (
        basic.join(neighbour_mad)
        .join(same_sign_fraction)
        .reset_index()
    )

    metrics["neighbour_scale"] = np.maximum(
        1.4826
        * metrics["neighbour_mad"],
        minimum_scale,
    )

    metrics = result[
        [
            "_row_id",
            anomaly_column,
            relevant_column,
        ]
    ].merge(
        metrics,
        on="_row_id",
        how="left",
        validate="one_to_one",
    )

    metrics["neighbour_count"] = (
        metrics["neighbour_count"]
        .fillna(0)
        .astype("int16")
    )

    metrics["spatial_residual"] = (
        metrics[anomaly_column]
        - metrics["neighbour_median"]
    )

    metrics["spatial_robust_z"] = (
        metrics["spatial_residual"]
        / metrics["neighbour_scale"]
    )

    relevant = (
        metrics[relevant_column]
        .fillna(False)
        .astype(bool)
    )

    evaluable = (
        relevant
        & metrics[anomaly_column].notna()
    )

    enough_neighbours = (
        metrics["neighbour_count"]
        .ge(minimum_same_day_neighbours)
    )

    same_direction = (
        np.sign(metrics[anomaly_column])
        .eq(
            np.sign(
                metrics["neighbour_median"]
            )
        )
        & metrics[anomaly_column].ne(0)
    )

    spatial_outlier = (
        evaluable
        & enough_neighbours
        & metrics[
            "spatial_residual"
        ].abs().ge(
            residual_threshold
        )
        & metrics[
            "spatial_robust_z"
        ].abs().ge(
            standardized_threshold
        )
    )

    regional_support = (
        evaluable
        & enough_neighbours
        & same_direction
        & metrics[
            "neighbour_median"
        ].abs().ge(
            support_median_threshold
        )
        & metrics[
            "spatial_residual"
        ].abs().le(
            support_residual_threshold
        )
        & metrics[
            "same_sign_fraction"
        ].ge(
            minimum_same_sign_fraction
        )
    )

    if variable == "tmin":
        cold_support = (
            regional_support
            & metrics[
                anomaly_column
            ].le(cold_target_threshold)
            & metrics[
                "neighbour_median"
            ].le(cold_neighbour_threshold)
        )
    else:
        cold_support = pd.Series(
            False,
            index=metrics.index,
            dtype=bool,
        )

    insufficient = (
        evaluable
        & ~enough_neighbours
    )

    result = pd.DataFrame(
        {
            "_row_id":
                metrics["_row_id"],

            output_columns[
                "neighbour_count"
            ]:
                metrics[
                    "neighbour_count"
                ],

            output_columns[
                "neighbour_median"
            ]:
                metrics[
                    "neighbour_median"
                ],

            output_columns[
                "neighbour_mad"
            ]:
                metrics[
                    "neighbour_mad"
                ],

            output_columns[
                "neighbour_scale"
            ]:
                metrics[
                    "neighbour_scale"
                ],

            output_columns[
                "neighbour_minimum"
            ]:
                metrics[
                    "neighbour_minimum"
                ],

            output_columns[
                "neighbour_maximum"
            ]:
                metrics[
                    "neighbour_maximum"
                ],

            output_columns[
                "same_sign_fraction"
            ]:
                metrics[
                    "same_sign_fraction"
                ],

            output_columns[
                "residual"
            ]:
                metrics[
                    "spatial_residual"
                ],

            output_columns[
                "robust_z"
            ]:
                metrics[
                    "spatial_robust_z"
                ],

            output_columns[
                "outlier"
            ]:
                spatial_outlier,

            output_columns[
                "regional_support"
            ]:
                regional_support,

            output_columns[
                "cold_support"
            ]:
                cold_support,

            output_columns[
                "insufficient"
            ]:
                insufficient,
        }
    )

    return result


def main() -> None:
    for required_path in [
        INPUT_FILE,
        RULE_FILE,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required file not found: "
                f"{required_path}"
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

    network_rules = rules[
        "station_network"
    ]

    residual_rules = rules[
        "spatial_residual"
    ]

    support_rules = rules[
        "regional_support"
    ]

    cold_rules = rules[
        "cold_event_support"
    ]

    maximum_neighbours = int(
        network_rules[
            "maximum_neighbours"
        ]
    )

    maximum_distance_km = float(
        network_rules[
            "maximum_distance_km"
        ]
    )

    minimum_fixed_neighbours = int(
        network_rules[
            "minimum_fixed_neighbours"
        ]
    )

    minimum_same_day_neighbours = int(
        network_rules[
            "minimum_same_day_neighbours"
        ]
    )

    minimum_scale = float(
        residual_rules[
            "minimum_robust_scale_celsius"
        ]
    )

    residual_threshold = float(
        residual_rules[
            "absolute_residual_threshold_celsius"
        ]
    )

    standardized_threshold = float(
        residual_rules[
            "standardized_residual_threshold"
        ]
    )

    support_median_threshold = float(
        support_rules[
            "minimum_absolute_neighbour_median_anomaly_celsius"
        ]
    )

    support_residual_threshold = float(
        support_rules[
            "maximum_residual_from_neighbour_median_celsius"
        ]
    )

    minimum_same_sign_fraction = float(
        support_rules[
            "minimum_same_sign_fraction"
        ]
    )

    cold_target_threshold = float(
        cold_rules[
            "target_tmin_anomaly_maximum_celsius"
        ]
    )

    cold_neighbour_threshold = float(
        cold_rules[
            "neighbour_median_tmin_anomaly_maximum_celsius"
        ]
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

    input_rows = len(data)

    original_values = data[
        [
            "station_uid",
            "date",
            "tmin_cleaned_stage1",
            "tmax_cleaned_stage1",
        ]
    ].copy()

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

    station_metadata = (
        data[
            [
                "station_uid",
                "station_id",
                "station_name_display",
                "metadata_status",
                "latitude",
                "longitude",
                "usable_for_spatial_analysis",
            ]
        ]
        .drop_duplicates(
            subset=["station_uid"]
        )
        .sort_values("station_uid")
        .reset_index(drop=True)
    )

    spatial_stations = station_metadata.loc[
        station_metadata[
            "usable_for_spatial_analysis"
        ].fillna(False).astype(bool)
        & station_metadata["latitude"].notna()
        & station_metadata["longitude"].notna()
    ].copy()

    if (
        len(spatial_stations)
        != EXPECTED_SPATIALLY_ELIGIBLE_STATIONS
    ):
        raise ValueError(
            "Expected "
            f"{EXPECTED_SPATIALLY_ELIGIBLE_STATIONS} "
            "spatially eligible stations; found "
            f"{len(spatial_stations)}."
        )

    network = build_neighbour_network(
        spatial_stations,
        maximum_neighbours,
        maximum_distance_km,
    )

    if network.empty:
        raise ValueError(
            "The fixed station-neighbour network is empty."
        )

    network.to_parquet(
        NEIGHBOUR_FILE,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    network.to_csv(
        REPORT_DIRECTORY
        / "step5b_fixed_station_neighbours.csv",
        index=False,
    )

    network_summary = (
        network.groupby(
            [
                "target_station_uid",
                "target_station_name",
            ],
            as_index=False,
        )
        .agg(
            fixed_neighbour_count=(
                "neighbour_station_uid",
                "count",
            ),
            nearest_neighbour_km=(
                "distance_km",
                "min",
            ),
            furthest_selected_neighbour_km=(
                "distance_km",
                "max",
            ),
            mean_selected_neighbour_km=(
                "distance_km",
                "mean",
            ),
        )
        .sort_values(
            "target_station_uid"
        )
    )

    network_summary[
        "minimum_network_requirement_met"
    ] = network_summary[
        "fixed_neighbour_count"
    ].ge(minimum_fixed_neighbours)

    network_summary.to_csv(
        REPORT_DIRECTORY
        / "step5b_station_neighbour_summary.csv",
        index=False,
    )

    data["_row_id"] = np.arange(
        len(data),
        dtype=np.int64,
    )

    data["step5b_tmin_relevant"] = (
        any_existing_flags(
            data,
            TMIN_RELEVANT_COLUMNS,
        )
    )

    data["step5b_tmax_relevant"] = (
        any_existing_flags(
            data,
            TMAX_RELEVANT_COLUMNS,
        )
    )

    candidates = data.loc[
        data[
            "step5a_candidate_review_required"
        ].fillna(False).astype(bool)
    ].copy()

    tmin_metrics = calculate_spatial_metrics(
        data=data,
        candidates=candidates,
        network=network,
        variable="tmin",
        relevant_column=(
            "step5b_tmin_relevant"
        ),
        minimum_same_day_neighbours=(
            minimum_same_day_neighbours
        ),
        minimum_scale=minimum_scale,
        residual_threshold=(
            residual_threshold
        ),
        standardized_threshold=(
            standardized_threshold
        ),
        support_median_threshold=(
            support_median_threshold
        ),
        support_residual_threshold=(
            support_residual_threshold
        ),
        minimum_same_sign_fraction=(
            minimum_same_sign_fraction
        ),
        cold_target_threshold=(
            cold_target_threshold
        ),
        cold_neighbour_threshold=(
            cold_neighbour_threshold
        ),
    )

    tmax_metrics = calculate_spatial_metrics(
        data=data,
        candidates=candidates,
        network=network,
        variable="tmax",
        relevant_column=(
            "step5b_tmax_relevant"
        ),
        minimum_same_day_neighbours=(
            minimum_same_day_neighbours
        ),
        minimum_scale=minimum_scale,
        residual_threshold=(
            residual_threshold
        ),
        standardized_threshold=(
            standardized_threshold
        ),
        support_median_threshold=(
            support_median_threshold
        ),
        support_residual_threshold=(
            support_residual_threshold
        ),
        minimum_same_sign_fraction=(
            minimum_same_sign_fraction
        ),
        cold_target_threshold=(
            cold_target_threshold
        ),
        cold_neighbour_threshold=(
            cold_neighbour_threshold
        ),
    )

    data = data.merge(
        tmin_metrics,
        on="_row_id",
        how="left",
        validate="one_to_one",
    )

    data = data.merge(
        tmax_metrics,
        on="_row_id",
        how="left",
        validate="one_to_one",
    )

    boolean_spatial_columns = [
        "qc_tmin_spatial_outlier",
        "qc_tmin_regional_support",
        "qc_tmin_regional_cold_support",
        "qc_tmin_spatial_insufficient",
        "qc_tmax_spatial_outlier",
        "qc_tmax_regional_support",
        "qc_tmax_regional_cold_support",
        "qc_tmax_spatial_insufficient",
    ]

    for column in boolean_spatial_columns:
        data[column] = (
            data[column]
            .fillna(False)
            .astype(bool)
        )

    count_columns = [
        "tmin_spatial_neighbour_count",
        "tmax_spatial_neighbour_count",
    ]

    for column in count_columns:
        data[column] = (
            pd.to_numeric(
                data[column],
                errors="coerce",
            )
            .fillna(0)
            .astype("int16")
        )

    candidate_flag = (
        data[
            "step5a_candidate_review_required"
        ].fillna(False).astype(bool)
    )

    spatial_metadata_available = (
        data[
            "usable_for_spatial_analysis"
        ].fillna(False).astype(bool)
        & data["latitude"].notna()
        & data["longitude"].notna()
    )

    evaluable_relevant_variable = (
        (
            data["step5b_tmin_relevant"]
            & data[
                "tmin_clim_anomaly"
            ].notna()
        )
        | (
            data["step5b_tmax_relevant"]
            & data[
                "tmax_clim_anomaly"
            ].notna()
        )
    )

    data[
        "step5b_spatial_metadata_unavailable"
    ] = (
        candidate_flag
        & ~spatial_metadata_available
    )

    data[
        "step5b_spatial_baseline_unavailable"
    ] = (
        candidate_flag
        & spatial_metadata_available
        & ~evaluable_relevant_variable
    )

    data[
        "step5b_isolated_spatial_outlier"
    ] = (
        data["qc_tmin_spatial_outlier"]
        | data["qc_tmax_spatial_outlier"]
    )

    data[
        "step5b_regional_signal_supported"
    ] = (
        data["qc_tmin_regional_support"]
        | data["qc_tmax_regional_support"]
    )

    data[
        "step5b_regional_cold_signal_supported"
    ] = data[
        "qc_tmin_regional_cold_support"
    ]

    data[
        "step5b_insufficient_same_day_neighbours"
    ] = (
        data["qc_tmin_spatial_insufficient"]
        | data["qc_tmax_spatial_insufficient"]
    )

    data["step5b_status"] = np.select(
        [
            ~candidate_flag,
            data[
                "step5b_spatial_metadata_unavailable"
            ],
            data[
                "step5b_spatial_baseline_unavailable"
            ],
            data[
                "step5b_isolated_spatial_outlier"
            ],
            data[
                "step5b_regional_signal_supported"
            ],
            data[
                "step5b_insufficient_same_day_neighbours"
            ],
        ],
        [
            "not_evaluated_not_step5a_candidate",
            "spatial_unavailable_metadata",
            "spatial_unavailable_baseline",
            "isolated_spatial_outlier",
            "regional_signal_supported",
            "insufficient_same_day_neighbours",
        ],
        default="spatially_inconclusive",
    )

    data["step5b_priority"] = np.select(
        [
            data[
                "step5b_isolated_spatial_outlier"
            ],
            data[
                "step5b_regional_signal_supported"
            ],
            data[
                "step5b_insufficient_same_day_neighbours"
            ],
            candidate_flag,
        ],
        [
            "high",
            "regional_support",
            "insufficient_data",
            "review",
        ],
        default="none",
    )

    data[
        "step5b_candidate_for_cleaning_decision"
    ] = data[
        "step5b_isolated_spatial_outlier"
    ]

    flag_codes = np.full(
        len(data),
        "",
        dtype=object,
    )

    flag_count = np.zeros(
        len(data),
        dtype=np.int16,
    )

    flag_definitions = [
        (
            "TMIN_SPATIAL_OUTLIER",
            "qc_tmin_spatial_outlier",
        ),
        (
            "TMAX_SPATIAL_OUTLIER",
            "qc_tmax_spatial_outlier",
        ),
        (
            "TMIN_REGIONAL_SUPPORT",
            "qc_tmin_regional_support",
        ),
        (
            "TMAX_REGIONAL_SUPPORT",
            "qc_tmax_regional_support",
        ),
        (
            "TMIN_REGIONAL_COLD_SUPPORT",
            "qc_tmin_regional_cold_support",
        ),
        (
            "SPATIAL_METADATA_UNAVAILABLE",
            "step5b_spatial_metadata_unavailable",
        ),
        (
            "SPATIAL_BASELINE_UNAVAILABLE",
            "step5b_spatial_baseline_unavailable",
        ),
        (
            "INSUFFICIENT_NEIGHBOURS",
            "step5b_insufficient_same_day_neighbours",
        ),
    ]

    for code, column in flag_definitions:
        mask = (
            data[column]
            .fillna(False)
            .astype(bool)
            .to_numpy()
        )

        flag_count[mask] += 1

        flag_codes[mask] = np.where(
            flag_codes[mask] == "",
            code,
            flag_codes[mask] + ";" + code,
        )

    data["step5b_flag_codes"] = (
        pd.Series(
            flag_codes,
            index=data.index,
            dtype="string",
        )
    )

    data["step5b_flag_count"] = (
        flag_count.astype("int16")
    )

    # Validate unchanged temperature values.
    output_values = data[
        [
            "station_uid",
            "date",
            "tmin_cleaned_stage1",
            "tmax_cleaned_stage1",
        ]
    ].copy()

    if not original_values.equals(
        output_values
    ):
        raise ValueError(
            "Temperature values or row ordering "
            "changed during Step 5B."
        )

    if len(data) != input_rows:
        raise ValueError(
            "Step 5B changed the row count."
        )

    if data.duplicated(
        subset=[
            "station_uid",
            "date",
        ]
    ).any():
        raise ValueError(
            "Step 5B introduced duplicate keys."
        )

    # Reports.
    report_columns = [
        "station_uid",
        "station_id",
        "station_name_display",
        "metadata_status",
        "date",
        "year",
        "month",
        "day",
        "latitude",
        "longitude",

        "tmin_cleaned_stage1",
        "tmax_cleaned_stage1",
        "tmin_clim_anomaly",
        "tmax_clim_anomaly",

        "step5a_flag_codes",
        "step5a_priority",

        "tmin_spatial_neighbour_count",
        "tmin_spatial_neighbour_median_anomaly",
        "tmin_spatial_neighbour_scale",
        "tmin_spatial_same_sign_fraction",
        "tmin_spatial_residual",
        "tmin_spatial_robust_z",

        "tmax_spatial_neighbour_count",
        "tmax_spatial_neighbour_median_anomaly",
        "tmax_spatial_neighbour_scale",
        "tmax_spatial_same_sign_fraction",
        "tmax_spatial_residual",
        "tmax_spatial_robust_z",

        "step5b_flag_codes",
        "step5b_priority",
        "step5b_status",
        "step5b_candidate_for_cleaning_decision",
        "old_source_rows",
        "new_source_cell",
        "step4b_decision",
    ]

    evaluated_candidates = data.loc[
        candidate_flag
    ].copy()

    evaluated_candidates[
        report_columns
    ].to_csv(
        REPORT_DIRECTORY
        / "step5b_all_spatially_evaluated_candidates.csv",
        index=False,
    )

    isolated_outliers = data.loc[
        data[
            "step5b_isolated_spatial_outlier"
        ]
    ].copy()

    isolated_outliers[
        report_columns
    ].to_csv(
        REPORT_DIRECTORY
        / "step5b_isolated_spatial_outliers.csv",
        index=False,
    )

    regional_support = data.loc[
        data[
            "step5b_regional_signal_supported"
        ]
    ].copy()

    regional_support[
        report_columns
    ].to_csv(
        REPORT_DIRECTORY
        / "step5b_regional_signal_supported.csv",
        index=False,
    )

    regional_cold_support = data.loc[
        data[
            "step5b_regional_cold_signal_supported"
        ]
    ].copy()

    regional_cold_support[
        report_columns
    ].to_csv(
        REPORT_DIRECTORY
        / "step5b_regional_cold_signal_supported.csv",
        index=False,
    )

    insufficient_support = data.loc[
        data[
            "step5b_insufficient_same_day_neighbours"
        ]
        | data[
            "step5b_spatial_metadata_unavailable"
        ]
        | data[
            "step5b_spatial_baseline_unavailable"
        ]
    ].copy()

    insufficient_support[
        report_columns
    ].to_csv(
        REPORT_DIRECTORY
        / "step5b_spatially_unavailable_or_insufficient.csv",
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
            step5a_candidates=(
                "step5a_candidate_review_required",
                "sum",
            ),
            tmin_spatial_outliers=(
                "qc_tmin_spatial_outlier",
                "sum",
            ),
            tmax_spatial_outliers=(
                "qc_tmax_spatial_outlier",
                "sum",
            ),
            regional_signal_supported=(
                "step5b_regional_signal_supported",
                "sum",
            ),
            regional_cold_supported=(
                "step5b_regional_cold_signal_supported",
                "sum",
            ),
            insufficient_spatial_support=(
                "step5b_insufficient_same_day_neighbours",
                "sum",
            ),
            cleaning_decision_candidates=(
                "step5b_candidate_for_cleaning_decision",
                "sum",
            ),
        )
        .reset_index()
        .sort_values("station_uid")
    )

    station_summary.to_csv(
        REPORT_DIRECTORY
        / "step5b_station_spatial_qc_summary.csv",
        index=False,
    )

    data = data.drop(
        columns=["_row_id"]
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

    status_counts = (
        data["step5b_status"]
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
        "neighbour_file":
            str(NEIGHBOUR_FILE),
        "input_rows":
            input_rows,
        "output_rows":
            output_parquet.metadata.num_rows,
        "unique_stations":
            station_count,
        "spatially_eligible_stations":
            len(spatial_stations),
        "fixed_neighbour_pairs":
            len(network),
        "stations_below_fixed_neighbour_minimum":
            int(
                (
                    ~network_summary[
                        "minimum_network_requirement_met"
                    ]
                ).sum()
            ),
        "step5a_candidate_records":
            len(evaluated_candidates),
        "isolated_spatial_outliers":
            len(isolated_outliers),
        "regional_signal_supported_records":
            len(regional_support),
        "regional_cold_supported_records":
            len(regional_cold_support),
        "unavailable_or_insufficient_records":
            len(insufficient_support),
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
        / "step5b_qc_summary.json"
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
        / "step5b_parquet_schema.json"
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
        "STEP 5B: SPATIAL CONSISTENCY QC",
        "=" * 38,
        f"Input rows: {input_rows:,}",
        (
            "Output rows: "
            f"{output_parquet.metadata.num_rows:,}"
        ),
        f"Unique stations: {station_count}",
        (
            "Spatially eligible stations: "
            f"{len(spatial_stations)}"
        ),
        (
            "Fixed directed neighbour pairs: "
            f"{len(network):,}"
        ),
        (
            "Stations below fixed-neighbour minimum: "
            f"{summary['stations_below_fixed_neighbour_minimum']}"
        ),
        (
            "Step 5A candidate records evaluated: "
            f"{len(evaluated_candidates):,}"
        ),
        (
            "Isolated spatial outliers: "
            f"{len(isolated_outliers):,}"
        ),
        (
            "Regional-signal supported records: "
            f"{len(regional_support):,}"
        ),
        (
            "Regional cold-signal supported records: "
            f"{len(regional_cold_support):,}"
        ),
        (
            "Unavailable or insufficient records: "
            f"{len(insufficient_support):,}"
        ),
        "",
        "Spatial statuses:",
        *[
            f"- {key}: {value:,}"
            for key, value
            in status_counts.items()
        ],
        "",
        "No temperature value was changed.",
        "",
        f"Output: {OUTPUT_FILE}",
    ]

    (
        REPORT_DIRECTORY
        / "step5b_qc_report.txt"
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

    if station_count != EXPECTED_STATIONS:
        failures.append(
            "Unexpected station count."
        )

    if (
        len(spatial_stations)
        != EXPECTED_SPATIALLY_ELIGIBLE_STATIONS
    ):
        failures.append(
            "Unexpected spatially eligible station count."
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

    if failures:
        raise SystemExit(
            "\nSTEP 5B FAILED:\n- "
            + "\n- ".join(failures)
        )

    print("\nSTEP 5B PASSED.")


if __name__ == "__main__":
    main()
