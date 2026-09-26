from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import yaml
from shapely import MultiPoint, get_parts, voronoi_polygons


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

RAW_GEOMETRY_DIRECTORY = (
    PROJECT_ROOT
    / "01_raw_data"
    / "geospatial"
)

RAW_BOUNDARY_FILE = (
    RAW_GEOMETRY_DIRECTORY
    / "geoBoundaries-BGD-ADM0.geojson"
)

BOUNDARY_METADATA_FILE = (
    PROJECT_ROOT
    / "00_admin"
    / "step7d_geoboundaries_metadata.json"
)

BOUNDARY_OUTPUT_FILE = (
    PROJECT_ROOT
    / "02_metadata"
    / "step7d_bangladesh_boundary.gpkg"
)

VORONOI_OUTPUT_FILE = (
    PROJECT_ROOT
    / "02_metadata"
    / "step7d_primary_station_voronoi_weights.gpkg"
)

WEIGHT_TABLE_FILE = (
    PROJECT_ROOT
    / "02_metadata"
    / "step7d_primary_station_area_weights.csv"
)

OUTSIDE_STATION_REPORT = (
    PROJECT_ROOT
    / "05_qc_reports"
    / "step7d_station_boundary_distance.csv"
)

DUPLICATE_COORDINATE_REPORT = (
    PROJECT_ROOT
    / "05_qc_reports"
    / "step7d_duplicate_station_coordinates.csv"
)

MAP_FILE = (
    PROJECT_ROOT
    / "05_qc_reports"
    / "step7d_voronoi_area_weight_map.png"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "05_qc_reports"
    / "step7d_voronoi_weight_summary.json"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "05_qc_reports"
    / "step7d_voronoi_weight_report.txt"
)


LATITUDE_CANDIDATES = [
    "latitude",
    "station_latitude",
    "official_latitude",
    "latitude_dd",
    "lat_dd",
    "lat",
    "Latitude",
    "LATITUDE",
]

LONGITUDE_CANDIDATES = [
    "longitude",
    "station_longitude",
    "official_longitude",
    "longitude_dd",
    "lon_dd",
    "lon",
    "lng",
    "Longitude",
    "LONGITUDE",
]


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


def detect_column(
    columns: list[str],
    candidates: list[str],
    description: str,
) -> str:
    exact_lookup = {
        str(column).lower():
            str(column)
        for column in columns
    }

    for candidate in candidates:
        match = exact_lookup.get(
            candidate.lower()
        )

        if match is not None:
            return match

    raise ValueError(
        f"Could not identify the {description} column.\n"
        f"Available columns:\n{columns}"
    )


def union_geometry(
    geometries: gpd.GeoSeries,
):
    if hasattr(
        geometries,
        "union_all",
    ):
        return geometries.union_all()

    return geometries.unary_union


def download_json(url: str) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "User-Agent":
                "Bangladesh-Cold-Wave-Research/1.0"
        },
    )

    with urlopen(
        request,
        timeout=120,
    ) as response:
        return json.loads(
            response.read().decode("utf-8")
        )


def download_binary(
    url: str,
    destination: Path,
) -> None:
    request = Request(
        url,
        headers={
            "User-Agent":
                "Bangladesh-Cold-Wave-Research/1.0"
        },
    )

    with urlopen(
        request,
        timeout=180,
    ) as response:
        destination.write_bytes(
            response.read()
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

    for directory in [
        RAW_GEOMETRY_DIRECTORY,
        BOUNDARY_OUTPUT_FILE.parent,
        VORONOI_OUTPUT_FILE.parent,
        MAP_FILE.parent,
        BOUNDARY_METADATA_FILE.parent,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    if tuple(
        shapely.geos_version
    ) < (3, 12, 0):
        raise RuntimeError(
            "GEOS 3.12 or newer is required for "
            "ordered Voronoi polygons. Installed: "
            f"{shapely.geos_version_string}"
        )

    policy = load_yaml(POLICY_FILE)

    boundary_rules = policy["boundary"]
    network_rules = policy["network"]
    projection_rules = policy[
        "spatial_projection"
    ]

    member_column = str(
        network_rules[
            "membership_column"
        ]
    )

    area_crs = str(
        projection_rules[
            "proj_string"
        ]
    )

    api_endpoint = str(
        boundary_rules["api_endpoint"]
    )

    expected_boundary_id = str(
        boundary_rules[
            "expected_boundary_id"
        ]
    )

    allow_boundary_change = bool(
        boundary_rules[
            "allow_boundary_id_change"
        ]
    )

    # --------------------------------------------------
    # Download and freeze boundary metadata
    # --------------------------------------------------

    if (
        not BOUNDARY_METADATA_FILE.exists()
        or not RAW_BOUNDARY_FILE.exists()
    ):
        metadata = download_json(
            api_endpoint
        )

        boundary_id = str(
            metadata["boundaryID"]
        )

        if (
            boundary_id
            != expected_boundary_id
            and not allow_boundary_change
        ):
            raise ValueError(
                "geoBoundaries boundary ID changed.\n"
                f"Expected: {expected_boundary_id}\n"
                f"Received: {boundary_id}\n"
                "Review the new boundary before updating "
                "the policy."
            )

        BOUNDARY_METADATA_FILE.write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        download_binary(
            str(metadata["gjDownloadURL"]),
            RAW_BOUNDARY_FILE,
        )
    else:
        metadata = json.loads(
            BOUNDARY_METADATA_FILE.read_text(
                encoding="utf-8"
            )
        )

    # --------------------------------------------------
    # Read station information
    # --------------------------------------------------

    data = pd.read_parquet(
        INPUT_FILE
    )

    if member_column not in data.columns:
        raise ValueError(
            f"Missing membership column: "
            f"{member_column}"
        )

    data[member_column] = parse_boolean(
        data[member_column]
    )

    latitude_column = detect_column(
        list(data.columns),
        LATITUDE_CANDIDATES,
        "latitude",
    )

    longitude_column = detect_column(
        list(data.columns),
        LONGITUDE_CANDIDATES,
        "longitude",
    )

    metadata_columns = [
        "station_uid",
        "station_name_display",
        latitude_column,
        longitude_column,
    ]

    if "station_id" in data.columns:
        metadata_columns.insert(
            1,
            "station_id",
        )

    primary = data.loc[
        data[member_column],
        metadata_columns,
    ].copy()

    coordinate_variation = (
        primary.groupby(
            "station_uid"
        )[
            [
                latitude_column,
                longitude_column,
            ]
        ]
        .nunique(
            dropna=False
        )
    )

    inconsistent = (
        coordinate_variation.gt(1)
        .any(axis=1)
    )

    if inconsistent.any():
        raise ValueError(
            "One or more stations have changing "
            "coordinates in the daily dataset: "
            f"{coordinate_variation.index[inconsistent].tolist()}"
        )

    stations = (
        primary.drop_duplicates(
            subset=["station_uid"]
        )
        .sort_values("station_uid")
        .reset_index(drop=True)
    )

    stations[latitude_column] = pd.to_numeric(
        stations[latitude_column],
        errors="coerce",
    )

    stations[longitude_column] = pd.to_numeric(
        stations[longitude_column],
        errors="coerce",
    )

    missing_coordinates = (
        stations[
            [
                latitude_column,
                longitude_column,
            ]
        ]
        .isna()
        .any(axis=1)
    )

    if missing_coordinates.any():
        raise ValueError(
            "Primary-network stations with missing "
            "coordinates:\n"
            + stations.loc[
                missing_coordinates,
                [
                    "station_uid",
                    "station_name_display",
                ],
            ].to_string(index=False)
        )

    duplicate_coordinates = stations.duplicated(
        subset=[
            latitude_column,
            longitude_column,
        ],
        keep=False,
    )

    stations.loc[
        duplicate_coordinates
    ].to_csv(
        DUPLICATE_COORDINATE_REPORT,
        index=False,
    )

    if duplicate_coordinates.any():
        raise ValueError(
            "Two or more primary stations have identical "
            "coordinates. Review:\n"
            f"{DUPLICATE_COORDINATE_REPORT}"
        )

    station_points = gpd.GeoDataFrame(
        stations.copy(),
        geometry=gpd.points_from_xy(
            stations[longitude_column],
            stations[latitude_column],
        ),
        crs="EPSG:4326",
    )

    # --------------------------------------------------
    # Read and project Bangladesh boundary
    # --------------------------------------------------

    boundary_source = gpd.read_file(
        RAW_BOUNDARY_FILE
    )

    if boundary_source.crs is None:
        boundary_source = (
            boundary_source.set_crs(
                "EPSG:4326"
            )
        )

    boundary_source = boundary_source.to_crs(
        "EPSG:4326"
    )

    boundary_wgs84 = union_geometry(
        boundary_source.geometry
    )

    if boundary_wgs84.is_empty:
        raise ValueError(
            "Bangladesh boundary geometry is empty."
        )

    boundary_gdf = gpd.GeoDataFrame(
        {
            "boundary_id": [
                metadata["boundaryID"]
            ],
            "boundary_name": [
                metadata["boundaryName"]
            ],
            "boundary_year": [
                metadata[
                    "boundaryYearRepresented"
                ]
            ],
            "boundary_source": [
                metadata["boundarySource"]
            ],
            "boundary_license": [
                metadata["boundaryLicense"]
            ],
        },
        geometry=[boundary_wgs84],
        crs="EPSG:4326",
    )

    boundary_projected = (
        boundary_gdf.to_crs(
            area_crs
        )
    )

    boundary_geometry = (
        boundary_projected.geometry.iloc[0]
    )

    points_projected = (
        station_points.to_crs(
            area_crs
        )
        .sort_values("station_uid")
        .reset_index(drop=True)
    )

    # Check station distance from Bangladesh.
    station_distances = (
        points_projected.geometry.distance(
            boundary_geometry
        )
        / 1000.0
    )

    distance_report = stations[
        [
            column
            for column in [
                "station_uid",
                "station_id",
                "station_name_display",
                latitude_column,
                longitude_column,
            ]
            if column in stations.columns
        ]
    ].copy()

    distance_report[
        "distance_outside_boundary_km"
    ] = station_distances.to_numpy()

    distance_report[
        "inside_or_on_boundary"
    ] = distance_report[
        "distance_outside_boundary_km"
    ].le(0.001)

    distance_report.to_csv(
        OUTSIDE_STATION_REPORT,
        index=False,
    )

    if station_distances.max() > 50.0:
        raise ValueError(
            "At least one primary station is more than "
            "50 km outside the Bangladesh boundary. "
            f"Review {OUTSIDE_STATION_REPORT}"
        )

    # --------------------------------------------------
    # Generate ordered Voronoi polygons
    # --------------------------------------------------

    point_coordinates = [
        (
            geometry.x,
            geometry.y,
        )
        for geometry
        in points_projected.geometry
    ]

    multipoint = MultiPoint(
        point_coordinates
    )

    voronoi_collection = voronoi_polygons(
        multipoint,
        extend_to=boundary_geometry.envelope,
        ordered=True,
    )

    voronoi_parts = list(
        get_parts(
            voronoi_collection
        )
    )

    if (
        len(voronoi_parts)
        != len(points_projected)
    ):
        raise ValueError(
            "Voronoi polygon count does not equal "
            "station count.\n"
            f"Stations: {len(points_projected)}\n"
            f"Polygons: {len(voronoi_parts)}"
        )

    clipped_geometries = []

    for polygon in voronoi_parts:
        clipped = polygon.intersection(
            boundary_geometry
        )

        if clipped.is_empty:
            raise ValueError(
                "A station Voronoi polygon became empty "
                "after clipping."
            )

        clipped_geometries.append(
            clipped
        )

    voronoi = gpd.GeoDataFrame(
        points_projected.drop(
            columns="geometry"
        ),
        geometry=clipped_geometries,
        crs=area_crs,
    )

    voronoi[
        "cell_area_km2"
    ] = (
        voronoi.geometry.area
        / 1_000_000.0
    )

    national_area_km2 = float(
        boundary_geometry.area
        / 1_000_000.0
    )

    voronoi[
        "national_area_weight"
    ] = (
        voronoi[
            "cell_area_km2"
        ]
        / national_area_km2
    )

    weight_sum = float(
        voronoi[
            "national_area_weight"
        ].sum()
    )

    coverage_union = union_geometry(
        voronoi.geometry
    )

    coverage_gap_km2 = float(
        boundary_geometry.difference(
            coverage_union
        ).area
        / 1_000_000.0
    )

    overlap_km2 = float(
        (
            voronoi.geometry.area.sum()
            - coverage_union.area
        )
        / 1_000_000.0
    )

    if not np.isclose(
        weight_sum,
        1.0,
        atol=1e-8,
        rtol=0.0,
    ):
        raise ValueError(
            "Voronoi weights do not sum to one: "
            f"{weight_sum}"
        )

    tolerance_area_km2 = max(
        0.01,
        national_area_km2 * 1e-7,
    )

    if (
        coverage_gap_km2
        > tolerance_area_km2
        or overlap_km2
        > tolerance_area_km2
    ):
        raise ValueError(
            "Voronoi polygons do not partition the "
            "Bangladesh boundary adequately.\n"
            f"Gap: {coverage_gap_km2} km2\n"
            f"Overlap: {overlap_km2} km2"
        )

    projected_centroids = (
        voronoi.geometry.centroid
    )

    centroid_wgs84 = gpd.GeoSeries(
        projected_centroids,
        crs=area_crs,
    ).to_crs(
        "EPSG:4326"
    )

    voronoi[
        "cell_centroid_longitude"
    ] = centroid_wgs84.x.to_numpy()

    voronoi[
        "cell_centroid_latitude"
    ] = centroid_wgs84.y.to_numpy()

    voronoi[
        "weight_rank_largest_first"
    ] = (
        voronoi[
            "national_area_weight"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    # --------------------------------------------------
    # Save geometry and weight table
    # --------------------------------------------------

    boundary_gdf.to_file(
        BOUNDARY_OUTPUT_FILE,
        layer="bangladesh_boundary",
        driver="GPKG",
    )

    voronoi_wgs84 = voronoi.to_crs(
        "EPSG:4326"
    )

    voronoi_wgs84.to_file(
        VORONOI_OUTPUT_FILE,
        layer="station_voronoi_weights",
        driver="GPKG",
    )

    weight_columns = [
        column
        for column in [
            "station_uid",
            "station_id",
            "station_name_display",
            latitude_column,
            longitude_column,
            "cell_area_km2",
            "national_area_weight",
            "cell_centroid_longitude",
            "cell_centroid_latitude",
            "weight_rank_largest_first",
        ]
        if column in voronoi.columns
    ]

    weight_table = (
        voronoi[
            weight_columns
        ]
        .sort_values("station_uid")
        .reset_index(drop=True)
    )

    weight_table.to_csv(
        WEIGHT_TABLE_FILE,
        index=False,
    )

    # --------------------------------------------------
    # Create the QC map
    # --------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(8.5, 10.0)
    )

    voronoi_wgs84.plot(
        ax=axis,
        column="national_area_weight",
        legend=True,
        edgecolor="black",
        linewidth=0.45,
        alpha=0.75,
        legend_kwds={
            "label":
                "Fraction of Bangladesh area",
            "shrink": 0.65,
        },
    )

    boundary_gdf.boundary.plot(
        ax=axis,
        linewidth=1.2,
        edgecolor="black",
    )

    station_points.plot(
        ax=axis,
        marker="o",
        markersize=16,
        edgecolor="black",
        linewidth=0.35,
    )

    axis.set_title(
        "Bangladesh primary-station Voronoi area weights"
    )

    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_aspect("equal")

    axis.text(
        0.01,
        0.01,
        (
            "Boundary: geoBoundaries "
            f"{metadata['boundaryID']}"
        ),
        transform=axis.transAxes,
        fontsize=7,
        verticalalignment="bottom",
    )

    figure.tight_layout()

    figure.savefig(
        MAP_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    summary = {
        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "boundary_id":
            metadata["boundaryID"],
        "boundary_year_represented":
            metadata[
                "boundaryYearRepresented"
            ],
        "boundary_source":
            metadata["boundarySource"],
        "boundary_license":
            metadata["boundaryLicense"],
        "area_projection":
            area_crs,
        "primary_station_count":
            len(voronoi),
        "national_boundary_area_km2":
            national_area_km2,
        "weight_sum":
            weight_sum,
        "minimum_station_weight":
            float(
                voronoi[
                    "national_area_weight"
                ].min()
            ),
        "maximum_station_weight":
            float(
                voronoi[
                    "national_area_weight"
                ].max()
            ),
        "median_station_weight":
            float(
                voronoi[
                    "national_area_weight"
                ].median()
            ),
        "coverage_gap_km2":
            coverage_gap_km2,
        "overlap_km2":
            overlap_km2,
        "maximum_station_distance_outside_km":
            float(
                station_distances.max()
            ),
        "latitude_column":
            latitude_column,
        "longitude_column":
            longitude_column,
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
        "STEP 7D-A: BANGLADESH VORONOI AREA WEIGHTS",
        "=" * 50,
        (
            "Boundary ID: "
            f"{metadata['boundaryID']}"
        ),
        (
            "Boundary year represented: "
            f"{metadata['boundaryYearRepresented']}"
        ),
        (
            "Primary stations: "
            f"{len(voronoi)}"
        ),
        (
            "Calculated national area: "
            f"{national_area_km2:,.3f} km2"
        ),
        (
            "Station weight sum: "
            f"{weight_sum:.12f}"
        ),
        (
            "Minimum station weight: "
            f"{voronoi['national_area_weight'].min():.6f}"
        ),
        (
            "Maximum station weight: "
            f"{voronoi['national_area_weight'].max():.6f}"
        ),
        (
            "Coverage gap: "
            f"{coverage_gap_km2:.8f} km2"
        ),
        (
            "Polygon overlap: "
            f"{overlap_km2:.8f} km2"
        ),
        "",
        f"Weight table: {WEIGHT_TABLE_FILE}",
        f"Voronoi polygons: {VORONOI_OUTPUT_FILE}",
        f"QC map: {MAP_FILE}",
    ]

    REPORT_FILE.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(report_lines))
    print("\nSTEP 7D-A PASSED.")


if __name__ == "__main__":
    main()
