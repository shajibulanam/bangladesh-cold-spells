from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "01_raw_data"
    / "era5"
    / "tropospheric_indices"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "03_intermediate"
    / "tropospheric_indices"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "08_outputs"
    / "tables"
    / "tropospheric_indices"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "05_qc_reports"
    / "tropospheric_indices"
)

for folder in [
    OUTPUT_DIR,
    TABLE_DIR,
    REPORT_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

DAILY_CSV = (
    TABLE_DIR
    / "table_22_daily_siberian_high_blocking_indices.csv"
)

CLIM_CSV = (
    TABLE_DIR
    / "table_22b_siberian_high_calendar_day_climatology.csv"
)

BLOCKING_NC = (
    OUTPUT_DIR
    / "ural_siberian_local_blocking_daily.nc"
)

SUMMARY_JSON = (
    REPORT_DIR
    / "step10a_daily_index_summary.json"
)

REPORT_TXT = (
    REPORT_DIR
    / "step10a_daily_index_report.txt"
)

GRAVITY = 9.80665

OFFSETS = [-5, 0, 5]

URAL_WEST = 40
URAL_EAST = 90

MINIMUM_BLOCK_WIDTH_DEGREES = 15


def collapse_expver(
    data: xr.DataArray,
) -> xr.DataArray:
    if "expver" not in data.dims:
        return data

    result = data.isel(expver=0)

    for index in range(
        1,
        data.sizes["expver"],
    ):
        result = result.combine_first(
            data.isel(expver=index)
        )

    return result


def standardize_dataarray(
    path: Path,
    variable_candidates: list[str],
) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        rename = {}

        if "valid_time" in ds.coords:
            rename["valid_time"] = "time"

        if "latitude" in ds.coords:
            rename["latitude"] = "lat"

        if "longitude" in ds.coords:
            rename["longitude"] = "lon"

        if "pressure_level" in ds.coords:
            rename["pressure_level"] = "level"

        ds = ds.rename(rename)

        variable_name = None

        for candidate in variable_candidates:
            if candidate in ds.data_vars:
                variable_name = candidate
                break

        if variable_name is None:
            raise KeyError(
                f"{path.name}: could not find any of "
                f"{variable_candidates}. Available: "
                f"{list(ds.data_vars)}"
            )

        data = collapse_expver(
            ds[variable_name]
        )

        if "level" in data.dims:
            if 500 in data["level"]:
                data = data.sel(level=500)
            else:
                data = data.squeeze(
                    "level",
                    drop=True,
                )

        data = data.squeeze(drop=True)

        required = {
            "time",
            "lat",
            "lon",
        }

        missing = required - set(data.dims)

        if missing:
            raise RuntimeError(
                f"{path.name}: missing dimensions "
                f"{sorted(missing)}."
            )

        data = data.transpose(
            "time",
            "lat",
            "lon",
        ).load()

    data = data.sortby(
        "lat",
        ascending=True,
    ).sortby(
        "lon",
        ascending=True,
    )

    return data


def area_weighted_mean(
    data: xr.DataArray,
    south: float,
    north: float,
    west: float,
    east: float,
) -> xr.DataArray:
    selected = data.where(
        (
            (data["lat"] >= south)
            & (data["lat"] <= north)
            & (data["lon"] >= west)
            & (data["lon"] <= east)
        ),
        drop=True,
    )

    weights = np.cos(
        np.deg2rad(
            selected["lat"]
        )
    )

    return selected.weighted(
        weights
    ).mean(
        dim=[
            "lat",
            "lon",
        ]
    )


def maximum_contiguous_width(
    blocked: np.ndarray,
    longitudes: np.ndarray,
) -> float:
    if len(blocked) == 0:
        return 0.0

    grid_spacing = float(
        np.median(
            np.diff(longitudes)
        )
    )

    maximum_count = 0
    current_count = 0

    for value in blocked.astype(bool):
        if value:
            current_count += 1
            maximum_count = max(
                maximum_count,
                current_count,
            )
        else:
            current_count = 0

    return maximum_count * grid_spacing


def consecutive_episode_flag(
    dates: pd.Series,
    daily_flag: pd.Series,
    minimum_days: int,
) -> pd.Series:
    output = pd.Series(
        False,
        index=daily_flag.index,
    )

    run_indices: list[int] = []

    for position in range(
        len(daily_flag)
    ):
        active = bool(
            daily_flag.iloc[position]
        )

        continuous = (
            position == 0
            or (
                dates.iloc[position]
                - dates.iloc[position - 1]
            ).days == 1
        )

        if not active or not continuous:
            if len(run_indices) >= minimum_days:
                output.iloc[
                    run_indices
                ] = True

            run_indices = []

        if active:
            run_indices.append(position)

    if len(run_indices) >= minimum_days:
        output.iloc[
            run_indices
        ] = True

    return output


def process_pair(
    z500_path: Path,
    mslp_path: Path,
) -> tuple[pd.DataFrame, xr.DataArray]:
    z500 = standardize_dataarray(
        z500_path,
        [
            "z",
            "geopotential",
        ],
    )

    mslp = standardize_dataarray(
        mslp_path,
        [
            "msl",
            "mean_sea_level_pressure",
        ],
    )

    if not np.array_equal(
        z500["time"].values,
        mslp["time"].values,
    ):
        raise RuntimeError(
            f"Time mismatch: {z500_path.name} "
            f"versus {mslp_path.name}"
        )

    z500_values = z500.astype("float64")

    if float(
        np.nanmedian(
            np.abs(z500_values.values)
        )
    ) > 20_000:
        z500_values = (
            z500_values
            / GRAVITY
        )

    mslp_values = mslp.astype("float64")

    if float(
        np.nanmedian(
            mslp_values.values
        )
    ) > 2_000:
        mslp_values = (
            mslp_values
            / 100.0
        )

    shi_mean = area_weighted_mean(
        mslp_values,
        south=40,
        north=60,
        west=80,
        east=120,
    )

    shi_region = mslp_values.where(
        (
            (mslp_values["lat"] >= 40)
            & (mslp_values["lat"] <= 65)
            & (mslp_values["lon"] >= 80)
            & (mslp_values["lon"] <= 120)
        ),
        drop=True,
    )

    shi_maximum = shi_region.max(
        dim=[
            "lat",
            "lon",
        ]
    )

    sector_lon = z500_values["lon"].where(
        (
            (z500_values["lon"] >= URAL_WEST)
            & (z500_values["lon"] <= URAL_EAST)
        ),
        drop=True,
    )

    offset_masks = []

    for offset in OFFSETS:
        phi_south = 40 + offset
        phi_central = 60 + offset
        phi_north = 80 + offset

        z_south = z500_values.interp(
            lat=phi_south
        )

        z_central = z500_values.interp(
            lat=phi_central
        )

        z_north = z500_values.interp(
            lat=phi_north
        )

        ghgs = (
            z_central
            - z_south
        ) / (
            phi_central
            - phi_south
        )

        ghgn = (
            z_north
            - z_central
        ) / (
            phi_north
            - phi_central
        )

        offset_masks.append(
            (
                (ghgs > 0)
                & (ghgn < -10)
            )
        )

    local_blocking = xr.concat(
        offset_masks,
        dim="offset",
    ).any(
        dim="offset"
    )

    local_sector = local_blocking.sel(
        lon=sector_lon
    )

    blocking_fraction = (
        local_sector.mean(
            dim="lon"
        )
    )

    rows = []

    for time_index, time_value in enumerate(
        pd.to_datetime(
            z500_values["time"].values
        )
    ):
        blocked_values = (
            local_sector
            .isel(time=time_index)
            .values
            .astype(bool)
        )

        longitude_values = (
            local_sector["lon"].values
        )

        width = maximum_contiguous_width(
            blocked_values,
            longitude_values,
        )

        rows.append(
            {
                "date": time_value,
                "shi_mean_mslp_hpa": float(
                    shi_mean.isel(
                        time=time_index
                    ).values
                ),
                "shi_maximum_mslp_hpa": float(
                    shi_maximum.isel(
                        time=time_index
                    ).values
                ),
                "ural_blocking_fraction": float(
                    blocking_fraction.isel(
                        time=time_index
                    ).values
                ),
                "ural_max_contiguous_width_deg": (
                    width
                ),
                "ural_sector_block_day": bool(
                    width
                    >= MINIMUM_BLOCK_WIDTH_DEGREES
                ),
            }
        )

    return (
        pd.DataFrame(rows),
        local_sector.astype("uint8"),
    )


def main() -> None:
    z500_files = sorted(
        INPUT_DIR.glob(
            "ERA5_TROPINDEX_Z500_*.nc"
        )
    )

    mslp_files = sorted(
        INPUT_DIR.glob(
            "ERA5_TROPINDEX_MSLP_*.nc"
        )
    )

    if len(z500_files) != 240:
        raise RuntimeError(
            f"Expected 240 Z500 files, "
            f"found {len(z500_files)}."
        )

    if len(mslp_files) != 240:
        raise RuntimeError(
            f"Expected 240 MSLP files, "
            f"found {len(mslp_files)}."
        )

    mslp_lookup = {
        path.name.replace(
            "MSLP",
            "Z500",
        ): path
        for path in mslp_files
    }

    daily_parts = []
    blocking_parts = []

    for index, z500_path in enumerate(
        z500_files,
        start=1,
    ):
        expected_mslp_name = (
            z500_path.name.replace(
                "Z500",
                "MSLP",
            )
        )

        mslp_path = (
            INPUT_DIR
            / expected_mslp_name
        )

        if not mslp_path.exists():
            raise FileNotFoundError(
                mslp_path
            )

        print(
            f"[{index:03d}/{len(z500_files)}] "
            f"{z500_path.name}"
        )

        daily_part, blocking_part = (
            process_pair(
                z500_path,
                mslp_path,
            )
        )

        daily_parts.append(
            daily_part
        )

        blocking_parts.append(
            blocking_part
        )

    daily = pd.concat(
        daily_parts,
        ignore_index=True,
    ).sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    if daily["date"].duplicated().any():
        raise RuntimeError(
            "Duplicate dates found in daily index."
        )

    daily["month_day"] = (
        daily["date"]
        .dt.strftime("%m-%d")
    )

    baseline = daily.loc[
        daily["date"].dt.year.between(
            1991,
            2020,
        )
    ].copy()

    climatology = (
        baseline.groupby(
            "month_day",
            as_index=False,
        )
        .agg(
            shi_mean_climatology_hpa=(
                "shi_mean_mslp_hpa",
                "mean",
            ),
            shi_mean_climatology_sd_hpa=(
                "shi_mean_mslp_hpa",
                "std",
            ),
            shi_maximum_climatology_hpa=(
                "shi_maximum_mslp_hpa",
                "mean",
            ),
            baseline_day_count=(
                "date",
                "count",
            ),
        )
    )

    daily = daily.merge(
        climatology,
        on="month_day",
        how="left",
        validate="many_to_one",
    )

    if daily[
        "shi_mean_climatology_hpa"
    ].isna().any():
        raise RuntimeError(
            "Missing calendar-day climatology."
        )

    daily[
        "shi_mean_anomaly_hpa"
    ] = (
        daily["shi_mean_mslp_hpa"]
        - daily[
            "shi_mean_climatology_hpa"
        ]
    )

    daily[
        "shi_standardized_anomaly"
    ] = (
        daily["shi_mean_anomaly_hpa"]
        / daily[
            "shi_mean_climatology_sd_hpa"
        ]
    )

    daily[
        "shi_maximum_anomaly_hpa"
    ] = (
        daily["shi_maximum_mslp_hpa"]
        - daily[
            "shi_maximum_climatology_hpa"
        ]
    )

    daily[
        "ural_block_episode_3day"
    ] = consecutive_episode_flag(
        dates=daily["date"],
        daily_flag=daily[
            "ural_sector_block_day"
        ],
        minimum_days=3,
    )

    daily[
        "ural_block_episode_5day"
    ] = consecutive_episode_flag(
        dates=daily["date"],
        daily_flag=daily[
            "ural_sector_block_day"
        ],
        minimum_days=5,
    )

    daily.to_csv(
        DAILY_CSV,
        index=False,
        date_format="%Y-%m-%d",
    )

    climatology.to_csv(
        CLIM_CSV,
        index=False,
    )

    blocking = xr.concat(
        blocking_parts,
        dim="time",
    ).sortby(
        "time"
    )

    blocking.name = (
        "local_blocking_tm90"
    )

    blocking.attrs.update(
        {
            "description": (
                "Local Tibaldi-Molteni-type "
                "blocking occurrence"
            ),
            "units": "0 or 1",
            "offsets_degrees": (
                "-5, 0, +5"
            ),
            "conditions": (
                "GHGS > 0 and GHGN < "
                "-10 m per degree latitude"
            ),
        }
    )

    blocking.to_dataset().to_netcdf(
        BLOCKING_NC,
        engine="netcdf4",
        encoding={
            "local_blocking_tm90": {
                "zlib": True,
                "complevel": 4,
            }
        },
    )

    summary = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "daily_record_count": int(
            len(daily)
        ),
        "first_date": str(
            daily["date"].min().date()
        ),
        "last_date": str(
            daily["date"].max().date()
        ),
        "three_day_block_episode_days": int(
            daily[
                "ural_block_episode_3day"
            ].sum()
        ),
        "five_day_block_episode_days": int(
            daily[
                "ural_block_episode_5day"
            ].sum()
        ),
        "mean_shi_hpa": float(
            daily["shi_mean_mslp_hpa"].mean()
        ),
        "mean_blocking_fraction": float(
            daily[
                "ural_blocking_fraction"
            ].mean()
        ),
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "STEP 10A: DAILY TROPOSPHERIC INDICES",
        "=" * 64,
        f"Daily records: {len(daily)}",
        (
            f"Date range: "
            f"{daily['date'].min().date()} to "
            f"{daily['date'].max().date()}"
        ),
        (
            "Mean Siberian High index: "
            f"{daily['shi_mean_mslp_hpa'].mean():.3f} hPa"
        ),
        (
            "Three-day blocking-episode days: "
            f"{daily['ural_block_episode_3day'].sum()}"
        ),
        (
            "Five-day blocking-episode days: "
            f"{daily['ural_block_episode_5day'].sum()}"
        ),
        "",
        f"Daily table: {DAILY_CSV}",
        f"Climatology: {CLIM_CSV}",
        f"Local blocking NetCDF: {BLOCKING_NC}",
    ]

    REPORT_TXT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("")
    print("\n".join(lines))
    print("")
    print(
        "STEP 10A-DAILY PASSED."
    )


if __name__ == "__main__":
    main()
