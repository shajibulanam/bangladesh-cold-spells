from __future__ import annotations

import calendar
import time
from pathlib import Path

import cdsapi
import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "01_raw_data"
    / "era5"
    / "tropospheric_indices"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PL = "reanalysis-era5-pressure-levels"
DATASET_SL = "reanalysis-era5-single-levels"

AREA = [90, 20, 20, 140]
GRID = [1.0, 1.0]

START_WINTER = 1985
END_WINTER = 2024

MAX_ATTEMPTS = 3


def winter_months() -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []

    for winter_start in range(
        START_WINTER,
        END_WINTER + 1,
    ):
        months.extend(
            [
                (winter_start, 10),
                (winter_start, 11),
                (winter_start, 12),
                (winter_start + 1, 1),
                (winter_start + 1, 2),
                (winter_start + 1, 3),
            ]
        )

    return sorted(set(months))


def valid_netcdf(
    path: Path,
    expected_days: int,
    variable_kind: str,
) -> bool:
    if not path.exists():
        return False

    if path.stat().st_size < 10_000:
        return False

    try:
        with xr.open_dataset(path) as ds:
            dimensions = set(ds.dims)
            variables = set(ds.data_vars)

            time_name = (
                "valid_time"
                if "valid_time" in ds.coords
                else "time"
            )

            if time_name not in ds.coords:
                return False

            if ds.sizes[time_name] != expected_days:
                return False

            if variable_kind == "z500":
                candidates = {
                    "z",
                    "geopotential",
                }
            else:
                candidates = {
                    "msl",
                    "mean_sea_level_pressure",
                }

            if not candidates.intersection(variables):
                return False

            lat_name = (
                "latitude"
                if "latitude" in ds.coords
                else "lat"
            )

            lon_name = (
                "longitude"
                if "longitude" in ds.coords
                else "lon"
            )

            latitude = np.asarray(
                ds[lat_name].values
            )

            longitude = np.asarray(
                ds[lon_name].values
            )

            if np.nanmax(latitude) < 89:
                return False

            if np.nanmin(latitude) > 21:
                return False

            if np.nanmin(longitude) > 21:
                return False

            if np.nanmax(longitude) < 139:
                return False

        return True

    except Exception:
        return False


def retrieve_with_retry(
    client: cdsapi.Client,
    dataset: str,
    request: dict,
    target: Path,
    expected_days: int,
    variable_kind: str,
) -> None:
    if valid_netcdf(
        target,
        expected_days,
        variable_kind,
    ):
        print(f"SKIP valid: {target.name}")
        return

    if target.exists():
        target.unlink()

    temporary = target.with_suffix(
        target.suffix + ".part"
    )

    if temporary.exists():
        temporary.unlink()

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):
        try:
            print(
                f"DOWNLOAD attempt {attempt}: "
                f"{target.name}"
            )

            client.retrieve(
                dataset,
                request,
                str(temporary),
            )

            temporary.replace(target)

            if not valid_netcdf(
                target,
                expected_days,
                variable_kind,
            ):
                raise RuntimeError(
                    "Downloaded file failed validation."
                )

            print(f"PASS: {target.name}")
            return

        except Exception as error:
            print(
                f"FAILED attempt {attempt}: "
                f"{target.name}: {error}"
            )

            if temporary.exists():
                temporary.unlink()

            if target.exists():
                target.unlink()

            if attempt == MAX_ATTEMPTS:
                raise

            time.sleep(30 * attempt)


def main() -> None:
    client = cdsapi.Client()

    months = winter_months()

    print(
        "STEP 10A-ERA5 DOWNLOAD"
    )
    print("=" * 70)
    print(f"Monthly periods: {len(months)}")
    print("Variables: MSLP and Z500")
    print("Grid: 1 degree")
    print("Domain: 20-90N, 20-140E")
    print("")

    for year, month in months:
        number_of_days = calendar.monthrange(
            year,
            month,
        )[1]

        days = [
            f"{day:02d}"
            for day in range(
                1,
                number_of_days + 1,
            )
        ]

        common = {
            "product_type": ["reanalysis"],
            "year": [str(year)],
            "month": [f"{month:02d}"],
            "day": days,
            "time": ["00:00"],
            "area": AREA,
            "grid": GRID,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

        z500_target = (
            OUTPUT_DIR
            / (
                f"ERA5_TROPINDEX_Z500_"
                f"{year}_{month:02d}.nc"
            )
        )

        z500_request = {
            **common,
            "variable": ["geopotential"],
            "pressure_level": ["500"],
        }

        retrieve_with_retry(
            client=client,
            dataset=DATASET_PL,
            request=z500_request,
            target=z500_target,
            expected_days=number_of_days,
            variable_kind="z500",
        )

        mslp_target = (
            OUTPUT_DIR
            / (
                f"ERA5_TROPINDEX_MSLP_"
                f"{year}_{month:02d}.nc"
            )
        )

        mslp_request = {
            **common,
            "variable": [
                "mean_sea_level_pressure"
            ],
        }

        retrieve_with_retry(
            client=client,
            dataset=DATASET_SL,
            request=mslp_request,
            target=mslp_target,
            expected_days=number_of_days,
            variable_kind="mslp",
        )

    print("")
    print(
        "STEP 10A DOWNLOAD PASSED."
    )


if __name__ == "__main__":
    main()
