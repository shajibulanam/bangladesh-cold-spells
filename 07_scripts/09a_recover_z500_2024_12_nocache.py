from __future__ import annotations

import shutil
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

REPORT_DIR = (
    PROJECT_ROOT
    / "05_qc_reports"
    / "tropospheric_indices"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_FILE = (
    OUTPUT_DIR
    / "ERA5_TROPINDEX_Z500_2024_12.nc"
)

FULL_PART_FILE = (
    OUTPUT_DIR
    / "ERA5_TROPINDEX_Z500_2024_12_full.part.nc"
)

FIRST_HALF_FILE = (
    OUTPUT_DIR
    / "ERA5_TROPINDEX_Z500_2024_12_days01_15.part.nc"
)

SECOND_HALF_FILE = (
    OUTPUT_DIR
    / "ERA5_TROPINDEX_Z500_2024_12_days16_31.part.nc"
)

MERGED_TEMP_FILE = (
    OUTPUT_DIR
    / "ERA5_TROPINDEX_Z500_2024_12_merged.part.nc"
)

REPORT_FILE = (
    REPORT_DIR
    / "step10a_z500_2024_12_nocache_recovery_report.txt"
)

DATASET = "reanalysis-era5-pressure-levels"

AREA = [90, 20, 20, 140]
GRID = [1.0, 1.0]

EXPECTED_YEAR = 2024
EXPECTED_MONTH = 12
EXPECTED_DAYS = 31

MAX_ATTEMPTS = 3


def identify_coordinate(
    dataset: xr.Dataset,
    candidates: list[str],
) -> str:
    for candidate in candidates:
        if candidate in dataset.coords or candidate in dataset.dims:
            return candidate

    raise KeyError(
        f"None of the coordinates {candidates} were found. "
        f"Coordinates: {list(dataset.coords)}; "
        f"dimensions: {dict(dataset.sizes)}"
    )


def identify_variable(
    dataset: xr.Dataset,
) -> str:
    for candidate in ["z", "geopotential"]:
        if candidate in dataset.data_vars:
            return candidate

    raise KeyError(
        "Geopotential variable not found. "
        f"Available variables: {list(dataset.data_vars)}"
    )


def validate_file(
    path: Path,
    expected_days: int,
    expected_first_day: int,
    expected_last_day: int,
) -> tuple[bool, str]:
    if not path.exists():
        return False, "file does not exist"

    if path.stat().st_size < 10_000:
        return False, "file is unexpectedly small"

    try:
        with path.open("rb") as stream:
            signature = stream.read(4)

        if signature[:2] == b"PK":
            return False, "file is a ZIP archive rather than NetCDF"

        with xr.open_dataset(path) as dataset:
            time_name = identify_coordinate(
                dataset,
                ["valid_time", "time"],
            )

            lat_name = identify_coordinate(
                dataset,
                ["latitude", "lat"],
            )

            lon_name = identify_coordinate(
                dataset,
                ["longitude", "lon"],
            )

            variable_name = identify_variable(dataset)

            if int(dataset.sizes[time_name]) != expected_days:
                return (
                    False,
                    "incorrect number of days: "
                    f"expected {expected_days}, "
                    f"found {dataset.sizes[time_name]}",
                )

            dates = pd_datetime_days(
                dataset[time_name].values
            )

            years = dates.astype("datetime64[Y]").astype(int) + 1970
            months = (
                dates.astype("datetime64[M]").astype(int) % 12
            ) + 1

            month_start = dates.astype("datetime64[M]")
            days = (
                dates
                - month_start
            ).astype("timedelta64[D]").astype(int) + 1

            if not np.all(years == EXPECTED_YEAR):
                return False, "unexpected year in downloaded data"

            if not np.all(months == EXPECTED_MONTH):
                return False, "unexpected month in downloaded data"

            if int(days.min()) != expected_first_day:
                return (
                    False,
                    f"first day is {days.min()}, "
                    f"expected {expected_first_day}",
                )

            if int(days.max()) != expected_last_day:
                return (
                    False,
                    f"last day is {days.max()}, "
                    f"expected {expected_last_day}",
                )

            if len(np.unique(dates)) != expected_days:
                return False, "duplicate or missing dates found"

            latitude = np.asarray(
                dataset[lat_name].values,
                dtype=float,
            )

            longitude = np.asarray(
                dataset[lon_name].values,
                dtype=float,
            )

            if np.nanmax(latitude) < 89:
                return False, "north boundary is incomplete"

            if np.nanmin(latitude) > 21:
                return False, "south boundary is incomplete"

            if np.nanmin(longitude) > 21:
                return False, "west boundary is incomplete"

            if np.nanmax(longitude) < 139:
                return False, "east boundary is incomplete"

            field = dataset[variable_name]

            if "pressure_level" in field.dims:
                levels = np.asarray(
                    field["pressure_level"].values
                )

                if 500 not in levels:
                    return False, "500 hPa level not found"

            if "level" in field.dims:
                levels = np.asarray(
                    field["level"].values
                )

                if 500 not in levels:
                    return False, "500 hPa level not found"

            sample = field.isel(
                {time_name: 0}
            ).values

            if not np.isfinite(sample).any():
                return False, "first field contains no finite data"

        return True, "valid"

    except Exception as error:
        return False, f"validation error: {error}"


def pd_datetime_days(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).astype("datetime64[D]")


def fresh_nocache_token() -> str:
    # Numeric string recommended for forcing a fresh CDS product.
    return str(time.time_ns())


def make_request(days: list[str]) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": ["geopotential"],
        "pressure_level": ["500"],
        "year": ["2024"],
        "month": ["12"],
        "day": days,
        "time": ["00:00"],
        "area": AREA,
        "grid": GRID,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "nocache": fresh_nocache_token(),
    }


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def download_request(
    client: cdsapi.Client,
    days: list[str],
    target: Path,
    expected_first_day: int,
    expected_last_day: int,
) -> None:
    remove_if_exists(target)

    expected_days = (
        expected_last_day
        - expected_first_day
        + 1
    )

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            request = make_request(days)

            print("")
            print(
                f"Downloading days "
                f"{expected_first_day:02d}–"
                f"{expected_last_day:02d}, "
                f"attempt {attempt}/{MAX_ATTEMPTS}"
            )
            print(
                f"nocache token: {request['nocache']}"
            )

            client.retrieve(
                DATASET,
                request,
                str(target),
            )

            valid, reason = validate_file(
                target,
                expected_days=expected_days,
                expected_first_day=expected_first_day,
                expected_last_day=expected_last_day,
            )

            if not valid:
                raise RuntimeError(
                    f"Downloaded file failed validation: {reason}"
                )

            print(
                f"VALID: {target.name} "
                f"({target.stat().st_size / 1024 / 1024:.2f} MB)"
            )
            return

        except Exception as error:
            print(
                f"FAILED attempt {attempt}: {error}"
            )

            remove_if_exists(target)

            if attempt == MAX_ATTEMPTS:
                raise

            wait_seconds = 30 * attempt

            print(
                f"Waiting {wait_seconds} seconds "
                "before a fresh request..."
            )

            time.sleep(wait_seconds)


def standardize_dataset(
    path: Path,
) -> xr.Dataset:
    dataset = xr.open_dataset(path)

    rename = {}

    if "valid_time" in dataset.coords:
        rename["valid_time"] = "time"

    if "latitude" in dataset.coords:
        rename["latitude"] = "lat"

    if "longitude" in dataset.coords:
        rename["longitude"] = "lon"

    if "pressure_level" in dataset.coords:
        rename["pressure_level"] = "level"

    dataset = dataset.rename(rename)

    variable_name = identify_variable(dataset)

    if variable_name != "z":
        dataset = dataset.rename(
            {variable_name: "z"}
        )

    if "expver" in dataset["z"].dims:
        combined = dataset["z"].isel(expver=0)

        for index in range(
            1,
            dataset.sizes["expver"],
        ):
            combined = combined.combine_first(
                dataset["z"].isel(expver=index)
            )

        dataset = combined.to_dataset(name="z")

    return dataset.load()


def merge_half_months() -> None:
    first = standardize_dataset(
        FIRST_HALF_FILE
    )

    second = standardize_dataset(
        SECOND_HALF_FILE
    )

    try:
        merged = xr.concat(
            [first, second],
            dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            combine_attrs="override",
        )

        merged = merged.sortby("time")

        dates = np.asarray(
            merged["time"].values
        ).astype("datetime64[D]")

        if len(dates) != EXPECTED_DAYS:
            raise RuntimeError(
                f"Merged dataset has {len(dates)} days, "
                f"expected {EXPECTED_DAYS}."
            )

        if len(np.unique(dates)) != EXPECTED_DAYS:
            raise RuntimeError(
                "Merged dataset contains duplicate dates."
            )

        remove_if_exists(MERGED_TEMP_FILE)

        encoding = {
            "z": {
                "zlib": True,
                "complevel": 4,
            }
        }

        merged.to_netcdf(
            MERGED_TEMP_FILE,
            engine="netcdf4",
            encoding=encoding,
        )

    finally:
        first.close()
        second.close()

        if "merged" in locals():
            merged.close()

    valid, reason = validate_file(
        MERGED_TEMP_FILE,
        expected_days=31,
        expected_first_day=1,
        expected_last_day=31,
    )

    if not valid:
        raise RuntimeError(
            f"Merged December file failed validation: {reason}"
        )

    MERGED_TEMP_FILE.replace(FINAL_FILE)


def main() -> None:
    print(
        "STEP 10A: RECOVER DECEMBER 2024 Z500 "
        "WITH FRESH CDS CACHE"
    )
    print("=" * 76)

    existing_valid, existing_reason = validate_file(
        FINAL_FILE,
        expected_days=31,
        expected_first_day=1,
        expected_last_day=31,
    )

    if existing_valid:
        print(
            f"FINAL FILE ALREADY VALID: {FINAL_FILE.name}"
        )
        return

    if FINAL_FILE.exists():
        backup = FINAL_FILE.with_name(
            FINAL_FILE.name
            + f".invalid_{time.strftime('%Y%m%d_%H%M%S')}"
        )

        shutil.move(
            str(FINAL_FILE),
            str(backup),
        )

        print(
            f"Moved invalid existing file to {backup.name}"
        )
    else:
        print(
            f"Final file status before recovery: {existing_reason}"
        )

    client = cdsapi.Client()

    full_month_error = None

    try:
        print("")
        print(
            "METHOD 1: Request the complete month "
            "with a unique nocache token."
        )

        download_request(
            client=client,
            days=[
                f"{day:02d}"
                for day in range(1, 32)
            ],
            target=FULL_PART_FILE,
            expected_first_day=1,
            expected_last_day=31,
        )

        FULL_PART_FILE.replace(
            FINAL_FILE
        )

    except Exception as error:
        full_month_error = str(error)

        print("")
        print(
            "Complete-month request still failed."
        )
        print(
            "METHOD 2: Download two independent "
            "half-month requests and merge them."
        )

        download_request(
            client=client,
            days=[
                f"{day:02d}"
                for day in range(1, 16)
            ],
            target=FIRST_HALF_FILE,
            expected_first_day=1,
            expected_last_day=15,
        )

        download_request(
            client=client,
            days=[
                f"{day:02d}"
                for day in range(16, 32)
            ],
            target=SECOND_HALF_FILE,
            expected_first_day=16,
            expected_last_day=31,
        )

        merge_half_months()

    final_valid, final_reason = validate_file(
        FINAL_FILE,
        expected_days=31,
        expected_first_day=1,
        expected_last_day=31,
    )

    report_lines = [
        "STEP 10A: DECEMBER 2024 Z500 RECOVERY",
        "=" * 64,
        f"Final file: {FINAL_FILE}",
        f"Final valid: {final_valid}",
        f"Validation result: {final_reason}",
        (
            "Recovery method: "
            + (
                "complete month with nocache"
                if full_month_error is None
                else "two half-month requests with nocache"
            )
        ),
    ]

    if full_month_error is not None:
        report_lines.extend(
            [
                "",
                "Complete-month failure before fallback:",
                full_month_error,
            ]
        )

    REPORT_FILE.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("")
    print("\n".join(report_lines))

    if not final_valid:
        raise RuntimeError(
            "December 2024 recovery failed. "
            f"Reason: {final_reason}"
        )

    for temporary in [
        FULL_PART_FILE,
        FIRST_HALF_FILE,
        SECOND_HALF_FILE,
        MERGED_TEMP_FILE,
    ]:
        remove_if_exists(temporary)

    print("")
    print(
        "STEP 10A DECEMBER 2024 Z500 "
        "RECOVERY PASSED."
    )


if __name__ == "__main__":
    main()
