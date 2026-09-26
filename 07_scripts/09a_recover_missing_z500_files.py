from __future__ import annotations

import calendar
import shutil
import time
from datetime import datetime
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

MISSING_REPORT = (
    REPORT_DIR
    / "step10a_missing_z500_before_download.txt"
)

FINAL_REPORT = (
    REPORT_DIR
    / "step10a_missing_z500_recovery_report.txt"
)

DATASET = "reanalysis-era5-pressure-levels"

START_WINTER = 1985
END_WINTER = 2024

# North, West, South, East
AREA = [90, 20, 20, 140]

GRID = [1.0, 1.0]

MAX_DOWNLOAD_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 30


def expected_months() -> list[tuple[int, int]]:
    """
    Return the unique October-March months needed for
    winters 1985/86 through 2024/25.
    """
    periods: set[tuple[int, int]] = set()

    for winter_start in range(
        START_WINTER,
        END_WINTER + 1,
    ):
        periods.update(
            {
                (winter_start, 10),
                (winter_start, 11),
                (winter_start, 12),
                (winter_start + 1, 1),
                (winter_start + 1, 2),
                (winter_start + 1, 3),
            }
        )

    return sorted(periods)


def expected_path(
    year: int,
    month: int,
) -> Path:
    return (
        OUTPUT_DIR
        / f"ERA5_TROPINDEX_Z500_{year}_{month:02d}.nc"
    )


def identify_time_name(
    dataset: xr.Dataset,
) -> str | None:
    for candidate in [
        "valid_time",
        "time",
    ]:
        if (
            candidate in dataset.coords
            or candidate in dataset.dims
        ):
            return candidate

    return None


def identify_latitude_name(
    dataset: xr.Dataset,
) -> str | None:
    for candidate in [
        "latitude",
        "lat",
    ]:
        if candidate in dataset.coords:
            return candidate

    return None


def identify_longitude_name(
    dataset: xr.Dataset,
) -> str | None:
    for candidate in [
        "longitude",
        "lon",
    ]:
        if candidate in dataset.coords:
            return candidate

    return None


def is_zip_file(
    path: Path,
) -> bool:
    try:
        with path.open("rb") as file:
            signature = file.read(4)

        return signature[:2] == b"PK"

    except OSError:
        return False


def validate_z500_file(
    path: Path,
    year: int,
    month: int,
    verbose: bool = False,
) -> tuple[bool, str]:
    """
    Validate one expected monthly Z500 file.
    """
    expected_day_count = calendar.monthrange(
        year,
        month,
    )[1]

    if not path.exists():
        return False, "missing"

    if path.stat().st_size < 10_000:
        return False, "file is unexpectedly small"

    if is_zip_file(path):
        return False, "file is a ZIP archive, not an unarchived NetCDF"

    try:
        with xr.open_dataset(path) as dataset:
            variable_candidates = {
                "z",
                "geopotential",
            }

            matched_variables = (
                variable_candidates
                .intersection(
                    set(dataset.data_vars)
                )
            )

            if not matched_variables:
                return (
                    False,
                    "geopotential variable not found; "
                    f"variables={list(dataset.data_vars)}",
                )

            time_name = identify_time_name(
                dataset
            )

            latitude_name = (
                identify_latitude_name(
                    dataset
                )
            )

            longitude_name = (
                identify_longitude_name(
                    dataset
                )
            )

            if time_name is None:
                return False, "time coordinate not found"

            if latitude_name is None:
                return False, "latitude coordinate not found"

            if longitude_name is None:
                return False, "longitude coordinate not found"

            actual_day_count = int(
                dataset.sizes[time_name]
            )

            if actual_day_count != expected_day_count:
                return (
                    False,
                    "incorrect time count: "
                    f"expected {expected_day_count}, "
                    f"found {actual_day_count}",
                )

            times = (
                dataset[time_name]
                .values
            )

            converted_times = np.asarray(
                times
            ).astype("datetime64[D]")

            years = (
                converted_times
                .astype("datetime64[Y]")
                .astype(int)
                + 1970
            )

            months = (
                converted_times
                .astype("datetime64[M]")
                .astype(int)
                % 12
                + 1
            )

            if not np.all(years == year):
                return (
                    False,
                    "file contains an unexpected year",
                )

            if not np.all(months == month):
                return (
                    False,
                    "file contains an unexpected month",
                )

            latitude = np.asarray(
                dataset[
                    latitude_name
                ].values,
                dtype=float,
            )

            longitude = np.asarray(
                dataset[
                    longitude_name
                ].values,
                dtype=float,
            )

            if np.nanmax(latitude) < 89:
                return False, "northern boundary is incomplete"

            if np.nanmin(latitude) > 21:
                return False, "southern boundary is incomplete"

            if np.nanmin(longitude) > 21:
                return False, "western boundary is incomplete"

            if np.nanmax(longitude) < 139:
                return False, "eastern boundary is incomplete"

            variable_name = sorted(
                matched_variables
            )[0]

            field = dataset[
                variable_name
            ]

            if "pressure_level" in field.dims:
                levels = np.asarray(
                    field[
                        "pressure_level"
                    ].values
                )

                if 500 not in levels:
                    return False, "500 hPa level is absent"

            elif "level" in field.dims:
                levels = np.asarray(
                    field["level"].values
                )

                if 500 not in levels:
                    return False, "500 hPa level is absent"

            sample = (
                field
                .isel(
                    {
                        time_name: 0,
                    }
                )
                .values
            )

            if not np.isfinite(sample).any():
                return False, "first daily field contains no finite data"

        if verbose:
            print(
                f"VALID: {path.name} "
                f"({path.stat().st_size / 1024 / 1024:.2f} MB)"
            )

        return True, "valid"

    except Exception as error:
        return (
            False,
            f"NetCDF validation error: {error}",
        )


def move_invalid_file(
    path: Path,
) -> Path:
    """
    Preserve an invalid file rather than deleting it.
    """
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup = path.with_name(
        f"{path.name}.invalid_{timestamp}"
    )

    shutil.move(
        str(path),
        str(backup),
    )

    return backup


def download_one_month(
    client: cdsapi.Client,
    year: int,
    month: int,
) -> None:
    target = expected_path(
        year,
        month,
    )

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

    request = {
        "product_type": [
            "reanalysis"
        ],
        "variable": [
            "geopotential"
        ],
        "pressure_level": [
            "500"
        ],
        "year": [
            str(year)
        ],
        "month": [
            f"{month:02d}"
        ],
        "day": days,
        "time": [
            "00:00"
        ],
        "area": AREA,
        "grid": GRID,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    temporary = target.with_name(
        target.name + ".part"
    )

    if temporary.exists():
        temporary.unlink()

    for attempt in range(
        1,
        MAX_DOWNLOAD_ATTEMPTS + 1,
    ):
        try:
            print("")
            print(
                f"Downloading {year}-{month:02d} "
                f"(attempt {attempt}/"
                f"{MAX_DOWNLOAD_ATTEMPTS})"
            )

            client.retrieve(
                DATASET,
                request,
                str(temporary),
            )

            if not temporary.exists():
                raise RuntimeError(
                    "CDS finished but the temporary "
                    "download file was not created."
                )

            temporary.replace(target)

            valid, reason = validate_z500_file(
                target,
                year,
                month,
                verbose=True,
            )

            if not valid:
                raise RuntimeError(
                    "Downloaded file failed validation: "
                    f"{reason}"
                )

            print(
                f"PASS: {target.name}"
            )

            return

        except Exception as error:
            print(
                f"FAILED: {year}-{month:02d}, "
                f"attempt {attempt}: {error}"
            )

            if temporary.exists():
                temporary.unlink()

            if target.exists():
                valid, _ = validate_z500_file(
                    target,
                    year,
                    month,
                )

                if not valid:
                    target.unlink()

            if attempt == MAX_DOWNLOAD_ATTEMPTS:
                raise RuntimeError(
                    f"Unable to download "
                    f"{year}-{month:02d} after "
                    f"{MAX_DOWNLOAD_ATTEMPTS} attempts."
                ) from error

            wait_seconds = (
                RETRY_WAIT_SECONDS
                * attempt
            )

            print(
                f"Waiting {wait_seconds} seconds "
                "before retrying..."
            )

            time.sleep(wait_seconds)


def main() -> None:
    periods = expected_months()

    if len(periods) != 240:
        raise RuntimeError(
            f"Expected 240 periods, generated "
            f"{len(periods)}."
        )

    missing_or_invalid: list[
        tuple[int, int, str]
    ] = []

    valid_before = 0

    print(
        "STEP 10A: FIND AND RECOVER "
        "MISSING Z500 FILES"
    )
    print("=" * 72)

    for year, month in periods:
        path = expected_path(
            year,
            month,
        )

        valid, reason = validate_z500_file(
            path,
            year,
            month,
        )

        if valid:
            valid_before += 1
        else:
            missing_or_invalid.append(
                (
                    year,
                    month,
                    reason,
                )
            )

    report_lines = [
        "STEP 10A MISSING Z500 CHECK",
        "=" * 60,
        f"Expected files: {len(periods)}",
        f"Valid before recovery: {valid_before}",
        (
            "Missing or invalid before recovery: "
            f"{len(missing_or_invalid)}"
        ),
        "",
    ]

    if missing_or_invalid:
        report_lines.append(
            "Files requiring download:"
        )

        for year, month, reason in (
            missing_or_invalid
        ):
            filename = expected_path(
                year,
                month,
            ).name

            report_lines.append(
                f"{filename} | {reason}"
            )
    else:
        report_lines.append(
            "No missing or invalid Z500 files."
        )

    MISSING_REPORT.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print(
        "\n".join(report_lines)
    )

    if not missing_or_invalid:
        print("")
        print(
            "STEP 10A RECOVERY PASSED: "
            "all 240 files were already valid."
        )
        return

    # Preserve invalid existing files before replacing them.
    for year, month, reason in (
        missing_or_invalid
    ):
        path = expected_path(
            year,
            month,
        )

        if path.exists():
            backup = move_invalid_file(
                path
            )

            print(
                f"Moved invalid file to: "
                f"{backup.name}"
            )

    client = cdsapi.Client()

    failed: list[
        tuple[int, int, str]
    ] = []

    for year, month, _ in (
        missing_or_invalid
    ):
        try:
            download_one_month(
                client,
                year,
                month,
            )

        except Exception as error:
            failed.append(
                (
                    year,
                    month,
                    str(error),
                )
            )

    final_invalid: list[
        tuple[int, int, str]
    ] = []

    valid_after = 0

    for year, month in periods:
        path = expected_path(
            year,
            month,
        )

        valid, reason = validate_z500_file(
            path,
            year,
            month,
        )

        if valid:
            valid_after += 1
        else:
            final_invalid.append(
                (
                    year,
                    month,
                    reason,
                )
            )

    final_lines = [
        "STEP 10A Z500 RECOVERY RESULT",
        "=" * 60,
        f"Expected files: {len(periods)}",
        f"Valid before recovery: {valid_before}",
        (
            "Requested for recovery: "
            f"{len(missing_or_invalid)}"
        ),
        f"Valid after recovery: {valid_after}",
        (
            "Still missing or invalid: "
            f"{len(final_invalid)}"
        ),
        "",
    ]

    if failed:
        final_lines.append(
            "Requests that failed:"
        )

        for year, month, error in failed:
            final_lines.append(
                f"{year}-{month:02d}: {error}"
            )

        final_lines.append("")

    if final_invalid:
        final_lines.append(
            "Files still requiring attention:"
        )

        for year, month, reason in (
            final_invalid
        ):
            final_lines.append(
                f"{expected_path(year, month).name}"
                f" | {reason}"
            )
    else:
        final_lines.append(
            "All 240 expected Z500 files "
            "are present and valid."
        )

    FINAL_REPORT.write_text(
        "\n".join(final_lines) + "\n",
        encoding="utf-8",
    )

    print("")
    print(
        "\n".join(final_lines)
    )

    if final_invalid:
        raise RuntimeError(
            "Z500 recovery incomplete. Read: "
            f"{FINAL_REPORT}"
        )

    print("")
    print(
        "STEP 10A Z500 RECOVERY PASSED."
    )


if __name__ == "__main__":
    main()
