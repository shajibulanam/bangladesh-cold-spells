from __future__ import annotations

import calendar
import json
import shutil
import time
from datetime import datetime, timezone
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
    / "step10a_missing_mslp_before_download.txt"
)

FINAL_REPORT = (
    REPORT_DIR
    / "step10a_missing_mslp_recovery_report.txt"
)

SUMMARY_JSON = (
    REPORT_DIR
    / "step10a_missing_mslp_recovery_summary.json"
)

DATASET = "reanalysis-era5-single-levels"

START_WINTER = 1985
END_WINTER = 2024

# North, West, South, East
AREA = [90, 20, 20, 140]
GRID = [1.0, 1.0]

MAX_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 30


def expected_months() -> list[tuple[int, int]]:
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


def expected_path(year: int, month: int) -> Path:
    return (
        OUTPUT_DIR
        / f"ERA5_TROPINDEX_MSLP_{year}_{month:02d}.nc"
    )


def identify_name(
    dataset: xr.Dataset,
    candidates: list[str],
) -> str | None:
    for candidate in candidates:
        if (
            candidate in dataset.coords
            or candidate in dataset.dims
        ):
            return candidate

    return None


def identify_variable(
    dataset: xr.Dataset,
) -> str | None:
    for candidate in [
        "msl",
        "mean_sea_level_pressure",
    ]:
        if candidate in dataset.data_vars:
            return candidate

    return None


def is_zip_file(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(2) == b"PK"
    except OSError:
        return False


def validate_mslp_file(
    path: Path,
    year: int,
    month: int,
    verbose: bool = False,
) -> tuple[bool, str]:
    expected_days = calendar.monthrange(
        year,
        month,
    )[1]

    if not path.exists():
        return False, "missing"

    if path.stat().st_size < 10_000:
        return False, "file is unexpectedly small"

    if is_zip_file(path):
        return (
            False,
            "file is a ZIP archive rather than NetCDF",
        )

    try:
        with xr.open_dataset(path) as dataset:
            time_name = identify_name(
                dataset,
                ["valid_time", "time"],
            )

            lat_name = identify_name(
                dataset,
                ["latitude", "lat"],
            )

            lon_name = identify_name(
                dataset,
                ["longitude", "lon"],
            )

            variable_name = identify_variable(
                dataset
            )

            if time_name is None:
                return False, "time coordinate not found"

            if lat_name is None:
                return False, "latitude coordinate not found"

            if lon_name is None:
                return False, "longitude coordinate not found"

            if variable_name is None:
                return (
                    False,
                    "MSLP variable not found; "
                    f"available={list(dataset.data_vars)}",
                )

            if int(
                dataset.sizes[time_name]
            ) != expected_days:
                return (
                    False,
                    "incorrect daily record count: "
                    f"expected {expected_days}, "
                    f"found {dataset.sizes[time_name]}",
                )

            dates = np.asarray(
                dataset[time_name].values
            ).astype("datetime64[D]")

            years = (
                dates.astype("datetime64[Y]")
                .astype(int)
                + 1970
            )

            months = (
                dates.astype("datetime64[M]")
                .astype(int)
                % 12
                + 1
            )

            if not np.all(years == year):
                return False, "unexpected year"

            if not np.all(months == month):
                return False, "unexpected month"

            if len(np.unique(dates)) != expected_days:
                return (
                    False,
                    "duplicate or missing daily dates",
                )

            latitude = np.asarray(
                dataset[lat_name].values,
                dtype=float,
            )

            longitude = np.asarray(
                dataset[lon_name].values,
                dtype=float,
            )

            if np.nanmax(latitude) < 89:
                return (
                    False,
                    "northern boundary is incomplete",
                )

            if np.nanmin(latitude) > 21:
                return (
                    False,
                    "southern boundary is incomplete",
                )

            if np.nanmin(longitude) > 21:
                return (
                    False,
                    "western boundary is incomplete",
                )

            if np.nanmax(longitude) < 139:
                return (
                    False,
                    "eastern boundary is incomplete",
                )

            field = dataset[variable_name]

            sample = field.isel(
                {time_name: 0}
            ).values

            if not np.isfinite(sample).any():
                return (
                    False,
                    "first daily field contains no finite data",
                )

            median_value = float(
                np.nanmedian(sample)
            )

            # ERA5 native MSLP is normally stored in Pa.
            if not (
                80_000 <= median_value <= 110_000
                or 800 <= median_value <= 1_100
            ):
                return (
                    False,
                    "unexpected MSLP magnitude: "
                    f"{median_value}",
                )

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


def fresh_nocache_token() -> str:
    return str(time.time_ns())


def build_request(
    year: int,
    month: int,
    days: list[str],
) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": [
            "mean_sea_level_pressure"
        ],
        "year": [str(year)],
        "month": [f"{month:02d}"],
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


def validate_partial_file(
    path: Path,
    year: int,
    month: int,
    first_day: int,
    last_day: int,
) -> tuple[bool, str]:
    expected_count = last_day - first_day + 1

    if not path.exists():
        return False, "missing"

    if path.stat().st_size < 10_000:
        return False, "file is unexpectedly small"

    if is_zip_file(path):
        return False, "received ZIP rather than NetCDF"

    try:
        with xr.open_dataset(path) as dataset:
            time_name = identify_name(
                dataset,
                ["valid_time", "time"],
            )

            variable_name = identify_variable(
                dataset
            )

            if time_name is None:
                return False, "time coordinate absent"

            if variable_name is None:
                return False, "MSLP variable absent"

            if int(
                dataset.sizes[time_name]
            ) != expected_count:
                return (
                    False,
                    "incorrect number of records: "
                    f"expected {expected_count}, "
                    f"found {dataset.sizes[time_name]}",
                )

            dates = np.asarray(
                dataset[time_name].values
            ).astype("datetime64[D]")

            years = (
                dates.astype("datetime64[Y]")
                .astype(int)
                + 1970
            )

            months = (
                dates.astype("datetime64[M]")
                .astype(int)
                % 12
                + 1
            )

            month_starts = dates.astype(
                "datetime64[M]"
            )

            day_numbers = (
                dates
                - month_starts
            ).astype(
                "timedelta64[D]"
            ).astype(int) + 1

            if not np.all(years == year):
                return False, "incorrect year"

            if not np.all(months == month):
                return False, "incorrect month"

            if int(day_numbers.min()) != first_day:
                return (
                    False,
                    "incorrect first day: "
                    f"{day_numbers.min()}",
                )

            if int(day_numbers.max()) != last_day:
                return (
                    False,
                    "incorrect last day: "
                    f"{day_numbers.max()}",
                )

            if len(np.unique(dates)) != expected_count:
                return (
                    False,
                    "duplicate or missing dates",
                )

        return True, "valid"

    except Exception as error:
        return False, str(error)


def download_request(
    client: cdsapi.Client,
    year: int,
    month: int,
    first_day: int,
    last_day: int,
    target: Path,
) -> None:
    days = [
        f"{day:02d}"
        for day in range(
            first_day,
            last_day + 1,
        )
    ]

    remove_if_exists(target)

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):
        request = build_request(
            year,
            month,
            days,
        )

        try:
            print("")
            print(
                f"Downloading {year}-{month:02d}, "
                f"days {first_day:02d}-{last_day:02d}, "
                f"attempt {attempt}/{MAX_ATTEMPTS}"
            )

            print(
                f"nocache token: "
                f"{request['nocache']}"
            )

            client.retrieve(
                DATASET,
                request,
                str(target),
            )

            valid, reason = validate_partial_file(
                target,
                year,
                month,
                first_day,
                last_day,
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
                f"FAILED attempt {attempt}: {error}"
            )

            remove_if_exists(target)

            if attempt == MAX_ATTEMPTS:
                raise

            wait_seconds = (
                RETRY_WAIT_SECONDS
                * attempt
            )

            print(
                f"Waiting {wait_seconds} seconds "
                "before retrying..."
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

    dataset = dataset.rename(rename)

    variable_name = identify_variable(
        dataset
    )

    if variable_name is None:
        dataset.close()

        raise KeyError(
            f"{path.name}: MSLP variable not found."
        )

    if variable_name != "msl":
        dataset = dataset.rename(
            {variable_name: "msl"}
        )

    if "expver" in dataset["msl"].dims:
        combined = dataset[
            "msl"
        ].isel(expver=0)

        for index in range(
            1,
            dataset.sizes["expver"],
        ):
            combined = combined.combine_first(
                dataset[
                    "msl"
                ].isel(expver=index)
            )

        dataset = combined.to_dataset(
            name="msl"
        )

    return dataset.load()


def merge_parts(
    first_path: Path,
    second_path: Path,
    final_path: Path,
    year: int,
    month: int,
) -> None:
    first = standardize_dataset(
        first_path
    )

    second = standardize_dataset(
        second_path
    )

    merged = None

    temporary = final_path.with_name(
        final_path.name + ".merged.part"
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

        expected_days = calendar.monthrange(
            year,
            month,
        )[1]

        if len(dates) != expected_days:
            raise RuntimeError(
                f"Merged file has {len(dates)} "
                f"records; expected {expected_days}."
            )

        if len(np.unique(dates)) != expected_days:
            raise RuntimeError(
                "Merged file contains duplicate dates."
            )

        remove_if_exists(temporary)

        merged.to_netcdf(
            temporary,
            engine="netcdf4",
            encoding={
                "msl": {
                    "zlib": True,
                    "complevel": 4,
                }
            },
        )

    finally:
        first.close()
        second.close()

        if merged is not None:
            merged.close()

    temporary.replace(final_path)

    valid, reason = validate_mslp_file(
        final_path,
        year,
        month,
        verbose=True,
    )

    if not valid:
        raise RuntimeError(
            "Merged monthly file failed "
            f"validation: {reason}"
        )


def recover_month(
    client: cdsapi.Client,
    year: int,
    month: int,
) -> str:
    final_path = expected_path(
        year,
        month,
    )

    total_days = calendar.monthrange(
        year,
        month,
    )[1]

    full_temp = final_path.with_name(
        final_path.name + ".full.part"
    )

    first_temp = final_path.with_name(
        final_path.name + ".first.part"
    )

    second_temp = final_path.with_name(
        final_path.name + ".second.part"
    )

    for path in [
        full_temp,
        first_temp,
        second_temp,
    ]:
        remove_if_exists(path)

    try:
        print("")
        print(
            f"METHOD 1: complete-month request "
            f"for {year}-{month:02d}"
        )

        download_request(
            client=client,
            year=year,
            month=month,
            first_day=1,
            last_day=total_days,
            target=full_temp,
        )

        full_temp.replace(final_path)

        valid, reason = validate_mslp_file(
            final_path,
            year,
            month,
            verbose=True,
        )

        if not valid:
            raise RuntimeError(reason)

        return "complete-month nocache request"

    except Exception as full_error:
        print("")
        print(
            "Complete-month request failed. "
            "Using two independent half-month requests."
        )

        midpoint = total_days // 2

        download_request(
            client=client,
            year=year,
            month=month,
            first_day=1,
            last_day=midpoint,
            target=first_temp,
        )

        download_request(
            client=client,
            year=year,
            month=month,
            first_day=midpoint + 1,
            last_day=total_days,
            target=second_temp,
        )

        merge_parts(
            first_path=first_temp,
            second_path=second_temp,
            final_path=final_path,
            year=year,
            month=month,
        )

        print(
            "Full-month failure before fallback:",
            full_error,
        )

        return "two half-month nocache requests"

    finally:
        for path in [
            full_temp,
            first_temp,
            second_temp,
        ]:
            remove_if_exists(path)


def move_invalid_file(path: Path) -> Path:
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


def main() -> None:
    periods = expected_months()

    if len(periods) != 240:
        raise RuntimeError(
            f"Expected 240 monthly periods, "
            f"generated {len(periods)}."
        )

    print(
        "STEP 10A: FIND AND RECOVER "
        "MISSING MSLP FILES"
    )
    print("=" * 72)

    missing_or_invalid: list[
        tuple[int, int, str]
    ] = []

    valid_before = 0

    for year, month in periods:
        path = expected_path(
            year,
            month,
        )

        valid, reason = validate_mslp_file(
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

    before_lines = [
        "STEP 10A MISSING MSLP CHECK",
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
        before_lines.append(
            "Files requiring recovery:"
        )

        for year, month, reason in (
            missing_or_invalid
        ):
            before_lines.append(
                f"{expected_path(year, month).name} "
                f"| {reason}"
            )
    else:
        before_lines.append(
            "No missing or invalid MSLP files."
        )

    MISSING_REPORT.write_text(
        "\n".join(before_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(before_lines))

    if not missing_or_invalid:
        print("")
        print(
            "STEP 10A MSLP RECOVERY PASSED: "
            "all 240 files were already valid."
        )
        return

    for year, month, _ in (
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

    recovery_rows = []

    for year, month, original_reason in (
        missing_or_invalid
    ):
        try:
            method = recover_month(
                client,
                year,
                month,
            )

            recovery_rows.append(
                {
                    "year": year,
                    "month": month,
                    "original_status": (
                        original_reason
                    ),
                    "recovery_method": method,
                    "status": "recovered",
                    "error": None,
                }
            )

        except Exception as error:
            recovery_rows.append(
                {
                    "year": year,
                    "month": month,
                    "original_status": (
                        original_reason
                    ),
                    "recovery_method": None,
                    "status": "failed",
                    "error": str(error),
                }
            )

    final_invalid = []
    valid_after = 0

    for year, month in periods:
        valid, reason = validate_mslp_file(
            expected_path(year, month),
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
        "STEP 10A MSLP RECOVERY RESULT",
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

    for row in recovery_rows:
        final_lines.append(
            f"{row['year']}-{row['month']:02d} | "
            f"{row['status']} | "
            f"{row['recovery_method'] or '-'}"
        )

        if row["error"]:
            final_lines.append(
                f"  Error: {row['error']}"
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
                f"{expected_path(year, month).name} "
                f"| {reason}"
            )
    else:
        final_lines.append(
            "All 240 expected MSLP files "
            "are present and valid."
        )

    FINAL_REPORT.write_text(
        "\n".join(final_lines) + "\n",
        encoding="utf-8",
    )

    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "expected_files": 240,
                "valid_before": valid_before,
                "requested_for_recovery": len(
                    missing_or_invalid
                ),
                "valid_after": valid_after,
                "remaining_invalid": len(
                    final_invalid
                ),
                "recovery_results": (
                    recovery_rows
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print("\n".join(final_lines))

    if final_invalid:
        raise RuntimeError(
            "MSLP recovery incomplete. Read: "
            f"{FINAL_REPORT}"
        )

    print("")
    print(
        "STEP 10A MSLP RECOVERY PASSED."
    )


if __name__ == "__main__":
    main()
