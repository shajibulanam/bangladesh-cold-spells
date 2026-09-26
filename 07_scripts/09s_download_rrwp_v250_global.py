from __future__ import annotations

import argparse
import calendar
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "01_raw_data" / "era5" / "rrwp_v250_global"
REPORT_DIR = PROJECT_ROOT / "05_qc_reports" / "rrwp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "reanalysis-era5-pressure-levels"
START_WINTER = 1985
END_WINTER = 2024
MONTHS_FIRST_YEAR = [9, 10, 11, 12]
MONTHS_SECOND_YEAR = [1, 2, 3, 4]
AREA = [80, 0, 20, 359]  # north, west, south, east
GRID = [1.0, 1.0]
MAX_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 30

INVENTORY_CSV = REPORT_DIR / "step10e_rrwp_v250_download_inventory.csv"
SUMMARY_JSON = REPORT_DIR / "step10e_rrwp_v250_download_summary.json"
REPORT_TXT = REPORT_DIR / "step10e_rrwp_v250_download_report.txt"


def expected_months() -> list[tuple[int, int]]:
    periods: set[tuple[int, int]] = set()
    for winter_start in range(START_WINTER, END_WINTER + 1):
        periods.update((winter_start, month) for month in MONTHS_FIRST_YEAR)
        periods.update((winter_start + 1, month) for month in MONTHS_SECOND_YEAR)
    return sorted(periods)


def output_path(year: int, month: int) -> Path:
    return OUTPUT_DIR / f"ERA5_RRWP_V250_{year}_{month:02d}.nc"


def coordinate_name(ds: xr.Dataset, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    return None


def variable_name(ds: xr.Dataset) -> str | None:
    for name in ["v", "v_component_of_wind", "meridional_wind"]:
        if name in ds.data_vars:
            return name
    return None


def is_zip(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(2) == b"PK"
    except OSError:
        return False


def validate_file(
    path: Path,
    year: int,
    month: int,
    expected_first_day: int = 1,
    expected_last_day: int | None = None,
) -> tuple[bool, str]:
    if expected_last_day is None:
        expected_last_day = calendar.monthrange(year, month)[1]
    expected_count = expected_last_day - expected_first_day + 1

    if not path.exists():
        return False, "missing"
    if path.stat().st_size < 10_000:
        return False, "file is unexpectedly small"
    if is_zip(path):
        return False, "received ZIP rather than unarchived NetCDF"

    try:
        with xr.open_dataset(path) as ds:
            time_name = coordinate_name(ds, ["valid_time", "time"])
            lat_name = coordinate_name(ds, ["latitude", "lat"])
            lon_name = coordinate_name(ds, ["longitude", "lon"])
            var_name = variable_name(ds)

            if time_name is None:
                return False, "time coordinate not found"
            if lat_name is None:
                return False, "latitude coordinate not found"
            if lon_name is None:
                return False, "longitude coordinate not found"
            if var_name is None:
                return False, f"V-wind variable not found; variables={list(ds.data_vars)}"
            if int(ds.sizes[time_name]) != expected_count:
                return False, (
                    f"incorrect time count: expected {expected_count}, "
                    f"found {ds.sizes[time_name]}"
                )

            dates = np.asarray(ds[time_name].values).astype("datetime64[D]")
            years = dates.astype("datetime64[Y]").astype(int) + 1970
            months = dates.astype("datetime64[M]").astype(int) % 12 + 1
            month_start = dates.astype("datetime64[M]")
            days = (dates - month_start).astype("timedelta64[D]").astype(int) + 1

            if not np.all(years == year):
                return False, "unexpected year"
            if not np.all(months == month):
                return False, "unexpected month"
            if int(days.min()) != expected_first_day:
                return False, f"first day is {int(days.min())}, expected {expected_first_day}"
            if int(days.max()) != expected_last_day:
                return False, f"last day is {int(days.max())}, expected {expected_last_day}"
            if len(np.unique(dates)) != expected_count:
                return False, "duplicate or missing dates"

            latitude = np.asarray(ds[lat_name].values, dtype=float)
            longitude = np.asarray(ds[lon_name].values, dtype=float)
            if np.nanmax(latitude) < 79:
                return False, "northern boundary incomplete"
            if np.nanmin(latitude) > 21:
                return False, "southern boundary incomplete"
            if len(longitude) < 359:
                return False, f"longitude count too small: {len(longitude)}"
            lon_mod = np.mod(longitude, 360.0)
            if np.nanmin(lon_mod) > 1.1 or np.nanmax(lon_mod) < 357.9:
                return False, "global longitude coverage incomplete"

            field = ds[var_name]
            for level_name in ["pressure_level", "level", "isobaricInhPa"]:
                if level_name in field.dims or level_name in ds.coords:
                    levels = np.asarray(ds[level_name].values)
                    if 250 not in levels:
                        return False, "250 hPa level absent"

            sample = field.isel({time_name: 0}).values
            if not np.isfinite(sample).any():
                return False, "first daily field contains no finite values"

        return True, "valid"
    except Exception as error:
        return False, f"validation error: {error}"


def fresh_nocache_token() -> str:
    return str(time.time_ns())


def request_payload(year: int, month: int, days: list[str]) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": ["v_component_of_wind"],
        "pressure_level": ["250"],
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


def remove(path: Path) -> None:
    if path.exists():
        path.unlink()


def download_part(
    client: cdsapi.Client,
    year: int,
    month: int,
    first_day: int,
    last_day: int,
    target: Path,
) -> None:
    days = [f"{day:02d}" for day in range(first_day, last_day + 1)]
    remove(target)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        payload = request_payload(year, month, days)
        try:
            print(
                f"Downloading {year}-{month:02d}, days {first_day:02d}-{last_day:02d}, "
                f"attempt {attempt}/{MAX_ATTEMPTS}"
            )
            client.retrieve(DATASET, payload, str(target))
            valid, reason = validate_file(
                target,
                year,
                month,
                expected_first_day=first_day,
                expected_last_day=last_day,
            )
            if not valid:
                raise RuntimeError(f"downloaded file failed validation: {reason}")
            print(f"PASS: {target.name}")
            return
        except Exception as error:
            print(f"FAILED attempt {attempt}: {error}")
            remove(target)
            if attempt == MAX_ATTEMPTS:
                raise
            wait_seconds = RETRY_WAIT_SECONDS * attempt
            print(f"Waiting {wait_seconds} seconds before retrying...")
            time.sleep(wait_seconds)


def standardize_part(path: Path) -> xr.Dataset:
    ds = xr.open_dataset(path)
    rename: dict[str, str] = {}
    if "valid_time" in ds.coords:
        rename["valid_time"] = "time"
    if "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename["longitude"] = "lon"
    if "pressure_level" in ds.coords:
        rename["pressure_level"] = "level_hpa"
    if "level" in ds.coords:
        rename["level"] = "level_hpa"
    ds = ds.rename(rename)
    var = variable_name(ds)
    if var is None:
        ds.close()
        raise KeyError(f"{path.name}: V-wind variable not found")
    if var != "v":
        ds = ds.rename({var: "v"})
    if "level_hpa" in ds["v"].dims:
        ds = ds.sel(level_hpa=250, drop=True)
    if "expver" in ds["v"].dims:
        combined = ds["v"].isel(expver=0)
        for index in range(1, ds.sizes["expver"]):
            combined = combined.combine_first(ds["v"].isel(expver=index))
        ds = combined.to_dataset(name="v")
    return ds[["v"]].load()


def merge_parts(first: Path, second: Path, final: Path, year: int, month: int) -> None:
    first_ds = standardize_part(first)
    second_ds = standardize_part(second)
    merged = None
    temporary = final.with_name(final.name + ".merged.part")
    try:
        merged = xr.concat(
            [first_ds, second_ds],
            dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            combine_attrs="override",
        ).sortby("time")
        expected_days = calendar.monthrange(year, month)[1]
        dates = np.asarray(merged["time"].values).astype("datetime64[D]")
        if len(dates) != expected_days or len(np.unique(dates)) != expected_days:
            raise RuntimeError("merged month has missing or duplicate dates")
        remove(temporary)
        merged.to_netcdf(
            temporary,
            engine="netcdf4",
            encoding={"v": {"zlib": True, "complevel": 4}},
        )
    finally:
        first_ds.close()
        second_ds.close()
        if merged is not None:
            merged.close()
    temporary.replace(final)
    valid, reason = validate_file(final, year, month)
    if not valid:
        raise RuntimeError(f"merged monthly file failed validation: {reason}")


def recover_month(client: cdsapi.Client, year: int, month: int) -> str:
    final = output_path(year, month)
    total_days = calendar.monthrange(year, month)[1]
    full = final.with_name(final.name + ".full.part")
    first = final.with_name(final.name + ".first.part")
    second = final.with_name(final.name + ".second.part")
    for path in [full, first, second]:
        remove(path)

    try:
        download_part(client, year, month, 1, total_days, full)
        full.replace(final)
        valid, reason = validate_file(final, year, month)
        if not valid:
            raise RuntimeError(reason)
        return "full-month request"
    except Exception as full_error:
        print(f"Full-month request failed for {year}-{month:02d}: {full_error}")
        print("Falling back to two half-month requests.")
        midpoint = total_days // 2
        download_part(client, year, month, 1, midpoint, first)
        download_part(client, year, month, midpoint + 1, total_days, second)
        merge_parts(first, second, final, year, month)
        return "two half-month requests"
    finally:
        for path in [full, first, second]:
            remove(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Validate and report without downloading missing files.",
    )
    args = parser.parse_args()

    periods = expected_months()
    if len(periods) != 320:
        raise RuntimeError(f"Expected 320 monthly periods, generated {len(periods)}")

    rows: list[dict[str, object]] = []
    missing: list[tuple[int, int, str]] = []
    for year, month in periods:
        path = output_path(year, month)
        valid, reason = validate_file(path, year, month)
        rows.append(
            {
                "year": year,
                "month": month,
                "file": str(path),
                "valid_before": valid,
                "status_before": reason,
                "recovery_method": None,
                "valid_after": valid,
                "status_after": reason,
            }
        )
        if not valid:
            missing.append((year, month, reason))

    print("STEP 10E: GLOBAL V250 DOWNLOAD/RECOVERY")
    print("=" * 72)
    print("Expected monthly files:", len(periods))
    print("Valid before recovery:", len(periods) - len(missing))
    print("Missing or invalid:", len(missing))

    if args.inventory_only:
        pd.DataFrame(rows).to_csv(INVENTORY_CSV, index=False)
        print("Inventory-only mode complete.")
        return

    client = cdsapi.Client() if missing else None
    for year, month, original_reason in missing:
        final = output_path(year, month)
        if final.exists():
            backup = final.with_name(
                final.name + f".invalid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            shutil.move(str(final), str(backup))
            print(f"Moved invalid file to {backup.name}")
        method = None
        error = None
        try:
            assert client is not None
            method = recover_month(client, year, month)
        except Exception as exc:
            error = str(exc)
        valid, reason = validate_file(final, year, month)
        for row in rows:
            if row["year"] == year and row["month"] == month:
                row["recovery_method"] = method
                row["valid_after"] = valid
                row["status_after"] = reason if error is None else f"{reason}; error={error}"
                break

    inventory = pd.DataFrame(rows)
    inventory.to_csv(INVENTORY_CSV, index=False)
    valid_after = int(inventory["valid_after"].sum())
    remaining = inventory.loc[~inventory["valid_after"]]

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "expected_files": len(periods),
        "valid_before": int(inventory["valid_before"].sum()),
        "requested_for_recovery": len(missing),
        "valid_after": valid_after,
        "remaining_invalid": int(len(remaining)),
        "domain": "20-80N, 0-359E",
        "pressure_level_hpa": 250,
        "time": "00 UTC",
        "grid": "1 degree",
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "STEP 10E: GLOBAL V250 DOWNLOAD/RECOVERY",
        "=" * 64,
        f"Expected files: {len(periods)}",
        f"Valid before: {summary['valid_before']}",
        f"Requested for recovery: {len(missing)}",
        f"Valid after: {valid_after}",
        f"Remaining invalid: {len(remaining)}",
        f"Inventory: {INVENTORY_CSV}",
    ]
    if not remaining.empty:
        lines.extend(["", "REMAINING INVALID FILES:"])
        lines.extend(
            f"{row.file} | {row.status_after}" for row in remaining.itertuples()
        )
    REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    if not remaining.empty:
        raise RuntimeError(f"Global V250 recovery incomplete. Read {REPORT_TXT}")
    print("STEP 10E-DOWNLOAD PASSED.")


if __name__ == "__main__":
    main()
