from __future__ import annotations

import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from importlib import import_module
C = import_module("11_stratosphere_common")

# Use the regular ERA5 pressure-level archive rather than the derived
# daily-statistics service. The latter can reject a hemispheric,
# multi-level, multi-variable monthly request with "cost limits exceeded".
#
# We retrieve four synoptic times per day on the requested 2.5-degree grid
# and form the daily mean locally. This keeps the downstream Step 12 file
# format unchanged while making each CDS request much smaller/cheaper.
DATASET = "reanalysis-era5-pressure-levels"

VARIABLES = [
    "geopotential",
    "u_component_of_wind",
    "temperature",
]

LEVELS = [str(v) for v in C.LEVELS]
TIMES = ["00:00", "06:00", "12:00", "18:00"]


def normalise_dataset(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for old, new in [
        ("valid_time", "time"),
        ("pressure_level", "level_hpa"),
        ("level", "level_hpa"),
        ("latitude", "lat"),
        ("longitude", "lon"),
    ]:
        if old in ds.coords or old in ds.dims:
            rename[old] = new

    if rename:
        ds = ds.rename(rename)

    candidates = {
        "geopotential_height_m": ["z", "geopotential"],
        "u_wind_ms": ["u", "u_component_of_wind"],
        "air_temperature_k": ["t", "temperature"],
    }

    out = xr.Dataset(
        coords={
            k: ds.coords[k]
            for k in ds.coords
            if k in ["time", "level_hpa", "lat", "lon"]
        }
    )

    for target, names in candidates.items():
        name = next((n for n in names if n in ds.data_vars), None)
        if name is None:
            continue

        da = ds[name].astype("float32")

        if target == "geopotential_height_m":
            da = da / C.G

        out[target] = da

    return out


def unpack_and_merge(path: Path) -> xr.Dataset:
    tempdir = Path(tempfile.mkdtemp(prefix="step12_unpack_"))

    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                zf.extractall(tempdir)
            files = sorted(tempdir.rglob("*.nc"))
        else:
            files = [path]

        if not files:
            raise RuntimeError(f"No NetCDF files found in {path}")

        pieces = []

        for nc in files:
            with xr.open_dataset(nc) as ds:
                pieces.append(normalise_dataset(ds).load())

        merged = xr.merge(
            pieces,
            compat="override",
            join="outer",
        )

        if "lat" in merged.coords and merged["lat"][0] < merged["lat"][-1]:
            merged = merged.sortby("lat", ascending=False)

        if "lon" in merged.coords:
            lon = ((merged["lon"] + 180) % 360) - 180
            merged = merged.assign_coords(lon=lon).sortby("lon")

        return merged

    finally:
        shutil.rmtree(
            tempdir,
            ignore_errors=True,
        )


def aggregate_to_daily(hourly: xr.Dataset) -> xr.Dataset:
    if "time" not in hourly.coords:
        raise RuntimeError("Downloaded ERA5 file has no time coordinate.")

    hourly = hourly.sortby("time")

    # Ensure the expected four samples exist on each UTC day.
    stamps = pd.to_datetime(hourly["time"].values)
    sample_count = (
        pd.Series(1, index=stamps.normalize())
        .groupby(level=0)
        .sum()
    )

    bad = sample_count[sample_count != len(TIMES)]

    if not bad.empty:
        raise RuntimeError(
            "Expected four ERA5 samples per UTC day "
            f"(00/06/12/18), but found bad days: {bad.head().to_dict()}"
        )

    daily = hourly.resample(time="1D").mean(
        dim="time",
        skipna=True,
        keep_attrs=True,
    )

    daily.attrs.update(
        {
            "step12_daily_mean_method": (
                "Arithmetic mean of ERA5 pressure-level fields at "
                "00, 06, 12 and 18 UTC."
            ),
            "step12_source_dataset": DATASET,
            "step12_source_times_utc": ",".join(TIMES),
            "step12_grid_degrees": "2.5 x 2.5",
        }
    )

    return daily


def validate_month(
    path: Path,
    year: int,
    month: int,
) -> tuple[bool, str]:

    try:
        with xr.open_dataset(path) as ds:
            required = {
                "geopotential_height_m",
                "u_wind_ms",
                "air_temperature_k",
            }

            if not required.issubset(ds.data_vars):
                return (
                    False,
                    f"missing variables {required - set(ds.data_vars)}",
                )

            if not {
                "time",
                "level_hpa",
                "lat",
                "lon",
            }.issubset(ds.coords):
                return False, "missing coordinates"

            levels = set(
                np.asarray(ds["level_hpa"])
                .astype(int)
                .tolist()
            )

            if not set(C.LEVELS).issubset(levels):
                return (
                    False,
                    f"missing levels {set(C.LEVELS) - levels}",
                )

            expected_days = len(
                C.days_in_month(year, month)
            )

            if ds.sizes.get("time", 0) != expected_days:
                return (
                    False,
                    f"time count {ds.sizes.get('time')} != {expected_days}",
                )

            times = pd.to_datetime(ds["time"].values)

            if not np.all(
                (times.year == year)
                & (times.month == month)
            ):
                return False, "wrong dates"

            if (
                ds.sizes.get("lat", 0) < 25
                or ds.sizes.get("lon", 0) < 100
            ):
                return False, "grid unexpectedly small"

        return True, "valid"

    except Exception as exc:
        return False, repr(exc)


def download_one(
    client,
    year: int,
    month: int,
    target: Path,
) -> None:

    request = {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "pressure_level": LEVELS,
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": C.days_in_month(year, month),
        "time": TIMES,
        "area": [90, -180, 20, 180],
        "grid": [2.5, 2.5],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    part = target.with_suffix(".download.part")
    part.unlink(missing_ok=True)

    client.retrieve(
        DATASET,
        request,
        str(part),
    )

    hourly = unpack_and_merge(part)

    required = {
        "geopotential_height_m",
        "u_wind_ms",
        "air_temperature_k",
    }

    if not required.issubset(hourly.data_vars):
        missing = required - set(hourly.data_vars)
        hourly.close()
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"Hourly download missing variables: {missing}"
        )

    daily = aggregate_to_daily(hourly)
    hourly.close()

    encoding = {
        name: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
        }
        for name in daily.data_vars
    }

    temp_nc = target.with_suffix(".nc.part")

    daily.to_netcdf(
        temp_nc,
        encoding=encoding,
    )

    daily.close()
    part.unlink(missing_ok=True)

    ok, reason = validate_month(
        temp_nc,
        year,
        month,
    )

    if not ok:
        temp_nc.unlink(missing_ok=True)
        raise RuntimeError(reason)

    temp_nc.replace(target)


def main() -> None:
    try:
        import cdsapi
    except ImportError as exc:
        raise SystemExit(
            "cdsapi is required: "
            "python -m pip install cdsapi"
        ) from exc

    client = cdsapi.Client()
    rows = []
    months = C.winter_months()

    for i, (year, month) in enumerate(months, 1):
        target = (
            C.RAW_DIR
            / f"ERA5_STRAT_DAILY_{year}_{month:02d}.nc"
        )

        ok, reason = (
            validate_month(target, year, month)
            if target.exists()
            else (False, "missing")
        )

        if ok:
            print(
                f"[{i}/{len(months)}] "
                f"SKIP valid {target.name}"
            )

        else:
            if target.exists():
                target.unlink()

            last = None

            for attempt in range(1, 4):
                try:
                    print(
                        f"[{i}/{len(months)}] "
                        f"Downloading {year}-{month:02d} "
                        f"from hourly ERA5 "
                        f"(00/06/12/18 UTC), "
                        f"attempt {attempt}/3"
                    )

                    download_one(
                        client,
                        year,
                        month,
                        target,
                    )

                    ok, reason = validate_month(
                        target,
                        year,
                        month,
                    )

                    if not ok:
                        raise RuntimeError(reason)

                    print(f"PASS: {target.name}")
                    break

                except Exception as exc:
                    last = exc
                    print(f"Attempt failed: {exc}")
                    time.sleep(5 * attempt)

            else:
                raise RuntimeError(
                    f"Failed {year}-{month:02d}: {last}"
                )

        rows.append(
            {
                "year": year,
                "month": month,
                "file": str(target),
                "valid": True,
                "size_mb": round(
                    target.stat().st_size / 1024**2,
                    2,
                ),
            }
        )

    inventory = pd.DataFrame(rows)

    inventory.to_csv(
        C.QC_DIR
        / "step12_stratosphere_download_inventory.csv",
        index=False,
    )

    report = [
        "STEP 12 ERA5 STRATOSPHERIC DOWNLOAD REPORT",
        f"Expected monthly files: {len(months)}",
        f"Valid monthly files: {int(inventory.valid.sum())}",
        "Domain: 20-90N, full longitude circle",
        "Grid: 2.5 degrees",
        f"Pressure levels: {C.LEVELS}",
        (
            "Variables: geopotential height, "
            "zonal wind, air temperature"
        ),
        (
            "Source: ERA5 hourly pressure-level archive; "
            "local daily means from 00, 06, 12 and 18 UTC."
        ),
        (
            "Reason for local aggregation: avoids CDS derived-daily "
            "statistics cost-limit failures for the hemispheric "
            "multi-level request."
        ),
    ]

    (
        C.QC_DIR
        / "step12_stratosphere_download_report.txt"
    ).write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("STEP 12-DOWNLOAD PASSED.")


if __name__ == "__main__":
    main()
