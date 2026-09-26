from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PL_STD_DIR = PROJECT_ROOT / "03_intermediate" / "era5_preprocessed" / "pressure_level_standardized"
SL_INSTANT_STD_DIR = PROJECT_ROOT / "03_intermediate" / "era5_preprocessed" / "single_level_instant_standardized"
SL_ACCUM_STD_DIR = PROJECT_ROOT / "03_intermediate" / "era5_preprocessed" / "single_level_accum_standardized"

OUT_ROOT = PROJECT_ROOT / "03_intermediate" / "era5_climatology"
REPORT_DIR = PROJECT_ROOT / "05_qc_reports" / "era5"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

LOG_CSV = REPORT_DIR / "step9c_era5_calendar_day_climatology_log.csv"
SAMPLE_COUNT_CSV = REPORT_DIR / "step9c_era5_calendar_day_sample_counts.csv"
SUMMARY_JSON = REPORT_DIR / "step9c_era5_calendar_day_climatology_summary.json"
REPORT_TXT = REPORT_DIR / "step9c_era5_calendar_day_climatology_report.txt"


BASELINE_START = pd.Timestamp("1991-01-01")
BASELINE_END = pd.Timestamp("2020-12-31")

# Use a leap year for January-March so that 02-29 is included.
# October-December can use any normal year.
CALENDAR_DAYS = (
    [d.strftime("%m-%d") for d in pd.date_range("2000-01-01", "2000-03-31")]
    + [d.strftime("%m-%d") for d in pd.date_range("2001-10-01", "2001-12-31")]
)

CALENDAR_DAY_TO_INDEX = {
    day: index for index, day in enumerate(CALENDAR_DAYS)
}


GROUPS = {
    "pressure_level": {
        "input_dir": PL_STD_DIR,
        "pattern": "ERA5_PL_STD_*.nc",
        "output_dir": OUT_ROOT / "pressure_level",
        "variables": [
            "geopotential_height_m",
            "air_temperature_k",
            "u_wind_ms",
            "v_wind_ms",
            "relative_vorticity_s-1",
            "potential_vorticity_pvu",
        ],
        "has_level": True,
    },
    "single_level_instant": {
        "input_dir": SL_INSTANT_STD_DIR,
        "pattern": "ERA5_SL_INSTANT_STD_*.nc",
        "output_dir": OUT_ROOT / "single_level_instant",
        "variables": [
            "t2m_c",
            "msl_hpa",
            "surface_pressure_hpa",
            "u10_ms",
            "v10_ms",
        ],
        "has_level": False,
    },
    "single_level_accum": {
        "input_dir": SL_ACCUM_STD_DIR,
        "pattern": "ERA5_SL_ACCUM_STD_*.nc",
        "output_dir": OUT_ROOT / "single_level_accum",
        "variables": [
            "tisr_j_m2",
        ],
        "has_level": False,
    },
}


def safe_variable_name(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace(" ", "_")
        .replace("-", "minus")
        .replace(".", "_")
    )


def get_template_coordinates(files: list[Path], has_level: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with xr.open_dataset(files[0], decode_times=True) as ds:
        lat = ds["lat"].values.astype("float32")
        lon = ds["lon"].values.astype("float32")
        levels = None

        if has_level:
            levels = ds["level_hpa"].values.astype("float32")

    return lat, lon, levels


def create_output_file(
    output_path: Path,
    variable: str,
    group_name: str,
    lat: np.ndarray,
    lon: np.ndarray,
    levels: np.ndarray | None,
) -> tuple[netCDF4.Dataset, netCDF4.Variable]:
    if output_path.exists():
        output_path.unlink()

    root = netCDF4.Dataset(output_path, "w", format="NETCDF4")

    root.createDimension("calendar_day", len(CALENDAR_DAYS))
    root.createDimension("lat", len(lat))
    root.createDimension("lon", len(lon))

    calendar_day_var = root.createVariable("calendar_day", str, ("calendar_day",))
    calendar_day_var[:] = np.array(CALENDAR_DAYS, dtype=object)
    calendar_day_var.long_name = "calendar day"
    calendar_day_var.description = "MM-DD calendar day"

    lat_var = root.createVariable("lat", "f4", ("lat",))
    lat_var[:] = lat
    lat_var.standard_name = "latitude"
    lat_var.units = "degrees_north"

    lon_var = root.createVariable("lon", "f4", ("lon",))
    lon_var[:] = lon
    lon_var.standard_name = "longitude"
    lon_var.units = "degrees_east"

    if levels is not None:
        root.createDimension("level_hpa", len(levels))
        level_var = root.createVariable("level_hpa", "f4", ("level_hpa",))
        level_var[:] = levels
        level_var.long_name = "pressure level"
        level_var.units = "hPa"

        data_var = root.createVariable(
            variable,
            "f4",
            ("calendar_day", "level_hpa", "lat", "lon"),
            zlib=True,
            complevel=4,
            fill_value=np.float32(np.nan),
            chunksizes=(1, 1, len(lat), len(lon)),
        )
    else:
        data_var = root.createVariable(
            variable,
            "f4",
            ("calendar_day", "lat", "lon"),
            zlib=True,
            complevel=4,
            fill_value=np.float32(np.nan),
            chunksizes=(1, len(lat), len(lon)),
        )

    data_var.climatology_baseline = "1991-2020"
    data_var.climatology_dimension = "calendar_day"

    root.title = f"ERA5 {group_name} calendar-day climatology: {variable}"
    root.step = "Step 9C"
    root.method = "Low-memory streaming mean by MM-DD calendar day"
    root.baseline_period = "1991-01-01 to 2020-12-31"
    root.temporal_resolution = "daily 00 UTC fields, not daily means"
    root.created_utc = datetime.now(timezone.utc).isoformat()
    root.note = "Anomalies should be computed by subtracting the matching calendar_day climatology."

    return root, data_var


def baseline_indices(times: pd.DatetimeIndex) -> np.ndarray:
    return np.where(
        (times >= BASELINE_START)
        & (times <= BASELINE_END)
        & (pd.Series(times.strftime("%m-%d")).isin(CALENDAR_DAY_TO_INDEX.keys()).to_numpy())
    )[0]


def update_accumulators(
    sum_array: np.ndarray,
    count_array: np.ndarray,
    data: np.ndarray,
    times: pd.DatetimeIndex,
) -> None:
    for index, timestamp in enumerate(times):
        calendar_day = timestamp.strftime("%m-%d")
        calendar_index = CALENDAR_DAY_TO_INDEX.get(calendar_day)

        if calendar_index is None:
            continue

        daily = data[index].astype("float64", copy=False)
        valid = np.isfinite(daily)

        sum_array[calendar_index][valid] += daily[valid]
        count_array[calendar_index][valid] += 1


def compute_single_level_climatology(
    files: list[Path],
    variable: str,
    output_path: Path,
    group_name: str,
) -> dict:
    lat, lon, _ = get_template_coordinates(files, has_level=False)

    row = {
        "group": group_name,
        "variable": variable,
        "input_file_count": len(files),
        "output_file": str(output_path),
        "status": "started",
        "calendar_day_count": len(CALENDAR_DAYS),
        "level_count": 0,
        "minimum_sample_count": np.nan,
        "maximum_sample_count": np.nan,
        "output_size_mb": np.nan,
        "error": "",
    }

    print("")
    print(f"LOW-MEMORY CLIMATOLOGY: {group_name} / {variable}")
    print(f"Output: {output_path}")

    try:
        root, data_var = create_output_file(
            output_path=output_path,
            variable=variable,
            group_name=group_name,
            lat=lat,
            lon=lon,
            levels=None,
        )

        sum_array = np.zeros((len(CALENDAR_DAYS), len(lat), len(lon)), dtype="float64")
        count_array = np.zeros((len(CALENDAR_DAYS), len(lat), len(lon)), dtype="uint16")

        for file_index, path in enumerate(files, start=1):
            print(f"  {file_index:03d}/{len(files)} {path.name}", flush=True)

            with xr.open_dataset(path, decode_times=True) as ds:
                times_all = pd.to_datetime(ds["time"].values)
                keep = baseline_indices(times_all)

                if len(keep) == 0:
                    continue

                times = times_all[keep]
                data = ds[variable].isel(time=keep).load().values

                update_accumulators(
                    sum_array=sum_array,
                    count_array=count_array,
                    data=data,
                    times=times,
                )

            gc.collect()

        with np.errstate(invalid="ignore", divide="ignore"):
            climatology = sum_array / count_array

        climatology[count_array == 0] = np.nan
        data_var[:, :, :] = climatology.astype("float32")

        row["minimum_sample_count"] = int(np.nanmin(count_array))
        row["maximum_sample_count"] = int(np.nanmax(count_array))

        root.close()

        row["status"] = "written"
        row["output_size_mb"] = round(output_path.stat().st_size / 1024 / 1024, 3)

        del sum_array, count_array, climatology
        gc.collect()

    except Exception as error:
        row["status"] = "failed"
        row["error"] = repr(error)

        try:
            root.close()
        except Exception:
            pass

        print(f"FAILED: {group_name} / {variable}: {repr(error)}", flush=True)

    return row


def compute_pressure_level_climatology(
    files: list[Path],
    variable: str,
    output_path: Path,
    group_name: str,
) -> dict:
    lat, lon, levels = get_template_coordinates(files, has_level=True)

    if levels is None:
        raise RuntimeError("Pressure-level input does not contain level_hpa coordinate.")

    row = {
        "group": group_name,
        "variable": variable,
        "input_file_count": len(files),
        "output_file": str(output_path),
        "status": "started",
        "calendar_day_count": len(CALENDAR_DAYS),
        "level_count": len(levels),
        "minimum_sample_count": np.nan,
        "maximum_sample_count": np.nan,
        "output_size_mb": np.nan,
        "error": "",
    }

    print("")
    print(f"LOW-MEMORY CLIMATOLOGY: {group_name} / {variable}")
    print(f"Output: {output_path}")

    try:
        root, data_var = create_output_file(
            output_path=output_path,
            variable=variable,
            group_name=group_name,
            lat=lat,
            lon=lon,
            levels=levels,
        )

        global_min_count = None
        global_max_count = None

        for level_index, level in enumerate(levels):
            print(f"  Level {level_index + 1}/{len(levels)}: {level:g} hPa", flush=True)

            sum_array = np.zeros((len(CALENDAR_DAYS), len(lat), len(lon)), dtype="float64")
            count_array = np.zeros((len(CALENDAR_DAYS), len(lat), len(lon)), dtype="uint16")

            for file_index, path in enumerate(files, start=1):
                print(f"    {file_index:03d}/{len(files)} {path.name}", flush=True)

                with xr.open_dataset(path, decode_times=True) as ds:
                    times_all = pd.to_datetime(ds["time"].values)
                    keep = baseline_indices(times_all)

                    if len(keep) == 0:
                        continue

                    times = times_all[keep]

                    data = (
                        ds[variable]
                        .sel(level_hpa=level)
                        .isel(time=keep)
                        .load()
                        .values
                    )

                    update_accumulators(
                        sum_array=sum_array,
                        count_array=count_array,
                        data=data,
                        times=times,
                    )

                gc.collect()

            with np.errstate(invalid="ignore", divide="ignore"):
                climatology = sum_array / count_array

            climatology[count_array == 0] = np.nan

            data_var[:, level_index, :, :] = climatology.astype("float32")

            level_min = int(np.nanmin(count_array))
            level_max = int(np.nanmax(count_array))

            global_min_count = level_min if global_min_count is None else min(global_min_count, level_min)
            global_max_count = level_max if global_max_count is None else max(global_max_count, level_max)

            root.sync()

            del sum_array, count_array, climatology
            gc.collect()

        root.close()

        row["minimum_sample_count"] = int(global_min_count)
        row["maximum_sample_count"] = int(global_max_count)

        row["status"] = "written"
        row["output_size_mb"] = round(output_path.stat().st_size / 1024 / 1024, 3)

    except Exception as error:
        row["status"] = "failed"
        row["error"] = repr(error)

        try:
            root.close()
        except Exception:
            pass

        print(f"FAILED: {group_name} / {variable}: {repr(error)}", flush=True)

    return row


def create_sample_count_table() -> pd.DataFrame:
    files = sorted(PL_STD_DIR.glob("ERA5_PL_STD_*.nc"))

    dates = []

    for path in files:
        with xr.open_dataset(path, decode_times=True) as ds:
            times = pd.to_datetime(ds["time"].values)
            times = times[
                (times >= BASELINE_START)
                & (times <= BASELINE_END)
            ]

            dates.extend(times)

    table = pd.DataFrame({"date": dates})
    table["calendar_day"] = table["date"].dt.strftime("%m-%d")
    table = table.loc[table["calendar_day"].isin(CALENDAR_DAYS)].copy()
    table["month"] = table["date"].dt.month
    table["day"] = table["date"].dt.day

    counts = (
        table.groupby(["calendar_day", "month", "day"])
        .size()
        .reset_index(name="baseline_sample_count")
        .sort_values(["month", "day"])
    )

    counts.to_csv(SAMPLE_COUNT_CSV, index=False)

    return counts


def main() -> None:
    rows = []

    sample_counts = create_sample_count_table()

    print("STEP 9C LOW-MEMORY ERA5 CALENDAR-DAY CLIMATOLOGY")
    print("=" * 64)
    print("This version avoids open_mfdataset and processes one variable/level/file at a time.")
    print("It is slower but much safer for low-RAM laptops.")
    print("")

    for group_name, config in GROUPS.items():
        input_dir = config["input_dir"]
        output_dir = config["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(input_dir.glob(config["pattern"]))

        if len(files) != 240:
            raise RuntimeError(
                f"{group_name}: expected 240 standardized files after October-March extension, found {len(files)}"
            )

        for variable in config["variables"]:
            safe_name = safe_variable_name(variable)

            output_path = (
                output_dir
                / f"ERA5_CLIM_{group_name}_{safe_name}_calendar_day_1991_2020.nc"
            )

            if config["has_level"]:
                row = compute_pressure_level_climatology(
                    files=files,
                    variable=variable,
                    output_path=output_path,
                    group_name=group_name,
                )
            else:
                row = compute_single_level_climatology(
                    files=files,
                    variable=variable,
                    output_path=output_path,
                    group_name=group_name,
                )

            rows.append(row)
            pd.DataFrame(rows).to_csv(LOG_CSV, index=False)

    log = pd.DataFrame(rows)
    log.to_csv(LOG_CSV, index=False)

    failed = log.loc[~log["status"].eq("written")]

    group_summary = (
        log.groupby("group")
        .agg(
            variables=("variable", "count"),
            written=("status", lambda s: int((s == "written").sum())),
            failed=("status", lambda s: int((s != "written").sum())),
            total_size_mb=("output_size_mb", "sum"),
        )
        .reset_index()
    )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "low_memory_streaming",
        "baseline_start": str(BASELINE_START.date()),
        "baseline_end": str(BASELINE_END.date()),
        "calendar_day_count": int(len(CALENDAR_DAYS)),
        "minimum_calendar_day_sample_count": int(sample_counts["baseline_sample_count"].min()),
        "maximum_calendar_day_sample_count": int(sample_counts["baseline_sample_count"].max()),
        "climatology_files_written": int(log["status"].eq("written").sum()),
        "climatology_files_failed": int((~log["status"].eq("written")).sum()),
        "group_summary": group_summary.to_dict(orient="records"),
        "log_csv": str(LOG_CSV),
        "sample_count_csv": str(SAMPLE_COUNT_CSV),
    }

    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "STEP 9C: ERA5 CALENDAR-DAY CLIMATOLOGY",
        "=" * 52,
        "Method: low-memory streaming calculation",
        "Baseline period: 1991-01-01 to 2020-12-31",
        "Calendar-day method: mean by MM-DD",
        "Temporal resolution: daily 00 UTC fields, not daily means",
        "",
        f"Calendar days: {summary['calendar_day_count']}",
        f"Minimum baseline sample count: {summary['minimum_calendar_day_sample_count']}",
        f"Maximum baseline sample count: {summary['maximum_calendar_day_sample_count']}",
        "",
        group_summary.to_string(index=False),
        "",
        f"Climatology files written: {summary['climatology_files_written']}",
        f"Climatology files failed: {summary['climatology_files_failed']}",
        "",
        f"Log CSV: {LOG_CSV}",
        f"Sample-count CSV: {SAMPLE_COUNT_CSV}",
        f"Summary JSON: {SUMMARY_JSON}",
    ]

    if not failed.empty:
        lines.extend(
            [
                "",
                "FAILED VARIABLES:",
                failed[["group", "variable", "error"]].to_string(index=False),
            ]
        )

    REPORT_TXT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("")
    print("\n".join(lines))

    if not failed.empty:
        raise SystemExit(
            "\nSTEP 9C FAILED. Check failed variables in the report."
        )

    print("\nSTEP 9C-1 PASSED. ERA5 LOW-MEMORY CLIMATOLOGY FILES CREATED.")


if __name__ == "__main__":
    main()
