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

EVENT_CATALOGUE = PROJECT_ROOT / "06_events" / "step7e_frozen_primary_event_catalogue.csv"

PL_STD_DIR = PROJECT_ROOT / "03_intermediate" / "era5_preprocessed" / "pressure_level_standardized"
SL_INSTANT_STD_DIR = PROJECT_ROOT / "03_intermediate" / "era5_preprocessed" / "single_level_instant_standardized"

CLIM_ROOT = PROJECT_ROOT / "03_intermediate" / "era5_climatology"

OUT_ROOT = PROJECT_ROOT / "03_intermediate" / "era5_event_composites"
OUT_NETCDF = OUT_ROOT / "composite_netcdf"
OUT_TABLES = OUT_ROOT / "composite_tables"

REPORT_DIR = PROJECT_ROOT / "05_qc_reports" / "era5"

OUT_NETCDF.mkdir(parents=True, exist_ok=True)
OUT_TABLES.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LOG_CSV = REPORT_DIR / "step9d_era5_event_lag_composite_log.csv"
EVENT_LAG_TABLE = OUT_TABLES / "step9d_event_lag_date_table.csv"
SUMMARY_JSON = REPORT_DIR / "step9d_era5_event_lag_composite_summary.json"
REPORT_TXT = REPORT_DIR / "step9d_era5_event_lag_composite_report.txt"


LAGS = list(range(-45, 11))

# Core fields for first publishable analysis.
# PV and vorticity can be added later after the first composite diagnostics.
PRESSURE_SELECTION = {
    "geopotential_height_m": [500, 300, 200, 100, 50, 10],
    "air_temperature_k": [850, 500, 100, 50, 10],
    "u_wind_ms": [850, 500, 300, 250, 200, 100, 50, 30, 10],
    "v_wind_ms": [850, 500, 300, 250, 200, 100, 50, 30, 10],
}

SINGLE_SELECTION = [
    "t2m_c",
    "msl_hpa",
    "u10_ms",
    "v10_ms",
]


def safe_variable_name(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace(" ", "_")
        .replace("-", "minus")
        .replace(".", "_")
    )


def pressure_file_for_date(date: pd.Timestamp) -> Path:
    return PL_STD_DIR / f"ERA5_PL_STD_{date.year}_{date.month:02d}.nc"


def single_file_for_date(date: pd.Timestamp) -> Path:
    return SL_INSTANT_STD_DIR / f"ERA5_SL_INSTANT_STD_{date.year}_{date.month:02d}.nc"


def clim_file(group: str, variable: str) -> Path:
    safe = safe_variable_name(variable)
    return CLIM_ROOT / group / f"ERA5_CLIM_{group}_{safe}_calendar_day_1991_2020.nc"


def build_event_lag_table() -> pd.DataFrame:
    events = pd.read_csv(EVENT_CATALOGUE)
    events["start_date"] = pd.to_datetime(events["start_date"])
    events["end_date"] = pd.to_datetime(events["end_date"])

    rows = []

    for _, event in events.iterrows():
        start_date = event["start_date"]
        end_date = event["end_date"]
        duration = int((end_date - start_date).days + 1)

        for lag in LAGS:
            date = start_date + pd.Timedelta(days=lag)

            if lag < -10:
                window_class = "extended_precursor_minus45_to_minus11"
            elif -10 <= lag <= -1:
                window_class = "synoptic_precursor_minus10_to_minus1"
            elif 0 <= lag <= duration - 1:
                window_class = "event_period"
            else:
                window_class = "post_event"

            rows.append(
                {
                    "event_id": str(event["event_id"]),
                    "winter_label": event.get("winter_label", ""),
                    "event_start_date": start_date.date().isoformat(),
                    "event_end_date": end_date.date().isoformat(),
                    "event_duration_days": duration,
                    "lag_day_from_onset": lag,
                    "era5_date": date.date().isoformat(),
                    "calendar_day": date.strftime("%m-%d"),
                    "year": date.year,
                    "month": date.month,
                    "window_class": window_class,
                }
            )

    table = pd.DataFrame(rows)
    table.to_csv(EVENT_LAG_TABLE, index=False)

    return table


def get_template_coordinates(path: Path, has_level: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with xr.open_dataset(path) as ds:
        lat = ds["lat"].values.astype("float32")
        lon = ds["lon"].values.astype("float32")
        levels = None

        if has_level:
            levels = ds["level_hpa"].values.astype("float32")

    return lat, lon, levels


def create_pressure_output(
    path: Path,
    variable: str,
    selected_levels: list[int],
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[netCDF4.Dataset, netCDF4.Variable, netCDF4.Variable]:
    if path.exists():
        path.unlink()

    root = netCDF4.Dataset(path, "w", format="NETCDF4")

    root.createDimension("lag", len(LAGS))
    root.createDimension("level_hpa", len(selected_levels))
    root.createDimension("lat", len(lat))
    root.createDimension("lon", len(lon))

    lag_var = root.createVariable("lag_day_from_onset", "i4", ("lag",))
    lag_var[:] = np.array(LAGS, dtype="int32")
    lag_var.long_name = "lag day from cold-spell onset"

    level_var = root.createVariable("level_hpa", "f4", ("level_hpa",))
    level_var[:] = np.array(selected_levels, dtype="float32")
    level_var.units = "hPa"

    lat_var = root.createVariable("lat", "f4", ("lat",))
    lat_var[:] = lat
    lat_var.units = "degrees_north"

    lon_var = root.createVariable("lon", "f4", ("lon",))
    lon_var[:] = lon
    lon_var.units = "degrees_east"

    comp_var = root.createVariable(
        "composite_mean_anomaly",
        "f4",
        ("lag", "level_hpa", "lat", "lon"),
        zlib=True,
        complevel=4,
        fill_value=np.float32(np.nan),
        chunksizes=(1, 1, len(lat), len(lon)),
    )

    count_var = root.createVariable(
        "event_count",
        "i4",
        ("lag", "level_hpa"),
        zlib=True,
        complevel=4,
    )

    comp_var.source_variable = variable
    comp_var.long_name = f"Composite mean anomaly of {variable}"
    comp_var.anomaly_method = "ERA5 daily 00 UTC field minus 1991-2020 MM-DD climatology"

    root.title = f"ERA5 event-centred lag composite anomaly: {variable}"
    root.step = "Step 9D"
    root.lag_window = "-45 to +10 days from cold-spell onset"
    root.baseline = "1991-2020"
    root.created_utc = datetime.now(timezone.utc).isoformat()

    return root, comp_var, count_var


def create_single_output(
    path: Path,
    variable: str,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[netCDF4.Dataset, netCDF4.Variable, netCDF4.Variable]:
    if path.exists():
        path.unlink()

    root = netCDF4.Dataset(path, "w", format="NETCDF4")

    root.createDimension("lag", len(LAGS))
    root.createDimension("lat", len(lat))
    root.createDimension("lon", len(lon))

    lag_var = root.createVariable("lag_day_from_onset", "i4", ("lag",))
    lag_var[:] = np.array(LAGS, dtype="int32")
    lag_var.long_name = "lag day from cold-spell onset"

    lat_var = root.createVariable("lat", "f4", ("lat",))
    lat_var[:] = lat
    lat_var.units = "degrees_north"

    lon_var = root.createVariable("lon", "f4", ("lon",))
    lon_var[:] = lon
    lon_var.units = "degrees_east"

    comp_var = root.createVariable(
        "composite_mean_anomaly",
        "f4",
        ("lag", "lat", "lon"),
        zlib=True,
        complevel=4,
        fill_value=np.float32(np.nan),
        chunksizes=(1, len(lat), len(lon)),
    )

    count_var = root.createVariable(
        "event_count",
        "i4",
        ("lag",),
        zlib=True,
        complevel=4,
    )

    comp_var.source_variable = variable
    comp_var.long_name = f"Composite mean anomaly of {variable}"
    comp_var.anomaly_method = "ERA5 daily 00 UTC field minus 1991-2020 MM-DD climatology"

    root.title = f"ERA5 event-centred lag composite anomaly: {variable}"
    root.step = "Step 9D"
    root.lag_window = "-45 to +10 days from cold-spell onset"
    root.baseline = "1991-2020"
    root.created_utc = datetime.now(timezone.utc).isoformat()

    return root, comp_var, count_var


def month_groups(table: pd.DataFrame):
    for (year, month), subset in table.groupby(["year", "month"], sort=True):
        yield int(year), int(month), subset.copy()


def compute_pressure_variable(table: pd.DataFrame, variable: str, selected_levels: list[int]) -> dict:
    first_file = sorted(PL_STD_DIR.glob("ERA5_PL_STD_*.nc"))[0]
    lat, lon, available_levels = get_template_coordinates(first_file, has_level=True)

    if available_levels is None:
        raise RuntimeError("No pressure levels found in pressure-level standardized files.")

    available_level_ints = [int(float(x)) for x in available_levels]
    missing_levels = sorted(set(selected_levels) - set(available_level_ints))

    if missing_levels:
        raise RuntimeError(f"{variable}: missing requested levels {missing_levels}")

    output_path = OUT_NETCDF / f"ERA5_COMP_pressure_level_{safe_variable_name(variable)}_lag_minus45_to_plus10.nc"

    row = {
        "group": "pressure_level",
        "variable": variable,
        "levels": ",".join(str(x) for x in selected_levels),
        "output_file": str(output_path),
        "status": "started",
        "event_count_min": np.nan,
        "event_count_max": np.nan,
        "output_size_mb": np.nan,
        "error": "",
    }

    print("")
    print(f"PRESSURE COMPOSITE: {variable}")
    print(f"Levels: {selected_levels}")
    print(f"Output: {output_path}")

    try:
        root, comp_var, count_var = create_pressure_output(
            path=output_path,
            variable=variable,
            selected_levels=selected_levels,
            lat=lat,
            lon=lon,
        )

        clim_path = clim_file("pressure_level", variable)

        with xr.open_dataset(clim_path) as clim:
            for level_index, level in enumerate(selected_levels):
                print(f"  Level {level_index + 1}/{len(selected_levels)}: {level} hPa", flush=True)

                climatology = clim[variable].sel(level_hpa=level).load()

                sum_array = np.zeros((len(LAGS), len(lat), len(lon)), dtype="float64")
                count_array = np.zeros((len(LAGS),), dtype="int32")

                for year, month, subset in month_groups(table):
                    era5_path = PL_STD_DIR / f"ERA5_PL_STD_{year}_{month:02d}.nc"

                    if not era5_path.exists():
                        continue

                    with xr.open_dataset(era5_path) as ds:
                        for _, item in subset.iterrows():
                            date = pd.Timestamp(item["era5_date"])
                            lag = int(item["lag_day_from_onset"])
                            lag_index = LAGS.index(lag)
                            calendar_day = str(item["calendar_day"])

                            try:
                                field = ds[variable].sel(time=date, level_hpa=level).load()
                                normal = climatology.sel(calendar_day=calendar_day)
                                anomaly = (field - normal).values.astype("float64")

                                valid = np.isfinite(anomaly)

                                sum_array[lag_index][valid] += anomaly[valid]
                                count_array[lag_index] += 1

                            except Exception:
                                continue

                    gc.collect()

                composite = np.full_like(sum_array, np.nan, dtype="float32")

                for lag_index in range(len(LAGS)):
                    if count_array[lag_index] > 0:
                        composite[lag_index, :, :] = (
                            sum_array[lag_index, :, :] / count_array[lag_index]
                        ).astype("float32")

                comp_var[:, level_index, :, :] = composite
                count_var[:, level_index] = count_array

                root.sync()

                del climatology, sum_array, count_array, composite
                gc.collect()

        root.close()

        with xr.open_dataset(output_path) as check:
            counts = check["event_count"].values
            row["event_count_min"] = int(np.nanmin(counts))
            row["event_count_max"] = int(np.nanmax(counts))

        row["status"] = "written"
        row["output_size_mb"] = round(output_path.stat().st_size / 1024 / 1024, 3)

    except Exception as error:
        row["status"] = "failed"
        row["error"] = repr(error)

        try:
            root.close()
        except Exception:
            pass

        print(f"FAILED {variable}: {repr(error)}")

    return row


def compute_single_variable(table: pd.DataFrame, variable: str) -> dict:
    first_file = sorted(SL_INSTANT_STD_DIR.glob("ERA5_SL_INSTANT_STD_*.nc"))[0]
    lat, lon, _ = get_template_coordinates(first_file, has_level=False)

    output_path = OUT_NETCDF / f"ERA5_COMP_single_level_instant_{safe_variable_name(variable)}_lag_minus45_to_plus10.nc"

    row = {
        "group": "single_level_instant",
        "variable": variable,
        "levels": "",
        "output_file": str(output_path),
        "status": "started",
        "event_count_min": np.nan,
        "event_count_max": np.nan,
        "output_size_mb": np.nan,
        "error": "",
    }

    print("")
    print(f"SINGLE-LEVEL COMPOSITE: {variable}")
    print(f"Output: {output_path}")

    try:
        root, comp_var, count_var = create_single_output(
            path=output_path,
            variable=variable,
            lat=lat,
            lon=lon,
        )

        clim_path = clim_file("single_level_instant", variable)

        with xr.open_dataset(clim_path) as clim:
            climatology = clim[variable].load()

            sum_array = np.zeros((len(LAGS), len(lat), len(lon)), dtype="float64")
            count_array = np.zeros((len(LAGS),), dtype="int32")

            for year, month, subset in month_groups(table):
                era5_path = SL_INSTANT_STD_DIR / f"ERA5_SL_INSTANT_STD_{year}_{month:02d}.nc"

                if not era5_path.exists():
                    continue

                with xr.open_dataset(era5_path) as ds:
                    for _, item in subset.iterrows():
                        date = pd.Timestamp(item["era5_date"])
                        lag = int(item["lag_day_from_onset"])
                        lag_index = LAGS.index(lag)
                        calendar_day = str(item["calendar_day"])

                        try:
                            field = ds[variable].sel(time=date).load()
                            normal = climatology.sel(calendar_day=calendar_day)
                            anomaly = (field - normal).values.astype("float64")

                            valid = np.isfinite(anomaly)

                            sum_array[lag_index][valid] += anomaly[valid]
                            count_array[lag_index] += 1

                        except Exception:
                            continue

                gc.collect()

            composite = np.full_like(sum_array, np.nan, dtype="float32")

            for lag_index in range(len(LAGS)):
                if count_array[lag_index] > 0:
                    composite[lag_index, :, :] = (
                        sum_array[lag_index, :, :] / count_array[lag_index]
                    ).astype("float32")

            comp_var[:, :, :] = composite
            count_var[:] = count_array

        root.close()

        with xr.open_dataset(output_path) as check:
            counts = check["event_count"].values
            row["event_count_min"] = int(np.nanmin(counts))
            row["event_count_max"] = int(np.nanmax(counts))

        row["status"] = "written"
        row["output_size_mb"] = round(output_path.stat().st_size / 1024 / 1024, 3)

        del climatology, sum_array, count_array, composite
        gc.collect()

    except Exception as error:
        row["status"] = "failed"
        row["error"] = repr(error)

        try:
            root.close()
        except Exception:
            pass

        print(f"FAILED {variable}: {repr(error)}")

    return row


def main() -> None:
    print("STEP 9D: ERA5 EVENT-CENTRED LAG COMPOSITES")
    print("=" * 60)
    print("Low-memory mode: one variable and one level at a time.")
    print("This creates composite anomaly files, not full daily anomaly files.")
    print("")

    table = build_event_lag_table()

    event_count = table["event_id"].nunique()
    required_rows = event_count * len(LAGS)

    if len(table) != required_rows:
        raise RuntimeError("Event-lag table row count mismatch.")

    print(f"Frozen events: {event_count}")
    print(f"Lags: {min(LAGS)} to {max(LAGS)}")
    print(f"Event-lag rows: {len(table)}")

    rows = []

    for variable, levels in PRESSURE_SELECTION.items():
        row = compute_pressure_variable(
            table=table,
            variable=variable,
            selected_levels=levels,
        )

        rows.append(row)
        pd.DataFrame(rows).to_csv(LOG_CSV, index=False)

    for variable in SINGLE_SELECTION:
        row = compute_single_variable(
            table=table,
            variable=variable,
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
            min_event_count=("event_count_min", "min"),
            max_event_count=("event_count_max", "max"),
        )
        .reset_index()
    )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "event_count": int(event_count),
        "lag_min": min(LAGS),
        "lag_max": max(LAGS),
        "lag_count": len(LAGS),
        "method": "event-day ERA5 minus 1991-2020 MM-DD climatology, then mean over events by lag",
        "composite_files_written": int(log["status"].eq("written").sum()),
        "composite_files_failed": int((~log["status"].eq("written")).sum()),
        "group_summary": group_summary.to_dict(orient="records"),
        "log_csv": str(LOG_CSV),
        "event_lag_table": str(EVENT_LAG_TABLE),
        "output_folder": str(OUT_NETCDF),
    }

    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "STEP 9D: ERA5 EVENT-CENTRED LAG COMPOSITES",
        "=" * 60,
        f"Frozen event count: {event_count}",
        f"Lag window: {min(LAGS)} to {max(LAGS)}",
        f"Lag count: {len(LAGS)}",
        "Anomaly method: event-day ERA5 minus 1991-2020 MM-DD climatology",
        "",
        group_summary.to_string(index=False),
        "",
        f"Composite files written: {summary['composite_files_written']}",
        f"Composite files failed: {summary['composite_files_failed']}",
        "",
        f"Event-lag table: {EVENT_LAG_TABLE}",
        f"Composite NetCDF folder: {OUT_NETCDF}",
        f"Log CSV: {LOG_CSV}",
        f"Summary JSON: {SUMMARY_JSON}",
    ]

    if not failed.empty:
        lines.extend(
            [
                "",
                "FAILED VARIABLES:",
                failed[["group", "variable", "levels", "error"]].to_string(index=False),
            ]
        )

    REPORT_TXT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("")
    print("\n".join(lines))

    if not failed.empty:
        raise SystemExit("\nSTEP 9D FAILED. Check failed variables in the report.")

    print("\nSTEP 9D-1 PASSED. ERA5 EVENT-CENTRED LAG COMPOSITES CREATED.")


if __name__ == "__main__":
    main()
