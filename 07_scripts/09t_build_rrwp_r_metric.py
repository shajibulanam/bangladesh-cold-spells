from __future__ import annotations

import argparse
import calendar
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "01_raw_data" / "era5" / "rrwp_v250_global"
GROUP_TABLE = PROJECT_ROOT / "08_outputs" / "tables" / "era5_persistence" / "table_18_persistence_groups.csv"
OUT_ROOT = PROJECT_ROOT / "03_intermediate" / "rrwp"
NETCDF_DIR = OUT_ROOT / "netcdf"
TABLE_DIR = PROJECT_ROOT / "08_outputs" / "tables" / "rrwp"
REPORT_DIR = PROJECT_ROOT / "05_qc_reports" / "rrwp"
for folder in [OUT_ROOT, NETCDF_DIR, TABLE_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

DAILY_NC = NETCDF_DIR / "ERA5_RRWP_R_metric_daily_1985_2025.nc"
COMPOSITE_NC = NETCDF_DIR / "ERA5_RRWP_event_composites_lag20_to10.nc"
TABLE43 = TABLE_DIR / "table_43_daily_rrwp_sector_metrics.csv"
TABLE44_LONG = TABLE_DIR / "table_44_event_rrwp_metrics_long.csv"
TABLE44_WIDE = TABLE_DIR / "table_44b_event_rrwp_predictors.csv"
TABLE47 = TABLE_DIR / "table_47_rrwp_episode_catalogue.csv"
TABLE48 = TABLE_DIR / "table_48_selected_lag_rrwp_metrics.csv"
SUMMARY_JSON = REPORT_DIR / "step10e_rrwp_metric_summary.json"
REPORT_TXT = REPORT_DIR / "step10e_rrwp_metric_report.txt"

START_WINTER = 1985
END_WINTER = 2024
EXPECTED_FILES = 320
LAT_MIN = 35.0
LAT_MAX = 65.0
TIME_FILTER_DAYS = 15
WAVENUMBER_MIN = 4
WAVENUMBER_MAX = 15
BASELINE_START = pd.Timestamp("1991-01-01")
BASELINE_END = pd.Timestamp("2020-12-31")
HIGH_PERCENTILE = 90.0
INSTANT_PEAK_PERCENTILE = 75.0
EPISODE_MIN_DAYS = 3
PEAK_MIN_DISTANCE_DAYS = 3
EVENT_LAGS = np.arange(-20, 11, dtype=int)
SELECTED_LAGS = [-15, -10, -7, -5, -3, 0, 3, 5, 10]
WINDOWS = {
    "pre15": (-15, -1),
    "pre10": (-10, -1),
    "pre5": (-5, -1),
    "first5": (0, 4),
    "extended": (-15, 10),
}
SECTORS = {
    "eurasian_upstream": {"label": "Eurasian upstream", "lon_min": 30.0, "lon_max": 60.0},
    "central_asia": {"label": "Central Asia", "lon_min": 55.0, "lon_max": 80.0},
    "bangladesh_sector": {"label": "Bangladesh-centred 60-degree sector", "lon_min": 60.0, "lon_max": 120.0},
    "asian_corridor": {"label": "Asian corridor", "lon_min": 30.0, "lon_max": 120.0},
    "east_asia": {"label": "East Asia downstream", "lon_min": 90.0, "lon_max": 150.0},
}


def expected_months() -> list[tuple[int, int]]:
    periods: list[tuple[int, int]] = []
    for winter in range(START_WINTER, END_WINTER + 1):
        periods.extend((winter, month) for month in [9, 10, 11, 12])
        periods.extend((winter + 1, month) for month in [1, 2, 3, 4])
    return periods


def variable_name(ds: xr.Dataset) -> str:
    for candidate in ["v", "v_component_of_wind", "meridional_wind"]:
        if candidate in ds.data_vars:
            return candidate
    raise KeyError(f"V-wind variable not found; variables={list(ds.data_vars)}")


def load_month_latitude_mean(path: Path) -> xr.DataArray:
    with xr.open_dataset(path) as source:
        rename: dict[str, str] = {}
        if "valid_time" in source.coords:
            rename["valid_time"] = "time"
        if "latitude" in source.coords:
            rename["latitude"] = "lat"
        if "longitude" in source.coords:
            rename["longitude"] = "lon"
        if "pressure_level" in source.coords:
            rename["pressure_level"] = "level_hpa"
        if "level" in source.coords:
            rename["level"] = "level_hpa"
        ds = source.rename(rename)
        name = variable_name(ds)
        field = ds[name]
        if "level_hpa" in field.dims:
            field = field.sel(level_hpa=250)
        if "expver" in field.dims:
            combined = field.isel(expver=0)
            for index in range(1, field.sizes["expver"]):
                combined = combined.combine_first(field.isel(expver=index))
            field = combined
        field = field.sortby("lat", ascending=True)
        lon_mod = np.mod(field["lon"].values.astype(float), 360.0)
        field = field.assign_coords(lon=lon_mod).sortby("lon")
        _, unique_indices = np.unique(field["lon"].values, return_index=True)
        field = field.isel(lon=np.sort(unique_indices))
        selected = field.where(
            (field["lat"] >= LAT_MIN) & (field["lat"] <= LAT_MAX),
            drop=True,
        )
        weights = np.cos(np.deg2rad(selected["lat"]))
        latitude_mean = selected.weighted(weights).mean("lat").load()
    latitude_mean.name = "v250_latmean_ms"
    return latitude_mean.transpose("time", "lon")


def analytic_wave(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.ndim != 2:
        raise ValueError(f"Expected time x longitude array, found shape {values.shape}")
    analytic = np.full(values.shape, np.nan + 0j, dtype="complex128")
    valid_rows = np.all(np.isfinite(values), axis=1)
    if np.any(valid_rows):
        spectrum = np.fft.fft(values[valid_rows], axis=1)
        selected = np.zeros_like(spectrum)
        selected[:, WAVENUMBER_MIN : WAVENUMBER_MAX + 1] = (
            2.0 * spectrum[:, WAVENUMBER_MIN : WAVENUMBER_MAX + 1]
        )
        analytic[valid_rows] = np.fft.ifft(selected, axis=1)
    return (
        np.abs(analytic).astype("float32"),
        np.real(analytic).astype("float32"),
        np.angle(analytic).astype("float32"),
    )


def calendar_climatology(values: np.ndarray, times: pd.DatetimeIndex) -> dict[str, np.ndarray | list[str]]:
    month_days = np.asarray(times.strftime("%m-%d"))
    calendar_days = sorted(np.unique(month_days).tolist())
    baseline = (times >= BASELINE_START) & (times <= BASELINE_END)
    mean = np.full((len(calendar_days), values.shape[1]), np.nan, dtype="float32")
    std = np.full_like(mean, np.nan)
    p75 = np.full_like(mean, np.nan)
    p90 = np.full_like(mean, np.nan)
    count = np.zeros(len(calendar_days), dtype="int16")

    for index, month_day in enumerate(calendar_days):
        selected = baseline & (month_days == month_day)
        subset = values[selected]
        count[index] = int(np.sum(np.any(np.isfinite(subset), axis=1)))
        if subset.size == 0 or not np.isfinite(subset).any():
            continue
        mean[index] = np.nanmean(subset, axis=0).astype("float32")
        std[index] = np.nanstd(subset, axis=0, ddof=1).astype("float32")
        p75[index] = np.nanpercentile(subset, 75, axis=0).astype("float32")
        p90[index] = np.nanpercentile(subset, 90, axis=0).astype("float32")

    return {
        "calendar_days": calendar_days,
        "mean": mean,
        "std": std,
        "p75": p75,
        "p90": p90,
        "count": count,
    }


def map_climatology_to_time(
    climatology: dict[str, np.ndarray | list[str]],
    times: pd.DatetimeIndex,
    key: str,
) -> np.ndarray:
    days = list(climatology["calendar_days"])
    lookup = {value: index for index, value in enumerate(days)}
    indices = np.asarray([lookup[value] for value in times.strftime("%m-%d")], dtype=int)
    return np.asarray(climatology[key])[indices]



def safe_nanmean_axis1(values: np.ndarray) -> np.ndarray:
    valid_count = np.sum(np.isfinite(values), axis=1)
    total = np.nansum(values, axis=1)
    return np.divide(
        total,
        valid_count,
        out=np.full(values.shape[0], np.nan, dtype="float64"),
        where=valid_count > 0,
    )


def safe_nanmax_axis1(values: np.ndarray) -> np.ndarray:
    output = np.full(values.shape[0], np.nan, dtype="float64")
    valid_rows = np.any(np.isfinite(values), axis=1)
    if np.any(valid_rows):
        output[valid_rows] = np.nanmax(values[valid_rows], axis=1)
    return output

def sector_mask(lon: np.ndarray, config: dict[str, float | str]) -> np.ndarray:
    return (lon >= float(config["lon_min"])) & (lon <= float(config["lon_max"]))


def longest_true_run(values: np.ndarray, dates: pd.DatetimeIndex) -> int:
    longest = 0
    current = 0
    for index, active in enumerate(values.astype(bool)):
        continuous = index == 0 or (dates[index] - dates[index - 1]).days == 1
        if active:
            if not continuous:
                current = 0
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def identify_episodes(
    dates: pd.DatetimeIndex,
    high_flag: np.ndarray,
    r_values: np.ndarray,
    thresholds: np.ndarray,
    sector: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, object]] = []
    episode_days = np.zeros(len(dates), dtype=bool)
    run: list[int] = []

    def finalize(indices: list[int]) -> None:
        if len(indices) < EPISODE_MIN_DAYS:
            return
        episode_days[indices] = True
        values = r_values[indices]
        excess = np.maximum(values - thresholds[indices], 0.0)
        peak_local = int(np.nanargmax(values))
        rows.append(
            {
                "sector": sector,
                "sector_label": SECTORS[sector]["label"],
                "episode_id": f"{sector}_{dates[indices[0]].strftime('%Y%m%d')}",
                "start_date": dates[indices[0]],
                "end_date": dates[indices[-1]],
                "duration_days": len(indices),
                "peak_date": dates[indices[peak_local]],
                "peak_r_ms": float(np.nanmax(values)),
                "mean_r_ms": float(np.nanmean(values)),
                "integrated_excess_ms_days": float(np.nansum(excess)),
            }
        )

    for index, active in enumerate(high_flag.astype(bool)):
        continuous = index == 0 or (dates[index] - dates[index - 1]).days == 1
        if not active or not continuous:
            finalize(run)
            run = []
        if active:
            run.append(index)
    finalize(run)
    return pd.DataFrame(rows), episode_days


def summary(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": np.nan, "minimum": np.nan, "maximum": np.nan}
    return {
        "mean": float(np.mean(finite)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
    }


def build_event_composite(
    events: pd.DataFrame,
    times: pd.DatetimeIndex,
    lon: np.ndarray,
    variables: dict[str, np.ndarray],
) -> xr.Dataset:
    time_lookup = {date.normalize(): index for index, date in enumerate(times)}
    groups = events["persistence_group"].astype(str).to_numpy()
    event_arrays: dict[str, np.ndarray] = {}

    for name, values in variables.items():
        data = np.full((len(events), len(EVENT_LAGS), len(lon)), np.nan, dtype="float32")
        for event_index, onset in enumerate(pd.to_datetime(events["start_date"])):
            for lag_index, lag in enumerate(EVENT_LAGS):
                index = time_lookup.get((onset + pd.Timedelta(days=int(lag))).normalize())
                if index is not None:
                    data[event_index, lag_index] = values[index]
        event_arrays[name] = data

    dataset = xr.Dataset(
        coords={
            "lag_day_from_onset": EVENT_LAGS.astype("int16"),
            "lon": lon.astype("float32"),
        }
    )
    for name, data in event_arrays.items():
        dataset[f"{name}_all_event_mean"] = (
            ("lag_day_from_onset", "lon"),
            np.nanmean(data, axis=0).astype("float32"),
        )
        dataset[f"{name}_short_mean"] = (
            ("lag_day_from_onset", "lon"),
            np.nanmean(data[groups == "short"], axis=0).astype("float32"),
        )
        dataset[f"{name}_persistent_mean"] = (
            ("lag_day_from_onset", "lon"),
            np.nanmean(data[groups == "persistent"], axis=0).astype("float32"),
        )
        dataset[f"{name}_persistent_minus_short"] = (
            ("lag_day_from_onset", "lon"),
            (
                np.nanmean(data[groups == "persistent"], axis=0)
                - np.nanmean(data[groups == "short"], axis=0)
            ).astype("float32"),
        )
    dataset.attrs.update(
        {
            "title": "Event-centred recurrent Rossby-wave metric composites",
            "latitude_average": f"{LAT_MIN:.0f}-{LAT_MAX:.0f}N cosine-weighted V250",
            "time_filter": f"{TIME_FILTER_DAYS}-day centred running mean",
            "wavenumber_filter": f"zonal wavenumbers {WAVENUMBER_MIN}-{WAVENUMBER_MAX}",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    return dataset


def smoke_test() -> None:
    files = sorted(INPUT_DIR.glob("ERA5_RRWP_V250_*.nc"))
    if not files:
        raise FileNotFoundError(f"No global V250 files found in {INPUT_DIR}")
    sample = load_month_latitude_mean(files[0])
    if sample.ndim != 2 or sample.sizes["lon"] < 359:
        raise RuntimeError(f"Unexpected sample dimensions: {dict(sample.sizes)}")
    values = sample.values.astype("float64")
    envelope, filtered, phase = analytic_wave(values)
    if not np.isfinite(envelope).any() or float(np.nanmax(envelope)) <= 0:
        raise RuntimeError("Synthetic envelope calculation failed on sample file")
    print("STEP 10E SMOKE TEST")
    print("=" * 60)
    print("Sample file:", files[0].name)
    print("Dimensions:", dict(sample.sizes))
    print("Envelope maximum:", f"{float(np.nanmax(envelope)):.3f} m s-1")
    print("Filtered field range:", f"{float(np.nanmin(filtered)):.3f} to {float(np.nanmax(filtered)):.3f} m s-1")
    print("Phase range:", f"{float(np.nanmin(phase)):.3f} to {float(np.nanmax(phase)):.3f} rad")
    print("STEP 10E SMOKE TEST PASSED.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
        return

    files = sorted(INPUT_DIR.glob("ERA5_RRWP_V250_*.nc"))
    if len(files) != EXPECTED_FILES:
        raise RuntimeError(f"Expected {EXPECTED_FILES} global V250 files, found {len(files)}")

    print("STEP 10E: BUILD RECURRENT ROSSBY-WAVE R METRIC")
    print("=" * 72)
    monthly_parts: list[xr.DataArray] = []
    for index, path in enumerate(files, start=1):
        monthly_parts.append(load_month_latitude_mean(path))
        if index % 20 == 0 or index == len(files):
            print(f"Loaded {index:03d}/{len(files)} monthly files")

    combined = xr.concat(monthly_parts, dim="time").sortby("time")
    if pd.DatetimeIndex(combined["time"].values).duplicated().any():
        raise RuntimeError("Duplicate dates found in global V250 archive")

    source_times = pd.DatetimeIndex(combined["time"].values).normalize()
    lon = combined["lon"].values.astype("float64")
    complete_times = pd.date_range(source_times.min(), source_times.max(), freq="D")
    complete = combined.reindex(time=complete_times)
    raw_values = complete.values.astype("float64")
    rolling = (
        complete.rolling(time=TIME_FILTER_DAYS, center=True, min_periods=TIME_FILTER_DAYS)
        .mean()
        .values.astype("float64")
    )

    instantaneous_envelope, instantaneous_filtered, _instantaneous_phase = analytic_wave(raw_values)
    r_metric, recurrent_filtered, recurrent_phase = analytic_wave(rolling)

    source_indices = complete_times.get_indexer(source_times)
    if np.any(source_indices < 0):
        raise RuntimeError("Could not map source dates onto complete daily calendar")
    raw_source = raw_values[source_indices].astype("float32")
    instant_env_source = instantaneous_envelope[source_indices]
    instant_filtered_source = instantaneous_filtered[source_indices]
    r_source = r_metric[source_indices]
    recurrent_filtered_source = recurrent_filtered[source_indices]
    recurrent_phase_source = recurrent_phase[source_indices]

    r_clim = calendar_climatology(r_source, source_times)
    instant_clim = calendar_climatology(instant_env_source, source_times)
    r_mean_time = map_climatology_to_time(r_clim, source_times, "mean")
    r_std_time = map_climatology_to_time(r_clim, source_times, "std")
    r_p90_time = map_climatology_to_time(r_clim, source_times, "p90")
    instant_p75_time = map_climatology_to_time(instant_clim, source_times, "p75")
    r_z = np.divide(
        r_source - r_mean_time,
        r_std_time,
        out=np.full(r_source.shape, np.nan, dtype="float32"),
        where=np.isfinite(r_std_time) & (r_std_time > 0),
    ).astype("float32")
    high_grid = (r_source >= r_p90_time) & np.isfinite(r_source) & np.isfinite(r_p90_time)

    daily_rows: list[dict[str, object]] = []
    episode_tables: list[pd.DataFrame] = []
    sector_arrays: dict[str, dict[str, np.ndarray]] = {}

    for sector_name, config in SECTORS.items():
        mask = sector_mask(lon, config)
        if int(mask.sum()) < 2:
            raise RuntimeError(f"Sector {sector_name} contains fewer than two longitudes")
        r_mean = safe_nanmean_axis1(r_source[:, mask])
        r_max = safe_nanmax_axis1(r_source[:, mask])
        instant_mean = safe_nanmean_axis1(instant_env_source[:, mask])
        instant_max = safe_nanmax_axis1(instant_env_source[:, mask])
        high_fraction = np.mean(high_grid[:, mask], axis=1).astype("float32")

        sector_r_clim = calendar_climatology(r_mean[:, None], source_times)
        sector_inst_clim = calendar_climatology(instant_mean[:, None], source_times)
        sector_r_mean_clim = map_climatology_to_time(sector_r_clim, source_times, "mean")[:, 0]
        sector_r_std_clim = map_climatology_to_time(sector_r_clim, source_times, "std")[:, 0]
        sector_r_p90 = map_climatology_to_time(sector_r_clim, source_times, "p90")[:, 0]
        sector_inst_p75 = map_climatology_to_time(sector_inst_clim, source_times, "p75")[:, 0]
        sector_z = np.divide(
            r_mean - sector_r_mean_clim,
            sector_r_std_clim,
            out=np.full(r_mean.shape, np.nan, dtype="float64"),
            where=np.isfinite(sector_r_std_clim) & (sector_r_std_clim > 0),
        )
        sector_high = (r_mean >= sector_r_p90) & np.isfinite(r_mean) & np.isfinite(sector_r_p90)
        episodes, episode_days = identify_episodes(
            source_times,
            sector_high,
            r_mean,
            sector_r_p90,
            sector_name,
        )
        if not episodes.empty:
            episode_tables.append(episodes)

        sector_arrays[sector_name] = {
            "r_mean_ms": r_mean.astype("float32"),
            "r_max_ms": r_max.astype("float32"),
            "r_standardized": sector_z.astype("float32"),
            "r_p90_ms": sector_r_p90.astype("float32"),
            "r_high_day": sector_high.astype("uint8"),
            "r_high_longitude_fraction": high_fraction.astype("float32"),
            "instantaneous_envelope_mean_ms": instant_mean.astype("float32"),
            "instantaneous_envelope_max_ms": instant_max.astype("float32"),
            "instantaneous_envelope_p75_ms": sector_inst_p75.astype("float32"),
            "rrwp_episode_3day": episode_days.astype("uint8"),
        }

    for time_index, date in enumerate(source_times):
        row: dict[str, object] = {"date": date}
        for sector_name, arrays in sector_arrays.items():
            for metric, values in arrays.items():
                row[f"{sector_name}_{metric}"] = values[time_index]
        daily_rows.append(row)
    daily_table = pd.DataFrame(daily_rows)
    daily_table.to_csv(TABLE43, index=False, date_format="%Y-%m-%d")

    episode_table = (
        pd.concat(episode_tables, ignore_index=True)
        if episode_tables
        else pd.DataFrame(
            columns=[
                "sector",
                "sector_label",
                "episode_id",
                "start_date",
                "end_date",
                "duration_days",
                "peak_date",
                "peak_r_ms",
                "mean_r_ms",
                "integrated_excess_ms_days",
            ]
        )
    )
    episode_table.to_csv(TABLE47, index=False, date_format="%Y-%m-%d")

    events = (
        pd.read_csv(GROUP_TABLE, parse_dates=["start_date", "end_date"])
        .sort_values("start_date")
        .reset_index(drop=True)
    )
    if len(events) != 85:
        raise RuntimeError(f"Expected 85 events, found {len(events)}")
    time_lookup = {date.normalize(): index for index, date in enumerate(source_times)}
    long_rows: list[dict[str, object]] = []
    wide_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for event_index, event in events.iterrows():
        onset = pd.Timestamp(event["start_date"])
        event_windows = dict(WINDOWS)
        event_windows["event"] = None
        wide: dict[str, object] = {
            "event_id": event["event_id"],
            "winter_label": event["winter_label"],
            "start_date": event["start_date"],
            "end_date": event["end_date"],
            "duration_days": int(event["duration_days"]),
            "persistence_group": event["persistence_group"],
        }

        for sector_name, arrays in sector_arrays.items():
            for window_name, lag_pair in event_windows.items():
                if window_name == "event":
                    dates = pd.date_range(event["start_date"], event["end_date"], freq="D")
                else:
                    assert lag_pair is not None
                    dates = pd.date_range(
                        onset + pd.Timedelta(days=lag_pair[0]),
                        onset + pd.Timedelta(days=lag_pair[1]),
                        freq="D",
                    )
                indices = np.asarray([time_lookup.get(date.normalize(), -1) for date in dates], dtype=int)
                if np.any(indices < 0):
                    missing_dates = [str(dates[i].date()) for i in np.where(indices < 0)[0]]
                    raise RuntimeError(
                        f"{event['event_id']} {sector_name} {window_name}: missing dates {missing_dates}"
                    )

                r_values = arrays["r_mean_ms"][indices].astype("float64")
                r_z_values = arrays["r_standardized"][indices].astype("float64")
                high_values = arrays["r_high_day"][indices].astype(bool)
                instant_values = arrays["instantaneous_envelope_mean_ms"][indices].astype("float64")
                instant_threshold = arrays["instantaneous_envelope_p75_ms"][indices].astype("float64")
                episode_values = arrays["rrwp_episode_3day"][indices].astype(bool)
                r_threshold = arrays["r_p90_ms"][indices].astype("float64")
                peaks, _ = find_peaks(
                    np.where(np.isfinite(instant_values), instant_values, -np.inf),
                    height=instant_threshold,
                    distance=PEAK_MIN_DISTANCE_DAYS,
                )
                metrics = {
                    "r_mean_ms": float(np.nanmean(r_values)),
                    "r_max_ms": float(np.nanmax(r_values)),
                    "r_standardized_mean": float(np.nanmean(r_z_values)),
                    "r_high_day_fraction": float(np.mean(high_values)),
                    "longest_high_r_run_days": longest_true_run(high_values, dates),
                    "instantaneous_envelope_mean_ms": float(np.nanmean(instant_values)),
                    "instantaneous_envelope_max_ms": float(np.nanmax(instant_values)),
                    "distinct_envelope_peak_count": int(len(peaks)),
                    "rrwp_episode_overlap_fraction": float(np.mean(episode_values)),
                    "integrated_r_excess_ms_days": float(
                        np.nansum(np.maximum(r_values - r_threshold, 0.0))
                    ),
                    "distinct_recurrence_flag": int(
                        (len(peaks) >= 2) and np.any(high_values)
                    ),
                }
                for metric, value in metrics.items():
                    long_rows.append(
                        {
                            "event_id": event["event_id"],
                            "winter_label": event["winter_label"],
                            "start_date": event["start_date"],
                            "end_date": event["end_date"],
                            "duration_days": int(event["duration_days"]),
                            "persistence_group": event["persistence_group"],
                            "sector": sector_name,
                            "sector_label": SECTORS[sector_name]["label"],
                            "window": window_name,
                            "window_day_count": len(dates),
                            "metric": metric,
                            "value": value,
                        }
                    )
                    wide[f"{sector_name}_{window_name}_{metric}"] = value

            for lag in SELECTED_LAGS:
                date = onset + pd.Timedelta(days=lag)
                index = time_lookup.get(date.normalize())
                if index is None:
                    raise RuntimeError(f"{event['event_id']} lag {lag}: date {date.date()} unavailable")
                for metric in [
                    "r_mean_ms",
                    "r_standardized",
                    "r_high_day",
                    "instantaneous_envelope_mean_ms",
                    "rrwp_episode_3day",
                ]:
                    selected_rows.append(
                        {
                            "event_id": event["event_id"],
                            "winter_label": event["winter_label"],
                            "duration_days": int(event["duration_days"]),
                            "persistence_group": event["persistence_group"],
                            "lag_day_from_onset": lag,
                            "date": date,
                            "sector": sector_name,
                            "sector_label": SECTORS[sector_name]["label"],
                            "metric": metric,
                            "value": arrays[metric][index],
                        }
                    )
        wide_rows.append(wide)
        if (event_index + 1) % 10 == 0 or event_index + 1 == len(events):
            print(f"Event metrics: {event_index + 1:02d}/{len(events)}")

    long_table = pd.DataFrame(long_rows)
    wide_table = pd.DataFrame(wide_rows)
    selected_table = pd.DataFrame(selected_rows)
    long_table.to_csv(TABLE44_LONG, index=False, date_format="%Y-%m-%d")
    wide_table.to_csv(TABLE44_WIDE, index=False, date_format="%Y-%m-%d")
    selected_table.to_csv(TABLE48, index=False, date_format="%Y-%m-%d")

    daily_ds = xr.Dataset(
        coords={
            "time": source_times,
            "lon": lon.astype("float32"),
            "calendar_day": np.asarray(r_clim["calendar_days"], dtype=str),
        }
    )
    daily_ds["v250_latmean_ms"] = (("time", "lon"), raw_source)
    daily_ds["v250_instantaneous_filtered_ms"] = (("time", "lon"), instant_filtered_source)
    daily_ds["instantaneous_envelope_ms"] = (("time", "lon"), instant_env_source)
    daily_ds["r_metric_ms"] = (("time", "lon"), r_source)
    daily_ds["r_standardized"] = (("time", "lon"), r_z)
    daily_ds["r_high_p90"] = (("time", "lon"), high_grid.astype("uint8"))
    daily_ds["recurrent_filtered_v250_ms"] = (("time", "lon"), recurrent_filtered_source)
    daily_ds["recurrent_phase_radians"] = (("time", "lon"), recurrent_phase_source)
    daily_ds["r_climatology_mean_ms"] = (
        ("calendar_day", "lon"),
        np.asarray(r_clim["mean"], dtype="float32"),
    )
    daily_ds["r_climatology_std_ms"] = (
        ("calendar_day", "lon"),
        np.asarray(r_clim["std"], dtype="float32"),
    )
    daily_ds["r_climatology_p90_ms"] = (
        ("calendar_day", "lon"),
        np.asarray(r_clim["p90"], dtype="float32"),
    )
    daily_ds["baseline_sample_count"] = (
        ("calendar_day",),
        np.asarray(r_clim["count"], dtype="int16"),
    )
    daily_ds.attrs.update(
        {
            "title": "Daily recurrent Rossby-wave R metric from ERA5 V250",
            "source_field": "00 UTC V250 cosine-weighted over 35-65N",
            "time_filter": "15-day centred running mean; daily approximation to 14.25-day filter",
            "wavenumber_filter": "positive zonal wavenumbers 4-15",
            "baseline": "1991-2020 calendar-day climatology",
            "caution": "High R usually indicates recurrence but can occasionally arise from a stationary amplified system; distinct peak metrics are supplementary.",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    temporary = DAILY_NC.with_suffix(".nc.part")
    if temporary.exists():
        temporary.unlink()
    daily_ds.to_netcdf(
        temporary,
        engine="netcdf4",
        encoding={name: {"zlib": True, "complevel": 4} for name in daily_ds.data_vars},
    )
    temporary.replace(DAILY_NC)
    daily_ds.close()

    composite_ds = build_event_composite(
        events,
        source_times,
        lon,
        {
            "r_metric_ms": r_source,
            "r_standardized": r_z,
            "recurrent_filtered_v250_ms": recurrent_filtered_source,
            "instantaneous_envelope_ms": instant_env_source,
        },
    )
    temporary = COMPOSITE_NC.with_suffix(".nc.part")
    if temporary.exists():
        temporary.unlink()
    composite_ds.to_netcdf(
        temporary,
        engine="netcdf4",
        encoding={name: {"zlib": True, "complevel": 4} for name in composite_ds.data_vars},
    )
    temporary.replace(COMPOSITE_NC)
    composite_ds.close()

    summary_data = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "monthly_file_count": len(files),
        "daily_record_count": len(source_times),
        "first_date": str(source_times.min().date()),
        "last_date": str(source_times.max().date()),
        "longitude_count": len(lon),
        "event_count": len(events),
        "short_event_count": int(events["persistence_group"].eq("short").sum()),
        "persistent_event_count": int(events["persistence_group"].eq("persistent").sum()),
        "episode_count": int(len(episode_table)),
        "event_metric_rows_long": int(len(long_table)),
        "event_predictor_rows_wide": int(len(wide_table)),
        "selected_lag_rows": int(len(selected_table)),
        "daily_netcdf": str(DAILY_NC),
        "event_composite_netcdf": str(COMPOSITE_NC),
    }
    SUMMARY_JSON.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    lines = [
        "STEP 10E: RECURRENT ROSSBY-WAVE R METRIC",
        "=" * 68,
        f"Monthly files: {len(files)}",
        f"Daily records: {len(source_times)}",
        f"Longitude points: {len(lon)}",
        f"Events analysed: {len(events)}",
        f"Detected >=3-day sector episodes: {len(episode_table)}",
        f"Event metric rows: {len(long_table)}",
        f"Selected-lag rows: {len(selected_table)}",
        "",
        f"Daily NetCDF: {DAILY_NC}",
        f"Event composite NetCDF: {COMPOSITE_NC}",
        f"Daily sectors: {TABLE43}",
        f"Event predictors: {TABLE44_WIDE}",
        f"Episode catalogue: {TABLE47}",
    ]
    REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("STEP 10E-METRIC PASSED.")


if __name__ == "__main__":
    main()
