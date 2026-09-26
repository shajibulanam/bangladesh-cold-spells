from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PL_DIR = PROJECT_ROOT / "03_intermediate" / "era5_preprocessed" / "pressure_level_standardized"
CLIM_DIR = PROJECT_ROOT / "03_intermediate" / "era5_climatology" / "pressure_level"
GROUP_TABLE = PROJECT_ROOT / "08_outputs" / "tables" / "era5_persistence" / "table_18_persistence_groups.csv"
STEP10B_HELPERS = PROJECT_ROOT / "07_scripts" / "09e_compute_850hpa_advection_event_diagnostics.py"
OUT_ROOT = PROJECT_ROOT / "03_intermediate" / "tn_waf"
NETCDF_DIR = OUT_ROOT / "netcdf"
TABLE_DIR = PROJECT_ROOT / "08_outputs" / "tables" / "tn_waf"
REPORT_DIR = PROJECT_ROOT / "05_qc_reports" / "tn_waf"
for folder in [OUT_ROOT, NETCDF_DIR, TABLE_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

TABLE37_LONG = TABLE_DIR / "table_37_event_tn_waf_metrics_long.csv"
TABLE37_WIDE = TABLE_DIR / "table_37b_event_tn_waf_predictors.csv"
TABLE40 = TABLE_DIR / "table_40_selected_lag_tn_waf_metrics.csv"
TABLE41 = TABLE_DIR / "table_41_tn_waf_significance_summary.csv"
TABLE42 = TABLE_DIR / "table_42_tn_waf_longitude_profiles.csv"
NETCDF_FILE = NETCDF_DIR / "ERA5_TN_WAF_250hPa_selected_lags.nc"
SUMMARY_JSON = REPORT_DIR / "step10d_tn_waf_computation_summary.json"
REPORT_TXT = REPORT_DIR / "step10d_tn_waf_computation_report.txt"

LEVELS = [200, 250, 300]
PRIMARY_LEVEL = 250
SELECTED_LAGS = [-10, -7, -5, -3, 0, 3, 5]
WINDOWS = {"pre10": (-10, -1), "pre5": (-5, -1), "first3": (0, 2)}
REGIONS = {
    "eurasian_upstream": {"label": "Eurasian upstream", "lat_min": 35.0, "lat_max": 60.0, "lon_min": 30.0, "lon_max": 60.0},
    "central_asia": {"label": "Central Asia", "lat_min": 30.0, "lat_max": 55.0, "lon_min": 55.0, "lon_max": 80.0},
    "south_asian_pathway": {"label": "South Asian pathway", "lat_min": 20.0, "lat_max": 45.0, "lon_min": 70.0, "lon_max": 100.0},
    "bangladesh_approach": {"label": "Bangladesh approach", "lat_min": 20.0, "lat_max": 35.0, "lon_min": 80.0, "lon_max": 100.0},
    "asian_corridor": {"label": "Asian corridor", "lat_min": 20.0, "lat_max": 60.0, "lon_min": 30.0, "lon_max": 110.0},
}
MAP_PROFILE_LAT_MIN = 25.0
MAP_PROFILE_LAT_MAX = 55.0
EARTH_RADIUS_M = 6_371_000.0
OMEGA = 7.2921159e-5
GRAVITY = 9.80665
COARSEN_FACTOR = 4
SMOOTHING_SIGMA_GRIDCELLS = 1.0
MIN_BASIC_FLOW_SPEED_MS = 5.0
MIN_ANALYSIS_LAT = 20.0
MAX_ANALYSIS_LAT = 70.0
N_BOOTSTRAP_MAP = 1000
BOOTSTRAP_BATCH_SIZE = 25
RANDOM_SEED = 20260801
ALPHA = 0.05
EVENT_VECTOR_AGREEMENT_THRESHOLD = 0.60
DIFFERENCE_LOO_THRESHOLD = 0.90

SCALAR_METRICS = [
    "waf_x_m2_s2",
    "waf_y_m2_s2",
    "waf_magnitude_m2_s2",
    "waf_southeast_projection_m2_s2",
    "waf_divergence_1e6_ms2",
    "z_anomaly_m",
    "eastward_fraction",
    "southeastward_fraction",
    "convergence_fraction",
    "directional_coherence",
]


def load_helpers():
    if not STEP10B_HELPERS.exists():
        raise FileNotFoundError(STEP10B_HELPERS)
    spec = importlib.util.spec_from_file_location("step10b_helpers", STEP10B_HELPERS)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Step 10B helper module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPERS = load_helpers()


class MonthlyFieldCache:
    def __init__(self, maximum_months: int = 2) -> None:
        self.maximum_months = maximum_months
        self.cache: OrderedDict[tuple[int, int], xr.Dataset] = OrderedDict()

    def get(self, date: pd.Timestamp) -> xr.Dataset:
        key = (int(date.year), int(date.month))
        if key in self.cache:
            dataset = self.cache.pop(key)
            self.cache[key] = dataset
            return dataset
        path = PL_DIR / f"ERA5_PL_STD_{key[0]}_{key[1]:02d}.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        with xr.open_dataset(path) as source:
            dataset = (
                source[["geopotential_height_m", "u_wind_ms", "v_wind_ms"]]
                .sel(level_hpa=LEVELS)
                .sortby("lat", ascending=True)
                .sortby("lon", ascending=True)
                .load()
            )
        self.cache[key] = dataset
        while len(self.cache) > self.maximum_months:
            _, old = self.cache.popitem(last=False)
            old.close()
        return dataset

    def close(self) -> None:
        for dataset in self.cache.values():
            dataset.close()
        self.cache.clear()


class ClimatologyStore:
    def __init__(self) -> None:
        self.datasets: list[xr.Dataset] = []
        self.z = self._open("geopotential_height_m")
        self.u = self._open("u_wind_ms")
        self.v = self._open("v_wind_ms")

    def _open(self, variable: str) -> xr.DataArray:
        matches = sorted(CLIM_DIR.glob(f"*{variable}*calendar_day_1991_2020.nc"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one climatology for {variable}; found {len(matches)}")
        dataset = xr.open_dataset(matches[0])
        self.datasets.append(dataset)
        return (
            dataset[variable]
            .sel(level_hpa=LEVELS)
            .sortby("lat", ascending=True)
            .sortby("lon", ascending=True)
        )

    def get(self, calendar_day: str) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        z = self.z.sel(calendar_day=calendar_day).load()
        u = self.u.sel(calendar_day=calendar_day).load()
        v = self.v.sel(calendar_day=calendar_day).load()
        return xr.align(z, u, v, join="exact")

    def close(self) -> None:
        for dataset in self.datasets:
            dataset.close()
        self.datasets.clear()


def coarsen(values: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(
            "Step 10D coarsening requires a 2-D latitude x longitude field; "
            f"received shape {values.shape}. Check that the requested date and "
            "pressure level were selected before coarsening."
        )
    return HELPERS.coarsen_field(values, lat, lon)


def smooth(values: np.ndarray) -> np.ndarray:
    if not np.isfinite(values).all():
        valid = np.isfinite(values)
        filled = np.where(valid, values, 0.0)
        weight = gaussian_filter(valid.astype("float64"), sigma=SMOOTHING_SIGMA_GRIDCELLS, mode="nearest")
        filtered = gaussian_filter(filled.astype("float64"), sigma=SMOOTHING_SIGMA_GRIDCELLS, mode="nearest")
        return np.divide(filtered, weight, out=np.full_like(filtered, np.nan), where=weight > 0.25)
    return gaussian_filter(values.astype("float64"), sigma=SMOOTHING_SIGMA_GRIDCELLS, mode="nearest")


def compute_tn_waf(
    z_anomaly_m: np.ndarray,
    basic_u_ms: np.ndarray,
    basic_v_ms: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    pressure_hpa: int,
) -> dict[str, np.ndarray]:
    lat_rad = np.deg2rad(lat.astype("float64"))
    lon_rad = np.deg2rad(lon.astype("float64"))
    cos_lat = np.cos(lat_rad)[:, None]
    f = (2.0 * OMEGA * np.sin(lat_rad))[:, None]
    z_prime = smooth(z_anomaly_m)
    u_bar = smooth(basic_u_ms)
    v_bar = smooth(basic_v_ms)
    speed = np.sqrt(u_bar**2 + v_bar**2)
    psi_valid = (
        (lat[:, None] >= MIN_ANALYSIS_LAT)
        & (lat[:, None] <= MAX_ANALYSIS_LAT)
        & (np.abs(f) > 1e-5)
        & np.isfinite(z_prime)
    )
    valid = (
        psi_valid
        & (speed >= MIN_BASIC_FLOW_SPEED_MS)
        & np.isfinite(u_bar)
        & np.isfinite(v_bar)
    )
    psi = np.full(z_prime.shape, np.nan, dtype="float64")
    psi[psi_valid] = GRAVITY * z_prime[psi_valid] / np.broadcast_to(f, z_prime.shape)[psi_valid]
    # Derivatives with respect to longitude lambda and latitude phi in radians.
    psi_lam = np.gradient(psi, lon_rad, axis=1, edge_order=2)
    psi_phi = np.gradient(psi, lat_rad, axis=0, edge_order=2)
    psi_lam_lam = np.gradient(psi_lam, lon_rad, axis=1, edge_order=2)
    psi_phi_phi = np.gradient(psi_phi, lat_rad, axis=0, edge_order=2)
    psi_lam_phi = np.gradient(psi_lam, lat_rad, axis=0, edge_order=2)
    term_lam = psi_lam**2 - psi * psi_lam_lam
    term_phi = psi_phi**2 - psi * psi_phi_phi
    term_cross = psi_lam * psi_phi - psi * psi_lam_phi
    pressure_ratio = float(pressure_hpa) / 1000.0
    coefficient = pressure_ratio * cos_lat / (2.0 * speed)
    fx = coefficient * (
        u_bar / (EARTH_RADIUS_M**2 * cos_lat**2) * term_lam
        + v_bar / (EARTH_RADIUS_M**2 * cos_lat) * term_cross
    )
    fy = coefficient * (
        u_bar / (EARTH_RADIUS_M**2 * cos_lat) * term_cross
        + v_bar / (EARTH_RADIUS_M**2) * term_phi
    )
    fx[~valid] = np.nan
    fy[~valid] = np.nan
    magnitude = np.sqrt(fx**2 + fy**2)
    # Horizontal divergence on the sphere for eastward/northward vector components.
    d_fx_dlambda = np.gradient(fx, lon_rad, axis=1, edge_order=2)
    d_fycos_dphi = np.gradient(fy * cos_lat, lat_rad, axis=0, edge_order=2)
    divergence = (d_fx_dlambda + d_fycos_dphi) / (EARTH_RADIUS_M * cos_lat)
    divergence[~valid] = np.nan
    southeast_projection = (fx - fy) / np.sqrt(2.0)
    return {
        "z_anomaly_m": z_prime.astype("float32"),
        "basic_u_ms": u_bar.astype("float32"),
        "basic_v_ms": v_bar.astype("float32"),
        "waf_x_m2_s2": fx.astype("float32"),
        "waf_y_m2_s2": fy.astype("float32"),
        "waf_magnitude_m2_s2": magnitude.astype("float32"),
        "waf_southeast_projection_m2_s2": southeast_projection.astype("float32"),
        "waf_divergence_1e6_ms2": (divergence * 1e6).astype("float32"),
        "valid_mask": valid.astype("uint8"),
    }


def compute_date_levels(
    date: pd.Timestamp,
    monthly_cache: MonthlyFieldCache,
    climatology: ClimatologyStore,
) -> tuple[dict[int, dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    monthly = monthly_cache.get(date)

    # MonthlyFieldCache returns the complete monthly dataset. Select the exact
    # event-relative day before extracting a pressure level; otherwise the
    # time dimension is broadcast against the climatology and a 3-D
    # (time, lat, lon) array reaches the 2-D coarsening routine.
    target_time = pd.Timestamp(date).normalize()

    try:
        daily = monthly.sel(time=target_time).squeeze(drop=True)
    except KeyError as error:
        available_start = pd.Timestamp(monthly["time"].values[0])
        available_end = pd.Timestamp(monthly["time"].values[-1])
        raise KeyError(
            f"Requested ERA5 date {target_time.date()} is not present in "
            f"{available_start.date()} to {available_end.date()}."
        ) from error

    z_bar, u_bar, v_bar = climatology.get(target_time.strftime("%m-%d"))
    daily_z, z_bar, u_bar, v_bar = xr.align(
        daily["geopotential_height_m"],
        z_bar,
        u_bar,
        v_bar,
        join="exact",
    )
    lat_full = daily_z["lat"].values.astype("float64")
    lon_full = daily_z["lon"].values.astype("float64")
    results: dict[int, dict[str, np.ndarray]] = {}
    coarse_lat = None
    coarse_lon = None
    for level in LEVELS:
        z_daily = daily_z.sel(level_hpa=level).values.astype("float64")
        z_clim = z_bar.sel(level_hpa=level).values.astype("float64")
        u_clim = u_bar.sel(level_hpa=level).values.astype("float64")
        v_clim = v_bar.sel(level_hpa=level).values.astype("float64")
        z_anom, lat, lon = coarsen(z_daily - z_clim, lat_full, lon_full)
        u_coarse, lat_u, lon_u = coarsen(u_clim, lat_full, lon_full)
        v_coarse, lat_v, lon_v = coarsen(v_clim, lat_full, lon_full)
        if not (np.array_equal(lat, lat_u) and np.array_equal(lat, lat_v) and np.array_equal(lon, lon_u) and np.array_equal(lon, lon_v)):
            raise RuntimeError("Coarsened coordinate mismatch.")
        if coarse_lat is None:
            coarse_lat, coarse_lon = lat, lon
        results[level] = compute_tn_waf(z_anom, u_coarse, v_coarse, lat, lon, level)
    return results, coarse_lat, coarse_lon


def build_region_masks(lat: np.ndarray, lon: np.ndarray) -> dict[str, np.ndarray]:
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    return {
        name: (
            (lat_grid >= cfg["lat_min"])
            & (lat_grid <= cfg["lat_max"])
            & (lon_grid >= cfg["lon_min"])
            & (lon_grid <= cfg["lon_max"])
        )
        for name, cfg in REGIONS.items()
    }


def area_mean(values: np.ndarray, lat: np.ndarray, mask: np.ndarray) -> float:
    weights = np.cos(np.deg2rad(lat))[:, None]
    valid = mask & np.isfinite(values)
    denominator = np.sum(np.where(valid, weights, 0.0))
    if denominator <= 0:
        return float("nan")
    numerator = np.nansum(np.where(valid, values * weights, np.nan))
    return float(numerator / denominator)


def region_metrics(fields: dict[str, np.ndarray], lat: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    fx = fields["waf_x_m2_s2"]
    fy = fields["waf_y_m2_s2"]
    magnitude = fields["waf_magnitude_m2_s2"]
    southeast = fields["waf_southeast_projection_m2_s2"]
    divergence = fields["waf_divergence_1e6_ms2"]
    for name, mask in masks.items():
        valid = mask & np.isfinite(fx) & np.isfinite(fy)
        mean_fx = area_mean(fx, lat, mask)
        mean_fy = area_mean(fy, lat, mask)
        mean_magnitude = area_mean(magnitude, lat, mask)
        coherence = float(np.sqrt(mean_fx**2 + mean_fy**2) / mean_magnitude) if np.isfinite(mean_magnitude) and mean_magnitude > 0 else float("nan")
        if valid.any():
            eastward_fraction = float(np.mean(fx[valid] > 0))
            southeastward_fraction = float(np.mean(southeast[valid] > 0))
            convergence_fraction = float(np.mean(divergence[valid] < 0))
        else:
            eastward_fraction = southeastward_fraction = convergence_fraction = float("nan")
        output[name] = {
            "waf_x_m2_s2": mean_fx,
            "waf_y_m2_s2": mean_fy,
            "waf_magnitude_m2_s2": mean_magnitude,
            "waf_southeast_projection_m2_s2": area_mean(southeast, lat, mask),
            "waf_divergence_1e6_ms2": area_mean(divergence, lat, mask),
            "z_anomaly_m": area_mean(fields["z_anomaly_m"], lat, mask),
            "eastward_fraction": eastward_fraction,
            "southeastward_fraction": southeastward_fraction,
            "convergence_fraction": convergence_fraction,
            "directional_coherence": coherence,
        }
    return output


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype="float64")
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"mean": np.nan, "minimum": np.nan, "maximum": np.nan}
    return {"mean": float(np.mean(finite)), "minimum": float(np.min(finite)), "maximum": float(np.max(finite))}


def vector_direction_agreement(
    fx_events: np.ndarray,
    fy_events: np.ndarray,
    mean_fx: np.ndarray,
    mean_fy: np.ndarray,
) -> np.ndarray:
    composite_mag = np.sqrt(mean_fx**2 + mean_fy**2)
    dot = fx_events * mean_fx[None, :, :] + fy_events * mean_fy[None, :, :]
    valid = np.isfinite(fx_events) & np.isfinite(fy_events) & np.isfinite(dot)
    count = valid.sum(axis=0)
    same = ((dot > 0) & valid).sum(axis=0)
    agreement = np.divide(same, count, out=np.full(count.shape, np.nan, dtype="float64"), where=count > 0)
    agreement[~np.isfinite(composite_mag) | (composite_mag == 0)] = np.nan
    return agreement.astype("float32")


def leave_one_winter_out_vector_direction(
    fx_events: np.ndarray,
    fy_events: np.ndarray,
    winters: np.ndarray,
    full_fx: np.ndarray,
    full_fy: np.ndarray,
    group_mask: np.ndarray | None = None,
) -> np.ndarray:
    unique_winters = np.unique(winters[group_mask] if group_mask is not None else winters)
    same = np.zeros(full_fx.shape, dtype="int16")
    valid_count = np.zeros(full_fx.shape, dtype="int16")
    for winter in unique_winters:
        keep = winters != winter
        if group_mask is not None:
            keep &= group_mask
        if keep.sum() == 0:
            continue
        fx = np.nanmean(fx_events[keep], axis=0)
        fy = np.nanmean(fy_events[keep], axis=0)
        dot = fx * full_fx + fy * full_fy
        valid = np.isfinite(dot) & np.isfinite(full_fx) & np.isfinite(full_fy)
        same[valid] += (dot[valid] > 0).astype("int16")
        valid_count[valid] += 1
    return np.divide(same, valid_count, out=np.full(full_fx.shape, np.nan, dtype="float64"), where=valid_count > 0).astype("float32")


def component_bootstrap(data: np.ndarray, winters: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    unique_winters = np.unique(winters)
    sums, counts = HELPERS.build_winter_arrays(data, winters, unique_winters)
    result = HELPERS.bootstrap_mean_map(sums, counts, seed)
    result["fdr_q"] = HELPERS.benjamini_hochberg_q(result["bootstrap_two_sided_p"])
    return result


def component_difference_bootstrap(
    data: np.ndarray,
    winters: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    union_winters = np.unique(winters[np.isin(groups, ["short", "persistent"])])
    short_mask = groups == "short"
    persistent_mask = groups == "persistent"
    short_sums, short_counts = HELPERS.build_winter_arrays(data, winters, union_winters, group_mask=short_mask)
    persistent_sums, persistent_counts = HELPERS.build_winter_arrays(data, winters, union_winters, group_mask=persistent_mask)
    result = HELPERS.bootstrap_difference_map(short_sums, short_counts, persistent_sums, persistent_counts, seed)
    result["fdr_q"] = HELPERS.benjamini_hochberg_q(result["bootstrap_two_sided_p"])
    return result


def scalar_map_analysis(data: np.ndarray, winters: np.ndarray, groups: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    all_mean = np.nanmean(data, axis=0).astype("float32")
    bootstrap = component_bootstrap(data, winters, seed)
    agreement = HELPERS.event_sign_agreement(data, all_mean)
    all_robust = ((bootstrap["fdr_q"] < ALPHA) & (agreement >= EVENT_VECTOR_AGREEMENT_THRESHOLD)).astype("uint8")
    short_mean = np.nanmean(data[groups == "short"], axis=0).astype("float32")
    persistent_mean = np.nanmean(data[groups == "persistent"], axis=0).astype("float32")
    difference = (persistent_mean - short_mean).astype("float32")
    diff_boot = component_difference_bootstrap(data, winters, groups, seed + 50)
    union_winters = np.unique(winters[np.isin(groups, ["short", "persistent"])])
    short_sums, short_counts = HELPERS.build_winter_arrays(data, winters, union_winters, group_mask=groups == "short")
    persistent_sums, persistent_counts = HELPERS.build_winter_arrays(data, winters, union_winters, group_mask=groups == "persistent")
    loo = HELPERS.leave_one_winter_out_difference_sign(short_sums, short_counts, persistent_sums, persistent_counts, difference)
    diff_robust = ((diff_boot["fdr_q"] < ALPHA) & (loo >= DIFFERENCE_LOO_THRESHOLD)).astype("uint8")
    return {
        "all_event_mean": all_mean,
        "all_event_fdr_q": bootstrap["fdr_q"],
        "all_event_sign_agreement": agreement,
        "all_event_robust": all_robust,
        "short_mean": short_mean,
        "persistent_mean": persistent_mean,
        "persistent_minus_short": difference,
        "difference_fdr_q": diff_boot["fdr_q"],
        "difference_loo_sign_fraction": loo,
        "difference_robust": diff_robust,
    }


def analyze_vector_maps(
    fx: np.ndarray,
    fy: np.ndarray,
    divergence: np.ndarray,
    z_anomaly: np.ndarray,
    basic_u: np.ndarray,
    basic_v: np.ndarray,
    winters: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    all_fx = np.nanmean(fx, axis=0).astype("float32")
    all_fy = np.nanmean(fy, axis=0).astype("float32")
    fx_boot = component_bootstrap(fx, winters, seed)
    fy_boot = component_bootstrap(fy, winters, seed + 1)
    agreement = vector_direction_agreement(fx, fy, all_fx, all_fy)
    robust_vector = (((fx_boot["fdr_q"] < ALPHA) | (fy_boot["fdr_q"] < ALPHA)) & (agreement >= EVENT_VECTOR_AGREEMENT_THRESHOLD)).astype("uint8")
    short_fx = np.nanmean(fx[groups == "short"], axis=0).astype("float32")
    short_fy = np.nanmean(fy[groups == "short"], axis=0).astype("float32")
    persistent_fx = np.nanmean(fx[groups == "persistent"], axis=0).astype("float32")
    persistent_fy = np.nanmean(fy[groups == "persistent"], axis=0).astype("float32")
    diff_fx = (persistent_fx - short_fx).astype("float32")
    diff_fy = (persistent_fy - short_fy).astype("float32")
    diff_fx_boot = component_difference_bootstrap(fx, winters, groups, seed + 20)
    diff_fy_boot = component_difference_bootstrap(fy, winters, groups, seed + 21)
    union_mask = np.isin(groups, ["short", "persistent"])
    # LOO direction of persistent-minus-short vector.
    unique_winters = np.unique(winters[union_mask])
    same = np.zeros(diff_fx.shape, dtype="int16")
    valid_count = np.zeros(diff_fx.shape, dtype="int16")
    for winter in unique_winters:
        short_keep = (groups == "short") & (winters != winter)
        persistent_keep = (groups == "persistent") & (winters != winter)
        if short_keep.sum() == 0 or persistent_keep.sum() == 0:
            continue
        loo_fx = np.nanmean(fx[persistent_keep], axis=0) - np.nanmean(fx[short_keep], axis=0)
        loo_fy = np.nanmean(fy[persistent_keep], axis=0) - np.nanmean(fy[short_keep], axis=0)
        dot = loo_fx * diff_fx + loo_fy * diff_fy
        valid = np.isfinite(dot) & np.isfinite(diff_fx) & np.isfinite(diff_fy)
        same[valid] += (dot[valid] > 0).astype("int16")
        valid_count[valid] += 1
    diff_loo = np.divide(same, valid_count, out=np.full(diff_fx.shape, np.nan, dtype="float64"), where=valid_count > 0).astype("float32")
    diff_robust_vector = (((diff_fx_boot["fdr_q"] < ALPHA) | (diff_fy_boot["fdr_q"] < ALPHA)) & (diff_loo >= DIFFERENCE_LOO_THRESHOLD)).astype("uint8")
    div_result = scalar_map_analysis(divergence, winters, groups, seed + 40)
    all_z = np.nanmean(z_anomaly, axis=0).astype("float32")
    all_u = np.nanmean(basic_u, axis=0).astype("float32")
    all_v = np.nanmean(basic_v, axis=0).astype("float32")
    short_z = np.nanmean(z_anomaly[groups == "short"], axis=0).astype("float32")
    persistent_z = np.nanmean(z_anomaly[groups == "persistent"], axis=0).astype("float32")
    short_u = np.nanmean(basic_u[groups == "short"], axis=0).astype("float32")
    short_v = np.nanmean(basic_v[groups == "short"], axis=0).astype("float32")
    persistent_u = np.nanmean(basic_u[groups == "persistent"], axis=0).astype("float32")
    persistent_v = np.nanmean(basic_v[groups == "persistent"], axis=0).astype("float32")
    # Coherent composite-field fluxes, calculated after averaging the anomaly/basic state.
    # These are visualization diagnostics and are kept separate from event-mean inference.
    return {
        "z_anomaly_all_event_composite": all_z,
        "basic_u_all_event_composite": all_u,
        "basic_v_all_event_composite": all_v,
        "waf_x_all_event_event_mean": all_fx,
        "waf_y_all_event_event_mean": all_fy,
        "waf_x_all_event_fdr_q": fx_boot["fdr_q"],
        "waf_y_all_event_fdr_q": fy_boot["fdr_q"],
        "waf_all_event_vector_agreement": agreement,
        "waf_all_event_robust_vector": robust_vector,
        "waf_x_short_event_mean": short_fx,
        "waf_y_short_event_mean": short_fy,
        "waf_x_persistent_event_mean": persistent_fx,
        "waf_y_persistent_event_mean": persistent_fy,
        "waf_x_persistent_minus_short": diff_fx,
        "waf_y_persistent_minus_short": diff_fy,
        "waf_x_difference_fdr_q": diff_fx_boot["fdr_q"],
        "waf_y_difference_fdr_q": diff_fy_boot["fdr_q"],
        "waf_difference_loo_direction_fraction": diff_loo,
        "waf_difference_robust_vector": diff_robust_vector,
        "z_anomaly_short_composite": short_z,
        "z_anomaly_persistent_composite": persistent_z,
        "z_anomaly_persistent_minus_short": (persistent_z - short_z).astype("float32"),
        "basic_u_short_composite": short_u,
        "basic_v_short_composite": short_v,
        "basic_u_persistent_composite": persistent_u,
        "basic_v_persistent_composite": persistent_v,
        "divergence_all_event_mean_1e6": div_result["all_event_mean"],
        "divergence_all_event_fdr_q": div_result["all_event_fdr_q"],
        "divergence_all_event_sign_agreement": div_result["all_event_sign_agreement"],
        "divergence_all_event_robust": div_result["all_event_robust"],
        "divergence_short_mean_1e6": div_result["short_mean"],
        "divergence_persistent_mean_1e6": div_result["persistent_mean"],
        "divergence_persistent_minus_short_1e6": div_result["persistent_minus_short"],
        "divergence_difference_fdr_q": div_result["difference_fdr_q"],
        "divergence_difference_loo_sign_fraction": div_result["difference_loo_sign_fraction"],
        "divergence_difference_robust": div_result["difference_robust"],
    }


def add_composite_field_flux(result: dict[str, np.ndarray], lat: np.ndarray, lon: np.ndarray) -> None:
    all_fields = compute_tn_waf(
        result["z_anomaly_all_event_composite"],
        result["basic_u_all_event_composite"],
        result["basic_v_all_event_composite"],
        lat,
        lon,
        PRIMARY_LEVEL,
    )
    result["waf_x_all_event_composite_field"] = all_fields["waf_x_m2_s2"]
    result["waf_y_all_event_composite_field"] = all_fields["waf_y_m2_s2"]
    result["divergence_all_event_composite_field_1e6"] = all_fields["waf_divergence_1e6_ms2"]
    short_fields = compute_tn_waf(
        result["z_anomaly_short_composite"],
        result["basic_u_short_composite"],
        result["basic_v_short_composite"],
        lat,
        lon,
        PRIMARY_LEVEL,
    )
    persistent_fields = compute_tn_waf(
        result["z_anomaly_persistent_composite"],
        result["basic_u_persistent_composite"],
        result["basic_v_persistent_composite"],
        lat,
        lon,
        PRIMARY_LEVEL,
    )
    result["waf_x_short_composite_field"] = short_fields["waf_x_m2_s2"]
    result["waf_y_short_composite_field"] = short_fields["waf_y_m2_s2"]
    result["waf_x_persistent_composite_field"] = persistent_fields["waf_x_m2_s2"]
    result["waf_y_persistent_composite_field"] = persistent_fields["waf_y_m2_s2"]
    result["waf_x_composite_field_persistent_minus_short"] = (persistent_fields["waf_x_m2_s2"] - short_fields["waf_x_m2_s2"]).astype("float32")
    result["waf_y_composite_field_persistent_minus_short"] = (persistent_fields["waf_y_m2_s2"] - short_fields["waf_y_m2_s2"]).astype("float32")


def write_netcdf(results: list[dict[str, np.ndarray]], lat: np.ndarray, lon: np.ndarray) -> None:
    dataset = xr.Dataset(
        coords={
            "lag_day_from_onset": np.asarray(SELECTED_LAGS, dtype="int32"),
            "lat": lat.astype("float32"),
            "lon": lon.astype("float32"),
        }
    )
    for variable in results[0]:
        if "robust" in variable:
            dtype = "uint8"
        else:
            dtype = "float32"
        dataset[variable] = (
            ("lag_day_from_onset", "lat", "lon"),
            np.stack([result[variable] for result in results], axis=0).astype(dtype),
        )
    dataset.attrs.update(
        {
            "title": "Takaya-Nakamura horizontal wave-activity flux for Bangladesh cold-spell events",
            "primary_pressure_level_hpa": PRIMARY_LEVEL,
            "selected_lags": str(SELECTED_LAGS),
            "waf_units": "m2 s-2",
            "divergence_units": "1e-6 m s-2",
            "basic_state": "1991-2020 calendar-day climatological wind",
            "perturbation_streamfunction": "gravity times geopotential-height anomaly divided by Coriolis parameter",
            "minimum_basic_flow_speed_ms": MIN_BASIC_FLOW_SPEED_MS,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    temporary = NETCDF_FILE.with_suffix(".nc.part")
    if temporary.exists():
        temporary.unlink()
    dataset.to_netcdf(
        temporary,
        engine="netcdf4",
        encoding={name: {"zlib": True, "complevel": 4} for name in dataset.data_vars},
    )
    temporary.replace(NETCDF_FILE)
    dataset.close()


def build_wide_row(event: pd.Series, lookup: dict[tuple[int, str, str, str], dict[str, float]]) -> dict[str, object]:
    row: dict[str, object] = {
        "event_id": event["event_id"],
        "winter_label": event["winter_label"],
        "start_date": event["start_date"],
        "end_date": event["end_date"],
        "duration_days": int(event["duration_days"]),
        "persistence_group": event["persistence_group"],
    }
    selected_metrics = [
        "waf_x_m2_s2",
        "waf_y_m2_s2",
        "waf_magnitude_m2_s2",
        "waf_southeast_projection_m2_s2",
        "waf_divergence_1e6_ms2",
        "eastward_fraction",
        "southeastward_fraction",
        "convergence_fraction",
        "directional_coherence",
    ]
    for level in LEVELS:
        for region in REGIONS:
            for window in ["pre10", "pre5", "first3", "event"]:
                for metric in selected_metrics:
                    summary = lookup[(level, region, window, metric)]
                    base = f"{region}_{level}hpa_{window}_{metric}"
                    row[f"{base}_mean"] = summary["mean"]
                    row[f"{base}_minimum"] = summary["minimum"]
                    row[f"{base}_maximum"] = summary["maximum"]
    return row


def smoke_test(events: pd.DataFrame, monthly_cache: MonthlyFieldCache, climatology: ClimatologyStore) -> None:
    date = pd.Timestamp(events.iloc[0]["start_date"])
    fields, lat, lon = compute_date_levels(date, monthly_cache, climatology)
    primary = fields[PRIMARY_LEVEL]
    finite = np.isfinite(primary["waf_x_m2_s2"]) & np.isfinite(primary["waf_y_m2_s2"])
    valid_fraction = float(np.mean(finite))
    max_flux = float(np.nanmax(primary["waf_magnitude_m2_s2"]))
    max_divergence = float(np.nanmax(np.abs(primary["waf_divergence_1e6_ms2"])))
    print("STEP 10D SMOKE TEST")
    print("=" * 60)
    print("Event:", events.iloc[0]["event_id"])
    print("Date:", date.date())
    print("Grid:", len(lat), "x", len(lon))
    print("Valid 250-hPa WAF fraction:", f"{valid_fraction:.3f}")
    print("Maximum flux magnitude:", f"{max_flux:.3f} m2 s-2")
    print("Maximum |divergence|:", f"{max_divergence:.3f} x 1e-6 m s-2")
    if valid_fraction < 0.20:
        raise RuntimeError("Too little valid WAF coverage in smoke test.")
    if not np.isfinite(max_flux) or max_flux <= 0:
        raise RuntimeError("Invalid WAF magnitude in smoke test.")
    print("STEP 10D SMOKE TEST PASSED.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    events = (
        pd.read_csv(GROUP_TABLE, parse_dates=["start_date", "end_date"])
        .sort_values("start_date")
        .reset_index(drop=True)
    )
    if len(events) != 85:
        raise RuntimeError(f"Expected 85 events, found {len(events)}")
    if events["persistence_group"].eq("short").sum() != 30:
        raise RuntimeError("Expected 30 short events.")
    if events["persistence_group"].eq("persistent").sum() != 26:
        raise RuntimeError("Expected 26 persistent events.")
    monthly_cache = MonthlyFieldCache(maximum_months=2)
    climatology = ClimatologyStore()
    try:
        if args.smoke_test:
            smoke_test(events, monthly_cache, climatology)
            return
        print("STEP 10D: TAKAYA-NAKAMURA WAVE-ACTIVITY FLUX")
        print("=" * 76)
        print("Events:", len(events))
        print("Primary level:", PRIMARY_LEVEL, "hPa")
        print("Sensitivity levels:", [level for level in LEVELS if level != PRIMARY_LEVEL])
        print("Selected lags:", SELECTED_LAGS)
        winters = events["winter_label"].astype(str).to_numpy()
        groups = events["persistence_group"].astype(str).to_numpy()
        map_results: list[dict[str, np.ndarray]] = []
        selected_rows: list[dict[str, object]] = []
        significance_rows: list[dict[str, object]] = []
        profile_rows: list[dict[str, object]] = []
        scalar_cache: dict[pd.Timestamp, dict[int, dict[str, dict[str, float]]]] = {}
        coarse_lat = None
        coarse_lon = None
        masks = None

        def cache_scalar(date: pd.Timestamp, level_fields: dict[int, dict[str, np.ndarray]], lat: np.ndarray, lon: np.ndarray) -> None:
            nonlocal masks
            if masks is None:
                masks = build_region_masks(lat, lon)
            scalar_cache[date] = {level: region_metrics(fields, lat, masks) for level, fields in level_fields.items()}

        for lag_index, lag in enumerate(SELECTED_LAGS):
            print(f"SELECTED LAG {lag:+d}")
            fx_events = []
            fy_events = []
            div_events = []
            z_events = []
            u_events = []
            v_events = []
            for event_index, event in events.iterrows():
                date = pd.Timestamp(event["start_date"]) + pd.Timedelta(days=lag)
                level_fields, lat, lon = compute_date_levels(date, monthly_cache, climatology)
                if coarse_lat is None:
                    coarse_lat, coarse_lon = lat, lon
                    masks = build_region_masks(lat, lon)
                elif not (np.array_equal(lat, coarse_lat) and np.array_equal(lon, coarse_lon)):
                    raise RuntimeError("Coarsened grid changed across dates.")
                if date not in scalar_cache:
                    cache_scalar(date, level_fields, lat, lon)
                primary = level_fields[PRIMARY_LEVEL]
                fx_events.append(primary["waf_x_m2_s2"])
                fy_events.append(primary["waf_y_m2_s2"])
                div_events.append(primary["waf_divergence_1e6_ms2"])
                z_events.append(primary["z_anomaly_m"])
                u_events.append(primary["basic_u_ms"])
                v_events.append(primary["basic_v_ms"])
                for region in REGIONS:
                    metrics = scalar_cache[date][PRIMARY_LEVEL][region]
                    for metric in SCALAR_METRICS:
                        selected_rows.append(
                            {
                                "event_id": event["event_id"],
                                "winter_label": event["winter_label"],
                                "duration_days": int(event["duration_days"]),
                                "persistence_group": event["persistence_group"],
                                "lag_day_from_onset": lag,
                                "date": date,
                                "pressure_level_hpa": PRIMARY_LEVEL,
                                "region": region,
                                "region_label": REGIONS[region]["label"],
                                "metric": metric,
                                "value": metrics[metric],
                            }
                        )
                if (event_index + 1) % 10 == 0 or event_index + 1 == len(events):
                    print(f"  {event_index + 1:02d}/{len(events)} events")
            fx_array = np.stack(fx_events, axis=0)
            fy_array = np.stack(fy_events, axis=0)
            div_array = np.stack(div_events, axis=0)
            z_array = np.stack(z_events, axis=0)
            u_array = np.stack(u_events, axis=0)
            v_array = np.stack(v_events, axis=0)
            result = analyze_vector_maps(
                fx_array,
                fy_array,
                div_array,
                z_array,
                u_array,
                v_array,
                winters,
                groups,
                RANDOM_SEED + lag_index * 100,
            )
            add_composite_field_flux(result, coarse_lat, coarse_lon)
            map_results.append(result)
            valid = np.isfinite(result["waf_x_all_event_event_mean"]) & np.isfinite(result["waf_y_all_event_event_mean"])
            for region, region_mask in masks.items():
                significance_rows.append(
                    {
                        "lag_day_from_onset": lag,
                        "region": region,
                        "region_label": REGIONS[region]["label"],
                        "all_event_robust_vector_area_fraction": HELPERS.weighted_mask_fraction(
                            result["waf_all_event_robust_vector"].astype(bool), valid, coarse_lat, region_mask
                        ),
                        "difference_robust_vector_area_fraction": HELPERS.weighted_mask_fraction(
                            result["waf_difference_robust_vector"].astype(bool), valid, coarse_lat, region_mask
                        ),
                        "all_event_robust_convergence_area_fraction": HELPERS.weighted_mask_fraction(
                            result["divergence_all_event_robust"].astype(bool) & (result["divergence_all_event_mean_1e6"] < 0), valid, coarse_lat, region_mask
                        ),
                        "difference_robust_convergence_area_fraction": HELPERS.weighted_mask_fraction(
                            result["divergence_difference_robust"].astype(bool) & (result["divergence_persistent_minus_short_1e6"] < 0), valid, coarse_lat, region_mask
                        ),
                    }
                )
            profile_lat_mask = (coarse_lat >= MAP_PROFILE_LAT_MIN) & (coarse_lat <= MAP_PROFILE_LAT_MAX)
            weights = np.cos(np.deg2rad(coarse_lat[profile_lat_mask]))[:, None]
            for group_name, group_mask in [("all", np.ones(len(events), dtype=bool)), ("short", groups == "short"), ("persistent", groups == "persistent")]:
                group_fx = np.nanmean(fx_array[group_mask], axis=0)[profile_lat_mask, :]
                numerator = np.nansum(group_fx * weights, axis=0)
                denominator = np.sum(np.where(np.isfinite(group_fx), weights, 0.0), axis=0)
                profile = np.divide(numerator, denominator, out=np.full(len(coarse_lon), np.nan), where=denominator > 0)
                for longitude, value in zip(coarse_lon, profile):
                    profile_rows.append({"lag_day_from_onset": lag, "persistence_group": group_name, "longitude": longitude, "mean_eastward_waf_m2_s2": value})
            del fx_array, fy_array, div_array, z_array, u_array, v_array
            gc.collect()

        print("BUILDING EVENT-WINDOW METRICS")
        long_rows: list[dict[str, object]] = []
        wide_rows: list[dict[str, object]] = []
        for event_index, event in events.iterrows():
            onset = pd.Timestamp(event["start_date"])
            window_dates = {
                name: pd.date_range(onset + pd.Timedelta(days=start), onset + pd.Timedelta(days=end), freq="D")
                for name, (start, end) in WINDOWS.items()
            }
            window_dates["event"] = pd.date_range(pd.Timestamp(event["start_date"]), pd.Timestamp(event["end_date"]), freq="D")
            lookup: dict[tuple[int, str, str, str], dict[str, float]] = {}
            for window_name, dates in window_dates.items():
                values = {
                    level: {region: {metric: [] for metric in SCALAR_METRICS} for region in REGIONS}
                    for level in LEVELS
                }
                for date_value in dates:
                    date = pd.Timestamp(date_value)
                    if date not in scalar_cache:
                        level_fields, lat, lon = compute_date_levels(date, monthly_cache, climatology)
                        cache_scalar(date, level_fields, lat, lon)
                    for level in LEVELS:
                        for region in REGIONS:
                            for metric in SCALAR_METRICS:
                                values[level][region][metric].append(scalar_cache[date][level][region][metric])
                for level in LEVELS:
                    for region in REGIONS:
                        for metric in SCALAR_METRICS:
                            summary = summarize(values[level][region][metric])
                            lookup[(level, region, window_name, metric)] = summary
                            long_rows.append(
                                {
                                    "event_id": event["event_id"],
                                    "winter_label": event["winter_label"],
                                    "start_date": event["start_date"],
                                    "end_date": event["end_date"],
                                    "duration_days": int(event["duration_days"]),
                                    "persistence_group": event["persistence_group"],
                                    "pressure_level_hpa": level,
                                    "region": region,
                                    "region_label": REGIONS[region]["label"],
                                    "window": window_name,
                                    "window_day_count": len(dates),
                                    "metric": metric,
                                    "mean": summary["mean"],
                                    "minimum": summary["minimum"],
                                    "maximum": summary["maximum"],
                                }
                            )
            wide_rows.append(build_wide_row(event, lookup))
            if (event_index + 1) % 10 == 0 or event_index + 1 == len(events):
                print(f"  {event_index + 1:02d}/{len(events)} events")

        long_table = pd.DataFrame(long_rows)
        wide_table = pd.DataFrame(wide_rows)
        selected_table = pd.DataFrame(selected_rows)
        significance_table = pd.DataFrame(significance_rows)
        profile_table = pd.DataFrame(profile_rows)
        long_table.to_csv(TABLE37_LONG, index=False, date_format="%Y-%m-%d")
        wide_table.to_csv(TABLE37_WIDE, index=False, date_format="%Y-%m-%d")
        selected_table.to_csv(TABLE40, index=False, date_format="%Y-%m-%d")
        significance_table.to_csv(TABLE41, index=False)
        profile_table.to_csv(TABLE42, index=False)
        write_netcdf(map_results, coarse_lat, coarse_lon)
        summary = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "event_count": len(events),
            "short_event_count": int(events["persistence_group"].eq("short").sum()),
            "persistent_event_count": int(events["persistence_group"].eq("persistent").sum()),
            "levels_hpa": LEVELS,
            "primary_level_hpa": PRIMARY_LEVEL,
            "selected_lags": SELECTED_LAGS,
            "unique_dates_processed": len(scalar_cache),
            "long_table_rows": len(long_table),
            "wide_table_rows": len(wide_table),
            "selected_lag_rows": len(selected_table),
            "significance_rows": len(significance_table),
            "profile_rows": len(profile_table),
            "netcdf_file": str(NETCDF_FILE),
        }
        SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        lines = [
            "STEP 10D: TAKAYA-NAKAMURA WAVE-ACTIVITY FLUX",
            "=" * 72,
            f"Events analysed: {len(events)}",
            f"Primary level: {PRIMARY_LEVEL} hPa",
            f"Sensitivity levels: {[level for level in LEVELS if level != PRIMARY_LEVEL]}",
            f"Selected lags: {SELECTED_LAGS}",
            f"Unique dates processed: {len(scalar_cache)}",
            f"Long event-metric rows: {len(long_table)}",
            f"Selected-lag rows: {len(selected_table)}",
            f"Significance rows: {len(significance_table)}",
            f"NetCDF: {NETCDF_FILE}",
            f"Event metrics: {TABLE37_LONG}",
            f"Event predictors: {TABLE37_WIDE}",
        ]
        REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        print("\nSTEP 10D-COMPUTE PASSED.")
    finally:
        monthly_cache.close()
        climatology.close()


if __name__ == "__main__":
    main()
