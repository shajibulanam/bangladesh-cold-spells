from __future__ import annotations

from pathlib import Path
import calendar
import json
import math
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / '01_raw_data' / 'era5' / 'stratosphere_daily_mean'
INTERMEDIATE_DIR = PROJECT_ROOT / '03_intermediate' / 'stratosphere'
NETCDF_DIR = INTERMEDIATE_DIR / 'netcdf'
TABLE_DIR = PROJECT_ROOT / '08_outputs' / 'tables' / 'stratosphere'
FIGURE_DIR = PROJECT_ROOT / '08_outputs' / 'figures' / 'stratosphere'
REPORT_OUT_DIR = PROJECT_ROOT / '08_outputs' / 'reports' / 'stratosphere'
QC_DIR = PROJECT_ROOT / '05_qc_reports' / 'stratosphere'
LOG_DIR = PROJECT_ROOT / '09_logs'
EVENT_CATALOGUE = PROJECT_ROOT / '06_events' / 'step7e_frozen_primary_event_catalogue.csv'
EVENT_DAYS = PROJECT_ROOT / '06_events' / 'step7e_frozen_primary_event_days.csv'
GROUPS = PROJECT_ROOT / '08_outputs' / 'tables' / 'era5_persistence' / 'table_18_persistence_groups.csv'

LEVELS = [10, 30, 50, 100, 250, 500, 1000]
WINTER_START_YEARS = list(range(1985, 2025))
SELECTED_LAGS = [-45, -30, -20, -15, -10, -5, 0, 5]
FULL_LAGS = np.arange(-45, 6, dtype=int)
G = 9.80665

for directory in [RAW_DIR, INTERMEDIATE_DIR, NETCDF_DIR, TABLE_DIR, FIGURE_DIR, REPORT_OUT_DIR, QC_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def winter_months() -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for y in WINTER_START_YEARS:
        rows.extend([(y, 10), (y, 11), (y, 12), (y + 1, 1), (y + 1, 2), (y + 1, 3), (y + 1, 4)])
    return rows


def days_in_month(year: int, month: int) -> list[str]:
    return [f'{day:02d}' for day in range(1, calendar.monthrange(year, month)[1] + 1)]


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    vals = p[valid]
    if vals.size == 0:
        return q
    order = np.argsort(vals)
    ranked = vals[order]
    n = ranked.size
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    q[valid] = restored
    return q


def bootstrap_two_sided(samples: np.ndarray) -> float:
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(min(1.0, 2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0))))


def winter_bootstrap_indices(winter_labels: np.ndarray, n_boot: int, rng: np.random.Generator):
    winters = np.unique(winter_labels)
    mapping = {w: np.where(winter_labels == w)[0] for w in winters}
    for _ in range(n_boot):
        selected = rng.choice(winters, size=len(winters), replace=True)
        yield np.concatenate([mapping[w] for w in selected])


def winter_bootstrap_diff(values: np.ndarray, groups: np.ndarray, winters: np.ndarray, n_boot: int = 5000, seed: int = 12011):
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    winters = np.asarray(winters)
    valid = np.isfinite(values) & np.isin(groups, ['short', 'persistent'])
    values, groups, winters = values[valid], groups[valid], winters[valid]
    short = values[groups == 'short']
    persistent = values[groups == 'persistent']
    observed = float(np.nanmean(persistent) - np.nanmean(short)) if short.size and persistent.size else np.nan
    rng = np.random.default_rng(seed)
    diffs = []
    for idx in winter_bootstrap_indices(winters, n_boot, rng):
        vg, gg = values[idx], groups[idx]
        s, p = vg[gg == 'short'], vg[gg == 'persistent']
        if s.size and p.size:
            diffs.append(float(np.nanmean(p) - np.nanmean(s)))
    arr = np.asarray(diffs, dtype=float)
    if arr.size:
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        pval = bootstrap_two_sided(arr)
    else:
        lo = hi = pval = np.nan
    return observed, float(lo), float(hi), float(pval)


def winter_bootstrap_spearman(x: np.ndarray, y: np.ndarray, winters: np.ndarray, n_boot: int = 5000, seed: int = 12012):
    from scipy.stats import spearmanr
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float); winters = np.asarray(winters)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y, winters = x[valid], y[valid], winters[valid]
    observed = float(spearmanr(x, y).statistic) if x.size >= 3 else np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for idx in winter_bootstrap_indices(winters, n_boot, rng):
        if idx.size >= 3:
            rho = spearmanr(x[idx], y[idx]).statistic
            if np.isfinite(rho): vals.append(float(rho))
    arr = np.asarray(vals, dtype=float)
    if arr.size:
        lo, hi = np.nanpercentile(arr, [2.5, 97.5]); pval = bootstrap_two_sided(arr)
    else:
        lo = hi = pval = np.nan
    return observed, float(lo), float(hi), float(pval)


def loo_sign_stability(values: np.ndarray, groups: np.ndarray, winters: np.ndarray) -> float:
    values = np.asarray(values, dtype=float); groups = np.asarray(groups); winters = np.asarray(winters)
    valid = np.isfinite(values) & np.isin(groups, ['short', 'persistent'])
    values, groups, winters = values[valid], groups[valid], winters[valid]
    if not np.any(groups == 'short') or not np.any(groups == 'persistent'):
        return np.nan
    full = np.nanmean(values[groups == 'persistent']) - np.nanmean(values[groups == 'short'])
    if not np.isfinite(full) or full == 0: return np.nan
    signs = []
    for w in np.unique(winters):
        keep = winters != w
        s = values[keep & (groups == 'short')]; p = values[keep & (groups == 'persistent')]
        if s.size and p.size:
            d = np.nanmean(p) - np.nanmean(s)
            if np.isfinite(d): signs.append(np.sign(d) == np.sign(full))
    return float(np.mean(signs)) if signs else np.nan


def weighted_mean_lat(values: np.ndarray, lat: np.ndarray, mask: np.ndarray | None = None) -> float:
    v = np.asarray(values, dtype=float); lat = np.asarray(lat, dtype=float)
    if mask is None: mask = np.ones(lat.shape, dtype=bool)
    weights = np.cos(np.deg2rad(lat[mask]))
    if v.ndim == 1:
        vv = v[mask]; good = np.isfinite(vv)
        return float(np.sum(vv[good] * weights[good]) / np.sum(weights[good])) if np.any(good) else np.nan
    vv = v[mask, :]
    zonal = np.nanmean(vv, axis=-1)
    good = np.isfinite(zonal)
    return float(np.sum(zonal[good] * weights[good]) / np.sum(weights[good])) if np.any(good) else np.nan


def consecutive_max(flag: np.ndarray) -> int:
    best = run = 0
    for value in np.asarray(flag, dtype=bool):
        run = run + 1 if value else 0
        best = max(best, run)
    return int(best)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
