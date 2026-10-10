"""
Shared helpers for the Step 16 termination analysis (scripts 15a-15g).

Imported by the Step 16 scripts through importlib, like 12_final_common.py.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------------------- paths
EVENT_CATALOGUE = PROJECT_ROOT / "06_events" / "step7e_frozen_primary_event_catalogue.csv"
STATION_WEIGHTS = PROJECT_ROOT / "02_metadata" / "step7d_primary_station_area_weights.csv"
STATION_MASTER = PROJECT_ROOT / "02_metadata" / "station_master_v2.csv"
RAW_TMAX_XLSX = PROJECT_ROOT / "01_raw_data" / "MaxT_MinT_1981_2024_raw.xlsx"
CLEAN_DJF = PROJECT_ROOT / "04_clean_data" / "temperature_daily_djf_with_area_weighted_event_membership.parquet"
DAILY_AREA_DIAG = PROJECT_ROOT / "03_intermediate" / "step7d_area_weighted_daily_cold_diagnostics.parquet"
ERA5_BOX_DAILY = PROJECT_ROOT / "03_intermediate" / "era5_box_daily" / "era5_box_daily_oct_mar.csv"
SH_BLOCK_TABLE = PROJECT_ROOT / "08_outputs" / "tables" / "tropospheric_indices" / "table_22_daily_siberian_high_blocking_indices.csv"
RRWP_TABLE = PROJECT_ROOT / "08_outputs" / "tables" / "rrwp" / "table_43_daily_rrwp_sector_metrics.csv"
STRAT_TABLE = PROJECT_ROOT / "08_outputs" / "tables" / "stratosphere" / "table_59_daily_stratospheric_indices.csv"

TERM_INTERMEDIATE = PROJECT_ROOT / "03_intermediate" / "termination"
TERM_TABLES = PROJECT_ROOT / "08_outputs" / "tables" / "termination"
TERM_FIGURES = PROJECT_ROOT / "08_outputs" / "figures" / "termination"
TERM_QC = PROJECT_ROOT / "05_qc_reports" / "termination"
ADMIN = PROJECT_ROOT / "00_admin"

DAILY_COVARIATES = TERM_INTERMEDIATE / "step16b_daily_covariates_djf.csv"

BASELINE_START, BASELINE_END = 1991, 2020
PERSISTENCE_GROUPS = {"short": (3, 3), "intermediate": (4, 5), "persistent": (6, 999)}


def ensure_dirs() -> None:
    for d in (TERM_INTERMEDIATE, TERM_TABLES, TERM_FIGURES, TERM_QC):
        d.mkdir(parents=True, exist_ok=True)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    """Read a parquet file; fall back to a CSV/CSV.GZ copy with the same stem."""
    path = Path(path)
    if path.suffix == ".parquet" and path.exists():
        return pd.read_parquet(path, **{k: v for k, v in kwargs.items() if k == "columns"})
    for alt in (path.with_suffix(".csv.gz"), path.with_suffix(".csv"), path):
        if alt.exists() and alt.suffix != ".parquet":
            usecols = kwargs.get("columns")
            return pd.read_csv(alt, usecols=usecols, low_memory=False)
    raise FileNotFoundError(path)


def write_policy(name: str, policy: dict) -> Path:
    policy = dict(policy)
    policy["created_utc"] = now_utc()
    out = ADMIN / name
    out.write_text(json.dumps(policy, indent=2, default=str))
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksums(name: str, paths: list[Path]) -> Path:
    out = ADMIN / name
    out.write_text("".join(f"{sha256(p)}  {p.relative_to(PROJECT_ROOT)}\n" for p in paths))
    return out


def load_events() -> pd.DataFrame:
    ev = pd.read_csv(EVENT_CATALOGUE, parse_dates=["start_date", "end_date"])
    ev["group"] = "intermediate"
    ev.loc[ev["duration_days"] == 3, "group"] = "short"
    ev.loc[ev["duration_days"] >= 6, "group"] = "persistent"
    return ev


# ----------------------------------------------------------------------------- climatology
def calendar_day(index: pd.DatetimeIndex) -> np.ndarray:
    """Day of a non-leap year (1-365); 29 February is mapped to 28 February (day 59)."""
    doy = pd.to_datetime("2001-" + index.strftime("%m-%d"), errors="coerce").dayofyear
    return pd.Series(doy).fillna(59).astype(int).to_numpy()


def calendar_anomalies(df: pd.DataFrame, half_window: int, group_col: str | None = None) -> pd.DataFrame:
    """
    Anomalies relative to a 1991-2020 calendar-day mean using a +/- half_window moving window.
    df must have a DatetimeIndex. If group_col is given (e.g. station_uid), climatology is per group.
    """
    value_cols = [c for c in df.columns if c != group_col]
    work = df.copy()
    work["_cd"] = calendar_day(work.index)
    base = work[(work.index.year >= BASELINE_START) & (work.index.year <= BASELINE_END)]
    keys = ["_cd"] if group_col is None else [group_col, "_cd"]
    parts = []
    for off in range(-half_window, half_window + 1):
        b = base.copy()
        b["_cd"] = ((b["_cd"] - 1 - off) % 365) + 1
        parts.append(b)
    clim = pd.concat(parts).groupby(keys)[value_cols].mean()
    merged = work.reset_index().merge(clim, left_on=keys, right_index=True, how="left", suffixes=("", "_clim"))
    merged = merged.set_index(work.index.name or "index")
    out = pd.DataFrame(index=work.index)
    if group_col is not None:
        out[group_col] = work[group_col]
    for c in value_cols:
        out[c] = merged[c].to_numpy() - merged[f"{c}_clim"].to_numpy()
    return out


# ----------------------------------------------------------------------------- statistics
def winter_bootstrap_counts(winters: np.ndarray, n_boot: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (unique winters, B x n_winters matrix of resampling counts)."""
    uniq = np.unique(winters)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    counts = np.zeros((n_boot, len(uniq)), dtype=np.int32)
    for j in range(len(uniq)):
        counts[:, j] = (draws == j).sum(axis=1)
    return uniq, counts


# Step 18 revision (October 2026). Season-end censoring is the primary hazard definition: a spell whose last
# cold day is the last day of February (the end of the analysed season) has an unobserved end, so its final
# risk-set row is right-censored (terminates = 0) instead of counted as a termination. Two of the 85 primary
# spells are affected. The uncensored definition (used up to v1.2.0) is kept as a sensitivity analysis.
CENSOR_SEASON_END = True


def season_end_censored(ev: pd.DataFrame) -> set:
    """Event ids whose last cold day is the last day of February."""
    end = pd.to_datetime(ev["end_date"])
    nxt = end + pd.Timedelta(days=1)
    return set(ev.loc[(end.dt.month == 2) & (nxt.dt.month == 3), "event_id"])


def apply_season_end_censoring(rs: pd.DataFrame, ev: pd.DataFrame, censor: bool | None = None) -> pd.DataFrame:
    """Set terminates = 0 on the final row of season-end spells when censoring is on (default: CENSOR_SEASON_END)."""
    if CENSOR_SEASON_END if censor is None else censor:
        ids = season_end_censored(ev)
        rs.loc[rs["event_id"].isin(ids) & (rs["terminates"] == 1), "terminates"] = 0
    return rs


# Step 18 revision (October 2026). Every bootstrap or Monte Carlo p-value is now a "plus-one" p-value,
#     p = min(1, 2 (min(#{t* <= 0}, #{t* >= 0}) + 1) / (B + 1)),
# which can never be exactly zero (the earlier 2 min(frac <= 0, frac >= 0) gave p = 0 whenever no
# resample crossed zero, and Benjamini-Hochberg then accepted every such test). Hypothesis tests use at
# least MIN_TEST_BOOT resamples so that the smallest attainable p (2 / (B + 1)) is far below the
# Benjamini-Hochberg thresholds of the largest test families (66 scalar predictors).
MIN_TEST_BOOT = 10000


def test_boot(n_boot: int) -> int:
    """Number of bootstrap resamples to use for a hypothesis test (never fewer than MIN_TEST_BOOT)."""
    return max(int(n_boot), MIN_TEST_BOOT)


def p_from_tail_counts(n_le, n_ge, n_valid):
    """Two-sided plus-one p-value from the counts of resamples <= null and >= null."""
    n_le, n_ge, n_valid = np.asarray(n_le, float), np.asarray(n_ge, float), np.asarray(n_valid, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.minimum(1.0, 2.0 * (np.minimum(n_le, n_ge) + 1.0) / (n_valid + 1.0))
    p = np.where(n_valid > 0, p, np.nan)
    return float(p) if p.ndim == 0 else p


def bootstrap_p(draws, null: float = 0.0, axis: int = 0):
    """Two-sided plus-one bootstrap p-value of H0: statistic = null. NaN draws are ignored."""
    d = np.asarray(draws, dtype=float)
    ok = np.isfinite(d)
    return p_from_tail_counts(((d <= null) & ok).sum(axis=axis), ((d >= null) & ok).sum(axis=axis),
                              ok.sum(axis=axis))


def logit_fit_batch(X: np.ndarray, y: np.ndarray, W: np.ndarray, max_iter: int = 100, tol: float = 1e-10,
                    chunk: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """
    Unpenalised weighted logistic regression for many weight vectors at once (Newton-Raphson with step
    halving). This is the same estimator as sklearn LogisticRegression(penalty=None) with sample_weight,
    but fits 10 000 winter-block bootstrap replicates in seconds instead of minutes.

    X: n x p predictors (an intercept is added), y: n (0/1), W: B x n non-negative weights (bootstrap counts).
    Returns (coef, converged): coef is B x p (intercept dropped); rows whose weighted outcome has only one class
    are NaN. Replicates with (quasi-)separation do not converge and keep their last (large) iterate, as sklearn
    does; the number of such replicates is reported by the callers.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    W = np.atleast_2d(np.asarray(W, dtype=float))
    n, p = X.shape
    Xi = np.column_stack([np.ones(n), X])
    k = p + 1
    coef = np.full((W.shape[0], k), np.nan)
    conv = np.zeros(W.shape[0], dtype=bool)
    ridge = 1e-10 * np.eye(k)

    def loglik(b, Wc):
        eta = b @ Xi.T
        return (Wc * (y * eta - np.logaddexp(0.0, eta))).sum(axis=1)

    for s in range(0, W.shape[0], chunk):
        Wc = W[s:s + chunk]
        pos, tot = Wc @ y, Wc.sum(axis=1)
        ok = (pos > 0) & (pos < tot)
        if not ok.any():
            continue
        Wc = Wc[ok]
        b = np.zeros((len(Wc), k))
        ybar = pos[ok] / tot[ok]
        b[:, 0] = np.log(ybar / (1.0 - ybar))
        ll = loglik(b, Wc)
        done = np.zeros(len(Wc), dtype=bool)
        for _ in range(max_iter):
            act = ~done
            if not act.any():
                break
            ba, Wa = b[act], Wc[act]
            mu = 1.0 / (1.0 + np.exp(-(ba @ Xi.T)))
            g = (Wa * (y - mu)) @ Xi
            H = np.einsum("cn,ni,nj->cij", Wa * mu * (1.0 - mu), Xi, Xi, optimize=True) + ridge
            step = np.linalg.solve(H, g[..., None])[..., 0]
            t = np.ones(len(ba))
            ll_old = ll[act]
            new = ba + step
            ll_new = loglik(new, Wa)
            for _h in range(30):  # step halving where the log-likelihood fell
                bad = ll_new < ll_old - 1e-12
                if not bad.any():
                    break
                t[bad] *= 0.5
                new[bad] = ba[bad] + t[bad, None] * step[bad]
                ll_new[bad] = loglik(new[bad], Wa[bad])
            idx = np.flatnonzero(act)
            b[idx], ll[idx] = new, ll_new
            done[idx] = np.max(np.abs(t[:, None] * step), axis=1) < tol
        rows = np.flatnonzero(ok) + s
        coef[rows] = b
        conv[rows] = done
    return coef[:, 1:], conv


def sign_flip_p(diff: np.ndarray, n_exact_max: int = 22, n_mc: int = 200000, seed: int = 20261010,
                chunk: int = 8192) -> np.ndarray:
    """
    Two-sided sign-flip (paired permutation) test of H0: mean = 0 for each column of diff (n x m),
    e.g. per-winter post-SSW minus rest-of-winter differences at m grid cells. Exact (all 2^n sign
    patterns) for n <= n_exact_max, otherwise Monte Carlo with a plus-one p-value. Columns with any NaN get NaN.
    """
    d = np.asarray(diff, dtype=float)
    if d.ndim == 1:
        d = d[:, None]
    n, m = d.shape
    good = np.isfinite(d).all(axis=0)
    dz = np.where(good[None, :], d, 0.0)
    obs = np.abs(dz.sum(axis=0))
    tol = 1e-9 * np.maximum(np.abs(dz).sum(axis=0), 1e-30)
    count = np.zeros(m)
    bits = np.arange(n)
    if n <= n_exact_max:
        total = 2 ** n
        for s in range(0, total, chunk):
            idx = np.arange(s, min(s + chunk, total), dtype=np.int64)
            signs = 1.0 - 2.0 * ((idx[:, None] >> bits[None, :]) & 1)
            count += (np.abs(signs @ dz) >= obs - tol).sum(axis=0)
        p = count / total
    else:
        rng = np.random.default_rng(seed)
        for s in range(0, n_mc, chunk):
            c = min(chunk, n_mc - s)
            signs = rng.choice([-1.0, 1.0], size=(c, n))
            count += (np.abs(signs @ dz) >= obs - tol).sum(axis=0)
        p = (count + 1.0) / (n_mc + 1.0)
    return np.where(good, np.minimum(p, 1.0), np.nan)


def group_difference_bootstrap(values: np.ndarray, is_a: np.ndarray, is_b: np.ndarray, winters: np.ndarray,
                               n_boot: int = 5000, seed: int = 20260923) -> dict:
    """Mean(A) - mean(B) with winter-block bootstrap CI and two-sided plus-one bootstrap p-value."""
    n_boot = test_boot(n_boot)
    ok = ~np.isnan(values)
    values, is_a, is_b, winters = values[ok], is_a[ok], is_b[ok], winters[ok]
    obs = values[is_a].mean() - values[is_b].mean()
    uniq, counts = winter_bootstrap_counts(winters, n_boot, seed)
    widx = np.searchsorted(uniq, winters)
    w = counts[:, widx].astype(float)  # B x n_events
    wa, wb = w * is_a, w * is_b
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = (wa @ values) / wa.sum(axis=1) - (wb @ values) / wb.sum(axis=1)
    diff = diff[np.isfinite(diff)]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    p = bootstrap_p(diff)
    return {"difference": obs, "ci_low": lo, "ci_high": hi, "p_bootstrap": p,
            "n_a": int(is_a.sum()), "n_b": int(is_b.sum()), "n_boot_valid": int(len(diff))}


# ----------------------------------------------------------------------------- terrain validity (Step 18)
# 850 hPa (standard height about 1,460 m) lies below or within a few hundred metres of the ground wherever the
# mean terrain height of an ERA5 0.25-degree cell exceeds about 1,000 m; ERA5 values there are extrapolated.
# Pressure-level fields at 1000 and 850 hPa and sea-level pressure are therefore masked where the ETOPO1
# cell-mean elevation exceeds TERRAIN_MAX_M; 700 hPa is masked above 2,500 m; 500 hPa and above are not masked.
# The rule was fixed before rerunning any analysis; 1,500 m is the pre-stated sensitivity threshold.
TERRAIN_FILE = PROJECT_ROOT / "01_raw_data" / "geospatial" / "etopo1_south_asia_0p1deg.nc"
TERRAIN_MAX_M = 1000.0
TERRAIN_700HPA_MAX_M = 2500.0
_ELEV_CACHE: dict = {}


def _read_terrain() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lat, lon, elevation[lat, lon]) from the ETOPO1 file, with xarray or, failing that, scipy (netCDF3)."""
    if not TERRAIN_FILE.exists():
        raise FileNotFoundError(f"{TERRAIN_FILE} is needed for the terrain mask")
    try:
        import xarray as xr
        with xr.open_dataset(TERRAIN_FILE) as ds:
            var = "altitude" if "altitude" in ds else list(ds.data_vars)[0]
            z = ds[var]
            dlat = next(d for d in z.dims if "lat" in d.lower())
            dlon = next(d for d in z.dims if "lon" in d.lower())
            return (ds[dlat].values.astype(float), ds[dlon].values.astype(float),
                    z.transpose(dlat, dlon).values.astype(float))
    except ImportError:
        from scipy.io import netcdf_file
        f = netcdf_file(TERRAIN_FILE, "r", mmap=False)
        v = f.variables
        dlat = next(k for k in v if "lat" in k.lower())
        dlon = next(k for k in v if "lon" in k.lower())
        var = "altitude" if "altitude" in v else next(k for k in v if k not in (dlat, dlon))
        return v[dlat][:].astype(float), v[dlon][:].astype(float), v[var][:].astype(float)


def grid_cell_elevation(lat: np.ndarray, lon: np.ndarray, half_width: float = 0.125) -> np.ndarray:
    """Mean ETOPO1 elevation (m) of the grid cell centred on each (lat, lon) point; returns nlat x nlon."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    key = (lat.tobytes(), lon.tobytes(), half_width)
    if key in _ELEV_CACHE:
        return _ELEV_CACHE[key]
    elat, elon, Z = _read_terrain()
    eps = 1e-6
    inside_lat = (lat >= elat.min() - eps) & (lat <= elat.max() + eps)
    inside_lon = (lon >= elon.min() - eps) & (lon <= elon.max() + eps)
    elev = np.full((len(lat), len(lon)), np.nan)
    rows = np.array([np.nanmean(Z[np.abs(elat - la) <= half_width + eps], axis=0) if ok else np.full(len(elon), np.nan)
                     for la, ok in zip(lat, inside_lat)])
    for j, (lo, ok) in enumerate(zip(lon, inside_lon)):
        if ok:
            elev[:, j] = np.nanmean(rows[:, np.abs(elon - lo) <= half_width + eps], axis=1)
    outside = ~(inside_lat[:, None] & inside_lon[None, :])
    if outside.any():
        # The ETOPO1 extract covers 0-55N, 45-125E. Cells outside it (Anatolia, Caucasus, Europe, north of 55N)
        # are left unmasked; they lie outside every analysis box and only enter the hemispheric maps.
        print(f"NOTE terrain mask: {outside.mean():.1%} of the grid lies outside the ETOPO1 extract and is not masked")
    _ELEV_CACHE[key] = elev
    return elev


def below_ground_mask(lat: np.ndarray, lon: np.ndarray, level_hpa: float | None,
                      max_m: float = TERRAIN_MAX_M, half_width: float = 0.125) -> np.ndarray:
    """
    True where a field is extrapolated below ground. level_hpa=None means sea-level pressure.
    half_width is half the grid spacing in degrees (0.125 for the native 0.25-degree grid, 0.5 for 1-degree
    block means, 1.0 for 2-degree block means).
    """
    elev = grid_cell_elevation(lat, lon, half_width)
    with np.errstate(invalid="ignore"):
        if level_hpa is None or float(level_hpa) >= 850.0:
            return np.nan_to_num(elev, nan=-1e9) > max_m
        if float(level_hpa) >= 700.0:
            return np.nan_to_num(elev, nan=-1e9) > TERRAIN_700HPA_MAX_M
    return np.zeros_like(elev, dtype=bool)


def dilate_mask(mask: np.ndarray) -> np.ndarray:
    """Add the four neighbours of every masked cell (centred differences use them)."""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


# ----------------------------------------------------------------------------- Step 16J mechanisms
MECHANISM_DAILY = PROJECT_ROOT / "03_intermediate" / "era5_box_daily" / "era5_mechanism_daily_oct_mar.csv"
MECHANISM_HALF_WINDOW = 7


def load_covariates_with_mechanisms() -> pd.DataFrame:
    """Step 16B daily covariates joined with Step 16J advection/jet anomalies and tendencies (DJF rows)."""
    cov = pd.read_csv(DAILY_COVARIATES, parse_dates=["date"]).set_index("date")
    mech = pd.read_csv(MECHANISM_DAILY, parse_dates=["date"]).set_index("date")
    mech.index.name = "date"
    anom = calendar_anomalies(mech, MECHANISM_HALF_WINDOW)
    anom.columns = [f"mech_{c}_anom" for c in anom.columns]
    tend = anom.diff()
    tend.columns = [c.replace("_anom", "_tend") for c in tend.columns]
    return cov.join(anom).join(tend)


# ----------------------------------------------------------------------------- merged Tmax (old archive + 2022-2025 file)
RAW_TMAX_NEW = PROJECT_ROOT / "01_raw_data" / "Daily_Maximum_Temperature_2022_2025_raw.csv"
NEW_SOURCE_START = pd.Timestamp("2022-01-01")
TMAX_MISSING_CODES = (0.0,)  # exact 0.0 degC is a missing value entered as zero in the 2022-2025 file


def _old_tmax() -> pd.DataFrame:
    master = pd.read_csv(STATION_MASTER)
    name_to_uid = dict(zip(master["old_xlsx_station_name"].astype(str).str.strip(), master["station_uid"]))
    raw = pd.read_excel(RAW_TMAX_XLSX, header=None, skiprows=2,
                        names=["station", "year", "month", "day", "tmax", "tmin"])
    raw["station_uid"] = raw["station"].astype(str).str.strip().map(name_to_uid)
    raw["date"] = pd.to_datetime(dict(year=raw["year"], month=raw["month"], day=raw["day"]), errors="coerce")
    raw = raw.dropna(subset=["date", "station_uid"]).drop_duplicates(["station_uid", "date"], keep="first")
    for c in ("tmax", "tmin"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    return raw[["station_uid", "date", "tmax", "tmin"]]


def _new_tmax() -> pd.DataFrame:
    master = pd.read_csv(STATION_MASTER)
    name_to_uid = dict(zip(master["new_tmin_station_name"].astype(str).str.strip(), master["station_uid"]))
    d = pd.read_csv(RAW_TMAX_NEW, skiprows=10, dtype=str)
    d.columns = [c.strip() for c in d.columns]
    days = [f"Day{i}" for i in range(1, 32)]
    L = d.melt(id_vars=["Station", "Year", "Month"], value_vars=days, var_name="day", value_name="v")
    L["date"] = pd.to_datetime(dict(year=pd.to_numeric(L["Year"]), month=pd.to_numeric(L["Month"]),
                                    day=L["day"].str[3:].astype(int)), errors="coerce")
    L["station_uid"] = L["Station"].astype(str).str.strip().map(name_to_uid)
    L = L.dropna(subset=["date", "station_uid"])
    L["tmax"] = pd.to_numeric(L["v"].astype(str).str.strip().replace({"**": None}), errors="coerce")
    L["missing_code"] = L["tmax"].isin(TMAX_MISSING_CODES)
    L.loc[L["missing_code"], "tmax"] = np.nan
    return L[["station_uid", "date", "tmax", "missing_code"]].drop_duplicates(["station_uid", "date"])


def raw_tmax_merged(return_parts: bool = False):
    """Raw daily Tmax from both BMD sources, merged with the Step 2C rule used for Tmin:
    before 2022 the old archive only; from 2022 the 2022-2025 file is selected where it has a value
    (including where the two disagree) and the old archive fills its gaps. Exact 0.0 degC in the
    2022-2025 file is treated as missing."""
    old, new = _old_tmax(), _new_tmax()
    m = old.merge(new.rename(columns={"tmax": "tmax_new"}), on=["station_uid", "date"], how="outer")
    m = m.rename(columns={"tmax": "tmax_old"})
    use_new = (m["date"] >= NEW_SOURCE_START) & m["tmax_new"].notna()
    m["tmax"] = np.where(use_new, m["tmax_new"], m["tmax_old"])
    m["tmax_source"] = np.where(use_new, "new_2022_2025", np.where(m["tmax_old"].notna(), "old_archive", "none"))
    out = m[["station_uid", "date", "tmax", "tmin", "tmax_source"]].sort_values(["station_uid", "date"])
    return (out, m) if return_parts else out
