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


def group_difference_bootstrap(values: np.ndarray, is_a: np.ndarray, is_b: np.ndarray, winters: np.ndarray,
                               n_boot: int = 5000, seed: int = 20260923) -> dict:
    """Mean(A) - mean(B) with winter-block bootstrap CI and two-sided bootstrap p-value."""
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
    p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
    return {"difference": obs, "ci_low": lo, "ci_high": hi, "p_bootstrap": p,
            "n_a": int(is_a.sum()), "n_b": int(is_b.sum()), "n_boot_valid": int(len(diff))}


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
