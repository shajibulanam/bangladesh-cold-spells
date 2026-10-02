"""
Step 16F - Cold-day (Tmax) spell catalogue and comparison with the primary cold-night (Tmin) catalogue.

Definition (mirrors Step 7A/7D, applied to Tmax)
    Station cold day   : Tmax strictly below the station calendar-day p10 (1991-2020, circular
                         5-day window, median-unbiased quantile; leave-one-calendar-year-out for
                         dates inside 1991-2020, full-baseline threshold outside it).
    National cold day  : >= 20% of the full national area cold (Step 7D Voronoi weights), with
                         >= 80% of the area and >= 15 stations observed.
    Cold-day spell     : >= 3 consecutive national cold days within one DJF season; no gap merging.
    Period             : 1985/86-2024/25 (Tmax: old BMD archive merged with the 2022-2025 file).
Tmax values are the Step 16B QC'd values for DJF; threshold windows that reach into November or
March use raw Tmax after the same range, Tmax<Tmin and 5-SD checks.

Comparison
    Event "type": pure cold-night (primary Tmin events with no day in a cold-day spell), pure
    cold-day (Tmax spells with no day in a primary event), and overlapping (share at least one day).
    Event-level means of station and ERA5 anomalies are compared between pure cold-day and pure
    cold-night events (winter-block bootstrap, 5,000 resamples, Benjamini-Hochberg FDR).

Run from the project root:
    python3 07_scripts/15f_cold_day_spell_catalogue.py 2>&1 | tee 09_logs/step16f_cold_day_spells.log
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("term_common", "15_termination_common.py")
B = _load("step16b", "15b_build_daily_covariates.py")

THRESHOLDS = C.TERM_INTERMEDIATE / "step16f_station_tmax_p10_thresholds.csv"
DAILY = C.TERM_INTERMEDIATE / "step16f_daily_cold_day_area.csv"
CATALOGUE = C.PROJECT_ROOT / "06_events" / "step16f_cold_day_spell_catalogue.csv"
T_WINTER = C.TERM_TABLES / "table_99_cold_day_cold_night_by_winter.csv"
T_TYPES = C.TERM_TABLES / "table_100_spell_type_summary.csv"
T_CONTRAST = C.TERM_TABLES / "table_101_cold_day_vs_cold_night_contrast.csv"
REPORT = C.TERM_QC / "step16f_cold_day_spells_report.txt"
SUMMARY = C.TERM_QC / "step16f_cold_day_spells_summary.json"

LAST_WINTER = 2024
P = 0.10
HALF_WINDOW = 2
MIN_AREA_OBS, MIN_STATIONS_OBS, MIN_COLD_AREA, MIN_DAYS = 0.80, 15, 0.20, 3
N_BOOT, SEED = 5000, 20260923
CONTRAST_VARS = [
    "nat_tmin_anom", "nat_tmax_anom", "nat_dtr_anom",
    "era5_bd_t2m_anom", "era5_bd_t850_anom", "era5_bd_stab_t2m_minus_t850_anom",
    "era5_bd_z500_anom", "era5_bd_mslp_anom", "era5_bd_u850_anom", "era5_bd_v850_anom",
    "era5_bd_wspd10_anom", "era5_sa_z500_anom", "era5_sa_mslp_anom", "era5_sa_t850_anom",
]

POLICY = {
    "step": "Step 16F", "title": "Cold-day (Tmax) spell catalogue",
    "station_threshold": {"percentile": P, "window_half_width_days": HALF_WINDOW, "circular_window": True,
                          "calendar": "366-day (29 Feb has its own window)", "quantile_method": "median_unbiased",
                          "baseline": "1991-2020", "baseline_bias": "leave_one_calendar_year_out inside baseline",
                          "comparison": "strictly_less_than"},
    "national_day": {"minimum_observed_area": MIN_AREA_OBS, "minimum_observed_stations": MIN_STATIONS_OBS,
                     "minimum_cold_area_full_denominator": MIN_COLD_AREA},
    "spell": {"minimum_consecutive_days": MIN_DAYS, "merge_gaps": False, "cross_winter": False},
    "period": f"1985/86-{LAST_WINTER}/{str(LAST_WINTER + 1)[-2:]}",
    "comparison": "pure cold-day vs pure cold-night events; event-level means; winter-block bootstrap; BH FDR",
}


def cal366(dates: pd.Series) -> np.ndarray:
    return pd.to_datetime("2000-" + dates.dt.strftime("%m-%d")).dt.dayofyear.to_numpy()


def raw_tmax_full_year() -> pd.DataFrame:
    raw = C.raw_tmax_merged()  # old archive + 2022-2025 file (Step 2C rule); see 15_termination_common.py
    raw.loc[~raw["tmax"].between(*B.TMAX_RANGE), "tmax"] = np.nan
    raw.loc[raw["tmin"].notna() & (raw["tmax"] < raw["tmin"]), "tmax"] = np.nan
    return raw[["station_uid", "date", "tmax"]]


def station_thresholds(raw: pd.DataFrame, stations: list[str]) -> pd.DataFrame:
    base = raw[raw["station_uid"].isin(stations) & raw["date"].dt.year.between(C.BASELINE_START, C.BASELINE_END)
               & raw["tmax"].notna()].copy()
    base["cd"] = cal366(base["date"])
    # 5-SD screen on the baseline sample (same rule as Step 16B)
    st = base.groupby(["station_uid", "cd"])["tmax"].agg(["mean", "std"])
    base = base.merge(st, left_on=["station_uid", "cd"], right_index=True)
    base = base[((base["tmax"] - base["mean"]) / base["std"]).abs().fillna(0) <= B.TMAX_CLIM_Z]
    base["year"] = base["date"].dt.year
    djf_cds = sorted(set(cal366(pd.Series(pd.date_range("2000-12-01", "2000-12-31")))) |
                     set(range(1, 61)))  # 1 Jan-29 Feb and December
    rows = []
    years = list(range(C.BASELINE_START, C.BASELINE_END + 1))
    for sid, g in base.groupby("station_uid"):
        cd = g["cd"].to_numpy()
        vals = g["tmax"].to_numpy()
        yrs = g["year"].to_numpy()
        for target in djf_cds:
            dist = np.abs(((cd - target + 183) % 366) - 183)
            m = dist <= HALF_WINDOW
            v, y = vals[m], yrs[m]
            if len(v) < 100:
                continue
            rec = {"station_uid": sid, "cd": target, "n": len(v),
                   "p10_full": np.quantile(v, P, method="median_unbiased")}
            for yr in years:
                vv = v[y != yr]
                rec[f"p10_loyo_{yr}"] = np.quantile(vv, P, method="median_unbiased") if len(vv) >= 96 else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def find_runs(flags: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Runs of consecutive calendar days flagged True, at least MIN_DAYS long."""
    runs, run = [], []
    for d, f in flags.sort_index().items():
        if f and run and (d - run[-1]).days == 1:
            run.append(d)
        else:
            if len(run) >= MIN_DAYS:
                runs.append((run[0], run[-1], len(run)))
            run = [d] if f else []
    if len(run) >= MIN_DAYS:
        runs.append((run[0], run[-1], len(run)))
    return runs


def main() -> None:
    C.ensure_dirs()
    C.write_policy("step16f_cold_day_spells_policy.json", POLICY)
    weights = pd.read_csv(C.STATION_WEIGHTS)
    wcol = "national_area_weight" if "national_area_weight" in weights.columns else \
        [c for c in weights.columns if "weight" in c][0]
    w = weights.set_index("station_uid")[wcol]

    st = B.load_station_temperatures()
    st, _ = B.qc_tmax(st)
    st = st[st["winter_start_year"] <= LAST_WINTER]
    stations = sorted(st["station_uid"].unique())

    thr = station_thresholds(raw_tmax_full_year(), stations)
    thr.to_csv(THRESHOLDS, index=False, float_format="%.3f")

    st["cd"] = cal366(st["date"])
    st = st.merge(thr, on=["station_uid", "cd"], how="left")
    in_base = st["date"].dt.year.between(C.BASELINE_START, C.BASELINE_END)
    loyo = np.full(len(st), np.nan)
    for yr in range(C.BASELINE_START, C.BASELINE_END + 1):
        m = (st["date"].dt.year == yr).to_numpy()
        loyo[m] = st.loc[m, f"p10_loyo_{yr}"].to_numpy()
    st["threshold"] = np.where(in_base, loyo, st["p10_full"])
    st["obs"] = st["tmax"].notna() & st["threshold"].notna()
    st["cold"] = st["obs"] & (st["tmax"] < st["threshold"])
    st["w"] = st["station_uid"].map(w).fillna(0.0)

    day = st.groupby("date").apply(lambda g: pd.Series({
        "observed_area": g.loc[g["obs"], "w"].sum(), "observed_stations": int(g["obs"].sum()),
        "cold_area": g.loc[g["cold"], "w"].sum()}), include_groups=False)
    day["valid"] = (day["observed_area"] >= MIN_AREA_OBS) & (day["observed_stations"] >= MIN_STATIONS_OBS)
    day["national_cold_day"] = day["valid"] & (day["cold_area"] >= MIN_COLD_AREA)
    day["winter_start_year"] = np.where(day.index.month == 12, day.index.year, day.index.year - 1)
    day.to_csv(DAILY, float_format="%.4f")

    ev = []
    for wy, g in day.groupby("winter_start_year"):
        for s, e, n in find_runs(g["national_cold_day"]):
            sub = g.loc[s:e]
            ev.append({"event_id": f"BDTX10A_{wy}_{len([x for x in ev if x['winter_start_year'] == wy]) + 1:02d}",
                       "winter_start_year": wy, "start_date": s.date(), "end_date": e.date(), "duration_days": n,
                       "peak_cold_area": sub["cold_area"].max(), "mean_cold_area": sub["cold_area"].mean()})
    tx = pd.DataFrame(ev)
    tx["start_date"] = pd.to_datetime(tx["start_date"])
    tx["end_date"] = pd.to_datetime(tx["end_date"])

    tn = C.load_events()
    tn = tn[tn["winter_start_year"] <= LAST_WINTER].copy()
    days_of = lambda df: {d for r in df.itertuples() for d in pd.date_range(r.start_date, r.end_date)}
    tn_days, tx_days = days_of(tn), days_of(tx)
    tx["overlaps_cold_night_spell"] = [bool(set(pd.date_range(r.start_date, r.end_date)) & tn_days)
                                       for r in tx.itertuples()]
    tn["overlaps_cold_day_spell"] = [bool(set(pd.date_range(r.start_date, r.end_date)) & tx_days)
                                     for r in tn.itertuples()]
    tx.to_csv(CATALOGUE, index=False, float_format="%.4f")

    winters = pd.DataFrame({"winter_start_year": range(1985, LAST_WINTER + 1)})
    winters["cold_night_spells"] = winters["winter_start_year"].map(tn.groupby("winter_start_year").size()).fillna(0).astype(int)
    winters["cold_day_spells"] = winters["winter_start_year"].map(tx.groupby("winter_start_year").size()).fillna(0).astype(int)
    winters["cold_night_days"] = winters["winter_start_year"].map(tn.groupby("winter_start_year")["duration_days"].sum()).fillna(0).astype(int)
    winters["cold_day_days"] = winters["winter_start_year"].map(tx.groupby("winter_start_year")["duration_days"].sum()).fillna(0).astype(int)
    winters.to_csv(T_WINTER, index=False)

    cov = pd.read_csv(C.DAILY_COVARIATES, parse_dates=["date"]).set_index("date")
    rows = []
    for label, df, pure in (("cold_night", tn, ~tn["overlaps_cold_day_spell"]),
                            ("cold_day", tx, ~tx["overlaps_cold_night_spell"])):
        for r, is_pure in zip(df.itertuples(), pure):
            m = cov.loc[r.start_date:r.end_date, CONTRAST_VARS].mean()
            rows.append({"event_id": r.event_id, "type": label, "pure": bool(is_pure),
                         "winter": r.winter_start_year, "duration": r.duration_days, **m.to_dict()})
    EVM = pd.DataFrame(rows)

    types = EVM.groupby(["type", "pure"]).agg(events=("event_id", "count"), mean_duration=("duration", "mean"),
                                              **{f"mean_{v}": (v, "mean") for v in CONTRAST_VARS[:3]}).reset_index()
    types.to_csv(T_TYPES, index=False, float_format="%.3f")

    P_ = EVM[EVM["pure"]]
    res = []
    for v in CONTRAST_VARS:
        r = C.group_difference_bootstrap(P_[v].to_numpy(float), (P_["type"] == "cold_day").to_numpy(),
                                         (P_["type"] == "cold_night").to_numpy(), P_["winter"].to_numpy(),
                                         N_BOOT, SEED)
        res.append({"variable": v, "cold_day_mean": P_.loc[P_["type"] == "cold_day", v].mean(),
                    "cold_night_mean": P_.loc[P_["type"] == "cold_night", v].mean(), **r})
    T = pd.DataFrame(res)
    T["q_fdr"] = C.benjamini_hochberg(T["p_bootstrap"].to_numpy())
    T.to_csv(T_CONTRAST, index=False, float_format="%.4f")

    trends = {}
    for col in ("cold_night_spells", "cold_day_spells", "cold_night_days", "cold_day_days"):
        r = spearmanr(winters["winter_start_year"], winters[col])
        slope = np.polyfit(winters["winter_start_year"], winters[col], 1)[0] * 10
        trends[col] = {"spearman_rho": float(r[0]), "p_value": float(r[1]), "ols_slope_per_decade": float(slope)}
    jan24 = tx[(tx["start_date"] <= "2024-01-31") & (tx["end_date"] >= "2024-01-01")]
    C.write_checksums("step16f_output_sha256.txt", [THRESHOLDS, DAILY, CATALOGUE, T_WINTER, T_TYPES, T_CONTRAST])
    summ = {"step": "Step 16F", "completed_utc": C.now_utc(), "cold_day_spells": int(len(tx)),
            "cold_night_spells_same_period": int(len(tn)),
            "cold_day_spells_overlapping_cold_night": int(tx["overlaps_cold_night_spell"].sum()),
            "cold_night_spells_overlapping_cold_day": int(tn["overlaps_cold_day_spell"].sum()),
            "shared_days": len(tn_days & tx_days), "cold_night_days": len(tn_days), "cold_day_days": len(tx_days),
            "trends_per_winter": trends,
            "january_2024_cold_day_spells": jan24[["start_date", "end_date", "duration_days"]].astype(str).to_dict("records"),
            "invalid_national_days": int((~day["valid"]).sum())}
    SUMMARY.write_text(json.dumps(summ, indent=2, default=float))
    lines = ["Step 16F - Cold-day spell catalogue", "=" * 36] + [f"{k}: {v}" for k, v in summ.items()] + [
        "", "Duration distribution (cold-day spells):", tx["duration_days"].value_counts().sort_index().to_string(),
        "", "Spell types:", types.round(2).to_string(index=False),
        "", "Pure cold-day minus pure cold-night (event means):",
        T[["variable", "cold_day_mean", "cold_night_mean", "difference", "ci_low", "ci_high", "q_fdr"]].round(3).to_string(index=False)]
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
