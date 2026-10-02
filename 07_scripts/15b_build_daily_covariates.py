"""
Step 16B - Tmax quality control and the daily DJF covariate dataset.

Builds one row per DJF day (1985/86-2024/25) containing:
  * national area-weighted Tmin, Tmax and diurnal-range (DTR) anomalies
    (Tmin = Step 6C cleaned analysis series; Tmax = raw BMD Tmax after the QC below);
  * national p10 cold-area fraction and winter day index (Step 7D);
  * ERA5 Bangladesh- and South Asia-box anomalies and day-to-day tendencies (Step 16A);
  * remote-driver indices: Siberian High, Ural blocking, NAM proxies (10, 100 hPa) and RRWP.
    RRWP R uses a 15-day CENTRED filter (Step 10E), so same-day R contains information from
    up to 7 days later. Causal versions are added: R lagged by 7 days and a trailing 3-day
    mean of the standardized instantaneous wave-packet envelope.

Tmax QC (applied to the 26 primary-network stations, DJF only)
  1. Missing codes ('****', blank) -> missing.
  2. Physical range: Tmax outside 0-40 degC -> rejected.
  3. Internal consistency: Tmax < cleaned Tmin on the same station-day -> rejected.
  4. Climatological outlier: |Tmax - calendar-day mean| > 5 SD (1991-2020, +/-7 days) -> rejected.
  5. Spatial check: station anomaly minus the median anomaly of all other stations > 10 degC in
     magnitude -> FLAGGED ONLY (fog is spatially patchy, so large local departures can be real).
Every rejected or flagged value is written to a decision log.

Run from the project root:
    python3 07_scripts/15b_build_daily_covariates.py 2>&1 | tee 09_logs/step16b_build_daily_covariates.log
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "term_common", Path(__file__).resolve().parent / "15_termination_common.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

TMAX_QC_LOG = C.TERM_QC / "step16b_tmax_qc_decisions.csv"
TMAX_AVAIL = C.TERM_QC / "step16b_tmax_availability_by_winter.csv"
REPORT = C.TERM_QC / "step16b_daily_covariates_report.txt"
SUMMARY = C.TERM_QC / "step16b_daily_covariates_summary.json"

ANOMALY_HALF_WINDOW = 7
MIN_OBSERVED_AREA = 0.80
TMAX_RANGE = (0.0, 40.0)
TMAX_CLIM_Z = 5.0
TMAX_SPATIAL_FLAG = 10.0
RRWP_SECTORS = ["eurasian_upstream", "central_asia", "bangladesh_sector", "asian_corridor"]

POLICY = {
    "step": "Step 16B",
    "title": "Tmax quality control and daily DJF covariate dataset",
    "tmin_source": "analysis_tmin_unadjusted from the Step 6C/7D cleaned DJF dataset",
    "tmax_source": ("raw BMD Tmax: MaxT_MinT_1981_2024_raw.xlsx (1981-2024) merged with "
                    "Daily_Maximum_Temperature_2022_2025_raw.csv (2022-2025) by the Step 2C rule used for Tmin "
                    "(from 2022 the 2022-2025 file is selected where it has a value, the old archive fills its gaps); "
                    "exact 0.0 degC in the 2022-2025 file is treated as missing"),
    "tmax_qc": {
        "physical_range_degC": TMAX_RANGE,
        "reject_if_below_cleaned_tmin": True,
        "climatological_outlier_sd": TMAX_CLIM_Z,
        "spatial_departure_flag_only_degC": TMAX_SPATIAL_FLAG,
    },
    "anomalies": f"calendar-day mean 1991-2020, +/-{ANOMALY_HALF_WINDOW}-day window; station temperature "
                 "climatologies use DJF data only, so windows are truncated at 1 Dec and 28 Feb",
    "national_mean": f"area-weighted with Step 7D Voronoi weights; missing if observed area < {MIN_OBSERVED_AREA}",
    "era5": "Step 16A 00 UTC box means; anomalies as above; tendency = anomaly(day) - anomaly(day-1)",
    "rrwp_causality": "same-day R is centred (+/-7 days) and flagged non-causal; causal versions are R(t-7) "
                      "and the trailing 3-day mean of the standardized instantaneous envelope",
}


def load_station_temperatures() -> pd.DataFrame:
    cols = ["station_uid", "date", "winter_start_year", "analysis_tmin_unadjusted", "national_area_weight"]
    tmin = C.read_table(C.CLEAN_DJF, columns=cols)
    tmin["date"] = pd.to_datetime(tmin["date"])
    tmin = tmin.rename(columns={"analysis_tmin_unadjusted": "tmin"})

    raw = C.raw_tmax_merged()[["station_uid", "date", "tmax", "tmax_source"]]
    return tmin.merge(raw, on=["station_uid", "date"], how="left")


def qc_tmax(st: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    st = st.copy()
    log = []

    def reject(mask, reason):
        for r in st.loc[mask, ["station_uid", "date", "tmax", "tmin"]].itertuples(index=False):
            log.append({"station_uid": r.station_uid, "date": r.date.date(), "tmax": r.tmax,
                        "tmin": r.tmin, "decision": "rejected", "reason": reason})
        st.loc[mask, "tmax"] = np.nan

    reject(st["tmax"].notna() & ~st["tmax"].between(*TMAX_RANGE), "physical_range")
    reject(st["tmax"].notna() & st["tmin"].notna() & (st["tmax"] < st["tmin"]), "tmax_below_cleaned_tmin")

    # climatological outlier (station, calendar day, +/-7 days, 1991-2020)
    st["_cd"] = C.calendar_day(pd.DatetimeIndex(st["date"]))
    base = st[st["date"].dt.year.between(C.BASELINE_START, C.BASELINE_END) & st["tmax"].notna()]
    parts = []
    for off in range(-ANOMALY_HALF_WINDOW, ANOMALY_HALF_WINDOW + 1):
        b = base[["station_uid", "_cd", "tmax"]].copy()
        b["_cd"] = ((b["_cd"] - 1 - off) % 365) + 1
        parts.append(b)
    clim = pd.concat(parts).groupby(["station_uid", "_cd"])["tmax"].agg(["mean", "std"])
    st = st.merge(clim, left_on=["station_uid", "_cd"], right_index=True, how="left")
    z = (st["tmax"] - st["mean"]) / st["std"]
    reject(z.abs() > TMAX_CLIM_Z, "climatological_outlier_5sd")

    # spatial departure (flag only)
    anom = st["tmax"] - st["mean"]
    st["anom_tmp"] = anom
    day_n = st.groupby("date")["anom_tmp"].transform("count")
    st["others_median_tmp"] = st.groupby("date")["anom_tmp"].transform("median")  # median robust to one station
    flag = st["anom_tmp"].notna() & (day_n >= 10) & ((st["anom_tmp"] - st["others_median_tmp"]).abs() > TMAX_SPATIAL_FLAG)
    for r in st.loc[flag, ["station_uid", "date", "tmax", "tmin", "anom_tmp", "others_median_tmp"]].itertuples(index=False):
        log.append({"station_uid": r.station_uid, "date": r.date.date(), "tmax": r.tmax, "tmin": r.tmin,
                    "decision": "flagged_retained",
                    "reason": f"spatial_departure {r.anom_tmp - r.others_median_tmp:+.1f} degC"})
    st = st.drop(columns=["_cd", "mean", "std", "anom_tmp", "others_median_tmp"])
    return st, pd.DataFrame(log)


def national_anomalies(st: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for var in ("tmin", "tmax"):
        s = st[["station_uid", "date", var]].set_index("date")
        a = C.calendar_anomalies(s, ANOMALY_HALF_WINDOW, group_col="station_uid")
        frames.append(a.rename(columns={var: f"{var}_anom"}).reset_index())
    anom = frames[0].merge(frames[1], on=["date", "station_uid"])
    anom = anom.merge(st[["station_uid", "date", "national_area_weight"]].drop_duplicates(),
                      on=["station_uid", "date"])
    anom["dtr_anom"] = anom["tmax_anom"] - anom["tmin_anom"]
    out = {}
    for var in ("tmin_anom", "tmax_anom", "dtr_anom"):
        ok = anom[var].notna()
        w = anom["national_area_weight"].where(ok, 0.0)
        num = (anom[var].fillna(0.0) * w).groupby(anom["date"]).sum()
        den = w.groupby(anom["date"]).sum()
        out[f"nat_{var}"] = (num / den).where(den >= MIN_OBSERVED_AREA)
        out[f"nat_{var}_observed_area"] = den
    return pd.DataFrame(out)


def era5_covariates() -> pd.DataFrame:
    e = pd.read_csv(C.ERA5_BOX_DAILY, parse_dates=["date"]).set_index("date")
    d = pd.DataFrame(index=e.index)
    for b in ("bd", "sa"):
        d[f"{b}_t2m"] = e[f"{b}_t2m_c"]
        d[f"{b}_t850"] = e[f"{b}_air_temperature_k_850"] - 273.15
        d[f"{b}_stab_t2m_minus_t850"] = d[f"{b}_t2m"] - d[f"{b}_t850"]
        d[f"{b}_thick_1000_500"] = e[f"{b}_geopotential_height_m_500"] - e[f"{b}_geopotential_height_m_1000"]
        d[f"{b}_z500"] = e[f"{b}_geopotential_height_m_500"]
        d[f"{b}_mslp"] = e[f"{b}_msl_hpa"]
        d[f"{b}_u850"] = e[f"{b}_u_wind_ms_850"]
        d[f"{b}_v850"] = e[f"{b}_v_wind_ms_850"]
        d[f"{b}_u200"] = e[f"{b}_u_wind_ms_200"]
        d[f"{b}_wspd10"] = e[f"{b}_wspd10_ms"]
    d.index.name = "date"
    anom = C.calendar_anomalies(d, ANOMALY_HALF_WINDOW)
    anom.columns = [f"era5_{c}_anom" for c in anom.columns]
    tend = anom.diff()
    tend.columns = [c.replace("_anom", "_tend") for c in tend.columns]
    return anom.join(tend)


def remote_indices() -> pd.DataFrame:
    sh = pd.read_csv(C.SH_BLOCK_TABLE, parse_dates=["date"]).set_index("date")
    st = pd.read_csv(C.STRAT_TABLE, parse_dates=["date"]).set_index("date")
    rr = pd.read_csv(C.RRWP_TABLE, parse_dates=["date"]).set_index("date").asfreq("D")
    out = pd.DataFrame(index=rr.index)
    out["siberian_high_std_anom"] = sh["shi_standardized_anomaly"]
    out["ural_blocking_fraction"] = sh["ural_blocking_fraction"]
    out["nam_proxy_10"] = st["nam_proxy_10"]
    out["nam_proxy_100"] = st["nam_proxy_100"]
    for s in RRWP_SECTORS:
        r = rr[f"{s}_r_standardized"]
        out[f"rrwp_{s}_R_centred_noncausal"] = r
        out[f"rrwp_{s}_R_lag7"] = r.shift(7)
        env = rr[f"{s}_instantaneous_envelope_mean_ms"].to_frame("env")
        env["_cd"] = C.calendar_day(env.index)
        base = env[(env.index.year >= C.BASELINE_START) & (env.index.year <= C.BASELINE_END)]
        parts = []
        for off in range(-ANOMALY_HALF_WINDOW, ANOMALY_HALF_WINDOW + 1):
            b = base.copy()
            b["_cd"] = ((b["_cd"] - 1 - off) % 365) + 1
            parts.append(b)
        clim = pd.concat(parts).groupby("_cd")["env"].agg(["mean", "std"])
        z = (env["env"].to_numpy() - clim["mean"].reindex(env["_cd"]).to_numpy()) / clim["std"].reindex(env["_cd"]).to_numpy()
        out[f"rrwp_{s}_envelope_z_trailing3"] = pd.Series(z, index=env.index).rolling(3).mean()
    return out


def main() -> None:
    C.ensure_dirs()
    C.write_policy("step16b_daily_covariates_policy.json", POLICY)

    st = load_station_temperatures()
    raw_tmax_n = int(st["tmax"].notna().sum())
    tmax_by_source = st.loc[st["tmax"].notna(), "tmax_source"].value_counts().to_dict()
    st, qc_log = qc_tmax(st)
    qc_log.to_csv(TMAX_QC_LOG, index=False)
    avail = st.groupby("winter_start_year")["tmax"].apply(lambda x: x.notna().mean()).rename("tmax_available_fraction")
    avail.to_csv(TMAX_AVAIL)

    nat = national_anomalies(st)
    area = C.read_table(C.DAILY_AREA_DIAG, columns=["date", "winter_start_year", "winter_day_index",
                                                    "p10_cold_national_area_fraction", "area_weighted_event_id"])
    area["date"] = pd.to_datetime(area["date"])
    area = area.set_index("date")

    cov = area.join(nat).join(era5_covariates()).join(remote_indices())
    cov.index.name = "date"
    cov.to_csv(C.DAILY_COVARIATES, float_format="%.5f")

    n_reject = int((qc_log["decision"] == "rejected").sum()) if len(qc_log) else 0
    n_flag = int((qc_log["decision"] == "flagged_retained").sum()) if len(qc_log) else 0
    summary = {
        "step": "Step 16B", "completed_utc": C.now_utc(),
        "rows": int(len(cov)), "columns": int(cov.shape[1]),
        "first_date": str(cov.index.min().date()), "last_date": str(cov.index.max().date()),
        "tmax_values_before_qc": raw_tmax_n, "tmax_values_by_source_djf": tmax_by_source, "tmax_rejected": n_reject, "tmax_flagged_retained": n_flag,
        "tmax_rejection_reasons": qc_log.loc[qc_log["decision"] == "rejected", "reason"].value_counts().to_dict()
        if len(qc_log) else {},
        "national_tmax_missing_days": int(cov["nat_tmax_anom"].isna().sum()),
        "national_tmin_missing_days": int(cov["nat_tmin_anom"].isna().sum()),
        "era5_missing_values": int(cov.filter(like="era5_").drop(columns=cov.filter(like="_tend").columns)
                                   .isna().sum().sum()),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    C.write_checksums("step16b_output_sha256.txt", [C.DAILY_COVARIATES, TMAX_QC_LOG, TMAX_AVAIL])

    lines = ["Step 16B - Tmax QC and daily covariate dataset", "=" * 48]
    lines += [f"{k}: {v}" for k, v in summary.items()]
    lines += ["", "Tmax availability by winter (last 5):",
              avail.tail(5).round(3).to_string()]
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
