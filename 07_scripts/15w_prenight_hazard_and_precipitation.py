"""
Step 16W - Analyses 7A and 7B.

7A  Termination hazard with cloud, humidity and radiation measured BEFORE the terminating night.
    Pre-night family (FDR within family):
        day-window (00-11 UTC = 06-17 LT) anomalies of downward longwave, downward solar, total cloud,
        2-m dewpoint, cloud-base <1 km and <300 m fractions, boundary-layer height (Step 16V);
        dawn (00 UTC) snapshots of total cloud, cloud-base <1 km, cloud-base <300 m and dewpoint
        depression (Step 16K3);
        day-to-day tendencies of the day-window longwave, dewpoint and total cloud.
    Night family (reported for comparison only; overlaps the terminating night, FDR within family):
        night-window (12-23 UTC) anomalies of longwave, total cloud, dewpoint, cloud-base fractions.
    Adjustment and inference exactly as Step 16D/16E (log event age, season day, Tmin anomaly, year;
    winter-block bootstrap, 2,000 resamples).

7B  Precipitation in pure cold-day versus pure cold-night spells (1985/86-2024/25):
    event-mean UTC-day precipitation, fraction of spell days with >= 1 mm, day-window precipitation;
    winter-block bootstrap (5,000). Sensitivity: the Table 1 contrast repeated using only DRY cold-day
    spells (event-mean precipitation < 1 mm/day).

Run from the project root (after Step 16V):
    python3 07_scripts/15w_prenight_hazard_and_precipitation.py 2>&1 | tee 09_logs/step16w_prenight_precip.log
"""
from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("term_common", "15_termination_common.py")
E = _load("step16e", "15e_hazard_robustness_and_trend.py")

WINDOWS = C.PROJECT_ROOT / "03_intermediate" / "era5_box_daily" / "era5_cloud_radiation_windows_oct_mar.csv"
DAILY_CLOUD = C.PROJECT_ROOT / "03_intermediate" / "era5_box_daily" / "era5_cloud_radiation_daily_oct_mar.csv"
TX_CATALOGUE = C.PROJECT_ROOT / "06_events" / "step16f_cold_day_spell_catalogue.csv"
T_HAZ = C.TERM_TABLES / "table_114_prenight_vs_night_hazard.csv"
T_PRECIP = C.TERM_TABLES / "table_115_precipitation_cold_day_vs_night.csv"
T_DRY = C.TERM_TABLES / "table_116_dry_cold_day_vs_cold_night_contrast.csv"
REPORT = C.TERM_QC / "step16w_prenight_precip_report.txt"
N_BOOT_HAZ, N_BOOT, SEED = 2000, 5000, 20260923
LAST_TMAX_WINTER = 2024
WET_MM = 1.0

PRENIGHT = ["win_strd_day_anom", "win_ssrd_day_anom", "win_tcc_day_anom", "win_td2m_day_anom",
            "win_low1000_day_anom", "win_fog300_day_anom", "win_blh_day_anom",
            "dly_tcc_00utc_anom", "dly_lowcloud1000_frac_00utc_anom", "dly_fog300_frac_00utc_anom",
            "dly_dpd_00utc_anom",
            "win_strd_day_tend", "win_td2m_day_tend", "win_tcc_day_tend"]
NIGHT = ["win_strd_night_anom", "win_tcc_night_anom", "win_td2m_night_anom",
         "win_low1000_night_anom", "win_fog300_night_anom"]
CONTRAST = ["nat_tmin_anom", "nat_tmax_anom", "nat_dtr_anom", "win_tcc_day_anom", "win_strd_day_anom",
            "win_ssrd_day_anom", "win_td2m_day_anom", "dly_lowcloud1000_frac_00utc_anom",
            "dly_fog300_frac_00utc_anom", "dly_dpd_00utc_anom"]


def load_covariates() -> pd.DataFrame:
    win = pd.read_csv(WINDOWS, parse_dates=["date"]).set_index("date")
    win = win.drop(columns=[c for c in win.columns if c.startswith("hours_")])
    win_anom = C.calendar_anomalies(win, 7)
    tend = win_anom[["strd_day", "td2m_day", "tcc_day"]].diff()
    tend.columns = [f"win_{c}_tend" for c in tend.columns]
    win_anom.columns = [f"win_{c}_anom" for c in win_anom.columns]
    dly = pd.read_csv(DAILY_CLOUD, parse_dates=["date"]).set_index("date")
    dly = dly[["tcc_00utc", "lowcloud1000_frac_00utc", "fog300_frac_00utc", "dpd_00utc"]]
    dly_anom = C.calendar_anomalies(dly, 7)
    dly_anom.columns = [f"dly_{c}_anom" for c in dly_anom.columns]
    raw_tp = win[[c for c in win.columns if c.startswith("tp_")]]
    return C.load_covariates_with_mechanisms().join(win_anom).join(tend).join(dly_anom).join(raw_tp)


def main() -> None:
    C.ensure_dirs()
    C.write_policy("step16w_prenight_precip_policy.json", {
        "step": "Step 16W", "prenight_family": PRENIGHT, "night_family_comparison_only": NIGHT,
        "hazard_adjustment": E.ADJ, "hazard_bootstrap": N_BOOT_HAZ, "bootstrap": N_BOOT, "seed": SEED,
        "wet_day_threshold_mm": WET_MM, "dry_cold_day_spell": f"event-mean UTC-day precipitation < {WET_MM} mm",
        "cold_day_period": f"1985/86-{LAST_TMAX_WINTER}/{str(LAST_TMAX_WINTER + 1)[-2:]}"})
    cov = load_covariates()
    ev = C.load_events()

    # 7A
    rs = E.risk_set(cov, ev, 3)
    rows = []
    for fam, preds in (("prenight", PRENIGHT), ("night_overlapping", NIGHT)):
        for p in preds:
            if p not in rs.columns or rs[p].isna().all():
                print(f"WARNING: {p} unavailable; skipped")
                continue
            rows.append({"family": fam, "predictor": p, **E.odds_ratio(rs, p, N_BOOT_HAZ)})
            print(f"{fam:18s} {p:40s} OR={rows[-1]['odds_ratio_per_sd']:.2f}", flush=True)
    H = pd.DataFrame(rows)
    H["q_fdr_family"] = np.nan
    for fam in H["family"].unique():
        m = H["family"] == fam
        H.loc[m, "q_fdr_family"] = C.benjamini_hochberg(H.loc[m, "p_bootstrap"].to_numpy())
    H.to_csv(T_HAZ, index=False, float_format="%.4f")

    # 7B
    precip_ok = "tp_utcday_mm" in cov.columns and cov["tp_utcday_mm"].notna().any()
    P = D = None
    if precip_ok:
        tx = pd.read_csv(TX_CATALOGUE, parse_dates=["start_date", "end_date"])
        tn = ev[ev["winter_start_year"] <= LAST_TMAX_WINTER]
        days = lambda df: {d for r in df.itertuples() for d in pd.date_range(r.start_date, r.end_date)}
        tn_days, tx_days = days(tn), days(tx)
        evm = []
        for kind, df, other in (("cold_day", tx, tn_days), ("cold_night", tn, tx_days)):
            for r in df.itertuples():
                rng_ = pd.date_range(r.start_date, r.end_date)
                if set(rng_) & other:
                    continue
                sub = cov.reindex(rng_)
                rec = {"type": kind, "winter": r.winter_start_year,
                       "tp_utcday_mm": sub["tp_utcday_mm"].mean(), "tp_day_mm": sub["tp_day_mm"].mean(),
                       "wet_day_fraction": (sub["tp_utcday_mm"] >= WET_MM).mean()}
                rec.update(sub[CONTRAST].mean().to_dict())
                evm.append(rec)
        EV = pd.DataFrame(evm)
        out = []
        for v in ("tp_utcday_mm", "tp_day_mm", "wet_day_fraction"):
            r = C.group_difference_bootstrap(EV[v].to_numpy(float), (EV["type"] == "cold_day").to_numpy(),
                                             (EV["type"] == "cold_night").to_numpy(), EV["winter"].to_numpy(), N_BOOT, SEED)
            out.append({"variable": v, "cold_day_mean": EV.loc[EV["type"] == "cold_day", v].mean(),
                        "cold_night_mean": EV.loc[EV["type"] == "cold_night", v].mean(), **r})
        P = pd.DataFrame(out)
        P["q_fdr"] = C.benjamini_hochberg(P["p_bootstrap"].to_numpy())
        n_cd = int((EV["type"] == "cold_day").sum())
        dry = EV[(EV["type"] == "cold_night") | ((EV["type"] == "cold_day") & (EV["tp_utcday_mm"] < WET_MM))]
        n_dry = int((dry["type"] == "cold_day").sum())
        P["pure_cold_day_spells"] = n_cd
        P["dry_cold_day_spells"] = n_dry
        P.to_csv(T_PRECIP, index=False, float_format="%.4f")
        out = []
        for v in CONTRAST:
            r = C.group_difference_bootstrap(dry[v].to_numpy(float), (dry["type"] == "cold_day").to_numpy(),
                                             (dry["type"] == "cold_night").to_numpy(), dry["winter"].to_numpy(), N_BOOT, SEED)
            out.append({"variable": v, "dry_cold_day_mean": dry.loc[dry["type"] == "cold_day", v].mean(),
                        "cold_night_mean": dry.loc[dry["type"] == "cold_night", v].mean(), **r})
        D = pd.DataFrame(out)
        D["q_fdr"] = C.benjamini_hochberg(D["p_bootstrap"].to_numpy())
        D.to_csv(T_DRY, index=False, float_format="%.4f")
    else:
        print("WARNING: precipitation not available - 7B skipped")

    paths = [T_HAZ] + ([T_PRECIP, T_DRY] if precip_ok else [])
    C.write_checksums("step16w_output_sha256.txt", paths)
    (C.TERM_QC / "step16w_summary.json").write_text(json.dumps({
        "completed_utc": C.now_utc(),
        "prenight_fdr_significant": H.loc[(H["family"] == "prenight") & (H["q_fdr_family"] < 0.05), "predictor"].tolist(),
        "precipitation_available": bool(precip_ok)}, indent=2))
    text = ("Step 16W - Pre-night hazard (7A) and precipitation check (7B)\n" + "=" * 60 + "\n7A. Hazard odds ratios\n" +
            H[["family", "predictor", "odds_ratio_per_sd", "ci_low", "ci_high", "p_bootstrap", "q_fdr_family"]]
            .round(3).to_string(index=False))
    if precip_ok:
        text += ("\n\n7B. Precipitation, pure cold-day minus pure cold-night\n" +
                 P.drop(columns=["n_boot_valid"]).round(3).to_string(index=False) +
                 "\n\n7B. Dry cold-day spells only (event-mean precipitation < 1 mm/day) vs cold-night\n" +
                 D[["variable", "dry_cold_day_mean", "cold_night_mean", "difference", "ci_low", "ci_high", "q_fdr", "n_a", "n_b"]]
                 .round(3).to_string(index=False))
    REPORT.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
