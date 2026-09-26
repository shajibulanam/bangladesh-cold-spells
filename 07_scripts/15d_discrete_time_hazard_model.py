"""
Step 16D - Discrete-time hazard (survival) model of cold-spell termination.

Question
    Given that a cold spell is ongoing on event day k, which conditions on day k are associated
    with day k being the LAST event day?

Risk set
    Every event day k >= 3 (events last at least 3 days by definition, so termination is only
    possible from day 3). Outcome = 1 if day k is the final event day, else 0.

Models
    One logistic model per candidate predictor (standardized, odds ratio per 1 SD), each adjusted for:
      log(event age k), winter day index (seasonal timing), same-day national Tmin anomaly
      (the current cold-spell state) and winter start year (long-term change).
    Winter-block bootstrap (resampling winters, implemented as integer sample weights) gives
    95% CIs and two-sided bootstrap p-values. Benjamini-Hochberg FDR across all CAUSAL predictors.
    Same-day centred RRWP R values use information from up to 7 days later; they are reported
    as non-causal diagnostics only and are excluded from the FDR family.

Run from the project root:
    python3 07_scripts/15d_discrete_time_hazard_model.py 2>&1 | tee 09_logs/step16d_hazard_model.log
"""
from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

_spec = importlib.util.spec_from_file_location(
    "term_common", Path(__file__).resolve().parent / "15_termination_common.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

RISK_SET = C.TERM_INTERMEDIATE / "step16d_hazard_risk_set.csv"
TABLE_HAZARD = C.TERM_TABLES / "table_91_termination_hazard_odds_ratios.csv"
TABLE_BASE = C.TERM_TABLES / "table_92_termination_hazard_base_model.csv"
REPORT = C.TERM_QC / "step16d_hazard_model_report.txt"
SUMMARY = C.TERM_QC / "step16d_hazard_model_summary.json"

N_BOOT = 2000
SEED = 20260923
MIN_DURATION_AT_RISK = 3
ADJUSTMENT = ["log_event_age", "winter_day_index", "nat_tmin_anom", "winter_start_year"]

FAMILIES = {
    "station_temperature": ["nat_tmax_anom", "nat_dtr_anom", "p10_cold_national_area_fraction"],
    "era5_state": [
        "era5_bd_t850_anom", "era5_bd_stab_t2m_minus_t850_anom", "era5_bd_thick_1000_500_anom",
        "era5_bd_z500_anom", "era5_bd_mslp_anom", "era5_bd_u850_anom", "era5_bd_v850_anom",
        "era5_bd_wspd10_anom", "era5_sa_t850_anom", "era5_sa_z500_anom", "era5_sa_mslp_anom",
        "era5_sa_u850_anom", "era5_sa_v850_anom",
    ],
    "era5_tendency": [
        "era5_bd_t850_tend", "era5_bd_stab_t2m_minus_t850_tend", "era5_bd_z500_tend", "era5_bd_mslp_tend",
        "era5_bd_u850_tend", "era5_bd_v850_tend", "era5_bd_wspd10_tend",
        "era5_sa_t850_tend", "era5_sa_z500_tend", "era5_sa_mslp_tend", "era5_sa_v850_tend",
    ],
    "remote_causal": [
        "siberian_high_std_anom", "ural_blocking_fraction", "nam_proxy_10", "nam_proxy_100",
        "rrwp_eurasian_upstream_R_lag7", "rrwp_central_asia_R_lag7", "rrwp_bangladesh_sector_R_lag7",
        "rrwp_asian_corridor_R_lag7", "rrwp_eurasian_upstream_envelope_z_trailing3",
        "rrwp_central_asia_envelope_z_trailing3", "rrwp_bangladesh_sector_envelope_z_trailing3",
        "rrwp_asian_corridor_envelope_z_trailing3",
    ],
    "noncausal_diagnostic": [
        "rrwp_eurasian_upstream_R_centred_noncausal", "rrwp_central_asia_R_centred_noncausal",
        "rrwp_bangladesh_sector_R_centred_noncausal", "rrwp_asian_corridor_R_centred_noncausal",
    ],
}

POLICY = {
    "step": "Step 16D",
    "title": "Discrete-time hazard model of cold-spell termination",
    "risk_set": f"event days k >= {MIN_DURATION_AT_RISK}; outcome = day k is the final event day",
    "adjustment": ADJUSTMENT,
    "predictor_families": FAMILIES,
    "standardization": "predictors and adjustment covariates z-scored on the full risk set",
    "bootstrap": {"unit": "winter", "repetitions": N_BOOT, "seed": SEED,
                  "implementation": "integer sample weights = number of times each winter is drawn"},
    "multiple_testing": "Benjamini-Hochberg across all causal predictors (families except noncausal_diagnostic)",
    "input": "Step 16B daily covariate dataset",
}


def build_risk_set(cov: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for e in ev.itertuples(index=False):
        for k in range(MIN_DURATION_AT_RISK, e.duration_days + 1):
            rows.append({"event_id": e.event_id, "winter_start_year": e.winter_start_year,
                         "group": e.group, "event_age_k": k, "log_event_age": np.log(k),
                         "date": e.start_date + pd.Timedelta(days=k - 1),
                         "terminates": int(k == e.duration_days)})
    rs = pd.DataFrame(rows)
    cov_cols = [c for c in cov.columns if c not in ("winter_start_year",)]
    return rs.merge(cov[cov_cols], left_on="date", right_index=True, how="left")


def fit_coef(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    m = LogisticRegression(penalty=None, max_iter=5000)
    m.fit(X, y, sample_weight=w)
    return m.coef_[0]


def main() -> None:
    C.ensure_dirs()
    C.write_policy("step16d_hazard_model_policy.json", POLICY)
    cov = pd.read_csv(C.DAILY_COVARIATES, parse_dates=["date"]).set_index("date")
    ev = C.load_events()
    rs = build_risk_set(cov, ev)
    rs.to_csv(RISK_SET, index=False, float_format="%.5f")

    uniq, counts = C.winter_bootstrap_counts(rs["winter_start_year"].to_numpy(), N_BOOT, SEED)

    def analyse(cols: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
        d = rs.dropna(subset=cols + ["terminates"])
        X = ((d[cols] - d[cols].mean()) / d[cols].std()).to_numpy()
        y = d["terminates"].to_numpy()
        est = fit_coef(X, y)
        widx = np.searchsorted(uniq, d["winter_start_year"].to_numpy())
        boots = []
        for b in range(N_BOOT):
            w = counts[b, widx].astype(float)
            keep = w > 0
            if y[keep].min() == y[keep].max():
                continue
            boots.append(fit_coef(X[keep], y[keep], w[keep]))
        return est, np.array(boots), len(d)

    # Base model
    est, boots, n = analyse(ADJUSTMENT)
    base = pd.DataFrame({"term": ADJUSTMENT, "odds_ratio_per_sd": np.exp(est),
                         "ci_low": np.exp(np.percentile(boots, 2.5, axis=0)),
                         "ci_high": np.exp(np.percentile(boots, 97.5, axis=0))})
    base["n_rows"] = n
    base.to_csv(TABLE_BASE, index=False, float_format="%.4f")

    results = []
    for family, preds in FAMILIES.items():
        for p in preds:
            if p not in rs.columns:
                print(f"WARNING: predictor {p} not found; skipped")
                continue
            est, boots, n = analyse([p] + ADJUSTMENT)
            b = boots[:, 0]
            results.append({
                "family": family, "predictor": p, "n_rows": n,
                "n_terminations": int(rs.dropna(subset=[p] + ADJUSTMENT)["terminates"].sum()),
                "odds_ratio_per_sd": np.exp(est[0]),
                "ci_low": np.exp(np.percentile(b, 2.5)), "ci_high": np.exp(np.percentile(b, 97.5)),
                "p_bootstrap": min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())),
                "n_boot_valid": len(b),
            })
            print(f"{family:22s} {p:48s} OR={np.exp(est[0]):.2f}")
    T = pd.DataFrame(results)
    causal = T["family"] != "noncausal_diagnostic"
    T["q_fdr_causal"] = np.nan
    T.loc[causal, "q_fdr_causal"] = C.benjamini_hochberg(T.loc[causal, "p_bootstrap"].to_numpy())
    T["ci_excludes_one"] = (T["ci_low"] > 1) | (T["ci_high"] < 1)
    T["fdr_significant"] = T["q_fdr_causal"] < 0.05
    T.to_csv(TABLE_HAZARD, index=False, float_format="%.4f")

    summary = {"step": "Step 16D", "completed_utc": C.now_utc(),
               "risk_set_rows": int(len(rs)), "terminations": int(rs["terminates"].sum()),
               "events": int(rs["event_id"].nunique()),
               "causal_predictors_tested": int(causal.sum()),
               "fdr_significant": T.loc[T["fdr_significant"], ["predictor", "odds_ratio_per_sd"]].to_dict("records")}
    SUMMARY.write_text(json.dumps(summary, indent=2, default=float))
    C.write_checksums("step16d_output_sha256.txt", [RISK_SET, TABLE_HAZARD, TABLE_BASE])

    lines = ["Step 16D - Discrete-time hazard model of termination", "=" * 52,
             f"Risk-set rows: {len(rs)}; terminations: {int(rs['terminates'].sum())}; "
             f"adjustment: {', '.join(ADJUSTMENT)}", "",
             "Base model:", base.round(3).to_string(index=False), "",
             T[["family", "predictor", "odds_ratio_per_sd", "ci_low", "ci_high", "p_bootstrap", "q_fdr_causal"]]
             .round(3).to_string(index=False)]
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
