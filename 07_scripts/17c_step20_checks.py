"""
17c_step20_checks.py  -  Step 20 check requested by the third pre-submission review (October 2026)

Final row of the two season-end spells. A spell whose last cold day is the last day of February is right-
censored: whether that day was its last is decided by the Tmin of 1 March, which lies outside the analysed
season. The primary risk set codes that final row as y = 0 (no termination observed). In a discrete-time
hazard model the cleaner treatment is to omit a row whose outcome is unobserved. This script refits the
primary-family (39) and the lagged (66, all families) hazard models with those two final rows removed
(258 rows, 83 terminations) and compares them with the published estimates.

Run from the project root (about 5-15 minutes):
    python 07_scripts/17c_step20_checks.py 2>&1 | tee 09_logs/step20_17c.log

Outputs
    08_outputs/tables/revision/table_R18_season_end_final_row_dropped.csv
    05_qc_reports/revision/step20a_checks.txt
"""
from __future__ import annotations

import importlib.util
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
W = _load("step16w", "15w_prenight_hazard_and_precipitation.py")
A = _load("step17a", "16a_revision_hazard_validation.py")

ROOT = C.PROJECT_ROOT
OUT_T = ROOT / "08_outputs" / "tables" / "revision"
OUT_Q = ROOT / "05_qc_reports" / "revision"
ADV_NW, JET = "mech_nigp_adv850_k_day_anom", "mech_jet200_core_speed_70_100e_anom"


def drop_final_censored(rs: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    ids = C.season_end_censored(ev)
    end = ev.set_index("event_id")["end_date"]
    last = rs["event_id"].isin(ids) & (pd.to_datetime(rs["date"]) == rs["event_id"].map(end))
    return rs.loc[~last].copy()


def main() -> None:
    OUT_T.mkdir(parents=True, exist_ok=True)
    OUT_Q.mkdir(parents=True, exist_ok=True)
    ev = C.load_events()
    cov = W.load_covariates()
    t91 = pd.read_csv(C.TERM_TABLES / "table_91_termination_hazard_odds_ratios.csv")
    t91 = t91[t91["family"] != "noncausal_diagnostic"]
    r3 = pd.read_csv(OUT_T / "table_R3_one_day_lead_all_families.csv")
    r11 = pd.read_csv(OUT_T / "table_R11_lead_advection_robustness.csv")

    rows = []
    rs0 = drop_final_censored(E.risk_set(cov, ev, 3), ev)
    for _, r in t91.iterrows():
        o = E.odds_ratio(rs0, r.predictor)
        rows.append({"analysis": "same day, primary family", "family": r.family, "predictor": r.predictor, **o,
                     "published_or": r.odds_ratio_per_sd, "published_q": r.q_fdr_causal})
        print("  same day", r.predictor, flush=True)
    S = pd.DataFrame(rows)
    S["q"] = C.benjamini_hochberg(S["p_bootstrap"].to_numpy())

    rows = []
    rs1 = drop_final_censored(E.risk_set(cov, ev, 3, lead=1), ev)
    for _, r in r3.iterrows():
        if r.predictor not in rs1.columns:
            continue
        o = E.odds_ratio(rs1, r.predictor, A.N_BOOT_OR)
        rows.append({"analysis": "predictor lag 1 day, all families", "family": r.family, "predictor": r.predictor,
                     **o, "published_or": r.odds_ratio_per_sd, "published_q": r.q_fdr_all})
        print("  lag 1", r.predictor, flush=True)
    L = pd.DataFrame(rows)
    L["q"] = C.benjamini_hochberg(L["p_bootstrap"].to_numpy())

    rows = []
    rs2 = drop_final_censored(E.risk_set(cov, ev, 3, lead=2), ev)
    for p in (ADV_NW, JET):
        pub = r11[(r11.analysis.astype(str).str.startswith("lead 2")) & (r11.predictor == p)]
        o = E.odds_ratio(rs2, p, A.N_BOOT_OR)
        rows.append({"analysis": "predictor lag 2 days", "family": "advection_jet", "predictor": p, **o,
                     "published_or": pub.odds_ratio_per_sd.iloc[0] if len(pub) else np.nan, "published_q": np.nan,
                     "q": np.nan})
    T = pd.concat([S, L, pd.DataFrame(rows)], ignore_index=True)
    T["ratio_to_published"] = T["odds_ratio_per_sd"] / T["published_or"]
    T.to_csv(OUT_T / "table_R18_season_end_final_row_dropped.csv", index=False, float_format="%.4f")

    lines = ["Step 20 check (17c): final rows of the two season-end spells removed", "=" * 66,
             f"Risk set: {len(rs0)} rows, {int(rs0['terminates'].sum())} terminations (published: 260 rows, 83)",
             "",
             f"Same-day primary family: q < 0.05 now {S.loc[S.q < 0.05, 'predictor'].tolist()}; "
             f"published {t91.loc[t91.q_fdr_causal < 0.05, 'predictor'].tolist()}",
             f"Lag 1, all 66: q < 0.05 now {L.loc[L.q < 0.05, 'predictor'].tolist()}; "
             f"published {r3.loc[r3.q_fdr_all < 0.05, 'predictor'].tolist()}",
             f"Largest relative change in any odds ratio: {np.nanmax(np.abs(np.log(T['ratio_to_published']))):.4f} (log scale)",
             "", "Headline estimates:",
             T[T.predictor.isin([ADV_NW, JET, "era5_bd_wspd10_anom", "era5_bd_stab_t2m_minus_t850_tend",
                                 "era5_sa_z500_tend", "nat_tmax_anom"])]
             [["analysis", "predictor", "odds_ratio_per_sd", "ci_low", "ci_high", "q", "published_or",
               "published_q", "n_rows", "n_terminations", "n_boot_valid", "n_boot_nonconverged"]]
             .round(4).to_string(index=False)]
    (OUT_Q / "step20a_checks.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
