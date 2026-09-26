"""
Step 16C - Leakage-free persistent-versus-short comparisons.

Why this step exists
    Short events last exactly 3 days (days 0-2). Comparing groups at lags +3 or +5 compares
    ended events with ongoing ones, so the difference is built into the event definition.
    Here groups are compared only in windows where their event status is identical:
      pre_onset   : days -3 to -1 relative to onset   (both not yet started)
      onset_0_2   : days 0 to 2 relative to onset      (both ongoing)
      last_day    : the final event day                (both on their last day)
      post_end    : days +1 to +2 after the final day  (both just ended)

Method
    Event-level window means; persistent (>=6 days) minus short (3 days); winter-block bootstrap
    (5,000 resamples of winters); two-sided bootstrap p; Benjamini-Hochberg FDR within each window.
    Also writes onset- and end-aligned lag composites (group means) for figures.

Run from the project root:
    python3 07_scripts/15c_leakage_free_persistence_tests.py 2>&1 | tee 09_logs/step16c_leakage_free_tests.log
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

TABLE_TESTS = C.TERM_TABLES / "table_89_leakage_free_persistence_tests.csv"
TABLE_LAGS = C.TERM_TABLES / "table_90_onset_end_aligned_lag_composites.csv"
REPORT = C.TERM_QC / "step16c_leakage_free_tests_report.txt"
SUMMARY = C.TERM_QC / "step16c_leakage_free_tests_summary.json"

N_BOOT = 5000
SEED = 20260923
WINDOWS = {
    "pre_onset": ("onset", -3, -1),
    "onset_0_2": ("onset", 0, 2),
    "last_day": ("end", 0, 0),
    "post_end": ("end", 1, 2),
}
VARIABLES = [
    "nat_tmin_anom", "nat_tmax_anom", "nat_dtr_anom",
    "era5_bd_t2m_anom", "era5_bd_t850_anom", "era5_bd_stab_t2m_minus_t850_anom",
    "era5_bd_thick_1000_500_anom", "era5_bd_z500_anom", "era5_bd_mslp_anom",
    "era5_bd_u850_anom", "era5_bd_v850_anom", "era5_bd_wspd10_anom",
    "era5_sa_t850_anom", "era5_sa_stab_t2m_minus_t850_anom", "era5_sa_z500_anom",
    "era5_sa_mslp_anom", "era5_sa_u850_anom", "era5_sa_v850_anom",
]
ONSET_LAGS = range(-5, 11)
END_LAGS = range(-5, 6)

POLICY = {
    "step": "Step 16C",
    "title": "Leakage-free persistent-versus-short comparisons",
    "groups": "short = 3 days, persistent >= 6 days (Step 9H frozen groups); intermediate excluded from contrasts",
    "windows": WINDOWS,
    "variables": VARIABLES,
    "bootstrap": {"unit": "winter", "repetitions": N_BOOT, "seed": SEED},
    "multiple_testing": "Benjamini-Hochberg FDR within each window across all variables",
    "input": "Step 16B daily covariate dataset",
}


def aligned_rows(cov: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for e in ev.itertuples(index=False):
        for align, ref, lags in (("onset", e.start_date, ONSET_LAGS), ("end", e.end_date, END_LAGS)):
            for lag in lags:
                d = ref + pd.Timedelta(days=lag)
                if d not in cov.index:
                    continue
                rec = {"event_id": e.event_id, "group": e.group, "winter": e.winter_start_year,
                       "duration": e.duration_days, "align": align, "lag": lag}
                rec.update(cov.loc[d, VARIABLES].to_dict())
                rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    C.ensure_dirs()
    C.write_policy("step16c_leakage_free_tests_policy.json", POLICY)
    cov = pd.read_csv(C.DAILY_COVARIATES, parse_dates=["date"]).set_index("date")
    ev = C.load_events()
    L = aligned_rows(cov, ev)

    results = []
    for wname, (align, a, b) in WINDOWS.items():
        sub = L[(L["align"] == align) & L["lag"].between(a, b) & L["group"].isin(["short", "persistent"])]
        evm = sub.groupby(["event_id", "group", "winter"])[VARIABLES].mean().reset_index()
        is_p = (evm["group"] == "persistent").to_numpy()
        is_s = (evm["group"] == "short").to_numpy()
        winters = evm["winter"].to_numpy()
        for v in VARIABLES:
            r = C.group_difference_bootstrap(evm[v].to_numpy(dtype=float), is_p, is_s, winters, N_BOOT, SEED)
            results.append({"window": wname, "variable": v,
                            "short_mean": evm.loc[is_s, v].mean(), "persistent_mean": evm.loc[is_p, v].mean(),
                            **r})
    T = pd.DataFrame(results)
    T["q_fdr_within_window"] = np.nan
    for w in WINDOWS:
        m = T["window"] == w
        T.loc[m, "q_fdr_within_window"] = C.benjamini_hochberg(T.loc[m, "p_bootstrap"].to_numpy())
    T["ci_excludes_zero"] = (T["ci_low"] > 0) | (T["ci_high"] < 0)
    T["fdr_significant"] = T["q_fdr_within_window"] < 0.05
    T.to_csv(TABLE_TESTS, index=False, float_format="%.4f")

    lagcomp = (L.groupby(["align", "group", "lag"])[VARIABLES].agg(["mean", "count"]))
    lagcomp.columns = [f"{v}_{s}" for v, s in lagcomp.columns]
    lagcomp.reset_index().to_csv(TABLE_LAGS, index=False, float_format="%.4f")

    n_fdr = int(T["fdr_significant"].sum())
    n_ci = int(T["ci_excludes_zero"].sum())
    summary = {"step": "Step 16C", "completed_utc": C.now_utc(), "tests": int(len(T)),
               "ci_excludes_zero": n_ci, "fdr_significant": n_fdr,
               "fdr_significant_results": T.loc[T["fdr_significant"], ["window", "variable", "difference"]]
               .to_dict("records")}
    SUMMARY.write_text(json.dumps(summary, indent=2, default=float))
    C.write_checksums("step16c_output_sha256.txt", [TABLE_TESTS, TABLE_LAGS])

    show = T[["window", "variable", "difference", "ci_low", "ci_high", "p_bootstrap", "q_fdr_within_window"]]
    lines = ["Step 16C - Leakage-free persistent-minus-short comparisons", "=" * 58,
             f"Tests: {len(T)}; CI excludes zero: {n_ci}; FDR q<0.05: {n_fdr}", "",
             show.round(3).to_string(index=False)]
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
