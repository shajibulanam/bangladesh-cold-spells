"""
Step 16M - Do remote and large-scale drivers control cold-spell duration?

Part A - Leakage-free duration tests. For every driver, event means in three windows where event
status is identical for all spells (pre10: days -10..-1, pre5: -5..-1, onset_0_2: days 0..2) are
related to duration in two ways: persistent (>=6 d) minus short (3 d) difference, and Spearman
rank correlation with duration over all 85 spells. Both use a winter-block bootstrap (5,000);
Benjamini-Hochberg FDR is applied within each window and statistic.
RRWP enters only in causal form (R lagged 7 days; trailing 3-day envelope).

Part B - Mechanism extension of the Step 16D hazard model. The 850-hPa advection and jet
diagnostics from Step 16J (anomalies and day-to-day tendencies) are tested as a SEPARATE,
second predictor family with the same adjustment set; FDR is applied within this family only,
so the primary Step 16D family and its conclusions are unchanged.

Run from the project root:
    python3 07_scripts/15m_remote_driver_duration_tests.py 2>&1 | tee 09_logs/step16m_remote_drivers.log
"""
from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("term_common", "15_termination_common.py")
E = _load("step16e", "15e_hazard_robustness_and_trend.py")

T_DUR = C.TERM_TABLES / "table_105_remote_driver_duration_tests.csv"
T_HAZ = C.TERM_TABLES / "table_106_mechanism_extension_hazard.csv"
REPORT = C.TERM_QC / "step16m_remote_drivers_report.txt"
N_BOOT, SEED, N_BOOT_HAZ = 5000, 20260923, 2000

WINDOWS = {"pre10": (-10, -1), "pre5": (-5, -1), "onset_0_2": (0, 2)}
DRIVERS = [
    "siberian_high_std_anom", "ural_blocking_fraction_anom", "mech_nigp_adv850_k_day_anom",
    "mech_bd_adv850_k_day_anom", "era5_bd_v850_anom", "era5_sa_z500_anom",
    "mech_jet200_core_lat_70_100e_anom", "mech_jet200_core_speed_70_100e_anom",
    "rrwp_asian_corridor_R_lag7", "rrwp_central_asia_R_lag7", "rrwp_eurasian_upstream_R_lag7",
    "rrwp_asian_corridor_envelope_z_trailing3", "nam_proxy_100", "nam_proxy_10",
]
MECH_FAMILY = [
    "mech_bd_adv850_k_day_anom", "mech_nigp_adv850_k_day_anom",
    "mech_jet200_core_lat_70_100e_anom", "mech_jet200_core_speed_70_100e_anom",
    "mech_bd_adv850_k_day_tend", "mech_nigp_adv850_k_day_tend",
    "mech_jet200_core_lat_70_100e_tend", "mech_jet200_core_speed_70_100e_tend",
]


def spearman_boot(x: np.ndarray, y: np.ndarray, winters: np.ndarray) -> tuple[float, float, float, float]:
    ok = ~np.isnan(x)
    x, y, winters = x[ok], y[ok], winters[ok]
    rho = spearmanr(x, y)[0]
    uniq, counts = C.winter_bootstrap_counts(winters, N_BOOT, SEED)
    widx = np.searchsorted(uniq, winters)
    out = []
    for b in range(N_BOOT):
        rep = np.repeat(np.arange(len(x)), counts[b, widx])
        if len(np.unique(y[rep])) > 1:
            out.append(spearmanr(x[rep], y[rep])[0])
    out = np.array(out)
    p = min(1.0, 2 * min((out <= 0).mean(), (out >= 0).mean()))
    return rho, np.percentile(out, 2.5), np.percentile(out, 97.5), p


def main() -> None:
    C.ensure_dirs()
    C.write_policy("step16m_remote_drivers_policy.json", {
        "step": "Step 16M", "title": "Remote-driver duration tests and mechanism-extended hazard",
        "windows": WINDOWS, "drivers": DRIVERS, "mechanism_family": MECH_FAMILY,
        "duration_statistics": ["persistent minus short difference", "Spearman rho with duration (all 85)"],
        "bootstrap": {"unit": "winter", "repetitions": N_BOOT, "hazard_repetitions": N_BOOT_HAZ, "seed": SEED},
        "multiple_testing": "BH within window x statistic (Part A); within the mechanism family (Part B)",
        "hazard_adjustment": E.ADJ})
    cov = C.load_covariates_with_mechanisms()
    cov["ural_blocking_fraction_anom"] = C.calendar_anomalies(cov[["ural_blocking_fraction"]], 7)[
        "ural_blocking_fraction"]
    ev = C.load_events()

    rows = []
    for w, (a, b) in WINDOWS.items():
        vals = []
        for e in ev.itertuples(index=False):
            d = pd.date_range(e.start_date + pd.Timedelta(days=a), e.start_date + pd.Timedelta(days=b))
            vals.append(cov.reindex(d)[DRIVERS].mean())
        M = pd.DataFrame(vals).reset_index(drop=True)
        is_p = (ev["group"] == "persistent").to_numpy()
        is_s = (ev["group"] == "short").to_numpy()
        winters = ev["winter_start_year"].to_numpy()
        dur = ev["duration_days"].to_numpy(float)
        for v in DRIVERS:
            x = M[v].to_numpy(float)
            g = C.group_difference_bootstrap(x, is_p, is_s, winters, N_BOOT, SEED)
            rho, lo, hi, p = spearman_boot(x, dur, winters)
            rows.append({"window": w, "driver": v, "persistent_minus_short": g["difference"],
                         "diff_ci_low": g["ci_low"], "diff_ci_high": g["ci_high"], "diff_p": g["p_bootstrap"],
                         "spearman_rho": rho, "rho_ci_low": lo, "rho_ci_high": hi, "rho_p": p})
        print("done window", w)
    A = pd.DataFrame(rows)
    for stat in ("diff", "rho"):
        A[f"{stat}_q"] = np.nan
        for w in WINDOWS:
            m = A["window"] == w
            A.loc[m, f"{stat}_q"] = C.benjamini_hochberg(A.loc[m, f"{stat}_p"].to_numpy())
    A.to_csv(T_DUR, index=False, float_format="%.4f")

    rs = E.risk_set(cov, ev, 3)
    E.N_BOOT = N_BOOT_HAZ
    H = pd.DataFrame([{"predictor": p, **E.odds_ratio(rs, p, N_BOOT_HAZ)} for p in MECH_FAMILY])
    H["q_fdr_family"] = C.benjamini_hochberg(H["p_bootstrap"].to_numpy())
    H.to_csv(T_HAZ, index=False, float_format="%.4f")

    C.write_checksums("step16m_output_sha256.txt", [T_DUR, T_HAZ])
    (C.TERM_QC / "step16m_remote_drivers_summary.json").write_text(json.dumps({
        "completed_utc": C.now_utc(),
        "duration_tests_fdr_significant": int(((A["diff_q"] < 0.05) | (A["rho_q"] < 0.05)).sum()),
        "mechanism_family_fdr_significant": H.loc[H["q_fdr_family"] < 0.05, "predictor"].tolist()}, indent=2))
    text = ("Step 16M - Remote-driver duration tests\n" + "=" * 40 + "\n" +
            A[["window", "driver", "persistent_minus_short", "diff_q", "spearman_rho", "rho_ci_low",
               "rho_ci_high", "rho_q"]].round(3).to_string(index=False) +
            "\n\nMechanism-extension hazard family (adjusted as Step 16D)\n" +
            H[["predictor", "odds_ratio_per_sd", "ci_low", "ci_high", "p_bootstrap", "q_fdr_family"]]
            .round(3).to_string(index=False) + "\n")
    REPORT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
