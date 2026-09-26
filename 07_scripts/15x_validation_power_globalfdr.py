"""
Step 16X - Analyses 7C, 7D and 7E.

7C  Out-of-sample (split-period) validation of the multivariable termination model of Step 16E.
    Predictors: the q < 0.05 causal predictors of Step 16D table_91 excluding nat_dtr_anom, plus the
    adjustment set. Fit on winters 1985/86-2004/05, test on 2005/06-2024/25, and the reverse.
    Scores on the held-out half: AUC and Brier score for adjustment-only vs adjustment + synoptic;
    95% CI of the AUC difference from a winter-block bootstrap of the held-out rows (2,000).

7D  Minimum detectable effects (80% power, two-sided alpha = 0.05) for the null results:
    - leakage-free group differences (Step 16C table_89): MDD = 2.80 x SE, with SE = CI width / 3.92;
      also expressed in units of the pooled between-event SD (standardized effect);
    - duration rank correlations (Step 16M table_105): minimum detectable |rho| = 2.80 x SE.

7E  One Benjamini-Hochberg correction across ALL causal hazard predictors from all families:
    primary (table_91), advection/jet (table_106), and cloud/radiation - either the UTC-day family
    (table_111) or, if present, the pre-night family (Step 16W table_114). Both variants are reported.

Run from the project root:
    python3 07_scripts/15x_validation_power_globalfdr.py 2>&1 | tee 09_logs/step16x_validation.log
"""
from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("term_common", "15_termination_common.py")
E = _load("step16e", "15e_hazard_robustness_and_trend.py")
LC = _load("step16c", "15c_leakage_free_persistence_tests.py")

T = C.TERM_TABLES
T_SPLIT = T / "table_117_split_period_validation.csv"
T_MDD = T / "table_118_minimum_detectable_differences.csv"
T_MDR = T / "table_119_minimum_detectable_correlations.csv"
T_GFDR = T / "table_120_global_fdr_all_families.csv"
REPORT = C.TERM_QC / "step16x_validation_report.txt"
N_BOOT, SEED = 2000, 20260923
Z80 = 1.959964 + 0.841621
SPLIT = 2004  # training winters <= 2004 (1985/86-2004/05)


def split_validation() -> pd.DataFrame:
    t91 = pd.read_csv(T / "table_91_termination_hazard_odds_ratios.csv")
    causal = t91[t91["family"] != "noncausal_diagnostic"]
    preds = [p for p in causal.loc[causal["q_fdr_causal"] < 0.05, "predictor"] if p != "nat_dtr_anom"]
    cov = pd.read_csv(C.DAILY_COVARIATES, parse_dates=["date"]).set_index("date")
    rs = E.risk_set(cov, C.load_events(), 3)
    cols_full = preds + E.ADJ
    d = rs.dropna(subset=cols_full).reset_index(drop=True)
    rows = []
    for train_early in (True, False):
        tr = (d["winter_start_year"] <= SPLIT) if train_early else (d["winter_start_year"] > SPLIT)
        te = ~tr
        y_tr, y_te = d.loc[tr, "terminates"].to_numpy(), d.loc[te, "terminates"].to_numpy()
        prob = {}
        for label, cols in (("adjustment_only", E.ADJ), ("adjustment_plus_synoptic", cols_full)):
            mu, sd = d.loc[tr, cols].mean(), d.loc[tr, cols].std()
            m = E.fit(((d.loc[tr, cols] - mu) / sd).to_numpy(), y_tr)
            prob[label] = m.predict_proba(((d.loc[te, cols] - mu) / sd).to_numpy())[:, 1]
        winters = d.loc[te, "winter_start_year"].to_numpy()
        uw = np.unique(winters)
        rng = np.random.default_rng(SEED)
        diffs = []
        for _ in range(N_BOOT):
            idx = np.concatenate([np.flatnonzero(winters == w) for w in rng.choice(uw, len(uw))])
            if y_te[idx].min() == y_te[idx].max():
                continue
            diffs.append(roc_auc_score(y_te[idx], prob["adjustment_plus_synoptic"][idx]) -
                         roc_auc_score(y_te[idx], prob["adjustment_only"][idx]))
        base = roc_auc_score(y_te, prob["adjustment_only"])
        full = roc_auc_score(y_te, prob["adjustment_plus_synoptic"])
        rows.append({
            "train": "1985/86-2004/05" if train_early else "2005/06-2024/25",
            "test": "2005/06-2024/25" if train_early else "1985/86-2004/05",
            "n_train_rows": int(tr.sum()), "n_test_rows": int(te.sum()), "test_terminations": int(y_te.sum()),
            "auc_adjustment_only": base, "auc_plus_synoptic": full, "auc_difference": full - base,
            "auc_diff_ci_low": np.percentile(diffs, 2.5), "auc_diff_ci_high": np.percentile(diffs, 97.5),
            "brier_adjustment_only": brier_score_loss(y_te, prob["adjustment_only"]),
            "brier_plus_synoptic": brier_score_loss(y_te, prob["adjustment_plus_synoptic"]),
            "brier_climatology": float(y_te.mean() * (1 - y_te.mean())),
            "synoptic_predictors": ";".join(preds)})
    return pd.DataFrame(rows)


def minimum_detectable() -> tuple[pd.DataFrame, pd.DataFrame]:
    t89 = pd.read_csv(T / "table_89_leakage_free_persistence_tests.csv")
    cov = pd.read_csv(C.DAILY_COVARIATES, parse_dates=["date"]).set_index("date")
    ev = C.load_events()
    L = LC.aligned_rows(cov, ev)
    sds = {}
    for w, (align, a, b) in LC.WINDOWS.items():
        sub = L[(L["align"] == align) & L["lag"].between(a, b) & L["group"].isin(["short", "persistent"])]
        evm = sub.groupby(["event_id", "group"])[LC.VARIABLES].mean().reset_index()
        for v in LC.VARIABLES:
            g = [evm.loc[evm["group"] == k, v].dropna() for k in ("short", "persistent")]
            pooled = np.sqrt(((len(g[0]) - 1) * g[0].var() + (len(g[1]) - 1) * g[1].var()) / (len(g[0]) + len(g[1]) - 2))
            sds[(w, v)] = pooled
    t89["se"] = (t89["ci_high"] - t89["ci_low"]) / 3.919928
    t89["mdd_80pct"] = Z80 * t89["se"]
    t89["pooled_between_event_sd"] = [sds.get((w, v), np.nan) for w, v in zip(t89["window"], t89["variable"])]
    t89["mdd_80pct_standardized"] = t89["mdd_80pct"] / t89["pooled_between_event_sd"]
    mdd = t89[["window", "variable", "difference", "ci_low", "ci_high", "se", "mdd_80pct",
               "pooled_between_event_sd", "mdd_80pct_standardized"]]
    t105 = pd.read_csv(T / "table_105_remote_driver_duration_tests.csv")
    t105["se_rho"] = (t105["rho_ci_high"] - t105["rho_ci_low"]) / 3.919928
    t105["min_detectable_abs_rho_80pct"] = Z80 * t105["se_rho"]
    mdr = t105[["window", "driver", "spearman_rho", "rho_ci_low", "rho_ci_high", "se_rho", "min_detectable_abs_rho_80pct"]]
    return mdd, mdr


def global_fdr() -> pd.DataFrame:
    parts = []
    t91 = pd.read_csv(T / "table_91_termination_hazard_odds_ratios.csv")
    parts.append(t91[t91["family"] != "noncausal_diagnostic"].assign(source="primary"))
    parts.append(pd.read_csv(T / "table_106_mechanism_extension_hazard.csv").assign(family="advection_jet", source="advection_jet"))
    cloud_day = pd.read_csv(T / "table_111_cloud_radiation_hazard.csv").assign(family="cloud_utc_day", source="cloud_utc_day")
    out = []
    variants = {"with_utc_day_cloud_family": cloud_day}
    t114 = T / "table_114_prenight_vs_night_hazard.csv"
    if t114.exists():
        pre = pd.read_csv(t114)
        variants["with_prenight_cloud_family"] = pre[pre["family"] == "prenight"].assign(source="cloud_prenight")
    for name, cloud in variants.items():
        allp = pd.concat(parts + [cloud], ignore_index=True)[["source", "predictor", "odds_ratio_per_sd", "ci_low",
                                                              "ci_high", "p_bootstrap"]]
        allp["q_global"] = C.benjamini_hochberg(allp["p_bootstrap"].to_numpy())
        allp["variant"] = name
        out.append(allp)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    C.ensure_dirs()
    C.write_policy("step16x_validation_policy.json", {
        "step": "Step 16X", "split_winter": SPLIT, "bootstrap": N_BOOT, "seed": SEED,
        "power": "80%, two-sided alpha 0.05, MDD = 2.80 x SE (SE from bootstrap CI width / 3.92)",
        "global_fdr": "Benjamini-Hochberg over all causal predictors of all families"})
    S = split_validation()
    S.to_csv(T_SPLIT, index=False, float_format="%.4f")
    mdd, mdr = minimum_detectable()
    mdd.to_csv(T_MDD, index=False, float_format="%.4f")
    mdr.to_csv(T_MDR, index=False, float_format="%.4f")
    G = global_fdr()
    G.to_csv(T_GFDR, index=False, float_format="%.4f")
    C.write_checksums("step16x_output_sha256.txt", [T_SPLIT, T_MDD, T_MDR, T_GFDR])
    sig = G[G["q_global"] < 0.05].groupby("variant")["predictor"].apply(list)
    text = ("Step 16X - Validation, power and global FDR\n" + "=" * 44 +
            "\n7C. Split-period validation\n" + S.drop(columns=["synoptic_predictors"]).round(3).to_string(index=False) +
            f"\nSynoptic predictors: {S['synoptic_predictors'].iloc[0]}" +
            "\n\n7D. Minimum detectable differences, days 0-2 window\n" +
            mdd[mdd["window"] == "onset_0_2"].round(3).to_string(index=False) +
            "\n\n7D. Median minimum detectable |rho| by window\n" +
            mdr.groupby("window")["min_detectable_abs_rho_80pct"].median().round(3).to_string() +
            "\n\n7E. Predictors with global q < 0.05\n" + sig.to_string() + "\n")
    REPORT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
