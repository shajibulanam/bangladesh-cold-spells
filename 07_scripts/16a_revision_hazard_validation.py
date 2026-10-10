"""
16a_revision_hazard_validation.py - revision analyses R1-R3 for the termination hazard model.

R1  Nested validation. Predictor selection, standardisation and fitting are repeated inside every training
    fold, so held-out winters never influence which predictors enter the model.
      - leave-one-winter-out (LOWO) and the two split-period directions (1985/86-2004/05 vs 2005/06-2024/25)
      - selection rule inside a fold: likelihood-ratio test of each of the 39 primary causal predictors
        (adjusted for log event age, day of season, same-day Tmin anomaly and winter start year),
        Benjamini-Hochberg q < 0.05 across the 39, DTR excluded (algebraically redundant with Tmax given Tmin),
        and a candidate is skipped if |r| > 0.8 with an already selected, stronger predictor
      - scores on held-out rows: AUC, Brier score, Brier skill against the TRAINING-fold event fraction,
        calibration slope (logistic regression of the outcome on the predicted log-odds) and observed/expected
      - winter-block bootstrap of held-out rows for the AUC and Brier differences
      - selection frequency of every predictor across folds
R2  Season-boundary censoring. Spells ending on the last DJF day cannot be known to terminate there; their
    last row is coded as censored (no termination). Variant B also drops spells starting on 1 December
    (left-truncated: their true age is unknown). All 39 primary predictors are re-estimated.
R3  One-day lead for every family: primary (39), advection and jet (8), pre-night cloud/humidity/radiation (14)
    and terminating-night (5, which at a one-day lead describe the previous night). Odds ratios, winter-block
    bootstrap intervals, and Benjamini-Hochberg q within each family and across all 66.

Outputs: 08_outputs/tables/revision/table_R1*.csv ... table_R3*.csv and 05_qc_reports/revision/step17a_report.txt
Run from the project root:
    python3 07_scripts/16a_revision_hazard_validation.py 2>&1 | tee 09_logs/step17a_revision_hazard.log
"""
from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("term_common", "15_termination_common.py")
D = _load("step16d", "15d_discrete_time_hazard_model.py")
E = _load("step16e", "15e_hazard_robustness_and_trend.py")
W = _load("step16w", "15w_prenight_hazard_and_precipitation.py")

OUT_T = C.PROJECT_ROOT / "08_outputs" / "tables" / "revision"
OUT_Q = C.PROJECT_ROOT / "05_qc_reports" / "revision"
ADJ = E.ADJ
PRIMARY = [p for fam, ps in D.FAMILIES.items() if fam != "noncausal_diagnostic" for p in ps]
ADVECTION = ["mech_bd_adv850_k_day_anom", "mech_nigp_adv850_k_day_anom", "mech_jet200_core_lat_70_100e_anom",
             "mech_jet200_core_speed_70_100e_anom", "mech_bd_adv850_k_day_tend", "mech_nigp_adv850_k_day_tend",
             "mech_jet200_core_lat_70_100e_tend", "mech_jet200_core_speed_70_100e_tend"]
FAMILIES_LEAD = {"primary": PRIMARY, "advection_jet": ADVECTION, "prenight": W.PRENIGHT, "night": W.NIGHT}
SPLIT = 2004
Q_SELECT, R_MAX = 0.05, 0.8
N_BOOT, N_BOOT_OR, SEED = 2000, 1000, 20261003
POLICY = {
    "step": "Step 17A (revision)", "adjustment": ADJ, "primary_candidates": len(PRIMARY),
    "nested_selection": f"inside each training fold: LR test per candidate, BH q < {Q_SELECT} across the 39, "
                        f"DTR excluded, skip |r| > {R_MAX} with a stronger selected predictor; training-only "
                        "standardisation",
    "validation": ["leave-one-winter-out", "train 1985/86-2004/05 test 2005/06-2024/25", "reverse"],
    "scores": ["AUC", "Brier", "Brier skill vs training event fraction", "calibration slope", "observed/expected"],
    "censoring": "last row of spells ending on the last DJF day coded censored; variant B drops 1 December starts",
    "lead": "all predictors on day k-1; adjustment (day of season, same-day Tmin anomaly, age, year) unchanged",
    "bootstrap": {"unit": "winter", "auc_brier": N_BOOT, "odds_ratios": N_BOOT_OR, "seed": SEED},
}


# ----------------------------------------------------------------------------- helpers
def zscore(train: pd.DataFrame, other: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = train[cols].mean(), train[cols].std().replace(0, 1)
    return ((train[cols] - mu) / sd).to_numpy(), ((other[cols] - mu) / sd).to_numpy()


def loglik(model, X: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def select_in_fold(tr: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    rows = []
    for p in PRIMARY:
        d = tr.dropna(subset=[p])
        if d[p].std() == 0 or len(d) < 30:
            continue
        yy = d["terminates"].to_numpy()
        Xa_d, _ = zscore(d, d, ADJ)
        Xf, _ = zscore(d, d, [p] + ADJ)
        m = E.fit(Xf, yy)
        lr = 2 * (loglik(m, Xf, yy) - loglik(E.fit(Xa_d, yy), Xa_d, yy))
        rows.append({"predictor": p, "coef": m.coef_[0][0], "p_lr": float(chi2.sf(max(lr, 0), 1))})
    R = pd.DataFrame(rows)
    R["q"] = C.benjamini_hochberg(R["p_lr"].to_numpy())
    chosen = []
    for p in R.sort_values("p_lr").query("q < @Q_SELECT")["predictor"]:
        if p == "nat_dtr_anom":
            continue
        if any(abs(tr[[p, s]].corr().iloc[0, 1]) > R_MAX for s in chosen):
            continue
        chosen.append(p)
    return chosen, R


def calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1))
    slope = E.fit(lp.reshape(-1, 1), y).coef_[0][0]
    return float(slope), float(y.sum() / p.sum())


def scores(y, p_adj, p_full, clim, winters) -> dict:
    out = {"auc_adjustment_only": roc_auc_score(y, p_adj), "auc_nested_model": roc_auc_score(y, p_full),
           "brier_adjustment_only": brier_score_loss(y, p_adj), "brier_nested_model": brier_score_loss(y, p_full),
           "brier_training_climatology": brier_score_loss(y, clim)}
    out["auc_difference"] = out["auc_nested_model"] - out["auc_adjustment_only"]
    out["brier_skill_vs_training_climatology"] = 1 - out["brier_nested_model"] / out["brier_training_climatology"]
    out["calibration_slope"], out["observed_over_expected"] = calibration(y, p_full)
    uw = np.unique(winters)
    rng = np.random.default_rng(SEED)
    da, db = [], []
    for _ in range(N_BOOT):
        idx = np.concatenate([np.flatnonzero(winters == w) for w in rng.choice(uw, len(uw))])
        if y[idx].min() == y[idx].max():
            continue
        da.append(roc_auc_score(y[idx], p_full[idx]) - roc_auc_score(y[idx], p_adj[idx]))
        db.append(brier_score_loss(y[idx], p_full[idx]) - brier_score_loss(y[idx], p_adj[idx]))
    out.update({"auc_diff_ci_low": np.percentile(da, 2.5), "auc_diff_ci_high": np.percentile(da, 97.5),
                "brier_diff_full_minus_adj": out["brier_nested_model"] - out["brier_adjustment_only"],
                "brier_diff_ci_low": np.percentile(db, 2.5), "brier_diff_ci_high": np.percentile(db, 97.5)})
    return out


def predict(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str]) -> np.ndarray:
    tr = tr.dropna(subset=cols)
    Xtr, Xte = zscore(tr, te, cols)
    return E.fit(Xtr, tr["terminates"].to_numpy()).predict_proba(np.nan_to_num(Xte))[:, 1]


# ----------------------------------------------------------------------------- R1 nested validation
def r1_nested(rs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = rs.dropna(subset=ADJ).reset_index(drop=True)
    results, picks = [], []
    # leave-one-winter-out
    p_adj, p_full, clim = np.full(len(d), np.nan), np.full(len(d), np.nan), np.full(len(d), np.nan)
    for w in sorted(d["winter_start_year"].unique()):
        te = d["winter_start_year"] == w
        tr = d[~te]
        chosen, _ = select_in_fold(tr)
        picks.append({"scheme": "lowo", "held_out": int(w), "selected": ";".join(chosen), "n_selected": len(chosen)})
        cols = chosen + ADJ
        te_rows = d[te]
        p_adj[te.to_numpy()] = predict(tr, te_rows, ADJ)
        p_full[te.to_numpy()] = predict(tr, te_rows, cols) if chosen else p_adj[te.to_numpy()]
        clim[te.to_numpy()] = tr["terminates"].mean()
    ok = ~np.isnan(p_full)
    results.append({"scheme": "leave-one-winter-out (nested)", "test_rows": int(ok.sum()),
                    "test_terminations": int(d.loc[ok, "terminates"].sum()),
                    **scores(d.loc[ok, "terminates"].to_numpy(), p_adj[ok], p_full[ok], clim[ok],
                             d.loc[ok, "winter_start_year"].to_numpy())})
    # split period
    for early in (True, False):
        tr_mask = (d["winter_start_year"] <= SPLIT) if early else (d["winter_start_year"] > SPLIT)
        tr, te = d[tr_mask], d[~tr_mask]
        chosen, _ = select_in_fold(tr)
        picks.append({"scheme": "split_train_early" if early else "split_train_late", "held_out": -1,
                      "selected": ";".join(chosen), "n_selected": len(chosen)})
        pa = predict(tr, te, ADJ)
        pf = predict(tr, te, chosen + ADJ) if chosen else pa
        results.append({"scheme": ("train 1985/86-2004/05, test 2005/06-2024/25 (nested)" if early else
                                   "train 2005/06-2024/25, test 1985/86-2004/05 (nested)"),
                        "test_rows": len(te), "test_terminations": int(te["terminates"].sum()),
                        "selected_predictors": ";".join(chosen),
                        **scores(te["terminates"].to_numpy(), pa, pf, np.full(len(te), tr["terminates"].mean()),
                                 te["winter_start_year"].to_numpy())})
    P = pd.DataFrame(picks)
    lowo = P[P["scheme"] == "lowo"]
    freq = pd.Series([p for s in lowo["selected"] for p in s.split(";") if p]).value_counts()
    F = pd.DataFrame({"predictor": freq.index, "folds_selected": freq.values,
                      "fraction_of_lowo_folds": freq.values / len(lowo)})
    return pd.DataFrame(results), pd.concat([P, F.assign(scheme="selection_frequency")], ignore_index=True)


# ----------------------------------------------------------------------------- R2 censoring
def censored_risk_set(cov: pd.DataFrame, ev: pd.DataFrame, drop_dec1: bool) -> tuple[pd.DataFrame, dict]:
    ev = ev.copy()
    ev["start_date"], ev["end_date"] = pd.to_datetime(ev["start_date"]), pd.to_datetime(ev["end_date"])
    season_end = ev["end_date"].apply(lambda d: d.month == 2 and (d + pd.Timedelta(days=1)).month == 3)
    dec1 = (ev["start_date"].dt.month == 12) & (ev["start_date"].dt.day == 1)
    info = {"right_censored_spells": ev.loc[season_end, "event_id"].tolist(),
            "left_truncated_spells": ev.loc[dec1, "event_id"].tolist()}
    cens = set(ev.loc[season_end, "event_id"])
    if drop_dec1:
        ev = ev[~dec1]
    rs = E.risk_set(cov, ev, 3, censor=True)
    return rs, info


def r2_censoring(cov: pd.DataFrame, ev: pd.DataFrame, rs_primary: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # Step 18: season-end censoring is now the primary definition; the uncensored model (primary up to
    # v1.2.0) is sensitivity A
    variants = {"primary (season-end spells censored)": rs_primary}
    _, info = censored_risk_set(cov, ev, False)
    rsB, _ = censored_risk_set(cov, ev, True)
    variants["A: not censored (v1.2.0 primary)"] = E.risk_set(cov, ev, 3, censor=False)
    variants["B: censored, 1 December starts removed"] = rsB
    rows = []
    for name, rs in variants.items():
        part = []
        for p in PRIMARY:
            if rs[p].notna().sum() < 30:
                continue
            part.append({"analysis": name, "predictor": p, **E.odds_ratio(rs, p, N_BOOT_OR)})
        Pt = pd.DataFrame(part)
        Pt["q_fdr"] = C.benjamini_hochberg(Pt["p_bootstrap"].to_numpy())
        rows.append(Pt)
        print(f"R2 {name}: {int(rs['terminates'].sum())} terminations, {len(rs)} rows; "
              f"q<0.05: {Pt.loc[Pt['q_fdr'] < 0.05, 'predictor'].tolist()}", flush=True)
    return pd.concat(rows, ignore_index=True), info


# ----------------------------------------------------------------------------- R3 lead for all families
def r3_lead(cov: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    rs = E.risk_set(cov, ev, 3, lead=1)
    rows = []
    for fam, preds in FAMILIES_LEAD.items():
        for p in preds:
            if p not in rs.columns or rs[p].notna().sum() < 30:
                print(f"R3 WARNING: {p} unavailable; skipped")
                continue
            rows.append({"family": fam, "predictor": p, **E.odds_ratio(rs, p, N_BOOT_OR)})
    L = pd.DataFrame(rows)
    L["q_fdr_family"] = np.nan
    for fam in L["family"].unique():
        m = L["family"] == fam
        L.loc[m, "q_fdr_family"] = C.benjamini_hochberg(L.loc[m, "p_bootstrap"].to_numpy())
    L["q_fdr_all"] = C.benjamini_hochberg(L["p_bootstrap"].to_numpy())
    L["ci_excludes_one"] = (L["ci_low"] > 1) | (L["ci_high"] < 1)
    return L


def main() -> None:
    OUT_T.mkdir(parents=True, exist_ok=True)
    OUT_Q.mkdir(parents=True, exist_ok=True)
    C.write_policy("step17a_revision_hazard_policy.json", POLICY)
    cov = W.load_covariates()
    ev = C.load_events()
    rs = E.risk_set(cov, ev, 3)
    print(f"Risk set: {len(rs)} rows, {int(rs['terminates'].sum())} terminations", flush=True)

    R1, P1 = r1_nested(rs)
    R1.to_csv(OUT_T / "table_R1_nested_validation.csv", index=False, float_format="%.4f")
    P1.to_csv(OUT_T / "table_R1b_nested_selection.csv", index=False, float_format="%.3f")
    print(R1[["scheme", "auc_adjustment_only", "auc_nested_model", "auc_difference", "auc_diff_ci_low",
              "auc_diff_ci_high", "brier_skill_vs_training_climatology", "calibration_slope",
              "observed_over_expected"]].round(3).to_string(index=False), flush=True)

    R2, info = r2_censoring(cov, ev, rs)
    R2.to_csv(OUT_T / "table_R2_censoring_sensitivity.csv", index=False, float_format="%.4f")

    R3 = r3_lead(cov, ev)
    R3.to_csv(OUT_T / "table_R3_one_day_lead_all_families.csv", index=False, float_format="%.4f")

    sel = P1[P1["scheme"] == "selection_frequency"][["predictor", "folds_selected", "fraction_of_lowo_folds"]]
    piv = R2.pivot_table(index="predictor", columns="analysis", values="odds_ratio_per_sd")
    qpiv = R2.pivot_table(index="predictor", columns="analysis", values="q_fdr")
    key = qpiv[(qpiv < 0.10).any(axis=1)].index
    lines = ["Step 17A - revision analyses R1-R3", "=" * 40,
             f"Risk set {len(rs)} rows, {int(rs['terminates'].sum())} terminations", "",
             "R1 Nested validation (selection repeated inside each training fold)",
             R1.drop(columns=[c for c in R1.columns if c.startswith("brier_diff")]).round(3).to_string(index=False),
             "", "LOWO selection frequency:", sel.round(2).to_string(index=False), "",
             f"R2 Censoring: right-censored {info['right_censored_spells']}; left-truncated {info['left_truncated_spells']}",
             "Odds ratios for predictors with q < 0.10 in any variant:", piv.loc[key].round(2).to_string(),
             "q values:", qpiv.loc[key].round(3).to_string(), "",
             "R3 One-day lead: predictors with nominal 95% interval excluding 1",
             R3[R3["ci_excludes_one"]][["family", "predictor", "odds_ratio_per_sd", "ci_low", "ci_high",
                                         "q_fdr_family", "q_fdr_all"]].round(3).to_string(index=False) or "none",
             f"Predictors with q_fdr_family < 0.05 at lead: {R3.loc[R3['q_fdr_family'] < 0.05, 'predictor'].tolist()}",
             f"Predictors with q_fdr_all < 0.05 at lead: {R3.loc[R3['q_fdr_all'] < 0.05, 'predictor'].tolist()}"]
    (OUT_Q / "step17a_report.txt").write_text("\n".join(lines) + "\n")
    (OUT_Q / "step17a_summary.json").write_text(json.dumps({"completed_utc": C.now_utc(), **info,
                                                            "lead_tested": len(R3)}, indent=2))
    print("\n".join(lines[-6:]))


if __name__ == "__main__":
    main()
