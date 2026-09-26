"""
Step 16E - Robustness of the termination hazard results and the duration trend.

Robustness predictor set (rule-based, read from Step 16D table_91):
    all CAUSAL predictors with q_fdr_causal < 0.10.

Parts
  A. Duration trend of the primary catalogue: Spearman rho and OLS slope (days per decade) with
     winter-block bootstrap CIs; decade summary.
  B. Event-definition sensitivity: hazard odds ratios re-estimated for every Step 7E sensitivity
     catalogue (10%/30% area, 4-day minimum, p05, one-day gap merge, BMD absolute threshold).
  C. Detrended covariates: every predictor and the Tmin state are linearly detrended over all DJF
     days before fitting.
  D. Leave-one-winter-out stability of the primary odds ratios (sign stability and range).
  E. One-day lead: predictors taken on day k-1 instead of day k.
  F. Multivariable model with all FDR-significant (q < 0.05) predictors, excluding nat_dtr_anom
     (algebraically redundant with Tmax once Tmin is adjusted for). Reports VIFs and
     leave-one-winter-out cross-validated AUC and Brier score versus the adjustment-only model.

Run from the project root:
    python3 07_scripts/15e_hazard_robustness_and_trend.py 2>&1 | tee 09_logs/step16e_hazard_robustness.log
"""
from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
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

SENS_CATALOGUE = C.PROJECT_ROOT / "06_events" / "step7e_sensitivity_event_catalogue.csv"
T_TREND = C.TERM_TABLES / "table_93_duration_trend.csv"
T_DECADE = C.TERM_TABLES / "table_94_duration_by_decade.csv"
T_ROBUST = C.TERM_TABLES / "table_95_hazard_robustness.csv"
T_LOWO = C.TERM_TABLES / "table_96_hazard_leave_one_winter_out.csv"
T_MULTI = C.TERM_TABLES / "table_97_hazard_multivariable_model.csv"
T_SKILL = C.TERM_TABLES / "table_98_hazard_cross_validated_skill.csv"
REPORT = C.TERM_QC / "step16e_hazard_robustness_report.txt"
SUMMARY = C.TERM_QC / "step16e_hazard_robustness_summary.json"

N_BOOT = 1000
SEED = 20260923
ROBUST_Q = 0.10
MULTI_Q = 0.05
ADJ = D.ADJUSTMENT  # log_event_age, winter_day_index, nat_tmin_anom, winter_start_year

POLICY = {
    "step": "Step 16E",
    "title": "Hazard robustness, one-day lead, multivariable model and duration trend",
    "robustness_predictors": f"causal predictors with q_fdr_causal < {ROBUST_Q} in Step 16D table_91",
    "multivariable_predictors": f"causal predictors with q_fdr_causal < {MULTI_Q}, excluding nat_dtr_anom",
    "adjustment": ADJ,
    "sensitivity_catalogues": "all non-primary scenarios in 06_events/step7e_sensitivity_event_catalogue.csv",
    "risk_set_start": "event day k >= minimum duration of the scenario",
    "detrending": "ordinary least-squares linear trend in time removed from each covariate over all DJF days",
    "lead": "predictors on day k-1",
    "bootstrap": {"unit": "winter", "repetitions": N_BOOT, "seed": SEED},
    "cross_validation": "leave-one-winter-out; AUC and Brier score on held-out risk-set rows",
}


def risk_set(cov: pd.DataFrame, ev: pd.DataFrame, min_k: int, lead: int = 0) -> pd.DataFrame:
    rows = []
    for e in ev.itertuples(index=False):
        for k in range(min_k, int(e.duration_days) + 1):
            rows.append({"event_id": e.event_id, "winter_start_year": int(e.winter_start_year),
                         "log_event_age": np.log(k), "terminates": int(k == e.duration_days),
                         "date": pd.Timestamp(e.start_date) + pd.Timedelta(days=k - 1)})
    rs = pd.DataFrame(rows)
    cols = [c for c in cov.columns if c != "winter_start_year"]
    rs = rs.merge(cov[cols], left_on="date", right_index=True, how="left")
    if lead:
        lagged = cov[cols].shift(lead)
        pred_cols = [c for c in cols if c not in ("winter_day_index", "nat_tmin_anom")]
        rs = rs.drop(columns=pred_cols).merge(lagged[pred_cols], left_on="date", right_index=True, how="left")
    return rs


def fit(X, y, w=None):
    return LogisticRegression(penalty=None, max_iter=5000).fit(X, y, sample_weight=w)


def odds_ratio(rs: pd.DataFrame, pred: str, n_boot: int = N_BOOT) -> dict:
    cols = [pred] + ADJ
    d = rs.dropna(subset=cols)
    X = ((d[cols] - d[cols].mean()) / d[cols].std()).to_numpy()
    y = d["terminates"].to_numpy()
    est = fit(X, y).coef_[0][0]
    uniq, counts = C.winter_bootstrap_counts(d["winter_start_year"].to_numpy(), n_boot, SEED)
    widx = np.searchsorted(uniq, d["winter_start_year"].to_numpy())
    boots = []
    for b in range(n_boot):
        w = counts[b, widx].astype(float)
        keep = w > 0
        if y[keep].min() == y[keep].max():
            continue
        boots.append(fit(X[keep], y[keep], w[keep]).coef_[0][0])
    boots = np.array(boots)
    return {"odds_ratio_per_sd": np.exp(est), "ci_low": np.exp(np.percentile(boots, 2.5)),
            "ci_high": np.exp(np.percentile(boots, 97.5)),
            "p_bootstrap": min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())),
            "n_rows": len(d), "n_terminations": int(y.sum())}


def detrend(cov: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = cov.copy()
    t = (out.index - out.index[0]).days.to_numpy(dtype=float)
    for c in cols:
        ok = out[c].notna().to_numpy()
        slope, intercept = np.polyfit(t[ok], out[c].to_numpy()[ok], 1)
        out[c] = out[c] - (slope * t + intercept)
    return out


def part_a_trend(ev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = ev["winter_start_year"].to_numpy(float)
    y = ev["duration_days"].to_numpy(float)
    rho = spearmanr(x, y)[0]
    slope = np.polyfit(x, y, 1)[0] * 10
    uniq, counts = C.winter_bootstrap_counts(x.astype(int), 5000, SEED)
    widx = np.searchsorted(uniq, x.astype(int))
    rhos, slopes = [], []
    for b in range(counts.shape[0]):
        rep = np.repeat(np.arange(len(x)), counts[b, widx])
        if len(np.unique(x[rep])) < 3:
            continue
        rhos.append(spearmanr(x[rep], y[rep])[0])
        slopes.append(np.polyfit(x[rep], y[rep], 1)[0] * 10)
    trend = pd.DataFrame([
        {"statistic": "spearman_rho_duration_vs_year", "estimate": rho,
         "ci_low": np.percentile(rhos, 2.5), "ci_high": np.percentile(rhos, 97.5)},
        {"statistic": "ols_slope_days_per_decade", "estimate": slope,
         "ci_low": np.percentile(slopes, 2.5), "ci_high": np.percentile(slopes, 97.5)},
    ])
    bins = [1984, 1994, 2004, 2014, 2025]
    labels = ["1985/86-1994/95", "1995/96-2004/05", "2005/06-2014/15", "2015/16-2024/25"]
    ev = ev.assign(decade=pd.cut(ev["winter_start_year"], bins=bins, labels=labels))
    dec = ev.groupby("decade", observed=False).agg(
        events=("event_id", "count"), mean_duration=("duration_days", "mean"),
        median_duration=("duration_days", "median"),
        persistent_events=("duration_days", lambda s: int((s >= 6).sum())),
        short_events=("duration_days", lambda s: int((s == 3).sum()))).reset_index()
    return trend, dec


def main() -> None:
    C.ensure_dirs()
    C.write_policy("step16e_hazard_robustness_policy.json", POLICY)
    cov = pd.read_csv(C.DAILY_COVARIATES, parse_dates=["date"]).set_index("date")
    ev = C.load_events()
    t91 = pd.read_csv(C.TERM_TABLES / "table_91_termination_hazard_odds_ratios.csv")
    causal = t91[t91["family"] != "noncausal_diagnostic"]
    robust_preds = causal.loc[causal["q_fdr_causal"] < ROBUST_Q, "predictor"].tolist()
    multi_preds = [p for p in causal.loc[causal["q_fdr_causal"] < MULTI_Q, "predictor"] if p != "nat_dtr_anom"]
    print("Robustness predictors:", robust_preds)
    print("Multivariable predictors:", multi_preds)

    # A
    trend, dec = part_a_trend(ev)
    trend.to_csv(T_TREND, index=False, float_format="%.4f")
    dec.to_csv(T_DECADE, index=False, float_format="%.3f")

    # B-E
    rows = []
    primary_rs = risk_set(cov, ev, 3)
    for p in robust_preds:
        rows.append({"analysis": "primary", "predictor": p, **odds_ratio(primary_rs, p)})
    sens = pd.read_csv(SENS_CATALOGUE, parse_dates=["start_date", "end_date"])
    sens = sens[~sens["primary_scenario"].astype(bool)]
    for sid, sev in sens.groupby("scenario_id"):
        min_k = int(sev["duration_days"].min())
        rs = risk_set(cov, sev, min_k)
        for p in robust_preds:
            rows.append({"analysis": f"catalogue:{sid}", "predictor": p, **odds_ratio(rs, p)})
        print("done scenario", sid)
    dt_cov = detrend(cov, list(dict.fromkeys(robust_preds + ["nat_tmin_anom"])))
    dt_rs = risk_set(dt_cov, ev, 3)
    for p in robust_preds:
        rows.append({"analysis": "detrended_covariates", "predictor": p, **odds_ratio(dt_rs, p)})
    lead_rs = risk_set(cov, ev, 3, lead=1)
    for p in robust_preds:
        rows.append({"analysis": "lead_1_day", "predictor": p, **odds_ratio(lead_rs, p)})
    R = pd.DataFrame(rows)
    R["ci_excludes_one"] = (R["ci_low"] > 1) | (R["ci_high"] < 1)
    prim = R[R["analysis"] == "primary"].set_index("predictor")["odds_ratio_per_sd"]
    R["same_direction_as_primary"] = np.sign(np.log(R["odds_ratio_per_sd"])) == np.sign(
        np.log(R["predictor"].map(prim)))
    R.to_csv(T_ROBUST, index=False, float_format="%.4f")

    # D: leave-one-winter-out
    lw = []
    for p in robust_preds:
        cols = [p] + ADJ
        d = primary_rs.dropna(subset=cols)
        X = ((d[cols] - d[cols].mean()) / d[cols].std())
        ors = []
        for w in d["winter_start_year"].unique():
            keep = (d["winter_start_year"] != w).to_numpy()
            ors.append(np.exp(fit(X[keep].to_numpy(), d["terminates"].to_numpy()[keep]).coef_[0][0]))
        ors = np.array(ors)
        sign = np.sign(np.log(prim[p]))
        lw.append({"predictor": p, "primary_or": prim[p], "lowo_min_or": ors.min(), "lowo_max_or": ors.max(),
                   "sign_stability": float((np.sign(np.log(ors)) == sign).mean()), "winters": len(ors)})
    LW = pd.DataFrame(lw)
    LW.to_csv(T_LOWO, index=False, float_format="%.4f")

    # F: multivariable model and cross-validated skill
    cols = multi_preds + ADJ
    d = primary_rs.dropna(subset=cols).reset_index(drop=True)
    Z = (d[cols] - d[cols].mean()) / d[cols].std()
    y = d["terminates"].to_numpy()
    full = fit(Z.to_numpy(), y)
    corr_inv = np.linalg.inv(np.corrcoef(Z.to_numpy(), rowvar=False))
    uniq, counts = C.winter_bootstrap_counts(d["winter_start_year"].to_numpy(), N_BOOT, SEED)
    widx = np.searchsorted(uniq, d["winter_start_year"].to_numpy())
    boots = []
    for b in range(N_BOOT):
        w = counts[b, widx].astype(float)
        keep = w > 0
        if y[keep].min() == y[keep].max():
            continue
        boots.append(fit(Z.to_numpy()[keep], y[keep], w[keep]).coef_[0])
    boots = np.array(boots)
    M = pd.DataFrame({"term": cols, "odds_ratio_per_sd": np.exp(full.coef_[0]),
                      "ci_low": np.exp(np.percentile(boots, 2.5, axis=0)),
                      "ci_high": np.exp(np.percentile(boots, 97.5, axis=0)),
                      "vif": np.diag(corr_inv)})
    M.to_csv(T_MULTI, index=False, float_format="%.4f")

    preds_base, preds_full = np.zeros(len(d)), np.zeros(len(d))
    for w in d["winter_start_year"].unique():
        test = (d["winter_start_year"] == w).to_numpy()
        for cols_, store in ((ADJ, preds_base), (multi_preds + ADJ, preds_full)):
            mu, sd = d.loc[~test, cols_].mean(), d.loc[~test, cols_].std()
            m = fit(((d.loc[~test, cols_] - mu) / sd).to_numpy(), y[~test])
            store[test] = m.predict_proba(((d.loc[test, cols_] - mu) / sd).to_numpy())[:, 1]
    diffs = []
    rng = np.random.default_rng(SEED)
    winters = d["winter_start_year"].to_numpy()
    uw = np.unique(winters)
    for _ in range(N_BOOT):
        pick = rng.choice(uw, len(uw))
        idx = np.concatenate([np.flatnonzero(winters == w) for w in pick])
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], preds_full[idx]) - roc_auc_score(y[idx], preds_base[idx]))
    S = pd.DataFrame([
        {"model": "adjustment_only", "cv_auc": roc_auc_score(y, preds_base), "cv_brier": brier_score_loss(y, preds_base)},
        {"model": "adjustment_plus_synoptic", "cv_auc": roc_auc_score(y, preds_full),
         "cv_brier": brier_score_loss(y, preds_full)},
        {"model": "auc_difference", "cv_auc": roc_auc_score(y, preds_full) - roc_auc_score(y, preds_base),
         "cv_brier": np.nan, "ci_low": np.percentile(diffs, 2.5), "ci_high": np.percentile(diffs, 97.5)},
    ])
    S["climatological_brier"] = y.mean() * (1 - y.mean())
    S.to_csv(T_SKILL, index=False, float_format="%.4f")

    C.write_checksums("step16e_output_sha256.txt", [T_TREND, T_DECADE, T_ROBUST, T_LOWO, T_MULTI, T_SKILL])
    summ = {"step": "Step 16E", "completed_utc": C.now_utc(), "robustness_predictors": robust_preds,
            "multivariable_predictors": multi_preds,
            "robust_rows_same_direction_fraction": float(R["same_direction_as_primary"].mean())}
    SUMMARY.write_text(json.dumps(summ, indent=2))

    piv = R.pivot_table(index="predictor", columns="analysis", values="odds_ratio_per_sd")
    lines = ["Step 16E - Hazard robustness, lead, multivariable model and duration trend", "=" * 72,
             "A. Duration trend", trend.round(3).to_string(index=False), "", dec.to_string(index=False), "",
             "B/C/E. Odds ratio per SD across analyses (CI in table_95)", piv.round(2).to_string(), "",
             "CI excludes 1, by analysis:",
             R.groupby("analysis")["ci_excludes_one"].sum().to_string(), "",
             "D. Leave-one-winter-out", LW.round(3).to_string(index=False), "",
             "F. Multivariable model", M.round(3).to_string(index=False), "",
             "F. Leave-one-winter-out cross-validated skill", S.round(3).to_string(index=False)]
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
