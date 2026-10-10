"""
Step 18A - Checks for the Step 18 revision patches (terrain mask, plus-one p-values, batched bootstrap fits).

Run from the project root, in two stages.

  --pre   BEFORE any rerun (uses the existing files):
          1. the batched logistic fitter (15_termination_common.logit_fit_batch) against sklearn on the real
             primary risk set: 300 winter-block replicates for four predictors;
          2. the terrain mask on the real ERA5 grid: masked area of the Bangladesh, NW IGP and South Asia boxes.

  --post  AFTER rerunning 15a, 15j and 15b (and 15j --terrain-max-m 1500 for the sensitivity):
          3. masked versus unmasked advection series (DJF correlation, SD);
          4. the lag-1 and same-day odds ratios of NW IGP and Bangladesh advection for the unmasked series, the
             1,000-m mask (primary) and the 1,500-m mask, with 10 000 winter-block resamples and plus-one p-values;
          5. TOTAL (not anomaly) 850-hPa advection and v850 around onset and termination: mean, share of events
             with warm advection or southerly flow, and the DJF climatology - the numbers needed to rewrite the
             "warm advection" and "southerly flow" sentences.

Outputs: 05_qc_reports/revision/step18a_pre_checks.txt
         05_qc_reports/revision/step18a_post_checks.txt
         08_outputs/tables/revision/table_R12_terrain_mask_advection_or.csv
         08_outputs/tables/revision/table_R13_total_advection_composites.csv

    python 07_scripts/17a_check_step18_patches.py --pre  2>&1 | tee 09_logs/step18a_pre.log
    python 07_scripts/17a_check_step18_patches.py --post 2>&1 | tee 09_logs/step18a_post.log
"""
from __future__ import annotations

import argparse
import importlib.util
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("term_common", "15_termination_common.py")
E = _load("step16e", "15e_hazard_robustness_and_trend.py")

ROOT = C.PROJECT_ROOT
OUT_Q = ROOT / "05_qc_reports" / "revision"
OUT_T = ROOT / "08_outputs" / "tables" / "revision"
BOX_DIR = ROOT / "03_intermediate" / "era5_box_daily"
MECH = BOX_DIR / "era5_mechanism_daily_oct_mar.csv"
MECH_UNMASKED = BOX_DIR / "era5_mechanism_daily_oct_mar_unmasked.csv"
MECH_1500 = BOX_DIR / "era5_mechanism_daily_oct_mar_terrain1500.csv"
PL_DIR = ROOT / "03_intermediate" / "era5_preprocessed" / "pressure_level_standardized"
ADV_NW, ADV_BD = "mech_nigp_adv850_k_day_anom", "mech_bd_adv850_k_day_anom"
BOXES = {"Bangladesh (20.5-26.8N, 88-93E)": (20.5, 26.8, 88.0, 93.0),
         "NW IGP (25-32N, 75-88E)": (25.0, 32.0, 75.0, 88.0),
         "South Asia (15-35N, 70-100E)": (15.0, 35.0, 70.0, 100.0)}


def box_fraction(mask: np.ndarray, lat: np.ndarray, lon: np.ndarray, box) -> float:
    sel = (lat[:, None] >= box[0]) & (lat[:, None] <= box[1]) & (lon[None, :] >= box[2]) & (lon[None, :] <= box[3])
    w = np.cos(np.deg2rad(lat))[:, None] * sel
    return float((w * mask).sum() / w.sum())


# ----------------------------------------------------------------------------- --pre
def pre() -> list[str]:
    lines = ["Step 18A - pre-run checks", "=" * 26, ""]
    # 1. fitter
    lines.append("1. Batched logistic fitter versus sklearn (primary risk set, 300 winter-block replicates)")
    cov = C.load_covariates_with_mechanisms()
    ev = C.load_events()
    rs = E.risk_set(cov, ev, 3)
    worst = 0.0
    for pred in ("nat_tmax_anom", "era5_sa_z500_tend", "era5_bd_wspd10_anom", ADV_BD):
        cols = [pred] + E.ADJ
        d = rs.dropna(subset=cols)
        X = ((d[cols] - d[cols].mean()) / d[cols].std()).to_numpy()
        y = d["terminates"].to_numpy()
        win = d["winter_start_year"].to_numpy()
        uniq, counts = C.winter_bootstrap_counts(win, 300, E.SEED)
        W = counts[:, np.searchsorted(uniq, win)].astype(float)
        t0 = time.time()
        coef, conv = C.logit_fit_batch(X, y, W)
        tb = time.time() - t0
        t0 = time.time()
        ref = np.full_like(coef, np.nan)   # sklearn converged tightly: the exact maximum-likelihood estimate
        old = np.full_like(coef, np.nan)   # sklearn with the settings used up to v1.2.0 (default tol 1e-4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for b in range(len(W)):
                keep = W[b] > 0
                if y[keep].min() == y[keep].max():
                    continue
                ref[b] = LogisticRegression(penalty=None, tol=1e-10, max_iter=20000).fit(
                    X[keep], y[keep], sample_weight=W[b][keep]).coef_[0]
                old[b] = E.fit(X[keep], y[keep], W[b][keep]).coef_[0]
        ts = time.time() - t0
        ok = np.isfinite(ref).all(axis=1) & np.isfinite(coef).all(axis=1)
        dor = float(np.max(np.abs(np.exp(coef[ok, 0]) - np.exp(ref[ok, 0]))))
        dold = float(np.max(np.abs(np.exp(old[ok, 0]) - np.exp(ref[ok, 0]))))
        worst = max(worst, dor)
        lines.append(f"   {pred:28s} max |OR batch - OR exact| {dor:.1e}  (old sklearn settings: {dold:.1e});"
                     f" non-converged {(~conv).sum()}; batch {tb:.2f} s vs sklearn {ts:.1f} s")
    lines.append(f"   {'PASS' if worst < 1e-4 else 'CHECK'}: batch fits agree with the exact estimates to {worst:.1e} "
                 "(PASS below 1e-4)")
    lines.append("")
    # 2. terrain on the real ERA5 grid
    lines.append("2. Terrain mask on the ERA5 grid (ETOPO1 cell-mean elevation)")
    files = sorted(PL_DIR.glob("ERA5_PL_STD_*.nc"))
    if not files:
        lines.append("   No ERA5 pressure-level file found; skipped")
        return lines
    import xarray as xr
    with xr.open_dataset(files[0]) as ds:
        lat, lon = ds["lat"].values, ds["lon"].values
    lines.append(f"   grid from {files[0].name}: {len(lat)} x {len(lon)}")
    for thr in (C.TERRAIN_MAX_M, 1500.0):
        m = C.below_ground_mask(lat, lon, 850.0, thr)
        md = C.dilate_mask(m)
        for name, box in BOXES.items():
            lines.append(f"   > {thr:4.0f} m  {name:34s} masked {box_fraction(m, lat, lon, box):6.1%}"
                         f"   with neighbours (advection) {box_fraction(md, lat, lon, box):6.1%}")
    return lines


# ----------------------------------------------------------------------------- --post
def covariates_with(mech_path: Path) -> pd.DataFrame:
    old = C.MECHANISM_DAILY
    C.MECHANISM_DAILY = mech_path
    try:
        return C.load_covariates_with_mechanisms()
    finally:
        C.MECHANISM_DAILY = old


def post() -> list[str]:
    lines = ["Step 18A - post-run checks (after 15a, 15j, 15b)", "=" * 48, ""]
    for f in (MECH, MECH_UNMASKED):
        if not f.exists():
            raise SystemExit(f"Missing {f}; rerun 15j first")
    m, u = (pd.read_csv(f, parse_dates=["date"]).set_index("date") for f in (MECH, MECH_UNMASKED))
    djf = m.index.month.isin([12, 1, 2])
    lines.append("3. Masked (1,000 m) versus unmasked daily advection, DJF")
    for c in ("bd_adv850_k_day", "nigp_adv850_k_day"):
        r = np.corrcoef(m.loc[djf, c], u.loc[djf, c])[0, 1]
        lines.append(f"   {c:20s} r = {r:.3f}; SD masked {m.loc[djf, c].std():.2f} vs unmasked {u.loc[djf, c].std():.2f} K/day;"
                     f" mean {m.loc[djf, c].mean():+.2f} vs {u.loc[djf, c].mean():+.2f}")
    lines.append("")

    # 4. odds ratios
    ev = C.load_events()
    variants = [("unmasked (v1.2.0 definition)", MECH_UNMASKED), ("terrain mask 1,000 m (primary)", MECH)]
    if MECH_1500.exists():
        variants.append(("terrain mask 1,500 m (sensitivity)", MECH_1500))
    rows = []
    for label, path in variants:
        cov = covariates_with(path)
        for lead in (0, 1, 2):
            rs = E.risk_set(cov, ev, 3, lead=lead)
            for pred in (ADV_NW, ADV_BD):
                r = E.odds_ratio(rs, pred, C.MIN_TEST_BOOT)
                rows.append({"advection_series": label, "lead_days": lead, "predictor": pred, **r})
                print(f"{label:36s} lead {lead} {pred:30s} OR {r['odds_ratio_per_sd']:.2f} "
                      f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}] p {r['p_bootstrap']:.4f}", flush=True)
    R = pd.DataFrame(rows)
    OUT_T.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT_T / "table_R12_terrain_mask_advection_or.csv", index=False, float_format="%.4f")
    lines.append("4. Advection odds ratios per SD (primary risk set, adjustment: age, season day, Tmin(k), year;")
    lines.append("   10 000 winter-block resamples; plus-one two-sided p). Lead 1 = advection on the day before.")
    lines.append(R[["advection_series", "lead_days", "predictor", "odds_ratio_per_sd", "ci_low", "ci_high",
                    "p_bootstrap", "n_rows", "n_terminations"]].round(4).to_string(index=False))
    lines.append("   Note: the global FDR q over all 66 lag-1 predictors comes from 16a (table_R3) after the full rerun.")
    lines.append("")

    # 5. total advection and v850 composites
    box = pd.read_csv(C.ERA5_BOX_DAILY, parse_dates=["date"]).set_index("date")
    tot = m[["nigp_adv850_k_day", "bd_adv850_k_day"]].join(box[["bd_v_wind_ms_850"]])
    winter = np.where(tot.index.month >= 10, tot.index.year, tot.index.year - 1)
    base = tot[tot.index.month.isin([12, 1, 2]) & (winter >= 1991) & (winter <= 2020)]
    clim = base.mean()
    sign = {"nigp_adv850_k_day": "warm", "bd_adv850_k_day": "warm", "bd_v_wind_ms_850": "southerly"}
    comp = []
    for anchor, col_ev, days in (("onset", "start_date", range(-3, 2)), ("last cold day", "end_date", range(-3, 3))):
        for var in tot.columns:
            for k in days:
                v = np.array([tot[var].get(t + pd.Timedelta(days=k), np.nan) for t in ev[col_ev]])
                v = v[np.isfinite(v)]
                comp.append({"anchor": anchor, "day": k, "variable": var, "mean_total": v.mean(),
                             "share_positive": (v > 0).mean(), "positive_means": sign[var],
                             "djf_climatology_1991_2020": clim[var], "n_events": len(v)})
    T = pd.DataFrame(comp)
    T.to_csv(OUT_T / "table_R13_total_advection_composites.csv", index=False, float_format="%.4f")
    lines.append("5. TOTAL 850-hPa advection (K/day, terrain-masked) and Bangladesh v850 (m/s) around onset and the")
    lines.append("   last cold day (day 0); share_positive = share of the 85 events with warm advection / southerly flow")
    lines.append(T.assign(mean_total=T["mean_total"].round(2), share_positive=(100 * T["share_positive"]).round(0),
                          djf_climatology_1991_2020=T["djf_climatology_1991_2020"].round(2))
                 .drop(columns=["positive_means", "n_events"]).to_string(index=False))
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pre", action="store_true")
    g.add_argument("--post", action="store_true")
    a = ap.parse_args()
    OUT_Q.mkdir(parents=True, exist_ok=True)
    lines = pre() if a.pre else post()
    out = OUT_Q / ("step18a_pre_checks.txt" if a.pre else "step18a_post_checks.txt")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWritten {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
