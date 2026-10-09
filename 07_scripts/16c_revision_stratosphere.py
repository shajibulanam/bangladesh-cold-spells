"""
16c_revision_stratosphere.py - revision analyses R4, R6 and R8 (stratosphere).

R4  Robustness of the post-SSW onset deficit
    - calendar-matched null as published (Step 16N rules), for windows of 0-30, 0-45, 0-60 and 15-60 days
    - decade-matched null: control winters restricted to within +/-5 years of the SSW winter
    - exposure: DJF days at risk of an onset (days not inside an ongoing spell) in the union of windows
    - exposure-based rate ratio, post-SSW windows vs all other at-risk DJF days (exact conditional-binomial CI)
    - within-winter rate ratio: post-SSW windows vs the rest of the SAME winters (removes interannual and decadal
      confounding); exact conditional-binomial CI
    - two-sided Monte Carlo p-values; number of SSWs with any DJF follow-up
R8  Stratosphere vs spell duration, with the Step 11 metric definitions (table_63)
    - the 8 primary Step 11 metrics; Spearman correlation with duration, winter-block bootstrap CI, BH q
    - partial Spearman controlling onset day of season and winter start year (confounding check)
    - persistent minus short difference for the pre-onset weak-vortex fraction
    - daily hazard version: trailing 20-day weak-vortex fraction and NAM10 mean at day k
R6  Pathway from the vortex to the South Asian circulation
    (a) box indices: winter-mean Spearman with DJF-mean NAM100 (and partial on DJF ONI when available);
        within-winter post-SSW minus rest-of-winter differences (bootstrap over SSW winters)
    (b) --maps: gridded DJF anomalies (u200, Z500, T850, MSLP; 2-degree means): regression of winter means on
        winter-mean NAM100 (t-test, BH across grid) and within-winter post-SSW minus rest-of-winter composite
        (bootstrap over winters, BH across grid); NetCDF and a figure

Run from the project root:
    python3 07_scripts/16c_revision_stratosphere.py 2>&1 | tee 09_logs/step17c.log           # R4, R6a, R8
    python3 07_scripts/16c_revision_stratosphere.py --maps 2>&1 | tee 09_logs/step17c_maps.log  # adds R6b
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, rankdata, spearmanr, t as t_dist

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("term_common", "15_termination_common.py")
E = _load("step16e", "15e_hazard_robustness_and_trend.py")
N = _load("step16n", "15n_stratosphere_ssw_power_nam.py")

ROOT = C.PROJECT_ROOT
STRAT = ROOT / "08_outputs" / "tables" / "stratosphere" / "table_59_daily_stratospheric_indices.csv"
ONI_MONTHLY = ROOT / "01_raw_data" / "climate_indices" / "oni_monthly.csv"
OUT_T = ROOT / "08_outputs" / "tables" / "revision"
OUT_Q = ROOT / "05_qc_reports" / "revision"
OUT_NC = ROOT / "03_intermediate" / "termination" / "step17c_stratospheric_pathway_fields.nc"
OUT_FIG = ROOT / "08_outputs" / "figures" / "termination" / "fig_R6_stratospheric_pathway"
WINDOWS = {"0-30": (0, 30), "0-45": (0, 45), "0-60": (0, 60), "15-60": (15, 60)}
N_MC, N_BOOT, SEED = 10000, 2000, 20261003
DECADE = 5
STEP11_WINDOWS = {"pre45": (-45, -1), "pre30": (-30, -1), "pre20": (-20, -1), "pre10": (-10, -1)}
BOX = ["mech_jet200_core_speed_70_100e_anom", "mech_jet200_core_lat_70_100e_anom", "era5_sa_z500_anom",
       "mech_nigp_adv850_k_day_anom", "era5_bd_v850_anom", "nat_tmin_anom"]
POLICY = {"step": "Step 17C (revision)", "windows_days_after_ssw": WINDOWS, "monte_carlo": N_MC,
          "decade_matched_null": f"control winters within +/-{DECADE} years of the SSW winter",
          "exposure": "DJF days at risk of onset (not inside an ongoing spell after its first day)",
          "rate_ratio_ci": "exact conditional binomial (Clopper-Pearson)", "bootstrap": N_BOOT, "seed": SEED,
          "duration_metrics": "Step 11 definitions (table_63)", "weak_vortex": "daily NAM10 <= -1 SD"}


# ----------------------------------------------------------------------------- shared
def rate_ratio_ci(o_in: int, t_in: float, o_out: int, t_out: float) -> tuple[float, float, float]:
    """Exact conditional-binomial CI for the Poisson rate ratio (in vs out)."""
    n = o_in + o_out
    rr = (o_in / t_in) / (o_out / t_out) if o_out > 0 and t_in > 0 else np.nan
    lo_p = beta_dist.ppf(0.025, o_in, n - o_in + 1) if o_in > 0 else 0.0
    hi_p = beta_dist.ppf(0.975, o_in + 1, n - o_in) if o_in < n else 1.0
    conv = lambda p: (p / (1 - p)) * (t_out / t_in) if p < 1 else np.inf
    return rr, conv(lo_p), conv(hi_p)


def at_risk_days(djf_days: pd.DatetimeIndex, ev: pd.DataFrame) -> pd.DatetimeIndex:
    inside = set()
    for r in ev.itertuples():
        inside.update(pd.date_range(r.start_date + pd.Timedelta(days=1), r.end_date))
    return pd.DatetimeIndex([d for d in djf_days if d not in inside])


def window_days(starts, a: int, b: int, pool: pd.DatetimeIndex) -> pd.DatetimeIndex:
    days = set()
    for s in starts:
        days.update(pd.date_range(s + pd.Timedelta(days=a), s + pd.Timedelta(days=b)))
    return pool[pool.isin(list(days))]


def onsets_in(onsets: np.ndarray, starts, a: int, b: int) -> int:
    st = np.array(starts, dtype="datetime64[ns]")
    lo, hi = st + np.timedelta64(a, "D"), st + np.timedelta64(b, "D")
    return int(((onsets[:, None] >= lo[None, :]) & (onsets[:, None] <= hi[None, :])).any(axis=1).sum())


# ----------------------------------------------------------------------------- R4
def null_dates(ssw: pd.Series, rng, decade: bool) -> list[pd.Timestamp]:
    ctrl = []
    for d in ssw:
        src = N.winter_of(d)
        pool = N.WINTERS[N.WINTERS != src]
        if decade:
            pool = pool[np.abs(pool - src) <= DECADE]
        for _ in range(100):
            cd = N.shift_to_winter(d, int(rng.choice(pool)))
            if np.all(np.abs((ssw - cd).dt.days) > N.EXCLUSION):
                break
        ctrl.append(cd)
    return ctrl


def r4_ssw(ev: pd.DataFrame, djf: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    ssw = pd.read_csv(N.SSW_TABLE, parse_dates=["central_date"])["central_date"]
    ssw = ssw[ssw.apply(N.winter_of).between(N.WINTERS.min(), N.WINTERS.max())].reset_index(drop=True)
    onsets = ev["start_date"].to_numpy(dtype="datetime64[ns]")
    risk = at_risk_days(djf, ev)
    rows = []
    info = {"ssw_total": len(ssw), "ssw_with_djf_followup_0_45": int(sum(len(window_days([s], 0, 45, djf)) > 0 for s in ssw))}
    for wname, (a, b) in WINDOWS.items():
        obs = onsets_in(onsets, ssw, a, b)
        exp_risk = window_days(ssw, a, b, risk)
        for null_name, decade in (("calendar-matched (published)", False), (f"decade-matched (+/-{DECADE} yr)", True)):
            rng = np.random.default_rng(SEED)
            counts, expo = np.empty(N_MC), np.empty(N_MC)
            for i in range(N_MC):
                ctrl = null_dates(ssw, rng, decade)
                counts[i] = onsets_in(onsets, ctrl, a, b)
                if i < 1000:  # exposure is smooth; 1000 draws suffice
                    expo[i] = len(window_days(ctrl, a, b, risk))
            mu, mexp = counts.mean(), expo[:1000].mean()
            p_lo = (1 + (counts <= obs).sum()) / (N_MC + 1)
            p_hi = (1 + (counts >= obs).sum()) / (N_MC + 1)
            rows.append({"window_days": wname, "null": null_name, "observed_onsets": obs, "null_mean": mu,
                         "null_2.5": np.percentile(counts, 2.5), "null_97.5": np.percentile(counts, 97.5),
                         "rate_ratio_vs_null": obs / mu, "p_deficit": p_lo, "p_excess": p_hi,
                         "p_two_sided": min(1.0, 2 * min(p_lo, p_hi)),
                         "observed_at_risk_days": len(exp_risk), "null_mean_at_risk_days": mexp,
                         "exposure_adjusted_ratio": (obs / len(exp_risk)) / (mu / mexp) if len(exp_risk) else np.nan})
            print(f"R4 {wname:6s} {null_name:28s} obs={obs:2d} null={mu:5.1f} ratio={obs / mu:.2f} "
                  f"p_two_sided={rows[-1]['p_two_sided']:.4f}", flush=True)
        # exposure-based and within-winter rate ratios
        on = pd.DatetimeIndex(ev["start_date"])
        in_win = on.isin(window_days(ssw, a, b, djf))
        t_in = len(exp_risk)
        t_out = len(risk) - t_in
        rr, lo, hi = rate_ratio_ci(int(in_win.sum()), t_in, int((~in_win).sum()), t_out)
        rows.append({"window_days": wname, "null": "rate ratio vs all other at-risk DJF days", "observed_onsets":
                     int(in_win.sum()), "observed_at_risk_days": t_in, "rate_ratio_vs_null": rr,
                     "rr_ci_low": lo, "rr_ci_high": hi})
        sw = {N.winter_of(s) for s in ssw if len(window_days([s], a, b, djf)) > 0}
        risk_w = risk[[N.winter_of(d) in sw for d in risk]]
        win_w = window_days(ssw, a, b, risk_w)
        on_w = on[[N.winter_of(d) in sw for d in on]]
        o_in = int(on_w.isin(win_w).sum())
        rr, lo, hi = rate_ratio_ci(o_in, len(win_w), len(on_w) - o_in, len(risk_w) - len(win_w))
        rows.append({"window_days": wname, "null": f"within-winter ({len(sw)} SSW winters)", "observed_onsets": o_in,
                     "observed_at_risk_days": len(win_w), "rate_ratio_vs_null": rr, "rr_ci_low": lo,
                     "rr_ci_high": hi, "onsets_rest_of_same_winters": len(on_w) - o_in,
                     "at_risk_days_rest_of_same_winters": len(risk_w) - len(win_w)})
    return pd.DataFrame(rows), info


# ----------------------------------------------------------------------------- R8
def partial_spearman(x, y, Z) -> float:
    rx, ry = rankdata(x), rankdata(y)
    Zr = np.column_stack([np.ones(len(x))] + [rankdata(z) for z in Z.T])
    ex = rx - Zr @ np.linalg.lstsq(Zr, rx, rcond=None)[0]
    ey = ry - Zr @ np.linalg.lstsq(Zr, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def r8_duration(ev: pd.DataFrame, strat: pd.DataFrame, cov: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ssw = pd.read_csv(N.SSW_TABLE, parse_dates=["central_date"])["central_date"]
    rows = []
    for r in ev.itertuples():
        m = {"event_id": r.event_id, "duration": r.duration_days, "winter": r.winter_start_year,
             "onset_day_of_season": (r.start_date - pd.Timestamp(year=r.winter_start_year, month=12, day=1)).days}
        for name, (a, b) in STEP11_WINDOWS.items():
            sub = strat.loc[r.start_date + pd.Timedelta(days=a): r.start_date + pd.Timedelta(days=b)]
            m[f"{name}_nam10_mean"] = sub["nam_proxy_10"].mean()
            m[f"{name}_nam100_mean"] = sub["nam_proxy_100"].mean()
            m[f"{name}_weak_vortex_fraction"] = sub["weak_vortex_nam10_le_m1"].astype(float).mean()
            m[f"{name}_negative_nam10_100_fraction"] = sub["negative_nam10_and_100"].astype(float).mean()
            m[f"{name}_vortex_displacement_mean"] = sub["vortex_min_displacement_deg_10hpa"].mean()
        m["prior45_ssw_flag"] = int(((r.start_date - ssw).dt.days.between(0, 45)).any())
        rows.append(m)
    M = pd.DataFrame(rows)
    metrics = ["pre45_nam10_mean", "pre30_nam10_mean", "pre20_nam10_mean", "pre10_nam100_mean",
               "pre20_weak_vortex_fraction", "pre20_negative_nam10_100_fraction", "pre45_vortex_displacement_mean",
               "prior45_ssw_flag"]
    winters = M["winter"].to_numpy()
    uw = np.unique(winters)
    rng = np.random.default_rng(SEED)
    boots = [np.concatenate([np.flatnonzero(winters == w) for w in rng.choice(uw, len(uw))]) for _ in range(N_BOOT)]
    Z = M[["onset_day_of_season", "winter"]].to_numpy(float)
    out = []
    for mtr in metrics:
        x, y = M[mtr].to_numpy(float), M["duration"].to_numpy(float)
        rho = spearmanr(x, y).correlation
        prho = partial_spearman(x, y, Z)
        br = np.array([spearmanr(x[i], y[i]).correlation for i in boots])
        bp = np.array([partial_spearman(x[i], y[i], Z[i]) for i in boots])
        out.append({"metric": mtr, "events_with_nonzero": int((x != 0).sum()), "spearman_rho": rho,
                    "rho_ci_low": np.nanpercentile(br, 2.5), "rho_ci_high": np.nanpercentile(br, 97.5),
                    "p_bootstrap": min(1, 2 * min(np.nanmean(br <= 0), np.nanmean(br >= 0))),
                    "partial_rho_season_year": prho, "partial_ci_low": np.nanpercentile(bp, 2.5),
                    "partial_ci_high": np.nanpercentile(bp, 97.5),
                    "partial_p_bootstrap": min(1, 2 * min(np.nanmean(bp <= 0), np.nanmean(bp >= 0))),
                    "rho_metric_vs_onset_day": spearmanr(x, M["onset_day_of_season"]).correlation})
    R = pd.DataFrame(out)
    R["q_fdr"] = C.benjamini_hochberg(R["p_bootstrap"].to_numpy())
    R["partial_q_fdr"] = C.benjamini_hochberg(R["partial_p_bootstrap"].to_numpy())
    g = M.merge(ev[["event_id", "group"]], on="event_id")
    diff = C.group_difference_bootstrap(g["pre20_weak_vortex_fraction"].to_numpy(float),
                                        (g["group"] == "persistent").to_numpy(), (g["group"] == "short").to_numpy(),
                                        g["winter"].to_numpy(), N_BOOT, SEED)
    extra = [{"metric": "pre20_weak_vortex_fraction persistent minus short", **diff}]
    # daily hazard version
    trail = pd.DataFrame(index=strat.index)
    trail["weak_vortex_fraction_trailing20"] = strat["weak_vortex_nam10_le_m1"].astype(float).rolling(20).mean().shift(1)
    trail["nam10_mean_trailing20"] = strat["nam_proxy_10"].rolling(20).mean().shift(1)
    cv = cov.join(trail, how="left")
    rs = E.risk_set(cv, ev, 3)
    for p in trail.columns:
        extra.append({"metric": f"hazard: {p}", **E.odds_ratio(rs, p, N_BOOT)})
    return R, pd.DataFrame(extra)


# ----------------------------------------------------------------------------- R6a box indices
def r6a_box(cov: pd.DataFrame, strat: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    djf = cov.index
    df = cov[[c for c in BOX if c in cov.columns]].copy()
    df["nam100"] = strat["nam_proxy_100"].reindex(djf)
    df["nam10"] = strat["nam_proxy_10"].reindex(djf)
    df["winter"] = [N.winter_of(d) for d in djf]
    wm = df.groupby("winter").mean()
    days = ev.groupby("winter_start_year")["duration_days"].sum().reindex(wm.index, fill_value=0)
    wm["cold_spell_days"] = days
    oni = None
    if ONI_MONTHLY.exists():
        o = pd.read_csv(ONI_MONTHLY)
        o = o[o["month"] == 1].set_index("year")["oni"]
        oni = o.reindex(wm.index + 1).to_numpy()
    rows = []
    for c in [x for x in BOX if x in wm.columns] + ["cold_spell_days"]:
        r = spearmanr(wm["nam100"], wm[c]).correlation
        pr = partial_spearman(wm["nam100"].to_numpy(), wm[c].to_numpy(), oni.reshape(-1, 1)) if oni is not None else np.nan
        rows.append({"analysis": "winter-mean vs DJF-mean NAM100 (40 winters)", "index": c, "spearman_rho": r,
                     "p_value": spearmanr(wm["nam100"], wm[c]).pvalue, "partial_rho_given_oni": pr})
    ssw = pd.read_csv(N.SSW_TABLE, parse_dates=["central_date"])["central_date"]
    post = window_days(ssw, 0, 45, djf)
    df["post"] = djf.isin(post)
    sw = sorted({N.winter_of(d) for d in post})
    rng = np.random.default_rng(SEED)
    for c in [x for x in BOX if x in df.columns]:
        per = np.array([df.loc[(df["winter"] == w) & df["post"], c].mean() - df.loc[(df["winter"] == w) & ~df["post"], c].mean()
                        for w in sw])
        per = per[np.isfinite(per)]
        bs = [rng.choice(per, len(per)).mean() for _ in range(N_BOOT)]
        rows.append({"analysis": f"post-SSW (0-45 d) minus rest of same winter ({len(per)} winters)", "index": c,
                     "mean_difference": per.mean(), "ci_low": np.percentile(bs, 2.5), "ci_high": np.percentile(bs, 97.5)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- R6b gridded maps
def r6b_maps(cov: pd.DataFrame, strat: pd.DataFrame) -> str:
    import xarray as xr
    S = _load("step16s", "15s_termination_composite_maps.py")
    S.DOMAIN, S.COARSEN = (10.0, 70.0, 30.0, 120.0), 8
    S.FIELDS = {"u200": ("pressure_level", "u_wind_ms", 200.0), "z500": ("pressure_level", "geopotential_height_m", 500.0),
                "t850": ("pressure_level", "air_temperature_k", 850.0), "msl": ("single_level_instant", "msl_hpa", None)}
    clim = S.load_climatology()
    djf = list(cov.index)
    winters = np.array([N.winter_of(d) for d in djf])
    ssw = pd.read_csv(N.SSW_TABLE, parse_dates=["central_date"])["central_date"]
    post = set(window_days(ssw, 0, 45, cov.index))
    keys = list(S.FIELDS)
    shape = next(iter(clim.values())).isel(calendar_day=0).shape
    uw = np.unique(winters)
    wsum = {k: np.zeros((len(uw),) + shape) for k in keys}
    wn = np.zeros(len(uw))
    psum = {k: np.zeros((len(uw),) + shape) for k in keys}
    pn = np.zeros(len(uw))
    for start in range(0, len(djf), 90):  # load in blocks to bound memory
        block = djf[start:start + 90]
        an = S.load_anomalies(block, clim)
        for d in block:
            i = int(np.searchsorted(uw, N.winter_of(d)))
            for k in keys:
                wsum[k][i] += an[d][k]
                if d in post:
                    psum[k][i] += an[d][k]
            wn[i] += 1
            pn[i] += d in post
    lat = next(iter(clim.values()))["lat"].values
    lon = next(iter(clim.values()))["lon"].values
    nam = pd.Series(strat["nam_proxy_100"].reindex(cov.index).to_numpy(), index=winters).groupby(level=0).mean().reindex(uw).to_numpy()
    fields = map_statistics(wsum, wn, psum, pn, nam, keys, shape)
    sw = int((pn > 0).sum())
    ds = xr.Dataset({k: (("lat", "lon"), v) for k, v in fields.items()}, coords={"lat": lat, "lon": lon})
    ds.attrs.update({"winters": len(uw), "ssw_winters_with_djf_followup": sw})
    ds.to_netcdf(OUT_NC)
    plot_maps(fields, lat, lon, keys)
    return f"maps: {len(uw)} winters, {sw} SSW winters with DJF follow-up; written {OUT_NC.name}"


def map_statistics(wsum, wn, psum, pn, nam, keys, shape) -> dict[str, np.ndarray]:
    """Regression of winter means on winter-mean NAM100 (t-test, BH) and within-winter post-SSW minus rest (bootstrap, BH)."""
    namz = (nam - nam.mean()) / nam.std()
    n = len(nam)
    sw = pn > 0
    rng = np.random.default_rng(SEED)
    out = {}
    for k in keys:
        X = (wsum[k] / wn[:, None, None]).reshape(n, -1)
        Xc = X - X.mean(0)
        slope = (namz[:, None] * Xc).sum(0) / (namz ** 2).sum()
        r = (namz[:, None] * Xc).sum(0) / np.sqrt((namz ** 2).sum() * (Xc ** 2).sum(0) + 1e-30)
        tstat = r * np.sqrt((n - 2) / np.maximum(1 - r ** 2, 1e-12))
        p = 2 * t_dist.sf(np.abs(tstat), n - 2)
        out[f"{k}_regression_on_nam100"] = slope.reshape(shape)
        out[f"{k}_regression_fdr05"] = (C.benjamini_hochberg(p) < 0.05).reshape(shape)
        rest = (wsum[k] - psum[k]) / np.maximum(wn - pn, 1)[:, None, None]
        diff = (psum[k][sw] / pn[sw][:, None, None] - rest[sw]).reshape(int(sw.sum()), -1)
        bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean(0) for _ in range(1000)])
        pb = np.minimum(1, 2 * np.minimum((bs <= 0).mean(0), (bs >= 0).mean(0)))
        out[f"{k}_post_ssw_minus_rest"] = diff.mean(0).reshape(shape)
        out[f"{k}_post_ssw_fdr05"] = (C.benjamini_hochberg(pb) < 0.05).reshape(shape)
    return out


def plot_maps(fields: dict, lat: np.ndarray, lon: np.ndarray, keys) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        proj = ccrs.PlateCarree()
    except ImportError:
        ccrs = None
    units = {"u200": "m s$^{-1}$", "z500": "m", "t850": "K", "msl": "hPa"}
    fig = plt.figure(figsize=(7.1, 7.6))
    letters = "abcdefgh"
    for row, (stat, sig, title) in enumerate([("regression_on_nam100", "regression_fdr05", "per SD of winter NAM100"),
                                              ("post_ssw_minus_rest", "post_ssw_fdr05", "after SSWs minus rest of winter")]):
        for col, k in enumerate(keys):
            ax = fig.add_subplot(4, 2, row * 4 + col + 1, **({"projection": proj} if ccrs else {}))
            f = fields[f"{k}_{stat}"]
            v = float(np.nanpercentile(np.abs(f), 98)) or 1.0
            m = ax.pcolormesh(lon, lat, f, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto",
                              **({"transform": proj} if ccrs else {}))
            yy, xx = np.where(fields[f"{k}_{sig}"])
            ax.scatter(lon[xx], lat[yy], s=1.2, c="k",
                       **({"transform": proj} if ccrs else {}))
            if ccrs:
                ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.set_title(f"({letters[row * 4 + col]}) {k.upper()} {title}", fontsize=7, loc="left")
            cb = fig.colorbar(m, ax=ax, shrink=0.8, pad=0.02)
            cb.set_label(units[k], fontsize=6)
            cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_FIG}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", action="store_true", help="also compute the gridded pathway maps (R6b)")
    args = ap.parse_args()
    OUT_T.mkdir(parents=True, exist_ok=True)
    OUT_Q.mkdir(parents=True, exist_ok=True)
    C.write_policy("step17c_revision_stratosphere_policy.json", POLICY)
    ev = C.load_events()
    cov = C.load_covariates_with_mechanisms()
    strat = pd.read_csv(STRAT, parse_dates=["date"]).set_index("date")
    for c in ("weak_vortex_nam10_le_m1", "negative_nam10_and_100"):
        strat[c] = strat[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    djf = cov.index

    R4, info = r4_ssw(ev, djf)
    R4.to_csv(OUT_T / "table_R4_ssw_robustness.csv", index=False, float_format="%.4f")
    R8, R8b = r8_duration(ev, strat, cov)
    R8.to_csv(OUT_T / "table_R8_stratosphere_duration.csv", index=False, float_format="%.4f")
    R8b.to_csv(OUT_T / "table_R8b_stratosphere_duration_extra.csv", index=False, float_format="%.4f")
    R6 = r6a_box(cov, strat, ev)
    R6.to_csv(OUT_T / "table_R6_pathway_box_indices.csv", index=False, float_format="%.4f")
    maps = r6b_maps(cov, strat) if args.maps else "maps not run (use --maps)"

    lines = ["Step 17C - revision analyses R4, R6, R8 (stratosphere)", "=" * 55, json.dumps(info), "",
             "R4 SSW robustness", R4.round(4).to_string(index=False), "",
             "R8 Stratosphere vs duration (Step 11 metrics)", R8.round(3).to_string(index=False), "",
             R8b.round(3).to_string(index=False), "", "R6a Pathway (box indices)", R6.round(3).to_string(index=False),
             "", maps]
    (OUT_Q / "step17c_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-12:]))


if __name__ == "__main__":
    main()
