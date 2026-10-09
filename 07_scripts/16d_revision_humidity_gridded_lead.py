"""
16d_revision_humidity_gridded_lead.py - revision analyses R5, R7 and R11.

R5  Absolute humidity of cold-day and cold-night regimes (the manuscript must not call an anomaly 'near-saturated')
    Bangladesh-box ERA5 2-m temperature and dewpoint (Step 16V windows, 06-17 LT day and 18-05 LT night):
    dewpoint depression (K) and relative humidity (%, Magnus formula over water) as event means for pure
    cold-day and pure cold-night spells (no shared day with the other type), the fraction of event nights with
    RH >= 95 %, and the DJF climatology for reference. Group differences: winter-block bootstrap.
R7  Gridded leakage-free test of persistent vs short cold-night spells
    Event-mean anomalies of Z500, T850, U200, MSLP and T2m (2-degree means, 10-70N, 30-120E) in the two windows in
    which both groups have identical event status: days -3 to -1 and days 0 to 2 after onset. Persistent (>= 6 d)
    minus short (3 d); winter-block bootstrap p per cell; Benjamini-Hochberg across cells. Also the upstream
    box 40-60N, 30-60E (positive anomaly seen in the earlier fixed-lag composites) with a bootstrap interval.
R11 Robustness of the one-day-lead precursor (850-hPa advection over the NW Indo-Gangetic Plain)
    (a) leads of 1 and 2 days; (b) lead 1 with season-end spells censored and on the stable 25-station catalogue;
    (c) lead-1 odds ratio adjusted additionally for the same-day core predictors (10-m wind, SA Z500 tendency,
        T2m-T850 tendency, Tmax anomaly); (d) end-aligned and onset-aligned composites of NW IGP and Bangladesh
        advection, days -5..+2, winter-block bootstrap intervals

Run from the project root (needs the Step 16V windows file and the gridded ERA5 files):
    python3 07_scripts/16d_revision_humidity_gridded_lead.py 2>&1 | tee 09_logs/step17d.log
"""
from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

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

ROOT = C.PROJECT_ROOT
OUT_T = ROOT / "08_outputs" / "tables" / "revision"
OUT_Q = ROOT / "05_qc_reports" / "revision"
OUT_F = ROOT / "08_outputs" / "figures" / "termination"
OUT_NC = ROOT / "03_intermediate" / "termination" / "step17d_gridded_leakage_free.nc"
STABLE_CAT = ROOT / "06_events" / "step17b_stable_network_event_catalogue.csv"
N_BOOT, N_BOOT_OR, SEED = 2000, 1000, 20261003
ADV_NW, ADV_BD, JET = "mech_nigp_adv850_k_day_anom", "mech_bd_adv850_k_day_anom", "mech_jet200_core_speed_70_100e_anom"
CORE_SAME_DAY = ["era5_bd_wspd10_anom", "era5_sa_z500_tend", "era5_bd_stab_t2m_minus_t850_tend", "nat_tmax_anom"]
GRID_WINDOWS = {"days -3 to -1": (-3, -1), "days 0 to 2": (0, 2)}
UPSTREAM_BOX = (40.0, 60.0, 30.0, 60.0)


# ----------------------------------------------------------------------------- R5
def rh(t, td):
    es = lambda x: np.exp(17.625 * x / (243.04 + x))
    return 100.0 * es(td) / es(t)


def r5_humidity(ev: pd.DataFrame) -> pd.DataFrame:
    win = pd.read_csv(W.WINDOWS, parse_dates=["date"]).set_index("date")
    for p in ("day", "night"):
        win[f"dpd_{p}"] = win[f"t2m_{p}"] - win[f"td2m_{p}"]
        win[f"rh_{p}"] = rh(win[f"t2m_{p}"], win[f"td2m_{p}"])
    win["night_rh_ge95"] = (win["rh_night"] >= 95).astype(float)
    cols = ["t2m_day", "td2m_day", "dpd_day", "rh_day", "t2m_night", "td2m_night", "dpd_night", "rh_night", "night_rh_ge95"]
    tx = pd.read_csv(W.TX_CATALOGUE, parse_dates=["start_date", "end_date"])
    tn = ev[ev["winter_start_year"] <= W.LAST_TMAX_WINTER]
    days = lambda df: {d for r in df.itertuples() for d in pd.date_range(r.start_date, r.end_date)}
    tn_days, tx_days = days(tn), days(tx)
    rows = []
    for kind, df, other in (("cold_day", tx, tn_days), ("cold_night", tn, tx_days)):
        for r in df.itertuples():
            rng_ = pd.date_range(r.start_date, r.end_date)
            if set(rng_) & other:
                continue
            rows.append({"type": kind, "winter": r.winter_start_year, **win.reindex(rng_)[cols].mean().to_dict()})
    EV = pd.DataFrame(rows)
    djf = win[win.index.month.isin([12, 1, 2])]
    out = []
    for v in cols:
        r = C.group_difference_bootstrap(EV[v].to_numpy(float), (EV["type"] == "cold_day").to_numpy(),
                                         (EV["type"] == "cold_night").to_numpy(), EV["winter"].to_numpy(), N_BOOT, SEED)
        out.append({"variable": v, "pure_cold_day_mean": EV.loc[EV["type"] == "cold_day", v].mean(),
                    "pure_cold_night_mean": EV.loc[EV["type"] == "cold_night", v].mean(),
                    "djf_climatology": djf[v].mean(), **r})
    R = pd.DataFrame(out)
    R["n_pure_cold_day"] = int((EV["type"] == "cold_day").sum())
    R["n_pure_cold_night"] = int((EV["type"] == "cold_night").sum())
    return R


# ----------------------------------------------------------------------------- R7
def r7_gridded(ev: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    import xarray as xr
    S = _load("step16s", "15s_termination_composite_maps.py")
    S.DOMAIN, S.COARSEN = (10.0, 70.0, 30.0, 120.0), 8
    S.FIELDS = {"z500": ("pressure_level", "geopotential_height_m", 500.0),
                "t850": ("pressure_level", "air_temperature_k", 850.0),
                "u200": ("pressure_level", "u_wind_ms", 200.0),
                "msl": ("single_level_instant", "msl_hpa", None), "t2m": ("single_level_instant", "t2m_c", None)}
    sub = ev[ev["group"].isin(["short", "persistent"])].reset_index(drop=True)
    dates = sorted({r.start_date + pd.Timedelta(days=k) for r in sub.itertuples() for k in range(-3, 3)})
    clim = S.load_climatology()
    an = S.load_anomalies(dates, clim)
    lat = next(iter(clim.values()))["lat"].values
    lon = next(iter(clim.values()))["lon"].values
    keys = list(S.FIELDS)
    is_p = (sub["group"] == "persistent").to_numpy()
    winters = sub["winter_start_year"].to_numpy()
    uniq, counts = C.winter_bootstrap_counts(winters, N_BOOT, SEED)
    Wb = counts[:, np.searchsorted(uniq, winters)].astype(float)
    ds = xr.Dataset(coords={"lat": lat, "lon": lon})
    rows = []
    in_box = (lat[:, None] >= UPSTREAM_BOX[0]) & (lat[:, None] <= UPSTREAM_BOX[1]) & \
             (lon[None, :] >= UPSTREAM_BOX[2]) & (lon[None, :] <= UPSTREAM_BOX[3])
    for wname, (a, b) in GRID_WINDOWS.items():
        for k in keys:
            X = np.stack([np.mean([an[r.start_date + pd.Timedelta(days=j)][k] for j in range(a, b + 1)], axis=0)
                          for r in sub.itertuples()])
            flat = X.reshape(len(sub), -1)
            diff = flat[is_p].mean(0) - flat[~is_p].mean(0)
            mp = (Wb[:, is_p] @ flat[is_p]) / Wb[:, is_p].sum(1, keepdims=True)
            ms = (Wb[:, ~is_p] @ flat[~is_p]) / Wb[:, ~is_p].sum(1, keepdims=True)
            bd = mp - ms
            p = np.minimum(1, 2 * np.minimum((bd <= 0).mean(0), (bd >= 0).mean(0)))
            q = C.benjamini_hochberg(p)
            tag = "pre" if a < 0 else "early"
            ds[f"{k}_{tag}_persistent_minus_short"] = (("lat", "lon"), diff.reshape(len(lat), len(lon)))
            ds[f"{k}_{tag}_fdr05"] = (("lat", "lon"), (q < 0.05).reshape(len(lat), len(lon)))
            boxw = np.cos(np.deg2rad(lat))[:, None] * in_box
            bflat = (boxw / boxw.sum()).ravel()
            bdiff = bd @ bflat
            rows.append({"window": wname, "field": k, "cells_fdr05": int((q < 0.05).sum()), "cells_total": q.size,
                         "fraction_fdr05": float((q < 0.05).mean()), "min_p": float(p.min()),
                         "upstream_box_difference": float(diff @ bflat), "upstream_box_ci_low": np.percentile(bdiff, 2.5),
                         "upstream_box_ci_high": np.percentile(bdiff, 97.5)})
            print(f"R7 {wname:14s} {k:5s} FDR cells {rows[-1]['cells_fdr05']}/{q.size}; upstream box "
                  f"{rows[-1]['upstream_box_difference']:.2f} [{rows[-1]['upstream_box_ci_low']:.2f}, "
                  f"{rows[-1]['upstream_box_ci_high']:.2f}]", flush=True)
    ds.attrs.update({"persistent_events": int(is_p.sum()), "short_events": int((~is_p).sum())})
    ds.to_netcdf(OUT_NC)
    plot_r7(ds, keys, lat, lon)
    return pd.DataFrame(rows), f"written {OUT_NC.name}"


def plot_r7(ds, keys, lat, lon) -> None:
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        kw = {"projection": ccrs.PlateCarree()}
        tr = {"transform": ccrs.PlateCarree()}
    except ImportError:
        ccrs, kw, tr = None, {}, {}
    units = {"z500": "m", "t850": "K", "u200": "m s$^{-1}$", "msl": "hPa", "t2m": "K"}
    fig = plt.figure(figsize=(7.1, 9.0))
    n = 0
    for k in keys:
        for tag, title in (("pre", "days -3 to -1"), ("early", "days 0 to 2")):
            n += 1
            ax = fig.add_subplot(len(keys), 2, n, **kw)
            f = ds[f"{k}_{tag}_persistent_minus_short"].values
            v = float(np.nanpercentile(np.abs(f), 98)) or 1.0
            m = ax.pcolormesh(lon, lat, f, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto", **tr)
            yy, xx = np.where(ds[f"{k}_{tag}_fdr05"].values)
            ax.scatter(lon[xx], lat[yy], s=1.2, c="k", **tr)
            if ccrs:
                ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.plot([30, 60, 60, 30, 30], [40, 40, 60, 60, 40], "k--", lw=0.6, **tr)
            ax.set_title(f"({chr(96 + n)}) {k.upper()} persistent - short, {title}", fontsize=7, loc="left")
            cb = fig.colorbar(m, ax=ax, shrink=0.8, pad=0.02)
            cb.set_label(units[k], fontsize=6)
            cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_F / f"fig_R7_gridded_leakage_free.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- R11
def or_with_controls(rs: pd.DataFrame, pred: str, controls: list[str]) -> dict:
    cols = [pred] + controls + E.ADJ
    d = rs.dropna(subset=cols)
    X = ((d[cols] - d[cols].mean()) / d[cols].std()).to_numpy()
    y = d["terminates"].to_numpy()
    est = E.fit(X, y).coef_[0][0]
    uniq, counts = C.winter_bootstrap_counts(d["winter_start_year"].to_numpy(), N_BOOT_OR, SEED)
    widx = np.searchsorted(uniq, d["winter_start_year"].to_numpy())
    boots = []
    for b in range(N_BOOT_OR):
        w = counts[b, widx].astype(float)
        keep = w > 0
        if y[keep].min() != y[keep].max():
            boots.append(E.fit(X[keep], y[keep], w[keep]).coef_[0][0])
    boots = np.array(boots)
    return {"odds_ratio_per_sd": np.exp(est), "ci_low": np.exp(np.percentile(boots, 2.5)),
            "ci_high": np.exp(np.percentile(boots, 97.5)),
            "p_bootstrap": min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())), "n_rows": len(d)}


def lead_risk_set(cov, ev, lead, censor_season_end=False):
    rs = E.risk_set(cov, ev, 3, lead=lead)
    if censor_season_end:
        end = pd.to_datetime(ev["end_date"])
        cens = set(ev.loc[(end.dt.month == 2) & ((end + pd.Timedelta(days=1)).dt.month == 3), "event_id"])
        rs.loc[rs["event_id"].isin(cens) & (rs["terminates"] == 1), "terminates"] = 0
    return rs


def r11_lead(cov: pd.DataFrame, ev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for lead in (1, 2):
        rs = lead_risk_set(cov, ev, lead)
        for p in (ADV_NW, ADV_BD, JET):
            rows.append({"analysis": f"lead {lead} day(s)", "predictor": p, **E.odds_ratio(rs, p, N_BOOT_OR)})
    rs = lead_risk_set(cov, ev, 1, censor_season_end=True)
    rows.append({"analysis": "lead 1, season-end spells censored", "predictor": ADV_NW, **E.odds_ratio(rs, ADV_NW, N_BOOT_OR)})
    if STABLE_CAT.exists():
        st = pd.read_csv(STABLE_CAT, parse_dates=["start_date", "end_date"])
        rs = lead_risk_set(cov, st, 1)
        rows.append({"analysis": "lead 1, stable 25-station catalogue", "predictor": ADV_NW,
                     **E.odds_ratio(rs, ADV_NW, N_BOOT_OR)})
    # (c) lead-1 NW IGP advection with same-day core predictors as controls
    rs1 = E.risk_set(cov, ev, 3)
    lagged = cov[[ADV_NW]].shift(1).rename(columns={ADV_NW: "nigp_adv_lead1"})
    rs1 = rs1.merge(lagged, left_on="date", right_index=True, how="left")
    rows.append({"analysis": "lead 1 + same-day core predictors", "predictor": "nigp_adv_lead1",
                 **or_with_controls(rs1, "nigp_adv_lead1", CORE_SAME_DAY)})
    rows.append({"analysis": "lead 1 + same-day NW IGP and BD advection", "predictor": "nigp_adv_lead1",
                 **or_with_controls(rs1, "nigp_adv_lead1", [ADV_NW, ADV_BD])})
    R = pd.DataFrame(rows)
    # (d) aligned composites
    comp = []
    for align in ("end", "onset"):
        for p, label in ((ADV_NW, "NW IGP"), (ADV_BD, "Bangladesh")):
            vals, wint = [], []
            for r in ev.itertuples():
                ref = r.end_date if align == "end" else r.start_date
                vals.append([cov[p].get(ref + pd.Timedelta(days=k), np.nan) for k in range(-5, 3)])
                wint.append(r.winter_start_year)
            V, wint = np.array(vals, dtype=float), np.array(wint)
            uniq, counts = C.winter_bootstrap_counts(wint, N_BOOT, SEED)
            Wb = counts[:, np.searchsorted(uniq, wint)].astype(float)
            ok = np.isfinite(V)
            bm = (Wb @ np.nan_to_num(V)) / (Wb @ ok.astype(float))
            for j, k in enumerate(range(-5, 3)):
                comp.append({"alignment": align, "box": label, "day": k, "mean": np.nanmean(V[:, j]),
                             "ci_low": np.percentile(bm[:, j], 2.5), "ci_high": np.percentile(bm[:, j], 97.5)})
    Cp = pd.DataFrame(comp)
    return R, Cp


def plot_r11(Cp: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), sharey=True)
    for ax, align, title in zip(axes, ("onset", "end"), ("(a) aligned on onset (day 0)", "(b) aligned on last event day (day 0)")):
        for box, col in (("NW IGP", "#0072B2"), ("Bangladesh", "#D55E00")):
            d = Cp[(Cp["alignment"] == align) & (Cp["box"] == box)]
            ax.fill_between(d["day"], d["ci_low"], d["ci_high"], color=col, alpha=0.2, lw=0)
            ax.plot(d["day"], d["mean"], "-o", color=col, ms=3, lw=1.2, label=box)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.axvline(0, color="0.5", lw=0.6, ls="--")
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xlabel("Day relative to alignment", fontsize=7)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("850-hPa advection anomaly (K day$^{-1}$)", fontsize=7)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_F / f"fig_R11_advection_pulse.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for d in (OUT_T, OUT_Q, OUT_F):
        d.mkdir(parents=True, exist_ok=True)
    C.write_policy("step17d_revision_policy.json", {
        "step": "Step 17D (revision)", "humidity": "Magnus RH over water; dewpoint depression; box means (Step 16V windows)",
        "gridded_windows": GRID_WINDOWS, "upstream_box": UPSTREAM_BOX, "grid": "2-degree means of 0.25-degree ERA5",
        "lead_controls": CORE_SAME_DAY, "bootstrap": N_BOOT, "seed": SEED})
    ev = C.load_events()
    cov = W.load_covariates()
    R5 = r5_humidity(ev)
    R5.to_csv(OUT_T / "table_R5_absolute_humidity.csv", index=False, float_format="%.4f")
    print(R5[["variable", "pure_cold_day_mean", "pure_cold_night_mean", "djf_climatology", "difference", "ci_low",
              "ci_high"]].round(2).to_string(index=False), flush=True)
    R11, Cp = r11_lead(cov, ev)
    R11.to_csv(OUT_T / "table_R11_lead_advection_robustness.csv", index=False, float_format="%.4f")
    Cp.to_csv(OUT_T / "table_R11b_aligned_advection_composites.csv", index=False, float_format="%.4f")
    plot_r11(Cp)
    print(R11[["analysis", "predictor", "odds_ratio_per_sd", "ci_low", "ci_high", "p_bootstrap"]].round(3).to_string(index=False), flush=True)
    try:
        R7, msg = r7_gridded(ev)
        R7.to_csv(OUT_T / "table_R7_gridded_leakage_free.csv", index=False, float_format="%.4f")
    except (ImportError, SystemExit, FileNotFoundError) as exc:
        R7, msg = pd.DataFrame(), f"R7 not run: {exc}"
    lines = ["Step 17D - revision analyses R5, R7, R11", "=" * 45,
             "R5 Absolute humidity (box means; pure spells)", R5.round(3).to_string(index=False), "",
             "R7 Gridded leakage-free persistent - short", R7.round(3).to_string(index=False) if len(R7) else msg, msg, "",
             "R11 Lead advection robustness", R11.round(3).to_string(index=False), "",
             "Aligned composites", Cp.pivot_table(index="day", columns=["alignment", "box"], values="mean").round(2).to_string(),
             "", "Sign convention: taken from 15j (confirm: positive = warm advection, -V.gradT)."]
    (OUT_Q / "step17d_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-14:]))


if __name__ == "__main__":
    main()
