"""
16f_final_stratosphere_figure.py - final figures: stratospheric association with cold-spell occurrence and its pathway.

Step 25 (October 2026): besides the seven-panel figure below (fig_final_R_stratosphere_pathway, unchanged), the
script now also writes the two figures used after the restructuring of the article:
  fig_final_R_stratosphere_occurrence  main-text Fig. 10 = panels (a)-(c) below
  FigS4_stratosphere_pathway_maps      Online Resource 1, Fig. S4 = panels (d)-(g) below, relabelled (a)-(d)
The statistics, seeds and Monte Carlo replicates are computed once and shared by all three figures.

Panels
  (a) cold-spell onsets within 45 days after the 25 major SSWs: calendar-matched null (grey histogram, as
      published) and decade-matched null (+/-5 winters; outline), observed count (red); text from table_R4
  (b) mean NAM100 on the 85 onset days vs calendar-matched random dates
  (c) winter-mean NAM100 vs cold-spell days, coloured by DJF ENSO phase (table_110)
  (d, e) Z500 and U200 regressed on winter-mean NAM100 with DJF ONI as a second predictor (ENSO removed);
         per SD of NAM100; dots: BH q < 0.05 (t-test, 37 degrees of freedom)
  (f, g) Z500 and U200 in the 0-45 days after SSWs minus the rest of the same winter (19 winters with DJF
         follow-up); dots: BH q < 0.05 (exact sign-flip test over the 19 winters, Step 18). A within-winter contrast
         needs no ENSO control.
Dashed boxes: Bangladesh (BD) and the northwestern Indo-Gangetic Plain (NW IGP).

Step 1 (first run only, ~1 h): DJF anomaly fields (2-degree means) are aggregated into winter means and
post-SSW / rest-of-winter means and cached in 03_intermediate/termination/step17f_winter_fields.nc.
Later runs only redo the statistics and the figure (and the SSW Monte Carlo, ~5 min).
Also writes table_R6b_enso_controlled_pathway.csv (regression with and without ONI at the two boxes).
Run from the project root:  python3 07_scripts/16f_final_stratosphere_figure.py 2>&1 | tee 09_logs/step17f.log
"""
from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import t as t_dist  # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("term_common", "15_termination_common.py")
G = _load("step16g", "15g_plot_termination_figures.py")
N = _load("step16n", "15n_stratosphere_ssw_power_nam.py")
M = _load("step17c", "16c_revision_stratosphere.py")

T = C.TERM_TABLES
REV = C.PROJECT_ROOT / "08_outputs" / "tables" / "revision"
CACHE = C.PROJECT_ROOT / "03_intermediate" / "termination" / "step17f_winter_fields.nc"
MM = 1 / 25.4
N_MC, SEED = 10000, 20261003
KEYS = ["z500", "u200", "t850", "msl"]
BOXES = {"BD": (20.5, 26.8, 88.0, 93.0), "NW IGP": (25.0, 32.0, 75.0, 88.0)}
SA_BOX = (15.0, 35.0, 70.0, 100.0)


# ----------------------------------------------------------------------------- step 1: cache
def build_cache(cov: pd.DataFrame) -> None:
    import xarray as xr
    S = _load("step16s", "15s_termination_composite_maps.py")
    S.DOMAIN, S.COARSEN = (10.0, 70.0, 30.0, 120.0), 8
    S.FIELDS = {"u200": ("pressure_level", "u_wind_ms", 200.0), "z500": ("pressure_level", "geopotential_height_m", 500.0),
                "t850": ("pressure_level", "air_temperature_k", 850.0), "msl": ("single_level_instant", "msl_hpa", None)}
    clim = S.load_climatology()
    djf = list(cov.index)
    uw = np.unique([N.winter_of(d) for d in djf])
    ssw = pd.read_csv(N.SSW_TABLE, parse_dates=["central_date"])["central_date"]
    post = set(M.window_days(ssw, 0, 45, cov.index))
    shape = next(iter(clim.values())).isel(calendar_day=0).shape
    wsum = {k: np.zeros((len(uw),) + shape) for k in KEYS}
    psum = {k: np.zeros((len(uw),) + shape) for k in KEYS}
    wn, pn = np.zeros(len(uw)), np.zeros(len(uw))
    for start in range(0, len(djf), 90):
        block = djf[start:start + 90]
        an = S.load_anomalies(block, clim)
        for d in block:
            i = int(np.searchsorted(uw, N.winter_of(d)))
            for k in KEYS:
                wsum[k][i] += an[d][k]
                if d in post:
                    psum[k][i] += an[d][k]
            wn[i] += 1
            pn[i] += d in post
    lat = next(iter(clim.values()))["lat"].values
    lon = next(iter(clim.values()))["lon"].values
    ds = xr.Dataset(coords={"winter": uw, "lat": lat, "lon": lon})
    for k in KEYS:
        ds[f"{k}_winter_mean"] = (("winter", "lat", "lon"), wsum[k] / wn[:, None, None])
        ds[f"{k}_post_ssw_mean"] = (("winter", "lat", "lon"),
                                     np.where(pn[:, None, None] > 0, psum[k] / np.maximum(pn, 1)[:, None, None], np.nan))
        ds[f"{k}_rest_mean"] = (("winter", "lat", "lon"), (wsum[k] - psum[k]) / np.maximum(wn - pn, 1)[:, None, None])
    ds["djf_days"] = ("winter", wn)
    ds["post_ssw_days"] = ("winter", pn)
    ds.to_netcdf(CACHE)
    print(f"cached {CACHE.name}", flush=True)


def read_cache() -> dict:
    import xarray as xr
    with xr.open_dataset(CACHE) as ds:
        out = {v: ds[v].values for v in ds.data_vars}
        out.update({"winter": ds["winter"].values, "lat": ds["lat"].values, "lon": ds["lon"].values})
    return out


# ----------------------------------------------------------------------------- step 2: statistics
def partial_regression(Y: np.ndarray, nam: np.ndarray, oni: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Y: winters x cells. Coefficient of standardised NAM (per SD) with optional ONI covariate; two-sided p."""
    z = lambda v: (v - v.mean()) / v.std()
    cols = [np.ones(len(nam)), z(nam)] + ([z(oni)] if oni is not None else [])
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    dof = len(nam) - X.shape[1]
    s2 = (resid ** 2).sum(0) / dof
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    p = 2 * t_dist.sf(np.abs(beta[1] / np.where(se > 0, se, np.inf)), dof)
    return beta[1], p


def post_minus_rest(post: np.ndarray, rest: np.ndarray, bad: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-winter post-SSW minus rest-of-winter differences (winters with post-SSW days only) and their two-sided
    p-value. Step 18: exact sign-flip test over the 19 SSW winters (all 2^19 sign patterns); the earlier
    2000-resample bootstrap gave p = 0 whenever no resample crossed zero, which produced spurious FDR stippling.
    """
    ok = np.isfinite(post).any(axis=1)
    diff = post[ok] - rest[ok]
    if bad is not None:
        diff[:, bad] = np.nan
    return diff.mean(0), C.sign_flip_p(diff)


def box_mean(field2d: np.ndarray, lat, lon, box) -> float:
    m = (lat[:, None] >= box[0]) & (lat[:, None] <= box[1]) & (lon[None, :] >= box[2]) & (lon[None, :] <= box[3])
    w = np.cos(np.deg2rad(lat))[:, None] * m * np.isfinite(field2d)
    return float(np.nansum(field2d * w) / w.sum())


def fdr_fraction(p: np.ndarray) -> float:
    """Fraction of tested (non-missing) cells with Benjamini-Hochberg q < 0.05."""
    ok = np.isfinite(p)
    return float((C.benjamini_hochberg(p)[ok] < 0.05).mean())


def statistics(cache: dict, nam: np.ndarray, oni: np.ndarray | None) -> tuple[dict, pd.DataFrame]:
    lat, lon = cache["lat"], cache["lon"]
    shape = (len(lat), len(lon))
    # Step 18: T850 and MSLP are missing where 850 hPa is below ground (2-degree cells above 1,000 m)
    terrain = C.below_ground_mask(lat, lon, 850.0, half_width=1.0).ravel()
    out, rows = {}, []
    for k in KEYS:
        bad = terrain if k in ("t850", "msl") else np.zeros_like(terrain)
        Y = cache[f"{k}_winter_mean"].reshape(len(nam), -1).copy()
        Y[:, bad] = 0.0
        for label, o in (("enso_removed", oni), ("no_enso_control", None)):
            if label == "enso_removed" and oni is None:
                continue
            b, p = partial_regression(Y, nam, o)
            b, p = np.where(bad, np.nan, b), np.where(bad, np.nan, p)
            out[f"{k}_nam_{label}"] = b.reshape(shape)
            out[f"{k}_nam_{label}_sig"] = (C.benjamini_hochberg(p) < 0.05).reshape(shape)
            rows.append({"field": k, "statistic": f"regression on NAM100 ({label})",
                         "south_asia_box": box_mean(b.reshape(shape), lat, lon, SA_BOX),
                         "bangladesh_box": box_mean(b.reshape(shape), lat, lon, BOXES["BD"]),
                         "fraction_cells_q05": fdr_fraction(p), "cells_q05": int((C.benjamini_hochberg(p) < 0.05).sum())})
        d, p = post_minus_rest(cache[f"{k}_post_ssw_mean"].reshape(len(nam), -1),
                               cache[f"{k}_rest_mean"].reshape(len(nam), -1), bad)
        out[f"{k}_post"] = d.reshape(shape)
        out[f"{k}_post_sig"] = (C.benjamini_hochberg(p) < 0.05).reshape(shape)
        rows.append({"field": k, "statistic": "post-SSW minus rest of winter",
                     "south_asia_box": box_mean(d.reshape(shape), lat, lon, SA_BOX),
                     "bangladesh_box": box_mean(d.reshape(shape), lat, lon, BOXES["BD"]),
                     "fraction_cells_q05": fdr_fraction(p), "cells_q05": int((C.benjamini_hochberg(p) < 0.05).sum())})
    return out, pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figure
def occurrence_data(ev, cov) -> dict:
    """SSW onset counts with their calendar- and decade-matched nulls, and onset-day NAM100 against matched dates."""
    ssw = pd.read_csv(N.SSW_TABLE, parse_dates=["central_date"])["central_date"]
    ssw = ssw[ssw.apply(N.winter_of).between(1985, 2024)].reset_index(drop=True)
    onsets = ev["start_date"].to_numpy(dtype="datetime64[ns]")
    obs = M.onsets_in(onsets, ssw, 0, 45)
    nulls = {}
    for name, decade in (("calendar", False), ("decade", True)):
        rng = np.random.default_rng(M.SEED)  # same seed and procedure as table_R4
        nulls[name] = np.array([M.onsets_in(onsets, M.null_dates(ssw, rng, decade), 0, 45) for _ in range(N_MC)])
    nam = cov["nam_proxy_100"]
    cd = pd.Series(cov.index.strftime("%m-%d"), index=cov.index)
    by_cd = {k: v.index.to_numpy() for k, v in cd.groupby(cd)}
    keys = ev["start_date"].dt.strftime("%m-%d").to_numpy()
    rng = np.random.default_rng(N.SEED)
    sims = np.array([nam.loc[[rng.choice(by_cd[k]) for k in keys]].mean() for _ in range(N_MC)])
    obs_nam = nam.loc[ev["start_date"]].mean()
    return {"obs": obs, "nulls": nulls, "sims": sims, "obs_nam": obs_nam}


def occurrence_panels(axes, ev, cov, wdf, r4, data: dict | None = None):
    data = data if data is not None else occurrence_data(ev, cov)
    obs, nulls, sims, obs_nam = data["obs"], data["nulls"], data["sims"], data["obs_nam"]
    ax = axes[0]
    bins = np.arange(min(nulls["calendar"].min(), obs) - 0.5, nulls["calendar"].max() + 1.5, 1)
    ax.hist(nulls["calendar"], bins=bins, color=G.OK["grey"], edgecolor="white", lw=0.3, label="calendar-matched")
    ax.hist(nulls["decade"], bins=bins, histtype="step", color="k", lw=0.8, label="decade-matched")
    ax.axvline(obs, color=G.OK["red"], lw=1.6)
    top = max(np.bincount(nulls["calendar"]).max(), np.bincount(nulls["decade"]).max())
    ax.set_ylim(0, top * 1.75)
    ax.set_xlabel("Onsets within 45 d after SSWs")
    ax.set_ylabel("Control replicates")
    ax.set_title("(a) After major SSWs", loc="left")
    rc = r4[(r4["window_days"] == "0-45") & r4["null"].str.startswith("calendar")].iloc[0]
    rd = r4[(r4["window_days"] == "0-45") & r4["null"].str.startswith("decade")].iloc[0]
    ax.text(0.98, 0.98, f"observed {obs}\nexpected {rc['null_mean']:.1f} (calendar), ratio {rc['rate_ratio_vs_null']:.2f}\n"
            f"expected {rd['null_mean']:.1f} (decade), ratio {rd['rate_ratio_vs_null']:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.5)
    ax.legend(frameon=False, fontsize=5.5, loc="upper right", bbox_to_anchor=(1.0, 0.80), handlelength=1.2)
    ax = axes[1]
    ax.hist(sims, bins=40, color=G.OK["grey"], edgecolor="white", lw=0.3)
    ax.axvline(obs_nam, color=G.OK["red"], lw=1.6)
    ax.set_xlabel("Mean NAM100 (SD)")
    ax.set_title("(b) Onset days vs matched", loc="left")
    nb = pd.read_csv(T / "table_108_nam_background_tests.csv")
    nb = nb[(nb["test"] == "across_winters_onset_vs_calendar_matched") & (nb["index"] == "nam_proxy_100")].iloc[0]
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.text(0.03, 0.98, f"onset − matched\n+{nb['estimate']:.2f} SD [{nb['ci_low']:.2f}, {nb['ci_high']:.2f}]",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.5)
    ax = axes[2]
    colors = {"el_nino": G.OK["red"], "la_nina": G.OK["blue"], "neutral": G.OK["grey"]}
    names = {"el_nino": "El Niño", "la_nina": "La Niña", "neutral": "Neutral"}
    for ph, g in wdf.groupby("enso_phase"):
        ax.scatter(g["mean_nam_proxy_100"], g["cold_spell_days"], s=12, color=colors[ph], label=names[ph],
                   edgecolor="k", linewidth=0.3)
    ax.set_xlabel("Winter-mean NAM100 (SD)")
    ax.set_ylabel("Cold-spell days")
    ax.set_title("(c) Winters", loc="left")
    ax.set_ylim(-1.5, wdf["cold_spell_days"].max() * 1.3)
    ax.legend(frameon=False, fontsize=5.5, loc="upper center", ncol=3, handletextpad=0.1, columnspacing=0.6,
              borderaxespad=0.1)
    return obs, nulls, obs_nam - sims.mean()


def map_panel(ax, field, sig, lat, lon, title, unit, ccrs, cfeature, unit_on_side: bool = False):
    kw = {"transform": ccrs.PlateCarree()} if ccrs else {}
    v = float(np.nanpercentile(np.abs(field), 98)) or 1.0
    m = ax.pcolormesh(lon, lat, field, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto", **kw)
    yy, xx = np.where(sig)
    ax.scatter(lon[xx], lat[yy], s=0.8, c="k", linewidths=0, **kw)
    for b in BOXES.values():
        ax.plot([b[2], b[3], b[3], b[2], b[2]], [b[0], b[0], b[1], b[1], b[0]], "k--", lw=0.6, **kw)
    if ccrs:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.35)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25)
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, color="0.6", xlocs=[40, 60, 80, 100, 120], ylocs=[20, 40, 60])
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 5.5}
    ax.set_title(title, loc="left", fontsize=7)
    cb = plt.colorbar(m, ax=ax, shrink=0.85, pad=0.02)
    if unit_on_side:  # Step 25c: unit beside the colour bar, so it cannot collide with a long panel title
        cb.set_label(unit, fontsize=6, labelpad=2)
    else:
        cb.ax.set_title(unit, fontsize=6, pad=3)
    cb.ax.tick_params(labelsize=5.5)


def _cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        ccrs = cfeature = None
    return ccrs, cfeature


def map_specs(enso_ok: bool, letters: str = "defg") -> list:
    tag = "nam_enso_removed" if enso_ok else "nam_no_enso_control"
    note = "ENSO removed" if enso_ok else "no ENSO control"
    a, b, c, d = letters
    return [((1, 0), f"z500_{tag}", f"({a}) Z500 per SD of winter NAM100 ({note})", "m"),
            ((1, 1), f"u200_{tag}", f"({b}) U200 per SD of winter NAM100 ({note})", "m s$^{-1}$"),
            ((2, 0), "z500_post", f"({c}) Z500, 0–45 d after SSWs minus rest of winter", "m"),
            ((2, 1), "u200_post", f"({d}) U200, 0–45 d after SSWs minus rest of winter", "m s$^{-1}$")]


def make_figure(ev, cov, wdf, r4, stats, lat, lon, enso_ok: bool, data: dict | None = None) -> None:
    ccrs, cfeature = _cartopy()
    fig = plt.figure(figsize=(174 * MM, 182 * MM))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.2, 1.2], hspace=0.48, wspace=0.75)
    top = [fig.add_subplot(gs[0, 2 * i:2 * i + 2]) for i in range(3)]
    occurrence_panels(top, ev, cov, wdf, r4, data)
    for (r, c), key, title, unit in map_specs(enso_ok):
        ax = fig.add_subplot(gs[r, 3 * c:3 * c + 3], **({"projection": ccrs.PlateCarree()} if ccrs else {}))
        map_panel(ax, stats[key], stats[f"{key}_sig"], lat, lon, title, unit, ccrs, cfeature)
    G.save(fig, "fig_final_R_stratosphere_pathway")


def make_split_figures(ev, cov, wdf, r4, stats, lat, lon, enso_ok: bool, data: dict) -> None:
    """Step 25: main-text Fig. 10 (occurrence, panels a-c) and Online Resource 1 Fig. S4 (pathway maps)."""
    ccrs, cfeature = _cartopy()
    fig = plt.figure(figsize=(174 * MM, 60 * MM))
    gs = fig.add_gridspec(1, 3, wspace=0.42)
    occurrence_panels([fig.add_subplot(gs[0, i]) for i in range(3)], ev, cov, wdf, r4, data)
    G.save(fig, "fig_final_R_stratosphere_occurrence")
    fig = plt.figure(figsize=(174 * MM, 100 * MM))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.18)
    for (r, c), key, title, unit in map_specs(enso_ok, "abcd"):
        ax = fig.add_subplot(gs[r - 1, c], **({"projection": ccrs.PlateCarree()} if ccrs else {}))
        map_panel(ax, stats[key], stats[f"{key}_sig"], lat, lon, title, unit, ccrs, cfeature, unit_on_side=True)
    G.save(fig, "FigS4_stratosphere_pathway_maps")


def main() -> None:
    C.ensure_dirs()
    REV.mkdir(parents=True, exist_ok=True)
    ev = C.load_events()
    cov = C.load_covariates_with_mechanisms()
    if not CACHE.exists():
        build_cache(cov)
    cache = read_cache()
    wdf = pd.read_csv(T / "table_110_winter_enso_nam_coldspells.csv").set_index("winter_start_year")
    winters = cache["winter"]
    nam = wdf["mean_nam_proxy_100"].reindex(winters).to_numpy(float)
    oni = wdf["djf_oni"].reindex(winters).to_numpy(float)
    enso_ok = bool(np.isfinite(oni).all())
    stats, tab = statistics(cache, nam, oni if enso_ok else None)
    tab.to_csv(REV / "table_R6b_enso_controlled_pathway.csv", index=False, float_format="%.3f")
    print(tab.round(3).to_string(index=False), flush=True)
    r4 = pd.read_csv(REV / "table_R4_ssw_robustness.csv")
    data = occurrence_data(ev, cov)
    make_figure(ev, cov, wdf.reset_index(), r4, stats, cache["lat"], cache["lon"], enso_ok, data)
    make_split_figures(ev, cov, wdf.reset_index(), r4, stats, cache["lat"], cache["lon"], enso_ok, data)


if __name__ == "__main__":
    main()
