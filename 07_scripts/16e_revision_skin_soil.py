"""
16e_revision_skin_soil.py - revision analysis R9: surface (skin) and soil temperature during cold spells.

Tests the boundary-layer/land-surface question raised in supervision: does the land surface store cold during
spells, and does its state relate to termination? ERA5 hourly skin temperature (skt) and level-1 soil temperature
(stl1, 0-7 cm) are read from the ECMWF ARCO archive for the Bangladesh box (as Step 16V), and summarised as
06-17 LT and 18-05 LT window means and 00 UTC values. Also the skin-minus-2 m temperature difference.

Part 1 (download, resumable per winter): 03_intermediate/era5_box_daily/era5_skin_soil_windows_oct_mar.csv
Part 2 (analysis):
  - end-aligned composites (days -5..+2 relative to the last event day) of skt, stl1 and skt - t2m anomalies
  - termination hazard (same adjustment as the paper) for: before-night variables (06-17 LT means, 00 UTC values,
    day-to-day tendencies) and terminating-night variables (18-05 LT); BH within each window
  - pure cold-day vs pure cold-night spell contrast
Requires the Step 16V setup (zarr, fsspec, aiohttp, requests, a ~/.cdsapirc key).
Run from the project root:
    python3 07_scripts/16e_revision_skin_soil.py 2>&1 | tee 09_logs/step17e.log
"""
from __future__ import annotations

import importlib.util
import time
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
V = _load("step16v", "15v_extract_arco_windows.py")
W = _load("step16w", "15w_prenight_hazard_and_precipitation.py")

ROOT = C.PROJECT_ROOT
OUT = ROOT / "03_intermediate" / "era5_box_daily" / "era5_skin_soil_windows_oct_mar.csv"
PART_DIR = ROOT / "03_intermediate" / "era5_box_daily" / "arco_skin_soil_parts"
OUT_T = ROOT / "08_outputs" / "tables" / "revision"
OUT_Q = ROOT / "05_qc_reports" / "revision"
OUT_F = ROOT / "08_outputs" / "figures" / "termination"
CANDIDATES = {"skt": ["skt", "skin_temperature"], "stl1": ["stl1", "soil_temperature_level_1"]}
N_BOOT, SEED = 1000, 20261003


def download() -> None:
    import xarray as xr
    PART_DIR.mkdir(parents=True, exist_ok=True)
    ds = xr.open_zarr(V.URL, consolidated=True, storage_options={"headers": {"Authorization": f"Bearer {V.cds_key()}"}})
    names = {k: next((c for c in cands if c in ds.data_vars), None) for k, cands in CANDIDATES.items()}
    names = {k: v for k, v in names.items() if v}
    if not names:
        raise SystemExit("Neither skin nor soil temperature found in the ARCO store")
    print("Using variables:", names, flush=True)
    LAT = "latitude" if "latitude" in ds.coords else "lat"
    LON = "longitude" if "longitude" in ds.coords else "lon"
    TIM = "valid_time" if "valid_time" in ds.coords else "time"
    lat = ds[LAT].values
    ls = slice(V.BOX[1], V.BOX[0]) if lat[0] > lat[-1] else slice(V.BOX[0], V.BOX[1])
    box = ds[list(names.values())].sel({LAT: ls, LON: slice(V.BOX[2], V.BOX[3])})
    w2d = np.cos(np.deg2rad(box[LAT].values))[:, None] * np.ones((1, box.sizes[LON]))
    w2d = w2d / w2d.sum()
    for winter in range(1985, 2025):
        part = PART_DIR / f"skin_soil_{winter}.csv"
        if part.exists():
            continue
        t0 = time.time()
        sub = box.sel({TIM: slice(f"{winter}-10-01", f"{winter + 1}-03-31T23:00")}).load()
        H = pd.DataFrame(index=pd.to_datetime(sub[TIM].values))
        for k, v in names.items():
            arr = sub[v].transpose(TIM, LAT, LON).values.astype("float64")
            ok = np.isfinite(arr)  # stl1 is undefined over water; land cells only
            H[k] = np.nansum(np.where(ok, arr, 0) * w2d[None], axis=(1, 2)) / np.sum(ok * w2d[None], axis=(1, 2)) - 273.15
        day, hour = H.index.normalize(), H.index.hour
        out = pd.DataFrame(index=pd.Index(day.unique(), name="date"))
        for label, hrs in (("day", V.DAY_HOURS), ("night", V.NIGHT_HOURS)):
            m = np.isin(hour, list(hrs))
            for k in names:
                out[f"{k}_{label}"] = H[m].groupby(day[m])[k].mean()
        for k in names:
            out[f"{k}_00utc"] = H[hour == 0].groupby(day[hour == 0])[k].mean()
        out.to_csv(part, float_format="%.4f")
        print(f"winter {winter}/{str(winter + 1)[-2:]}: {(time.time() - t0) / 60:.1f} min", flush=True)
    pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in sorted(PART_DIR.glob("skin_soil_*.csv"))]) \
        .sort_values("date").to_csv(OUT, index=False, float_format="%.4f")


def analyse() -> list[str]:
    ss = pd.read_csv(OUT, parse_dates=["date"]).set_index("date")
    win = pd.read_csv(W.WINDOWS, parse_dates=["date"]).set_index("date")
    for p in ("day", "night"):
        if f"skt_{p}" in ss and f"t2m_{p}" in win:
            ss[f"skt_minus_t2m_{p}"] = ss[f"skt_{p}"] - win[f"t2m_{p}"].reindex(ss.index)
    anom = C.calendar_anomalies(ss, 7)
    tend = anom[[c for c in anom.columns if c.endswith("_day")]].diff()
    tend.columns = [f"ss_{c}_tend" for c in tend.columns]
    anom.columns = [f"ss_{c}_anom" for c in anom.columns]
    cov = W.load_covariates().join(anom).join(tend)
    ev = C.load_events()
    pre = [c for c in anom.columns if c.endswith(("_day_anom", "_00utc_anom"))] + list(tend.columns)
    night = [c for c in anom.columns if c.endswith("_night_anom")]
    rs = E.risk_set(cov, ev, 3)
    rows = []
    for fam, preds in (("before terminating night", pre), ("terminating night (concurrent)", night)):
        part = [{"window": fam, "predictor": p, **E.odds_ratio(rs, p, N_BOOT)} for p in preds if rs[p].notna().sum() > 30]
        P = pd.DataFrame(part)
        P["q_fdr_window"] = C.benjamini_hochberg(P["p_bootstrap"].to_numpy())
        rows.append(P)
    H = pd.concat(rows, ignore_index=True)
    H.to_csv(OUT_T / "table_R9_skin_soil_hazard.csv", index=False, float_format="%.4f")
    # end-aligned composites
    comp = []
    show = [c for c in ("ss_skt_00utc_anom", "ss_stl1_00utc_anom", "ss_skt_minus_t2m_night_anom", "ss_stl1_day_anom")
            if c in cov.columns]
    wint = ev["winter_start_year"].to_numpy()
    uniq, counts = C.winter_bootstrap_counts(wint, N_BOOT, SEED)
    Wb = counts[:, np.searchsorted(uniq, wint)].astype(float)
    for c in show:
        Vv = np.array([[cov[c].get(r.end_date + pd.Timedelta(days=k), np.nan) for k in range(-5, 3)] for r in ev.itertuples()])
        ok = np.isfinite(Vv)
        bm = (Wb @ np.nan_to_num(Vv)) / (Wb @ ok.astype(float))
        for j, k in enumerate(range(-5, 3)):
            comp.append({"variable": c, "day": k, "mean": np.nanmean(Vv[:, j]), "ci_low": np.percentile(bm[:, j], 2.5),
                         "ci_high": np.percentile(bm[:, j], 97.5)})
    Cp = pd.DataFrame(comp)
    Cp.to_csv(OUT_T / "table_R9b_skin_soil_end_aligned.csv", index=False, float_format="%.4f")
    fig, axes = plt.subplots(1, len(show), figsize=(7.1, 2.2), sharex=True)
    for ax, c in zip(np.atleast_1d(axes), show):
        d = Cp[Cp["variable"] == c]
        ax.fill_between(d["day"], d["ci_low"], d["ci_high"], color="0.8")
        ax.plot(d["day"], d["mean"], "-o", color="k", ms=2.5, lw=1)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.axvline(0, color="0.5", lw=0.6, ls="--")
        ax.set_title(c.replace("ss_", "").replace("_anom", ""), fontsize=7)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_F / f"fig_R9_skin_soil_end_aligned.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    # cold-day vs cold-night
    tx = pd.read_csv(W.TX_CATALOGUE, parse_dates=["start_date", "end_date"])
    tn = ev[ev["winter_start_year"] <= W.LAST_TMAX_WINTER]
    days = lambda df: {d for r in df.itertuples() for d in pd.date_range(r.start_date, r.end_date)}
    tn_d, tx_d = days(tn), days(tx)
    recs = []
    for kind, df, other in (("cold_day", tx, tn_d), ("cold_night", tn, tx_d)):
        for r in df.itertuples():
            rng_ = pd.date_range(r.start_date, r.end_date)
            if not (set(rng_) & other):
                recs.append({"type": kind, "winter": r.winter_start_year, **cov.reindex(rng_)[show].mean().to_dict()})
    EVm = pd.DataFrame(recs)
    con = [{"variable": c, **C.group_difference_bootstrap(EVm[c].to_numpy(float), (EVm["type"] == "cold_day").to_numpy(),
                                                          (EVm["type"] == "cold_night").to_numpy(), EVm["winter"].to_numpy(),
                                                          N_BOOT, SEED)} for c in show]
    Cn = pd.DataFrame(con)
    Cn.to_csv(OUT_T / "table_R9c_skin_soil_regimes.csv", index=False, float_format="%.4f")
    return ["Step 17E - skin and soil temperature (R9)", "=" * 45, "Termination hazard:",
            H[["window", "predictor", "odds_ratio_per_sd", "ci_low", "ci_high", "q_fdr_window"]].round(3).to_string(index=False),
            "", "End-aligned composites (mean):", Cp.pivot_table(index="day", columns="variable", values="mean").round(2).to_string(),
            "", "Pure cold-day minus pure cold-night:", Cn.round(3).to_string(index=False)]


def main() -> None:
    for d in (OUT_T, OUT_Q, OUT_F):
        d.mkdir(parents=True, exist_ok=True)
    C.write_policy("step17e_skin_soil_policy.json", {"step": "Step 17E (revision)", "source": V.URL, "box": V.BOX,
                                                     "variables": CANDIDATES, "windows_utc": {"day": "00-11", "night": "12-23"},
                                                     "bootstrap": N_BOOT, "seed": SEED})
    if not OUT.exists():
        download()
    lines = analyse()
    (OUT_Q / "step17e_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
