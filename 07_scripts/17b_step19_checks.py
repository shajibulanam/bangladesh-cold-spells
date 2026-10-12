"""
17b_step19_checks.py  -  Step 19 checks requested by the second pre-submission review (October 2026)

A  Pressure-level validity with ERA5 surface pressure (review: "an elevation cutoff does not guarantee
   that every pressure level lies above ground; check p_surface at each level").
   For the Bangladesh (BD), South Asia (SA) and northwestern Indo-Gangetic Plain (NW IGP) boxes and every
   DJF day at 00 UTC: the area fraction of the grid points RETAINED by the terrain mask at which the
   1000-, 850- and 700-hPa surfaces lie below ground (surface pressure < level), and the lowest surface
   pressure among retained points. For NW IGP the retained points are those of the 850-hPa advection
   (1000-m mask plus the four neighbours). The BD 1000-500-hPa thickness is recomputed with a daily
   surface-pressure mask (only points with p_surface >= 1000 hPa) and compared with the published series
   (anomaly correlation, and its termination odds ratio on the primary risk set).
B  Upstream advection across the six alternative event catalogues (review: "event-definition robustness
   of the upstream association is not shown"): odds ratio of termination per SD of NW IGP 850-hPa
   advection taken one and two days earlier, same adjustment and risk-set rules as Table S12.
C  SSW null distributions: how often the 100-attempt redraw limit was reached (calendar-, decade- and
   ENSO-phase-matched nulls, same procedure and seeds as 15n, 16c and 15o).
D  Original-unit standard deviations of the headline predictors on their risk sets (to translate the
   per-SD odds ratios).

Run from the project root (about 10-20 minutes; part A reads the ERA5 monthly files):
    python 07_scripts/17b_step19_checks.py 2>&1 | tee 09_logs/step19_17b.log

Outputs
    08_outputs/tables/revision/table_R14_pressure_level_validity.csv
    08_outputs/tables/revision/table_R14b_thickness_surface_pressure_mask.csv
    08_outputs/tables/revision/table_R15_upstream_advection_alternative_catalogues.csv
    08_outputs/tables/revision/table_R16_ssw_null_redraws.csv
    08_outputs/tables/revision/table_R17_predictor_sd_original_units.csv
    05_qc_reports/revision/step19a_checks.txt
"""
from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

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
N = _load("step16n", "15n_stratosphere_ssw_power_nam.py")

ROOT = C.PROJECT_ROOT
PL_DIR = ROOT / "03_intermediate" / "era5_preprocessed" / "pressure_level_standardized"
SL_DIR = ROOT / "03_intermediate" / "era5_preprocessed" / "single_level_instant_standardized"
OUT_T = ROOT / "08_outputs" / "tables" / "revision"
OUT_Q = ROOT / "05_qc_reports" / "revision"
SENS_CATALOGUE = ROOT / "06_events" / "step7e_sensitivity_event_catalogue.csv"

BOXES = {"bd": (20.5, 26.8, 88.0, 93.0), "sa": (15.0, 35.0, 70.0, 100.0), "nigp": (25.0, 32.0, 75.0, 88.0)}
LEVELS = (1000.0, 850.0, 700.0)
MONTHS = (11, 12, 1, 2, 3)            # Nov and Mar only for the +/-7-day climatology at the DJF edges
ADV_NW = "mech_nigp_adv850_k_day_anom"
THICK = "era5_bd_thick_1000_500_anom"
SP_NAMES = ("surface_pressure_hpa", "sp", "surface_pressure", "sp_hpa")
RD_OVER_G = 287.05 / 9.80665          # m K-1, hypsometric scale per kelvin

report: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    report.append(s)


def subset(ds: xr.Dataset, box) -> xr.Dataset:
    la0, la1, lo0, lo1 = box
    lat = ds["lat"].values
    ls = slice(la1, la0) if lat[0] > lat[-1] else slice(la0, la1)
    return ds.sel(lat=ls, lon=slice(lo0, lo1))


def surface_pressure(sl: xr.Dataset, elev: np.ndarray) -> tuple[np.ndarray, str]:
    """Surface pressure (hPa, time x lat x lon). ERA5 sp if present, otherwise a hypsometric estimate."""
    for n in SP_NAMES:
        if n in sl:
            v = sl[n].transpose("time", "lat", "lon").values.astype(float)
            return (v / 100.0 if np.nanmedian(v) > 2000 else v), f"ERA5 {n}"
    msl = sl["msl_hpa"].transpose("time", "lat", "lon").values.astype(float)
    t2 = sl["t2m_c"].transpose("time", "lat", "lon").values.astype(float) + 273.15
    z = np.nan_to_num(elev, nan=0.0)[None]
    tv = t2 + 0.0065 * z / 2.0                     # mean temperature of the column below the surface
    return msl * np.exp(-z / (RD_OVER_G * tv)), "estimated from MSLP, T2m and ETOPO1 (no sp in files)"


# ----------------------------------------------------------------------------- A
def part_a(cov: pd.DataFrame, rs0: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, thick_rows, sp_source = [], [], set()
    for winter in range(1985, 2025):
        for month in MONTHS:
            year = winter if month >= 10 else winter + 1
            plf, slf = PL_DIR / f"ERA5_PL_STD_{year}_{month:02d}.nc", SL_DIR / f"ERA5_SL_INSTANT_STD_{year}_{month:02d}.nc"
            if not (plf.exists() and slf.exists()):
                say(f"  missing {year}-{month:02d}")
                continue
            with xr.open_dataset(plf) as pl, xr.open_dataset(slf) as sl:
                days = pd.to_datetime(sl["time"].values).normalize()
                djf = np.isin(days.month, (12, 1, 2))
                for key, box in BOXES.items():
                    s = subset(sl, box)
                    lat, lon = s["lat"].values, s["lon"].values
                    w = np.cos(np.deg2rad(lat))[:, None] * np.ones(len(lon))[None, :]
                    elev = C.grid_cell_elevation(lat, lon)
                    sp, src = surface_pressure(s, elev)
                    sp_source.add(src)
                    for lev in LEVELS:
                        bad = C.below_ground_mask(lat, lon, lev, C.TERRAIN_MAX_M)
                        if key == "nigp" and lev == 850.0:
                            bad = C.dilate_mask(bad)
                        keep = ~bad
                        wk = w * keep
                        below = sp < lev
                        frac_ret = (below * wk[None]).sum(axis=(1, 2)) / wk.sum()
                        frac_all = (below * w[None]).sum(axis=(1, 2)) / w.sum()
                        spmin = np.where(keep[None], sp, np.inf).min(axis=(1, 2))
                        for i in np.where(djf)[0]:
                            rows.append({"date": days[i], "box": key, "level_hpa": lev,
                                         "retained_fraction_of_box": float(wk.sum() / w.sum()),
                                         "below_ground_fraction_retained": float(frac_ret[i]),
                                         "below_ground_fraction_all_points": float(frac_all[i]),
                                         "min_surface_pressure_retained_hpa": float(spmin[i])})
                    if key == "bd":                                # thickness with a daily sp mask
                        p = subset(pl, box)
                        z1000 = p["geopotential_height_m"].sel(level_hpa=1000.0).transpose("time", "lat", "lon").values
                        z500 = p["geopotential_height_m"].sel(level_hpa=500.0).transpose("time", "lat", "lon").values
                        th = z500 - z1000
                        terr = ~C.below_ground_mask(lat, lon, 1000.0, C.TERRAIN_MAX_M)
                        ok_sp = terr[None] & (sp >= 1000.0)
                        w3 = w[None] * terr[None]
                        pub = (th * w3).sum(axis=(1, 2)) / w3.sum()
                        wsp = w[None] * ok_sp
                        with np.errstate(invalid="ignore", divide="ignore"):
                            spm = (np.where(ok_sp, th, 0.0) * wsp).sum(axis=(1, 2)) / wsp.sum(axis=(1, 2))
                        for i in range(len(days)):
                            thick_rows.append({"date": days[i], "thick_terrain_mask": pub[i],
                                               "thick_sp_mask": spm[i],
                                               "valid_fraction_sp_mask": float(wsp[i].sum() / (w.sum()))})
            print(f"  A {year}-{month:02d}", flush=True)
    say(f"Surface pressure source: {'; '.join(sorted(sp_source))}")
    R = pd.DataFrame(rows)
    S = (R.groupby(["box", "level_hpa"])
          .agg(retained_fraction_of_box=("retained_fraction_of_box", "first"),
               days=("date", "size"),
               mean_below_ground_fraction_retained=("below_ground_fraction_retained", "mean"),
               p95_below_ground_fraction_retained=("below_ground_fraction_retained", lambda x: np.percentile(x, 95)),
               max_below_ground_fraction_retained=("below_ground_fraction_retained", "max"),
               days_with_any_retained_point_below=("below_ground_fraction_retained", lambda x: int((x > 0).sum())),
               mean_below_ground_fraction_all_points=("below_ground_fraction_all_points", "mean"),
               lowest_surface_pressure_retained_hpa=("min_surface_pressure_retained_hpa", "min"))
          .reset_index())
    T = pd.DataFrame(thick_rows).set_index("date").sort_index()
    T = T[~T.index.duplicated()]
    an = C.calendar_anomalies(T[["thick_terrain_mask", "thick_sp_mask"]], 7)
    an = an[np.isin(an.index.month, (12, 1, 2))]
    pub_cov = cov[THICK].reindex(an.index)
    out = [{"quantity": "anomaly correlation, sp mask vs terrain mask (DJF days)",
            "value": float(an["thick_terrain_mask"].corr(an["thick_sp_mask"]))},
           {"quantity": "anomaly correlation, recomputed terrain-mask series vs published covariate",
            "value": float(an["thick_terrain_mask"].corr(pub_cov))},
           {"quantity": "mean valid area fraction with sp >= 1000 hPa (of the whole BD box)",
            "value": float(T.loc[np.isin(T.index.month, (12, 1, 2)), "valid_fraction_sp_mask"].mean())},
           {"quantity": "RMS anomaly difference (m)",
            "value": float(np.sqrt(((an["thick_terrain_mask"] - an["thick_sp_mask"]) ** 2).mean()))}]
    for name, series in (("published (terrain mask)", None), ("surface-pressure mask", an["thick_sp_mask"])):
        rs = rs0.copy()
        if series is not None:
            rs[THICK] = rs["date"].map(series)
        o = E.odds_ratio(rs, THICK)
        out.append({"quantity": f"termination OR per SD of BD thickness anomaly, {name}",
                    "value": o["odds_ratio_per_sd"], "ci_low": o["ci_low"], "ci_high": o["ci_high"],
                    "p": o["p_bootstrap"]})
    return S, pd.DataFrame(out)


# ----------------------------------------------------------------------------- B
def part_b(cov: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cats = [("primary", ev)]
    sens = pd.read_csv(SENS_CATALOGUE, parse_dates=["start_date", "end_date"])
    sens = sens[~sens["primary_scenario"].astype(bool)]
    cats += [(sid, sev) for sid, sev in sens.groupby("scenario_id")]
    for name, cat in cats:
        min_k = int(cat["duration_days"].min()) if name != "primary" else 3
        for lag in (1, 2):
            rs = E.risk_set(cov, cat, min_k, lead=lag)
            o = E.odds_ratio(rs, ADV_NW)
            rows.append({"catalogue": name, "spells": len(cat), "predictor_lag_days": lag, **o,
                         "ci_excludes_one": (o["ci_low"] > 1) or (o["ci_high"] < 1)})
        print(f"  B {name}", flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- C
def part_c(ev: pd.DataFrame) -> pd.DataFrame:
    ssw = pd.read_csv(N.SSW_TABLE, parse_dates=["central_date"])["central_date"]
    ssw = ssw[ssw.apply(N.winter_of).between(1985, 2024)].reset_index(drop=True)
    O = _load("step16o", "15o_enso_controls.py")
    oni = O.get_oni()
    djf = oni[oni["SEAS"] == "DJF"].set_index("YR")["ANOM"]
    ph = {w: O.phase(djf.get(w + 1, np.nan)) for w in N.WINTERS}
    pools = {"calendar-matched": lambda src: N.WINTERS[N.WINTERS != src],
             "decade-matched (+/-5 winters)": lambda src: N.WINTERS[(N.WINTERS != src) & (np.abs(N.WINTERS - src) <= 5)],
             "ENSO-phase-matched": lambda src: np.array([w for w in N.WINTERS if w != src and ph[w] == ph[src]]
                                                        or [w for w in N.WINTERS if w != src])}
    rows = []
    for name, pool in pools.items():
        rng = np.random.default_rng(N.SEED)
        attempts, failures = [], 0
        for _ in range(N.N_MC):
            for d in ssw:
                src = N.winter_of(d)
                cand = pool(src)
                for a in range(1, 101):
                    cd = N.shift_to_winter(d, int(rng.choice(cand)))
                    if np.all(np.abs((ssw - cd).dt.days) > N.EXCLUSION):
                        break
                else:
                    failures += 1
                attempts.append(a)
        attempts = np.array(attempts)
        rows.append({"null": name, "replicates": N.N_MC, "ssw_dates": len(ssw),
                     "control_dates_drawn": len(attempts), "limit_reached": failures,
                     "max_attempts": int(attempts.max()), "mean_attempts": float(attempts.mean()),
                     "share_needing_redraw": float((attempts > 1).mean()),
                     "same_winter_ssws_moved": "independently (each SSW draws its own control winter)"})
        print(f"  C {name}", flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- D
def part_d(cov: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    raw_mech = pd.read_csv(C.MECHANISM_DAILY, parse_dates=["date"]).set_index("date")
    rows = []
    rs1 = E.risk_set(cov, ev, 3, lead=1)
    rs0 = E.risk_set(cov, ev, 3)
    spec = [(rs1, ADV_NW, "K day-1", "NW IGP 850-hPa advection anomaly, previous day"),
            (rs0, "era5_bd_wspd10_anom", "m s-1", "BD 10-m wind speed anomaly"),
            (rs0, "era5_bd_stab_t2m_minus_t850_tend", "K day-1", "day-to-day change in BD T2m - T850"),
            (rs0, "era5_sa_z500_tend", "m day-1", "day-to-day change in SA Z500"),
            (rs0, "nat_tmax_anom", "degC", "national Tmax anomaly")]
    for rs, col, unit, label in spec:
        if col not in rs:
            cands = [c for c in rs.columns if col.split("_anom")[0].split("_tend")[0][:12] in c]
            rows.append({"predictor": col, "label": label, "sd": np.nan, "unit": unit,
                         "note": f"column not found; candidates: {cands[:6]}"})
            continue
        d = rs.dropna(subset=[col] + E.ADJ)
        rows.append({"predictor": col, "label": label, "sd": float(d[col].std()), "unit": unit,
                     "n_rows": len(d), "note": ""})
    tot = raw_mech["nigp_adv850_k_day"]
    tot = tot[np.isin(tot.index.month, (12, 1, 2))]
    rows.append({"predictor": tot.name, "label": "NW IGP 850-hPa total advection, all DJF days", "sd": float(tot.std()),
                 "unit": "K day-1", "n_rows": int(tot.notna().sum()), "note": "for context"})
    return pd.DataFrame(rows)


def main() -> None:
    OUT_T.mkdir(parents=True, exist_ok=True)
    OUT_Q.mkdir(parents=True, exist_ok=True)
    say("Step 19 checks (17b)\n" + "=" * 20)
    ev = C.load_events()
    cov = W.load_covariates()
    rs0 = E.risk_set(cov, ev, 3)
    say(f"Primary risk set: {len(rs0)} rows, {int(rs0['terminates'].sum())} terminations")

    say("\nD  Original-unit SDs")
    D = part_d(cov, ev)
    D.to_csv(OUT_T / "table_R17_predictor_sd_original_units.csv", index=False, float_format="%.4f")
    say(D.round(3).to_string(index=False))

    say("\nB  Upstream advection, alternative catalogues")
    B = part_b(cov, ev)
    B.to_csv(OUT_T / "table_R15_upstream_advection_alternative_catalogues.csv", index=False, float_format="%.4f")
    say(B[["catalogue", "spells", "predictor_lag_days", "n_rows", "n_terminations", "odds_ratio_per_sd",
           "ci_low", "ci_high", "p_bootstrap"]].round(4).to_string(index=False))

    say("\nC  SSW null redraws")
    Cc = part_c(ev)
    Cc.to_csv(OUT_T / "table_R16_ssw_null_redraws.csv", index=False, float_format="%.4f")
    say(Cc.round(4).to_string(index=False))

    say("\nA  Pressure-level validity (surface pressure)")
    S, T = part_a(cov, rs0)
    S.to_csv(OUT_T / "table_R14_pressure_level_validity.csv", index=False, float_format="%.4f")
    T.to_csv(OUT_T / "table_R14b_thickness_surface_pressure_mask.csv", index=False, float_format="%.4f")
    say(S.round(4).to_string(index=False))
    say("")
    say(T.round(4).to_string(index=False))

    (OUT_Q / "step19a_checks.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    say(f"\nWritten: {(OUT_Q / 'step19a_checks.txt').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
