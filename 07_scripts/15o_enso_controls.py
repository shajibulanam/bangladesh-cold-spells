"""
Step 16O - ENSO controls for the stratospheric results.

Why: El Nino winters tend to have a weaker polar vortex and more SSWs. If ENSO also changes
Bangladesh cold-spell frequency, the strong-vortex association and the SSW deficit (Step 16N)
could be ENSO effects rather than stratospheric ones.

Data: NOAA CPC Oceanic Nino Index (https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt),
downloaded once to 01_raw_data/climate_indices/. The DJF value (row SEAS=DJF, YR=winter_start+1)
is the winter ENSO state. A monthly CSV (year, month, oni; centre month of each 3-month season)
is also written so the earlier optional script 13g can run.
ENSO phase: El Nino if DJF ONI >= 0.5, La Nina if <= -0.5, otherwise neutral.

Tests
 1. Winter level (40 winters): Spearman correlation of cold-spell days with DJF ONI; partial
    Spearman correlation of cold-spell days with winter-mean NAM (10, 100 hPa) controlling for ONI
    (rank-residual method), p-values from 10,000 permutations of winters.
 2. Onset-day NAM against calendar-matched dates drawn only from winters of the SAME ENSO phase.
 3. SSW-centred onset test (as Step 16N) with control dates drawn only from other winters of the
    SAME ENSO phase as the SSW's own winter.
 The within-winter NAM test of Step 16N compares days of the same winter and is ENSO-free by design.

Run from the project root:
    python3 07_scripts/15o_enso_controls.py 2>&1 | tee 09_logs/step16o_enso_controls.log
"""
from __future__ import annotations

import importlib.util
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("term_common", "15_termination_common.py")
N = _load("step16n", "15n_stratosphere_ssw_power_nam.py")

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
ONI_DIR = C.PROJECT_ROOT / "01_raw_data" / "climate_indices"
ONI_RAW = ONI_DIR / "oni.ascii.txt"
ONI_MONTHLY = ONI_DIR / "oni_monthly.csv"
T_OUT = C.TERM_TABLES / "table_109_enso_controls.csv"
T_WINTER = C.TERM_TABLES / "table_110_winter_enso_nam_coldspells.csv"
REPORT = C.TERM_QC / "step16o_enso_controls_report.txt"
N_PERM, N_MC, SEED = 10000, 10000, 20260923
CENTRE = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
          "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}


def get_oni() -> pd.DataFrame:
    ONI_DIR.mkdir(parents=True, exist_ok=True)
    if not ONI_RAW.exists():
        print("Downloading", ONI_URL)
        with urllib.request.urlopen(ONI_URL, timeout=60) as r:
            ONI_RAW.write_bytes(r.read())
    oni = pd.read_csv(ONI_RAW, sep=r"\s+")
    oni.columns = [c.strip().upper() for c in oni.columns]
    oni["month"] = oni["SEAS"].map(CENTRE)
    oni.rename(columns={"YR": "year", "ANOM": "oni"})[["year", "month", "oni"]].to_csv(ONI_MONTHLY, index=False)
    return oni


def phase(x: float) -> str:
    return "el_nino" if x >= 0.5 else ("la_nina" if x <= -0.5 else "neutral")


def partial_spearman(x, y, z) -> float:
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(np.corrcoef(ex, ey)[0, 1])


def main() -> None:
    C.ensure_dirs()
    C.write_policy("step16o_enso_controls_policy.json", {
        "step": "Step 16O", "title": "ENSO controls for stratospheric results", "oni_source": ONI_URL,
        "winter_enso": "DJF ONI (SEAS=DJF, YR=winter_start+1)",
        "phases": {"el_nino": ">= 0.5", "la_nina": "<= -0.5", "neutral": "otherwise"},
        "permutations": N_PERM, "monte_carlo": N_MC, "seed": SEED})
    oni = get_oni()
    djf = oni[oni["SEAS"] == "DJF"].set_index("YR")["ANOM"]
    ev = C.load_events()
    cov = pd.read_csv(C.DAILY_COVARIATES, parse_dates=["date"]).set_index("date")
    winters = np.arange(1985, 2025)
    wdf = pd.DataFrame({"winter_start_year": winters})
    wdf["djf_oni"] = [djf.get(w + 1, np.nan) for w in winters]
    if wdf["djf_oni"].isna().any():
        raise RuntimeError(f"Missing DJF ONI for winters: {wdf.loc[wdf.djf_oni.isna(), 'winter_start_year'].tolist()}")
    wdf["enso_phase"] = wdf["djf_oni"].apply(phase)
    wdf["cold_spell_days"] = wdf["winter_start_year"].map(ev.groupby("winter_start_year")["duration_days"].sum()).fillna(0)
    wdf["cold_spells"] = wdf["winter_start_year"].map(ev.groupby("winter_start_year").size()).fillna(0)
    for col in ("nam_proxy_10", "nam_proxy_100"):
        wdf[f"mean_{col}"] = wdf["winter_start_year"].map(cov.groupby("winter_start_year")[col].mean())
    wdf.to_csv(T_WINTER, index=False, float_format="%.4f")

    rng = np.random.default_rng(SEED)
    rows = []
    r = spearmanr(wdf["djf_oni"], wdf["cold_spell_days"])
    rows.append({"test": "winter_cold_spell_days_vs_djf_oni_spearman", "estimate": r[0], "p": r[1]})
    for col in ("nam_proxy_10", "nam_proxy_100"):
        r = spearmanr(wdf["djf_oni"], wdf[f"mean_{col}"])
        rows.append({"test": f"winter_mean_{col}_vs_djf_oni_spearman", "estimate": r[0], "p": r[1]})
        x, y, z = wdf[f"mean_{col}"].to_numpy(), wdf["cold_spell_days"].to_numpy(), wdf["djf_oni"].to_numpy()
        obs = partial_spearman(x, y, z)
        perm = np.array([partial_spearman(rng.permutation(x), y, z) for _ in range(N_PERM)])
        rows.append({"test": f"partial_spearman_cold_spell_days_vs_{col}_given_oni", "estimate": obs,
                     "p": (1 + (np.abs(perm) >= abs(obs)).sum()) / (N_PERM + 1)})

    # onset-day NAM vs calendar-matched dates from same-ENSO-phase winters
    ph = dict(zip(wdf["winter_start_year"], wdf["enso_phase"]))
    cd = pd.Series(cov.index.strftime("%m-%d"), index=cov.index)
    cov_phase = cov["winter_start_year"].map(ph)
    pools = {(k, p): cov.index[(cd == k) & (cov_phase == p)].to_numpy()
             for k in cd.unique() for p in ("el_nino", "la_nina", "neutral")}
    keys = ev["start_date"].dt.strftime("%m-%d").to_numpy()
    phs = ev["winter_start_year"].map(ph).to_numpy()
    for col in ("nam_proxy_10", "nam_proxy_100"):
        obs = cov.loc[ev["start_date"], col].mean()
        sims = np.empty(N_MC)
        for i in range(N_MC):
            picks = [rng.choice(pools[(k, p)]) for k, p in zip(keys, phs)]
            sims[i] = cov.loc[picks, col].mean()
        rows.append({"test": f"onset_{col}_vs_calendar_and_enso_phase_matched", "estimate": obs - sims.mean(),
                     "ci_low": obs - np.percentile(sims, 97.5), "ci_high": obs - np.percentile(sims, 2.5),
                     "p": min(1.0, 2 * min((sims >= obs).mean(), (sims <= obs).mean()))})

    # SSW test with ENSO-phase-matched control winters
    ssw = pd.read_csv(N.SSW_TABLE, parse_dates=["central_date"])["central_date"]
    ssw = ssw[ssw.apply(N.winter_of).between(1985, 2024)].reset_index(drop=True)
    onsets = ev["start_date"].to_numpy(dtype="datetime64[ns]")
    observed = N.unique_onsets_in_windows(onsets, ssw.to_numpy(dtype="datetime64[ns]"))
    null = np.empty(N_MC, dtype=int)
    for i in range(N_MC):
        ctrl = []
        for d in ssw:
            src = N.winter_of(d)
            same = [w for w in winters if w != src and ph[w] == ph[src]]
            cand = same if same else [w for w in winters if w != src]
            for _ in range(100):
                c = N.shift_to_winter(d, int(rng.choice(cand)))
                if np.all(np.abs((ssw - c).dt.days) > N.EXCLUSION):
                    break
            ctrl.append(c)
        null[i] = N.unique_onsets_in_windows(onsets, np.array(ctrl, dtype="datetime64[ns]"))
    rows.append({"test": "ssw_onsets_day0_45_enso_phase_matched_null", "estimate": observed / null.mean(),
                 "observed": observed, "null_mean": null.mean(), "ci_low": np.percentile(null, 2.5),
                 "ci_high": np.percentile(null, 97.5), "p": (1 + (null <= observed).sum()) / (N_MC + 1)})
    ssw_phases = pd.Series([ph[N.winter_of(d)] for d in ssw]).value_counts().to_dict()

    T = pd.DataFrame(rows)
    T.to_csv(T_OUT, index=False, float_format="%.4f")
    C.write_checksums("step16o_output_sha256.txt", [T_OUT, T_WINTER, ONI_RAW])
    text = ("Step 16O - ENSO controls\n" + "=" * 24 + f"\nWinter ENSO phases: {wdf['enso_phase'].value_counts().to_dict()}\n"
            f"SSWs by ENSO phase: {ssw_phases}\n\n" + T.round(4).to_string(index=False) +
            "\n\nNote: for the SSW test, 'estimate' is the observed/null rate ratio and 'p' is lower one-sided.\n")
    REPORT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
