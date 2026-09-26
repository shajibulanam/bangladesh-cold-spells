"""
Step 16Q - How do cold spells end, and are cold-day spells fog/low-cloud events?

Input: Step 16K3 daily ERA5 ARCO Bangladesh-box series (cloud, cloud-base fractions, humidity,
boundary layer, downward radiation). Anomalies: 1991-2020 calendar-day mean, +/-7-day window;
tendency = day-to-day change in anomaly.

Part A - Cloud/radiation family of the termination hazard model. Each predictor is added to the
         Step 16D adjustment set (log event age, season day, Tmin anomaly, year); winter-block
         bootstrap (2,000); Benjamini-Hochberg FDR within this family only.
Part B - End-aligned all-event composites (days -5..+5) with 95% winter-block bootstrap bands
         (5,000) and a figure (fig_m7b_termination_cloud_radiation).
Part C - Pure cold-day minus pure cold-night spell means (1985/86-2023/24), winter-block bootstrap
         (5,000), BH FDR across variables. Pure = shares no day with a spell of the other type.

Run from the project root:
    python3 07_scripts/15q_cloud_radiation_analysis.py 2>&1 | tee 09_logs/step16q_cloud_radiation.log
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

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("term_common", "15_termination_common.py")
E = _load("step16e", "15e_hazard_robustness_and_trend.py")

CLOUD = C.PROJECT_ROOT / "03_intermediate" / "era5_box_daily" / "era5_cloud_radiation_daily_oct_mar.csv"
TX_CATALOGUE = C.PROJECT_ROOT / "06_events" / "step16f_cold_day_spell_catalogue.csv"
T_HAZ = C.TERM_TABLES / "table_111_cloud_radiation_hazard.csv"
T_END = C.TERM_TABLES / "table_112_end_aligned_cloud_radiation.csv"
T_TYPE = C.TERM_TABLES / "table_113_cold_day_vs_cold_night_cloud_radiation.csv"
REPORT = C.TERM_QC / "step16q_cloud_radiation_report.txt"
FIG = "fig_m7b_termination_cloud_radiation"
N_BOOT, N_BOOT_HAZ, SEED = 5000, 2000, 20260923
LAST_TMAX_WINTER = 2023

STATE = ["strd_wm2", "ssrd_wm2", "tcc_mean", "tcc_00utc", "lowcloud1000_frac_00utc", "fog300_frac_00utc",
         "td2m_mean", "dpd_00utc", "blh_06utc"]
TEND = ["strd_wm2", "tcc_mean", "lowcloud1000_frac_00utc", "td2m_mean", "dpd_00utc"]
LABELS = {"strd_wm2": ("Downward longwave", "W m⁻²"), "ssrd_wm2": ("Downward solar", "W m⁻²"),
          "tcc_mean": ("Total cloud (daily)", "fraction"), "tcc_00utc": ("Total cloud, 06 LT", "fraction"),
          "lowcloud1000_frac_00utc": ("Cloud base <1 km, 06 LT", "area fraction"),
          "lowcloud1000_frac_mean": ("Cloud base <1 km (daily)", "area fraction"),
          "fog300_frac_00utc": ("Cloud base <300 m, 06 LT", "area fraction"),
          "td2m_mean": ("2-m dewpoint", "°C"), "dpd_00utc": ("Dewpoint depression, 06 LT", "K"),
          "blh_06utc": ("Boundary layer, 12 LT", "m")}
PANEL_VARS = ["strd_wm2", "ssrd_wm2", "tcc_mean", "lowcloud1000_frac_00utc", "td2m_mean", "dpd_00utc",
              "blh_06utc", "fog300_frac_00utc"]


def load_cloud_anomalies() -> pd.DataFrame:
    raw = pd.read_csv(CLOUD, parse_dates=["date"]).set_index("date")
    cols = sorted(set(STATE + TEND + ["lowcloud1000_frac_mean"]))
    anom = C.calendar_anomalies(raw[cols], 7)
    tend = anom[TEND].diff()
    tend.columns = [f"cld_{c}_tend" for c in TEND]
    anom.columns = [f"cld_{c}_anom" for c in anom.columns]
    return anom.join(tend)


def boot_mean(x: np.ndarray, winters: np.ndarray) -> tuple[float, float, float]:
    ok = ~np.isnan(x)
    uniq, counts = C.winter_bootstrap_counts(winters[ok], N_BOOT, SEED)
    W = counts[:, np.searchsorted(uniq, winters[ok])].astype(float)
    b = (W @ x[ok]) / W.sum(axis=1)
    return float(x[ok].mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> None:
    C.ensure_dirs()
    C.write_policy("step16q_cloud_radiation_policy.json", {
        "step": "Step 16Q", "input": str(CLOUD.relative_to(C.PROJECT_ROOT)),
        "hazard_family": [f"cld_{v}_anom" for v in STATE] + [f"cld_{v}_tend" for v in TEND],
        "hazard_adjustment": E.ADJ, "hazard_bootstrap": N_BOOT_HAZ, "bootstrap": N_BOOT, "seed": SEED,
        "multiple_testing": "BH within the cloud/radiation hazard family; BH across variables in Part C",
        "cold_day_period": f"1985/86-{LAST_TMAX_WINTER}/{str(LAST_TMAX_WINTER + 1)[-2:]}"})
    cloud = load_cloud_anomalies()
    cov = C.load_covariates_with_mechanisms().join(cloud)
    ev = C.load_events()

    # Part A
    rs = E.risk_set(cov, ev, 3)
    fam = [f"cld_{v}_anom" for v in STATE] + [f"cld_{v}_tend" for v in TEND]
    H = pd.DataFrame([{"predictor": p, **E.odds_ratio(rs, p, N_BOOT_HAZ)} for p in fam])
    H["q_fdr_family"] = C.benjamini_hochberg(H["p_bootstrap"].to_numpy())
    H.to_csv(T_HAZ, index=False, float_format="%.4f")
    print("Part A done", flush=True)

    # Part B
    winters = ev["winter_start_year"].to_numpy()
    rows = []
    for lag in range(-5, 6):
        vals = cov.reindex(ev["end_date"] + pd.Timedelta(days=lag))
        for v in PANEL_VARS:
            m, lo, hi = boot_mean(vals[f"cld_{v}_anom"].to_numpy(float), winters)
            rows.append({"variable": v, "lag": lag, "mean": m, "ci_low": lo, "ci_high": hi})
    B = pd.DataFrame(rows)
    B.to_csv(T_END, index=False, float_format="%.4f")

    MM = 1 / 25.4
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.titlesize": 7,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 4, figsize=(180 * MM, 85 * MM), sharex=True)
    for ax, v, letter in zip(axes.flat, PANEL_VARS, "abcdefgh"):
        b = B[B["variable"] == v]
        ax.axvspan(-0.5, 0.5, color="#999999", alpha=0.2, lw=0)
        ax.fill_between(b["lag"], b["ci_low"], b["ci_high"], color="#0072B2", alpha=0.2, lw=0)
        ax.plot(b["lag"], b["mean"], color="k", marker="o", ms=2.5)
        ax.axhline(0, color="k", lw=0.5)
        name, unit = LABELS[v]
        ax.set_title(f"({letter}) {name} ({unit})", loc="left")
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    for ax in axes[-1]:
        ax.set_xlabel("Days from last event day")
    fig.tight_layout()
    paths = [C.TERM_FIGURES / f"{FIG}.png", C.TERM_FIGURES / f"{FIG}.pdf"]
    for p in paths:
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("Part B done", flush=True)

    # Part C
    tx = pd.read_csv(TX_CATALOGUE, parse_dates=["start_date", "end_date"])
    tn = ev[ev["winter_start_year"] <= LAST_TMAX_WINTER]
    days = lambda df: {d for r in df.itertuples() for d in pd.date_range(r.start_date, r.end_date)}
    tn_days, tx_days = days(tn), days(tx)
    evm = []
    for kind, df, other in (("cold_day", tx, tn_days), ("cold_night", tn, tx_days)):
        for r in df.itertuples():
            rng_ = pd.date_range(r.start_date, r.end_date)
            if set(rng_) & other:
                continue
            m = cov.reindex(rng_)[[f"cld_{v}_anom" for v in PANEL_VARS]].mean()
            evm.append({"type": kind, "winter": r.winter_start_year, **m.to_dict()})
    EV = pd.DataFrame(evm)
    out = []
    for v in PANEL_VARS:
        col = f"cld_{v}_anom"
        r = C.group_difference_bootstrap(EV[col].to_numpy(float), (EV["type"] == "cold_day").to_numpy(),
                                         (EV["type"] == "cold_night").to_numpy(), EV["winter"].to_numpy(),
                                         N_BOOT, SEED)
        out.append({"variable": v, "cold_day_mean": EV.loc[EV["type"] == "cold_day", col].mean(),
                    "cold_night_mean": EV.loc[EV["type"] == "cold_night", col].mean(), **r})
    T = pd.DataFrame(out)
    T["q_fdr"] = C.benjamini_hochberg(T["p_bootstrap"].to_numpy())
    T.to_csv(T_TYPE, index=False, float_format="%.4f")

    C.write_checksums("step16q_output_sha256.txt", [T_HAZ, T_END, T_TYPE] + paths)
    wide = B.pivot(index="lag", columns="variable", values="mean")[PANEL_VARS].round(3)
    text = ("Step 16Q - Cloud, humidity and radiation\n" + "=" * 40 +
            "\nA. Termination hazard, cloud/radiation family\n" +
            H[["predictor", "odds_ratio_per_sd", "ci_low", "ci_high", "p_bootstrap", "q_fdr_family"]]
            .round(3).to_string(index=False) +
            "\n\nB. End-aligned all-event anomalies (means)\n" + wide.to_string() +
            "\n\nC. Pure cold-day minus pure cold-night\n" +
            T[["variable", "cold_day_mean", "cold_night_mean", "difference", "ci_low", "ci_high", "q_fdr",
               "n_a", "n_b"]].round(3).to_string(index=False) + "\n")
    REPORT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
