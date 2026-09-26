"""
Step 16U - Final versions of manuscript Fig. 6 and Fig. 8 (redraw only; no statistics recomputed).

Fig. 6  fig_final_06_hazard_forest_three_families
        (a) primary family (Step 16D, table_91), (b) advection/jet family (Step 16M, table_106),
        (c) cloud/humidity/radiation family (Step 16Q, table_111). Red = FDR q < 0.05 within family;
        open grey = non-causal centred RRWP R (panel a only).
Fig. 8  fig_final_08_termination_sequence
        End-aligned all-event composites with 95% winter-block bootstrap bands, 3 x 4 panels:
        surface temperatures and wind (Step 16H table_102), dynamics (table_102) and moisture/radiation
        (Step 16Q table_112).

Run from the project root:
    python3 07_scripts/15u_plot_final_hazard_and_termination.py 2>&1 | tee 09_logs/step16u_final_figs.log
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("term_common", "15_termination_common.py")
G = _load("step16g", "15g_plot_termination_figures.py")  # shared style, labels, save()

T = C.TERM_TABLES
MM = 1 / 25.4
EXTRA_LABELS = {
    "mech_bd_adv850_k_day_anom": "Adv850, Bangladesh", "mech_nigp_adv850_k_day_anom": "Adv850, NW IGP",
    "mech_jet200_core_lat_70_100e_anom": "Jet latitude", "mech_jet200_core_speed_70_100e_anom": "Jet speed",
    "mech_bd_adv850_k_day_tend": "ΔAdv850, Bangladesh", "mech_nigp_adv850_k_day_tend": "ΔAdv850, NW IGP",
    "mech_jet200_core_lat_70_100e_tend": "ΔJet latitude", "mech_jet200_core_speed_70_100e_tend": "ΔJet speed",
    "cld_strd_wm2_anom": "Downward longwave", "cld_ssrd_wm2_anom": "Downward solar",
    "cld_tcc_mean_anom": "Total cloud (daily)", "cld_tcc_00utc_anom": "Total cloud, 06 LT",
    "cld_lowcloud1000_frac_00utc_anom": "Cloud base <1 km, 06 LT", "cld_fog300_frac_00utc_anom": "Cloud base <300 m, 06 LT",
    "cld_td2m_mean_anom": "2-m dewpoint", "cld_dpd_00utc_anom": "Dewpoint depression, 06 LT",
    "cld_blh_06utc_anom": "Boundary layer, 12 LT", "cld_strd_wm2_tend": "ΔDownward longwave",
    "cld_tcc_mean_tend": "ΔTotal cloud", "cld_lowcloud1000_frac_00utc_tend": "ΔCloud base <1 km, 06 LT",
    "cld_td2m_mean_tend": "Δ2-m dewpoint", "cld_dpd_00utc_tend": "ΔDewpoint depression, 06 LT",
    "win_strd_day_anom": "Downward longwave, 06–17 LT", "win_ssrd_day_anom": "Downward solar, 06–17 LT",
    "win_tcc_day_anom": "Total cloud, 06–17 LT", "win_td2m_day_anom": "2-m dewpoint, 06–17 LT",
    "win_low1000_day_anom": "Cloud base <1 km, 06–17 LT", "win_fog300_day_anom": "Cloud base <300 m, 06–17 LT",
    "win_blh_day_anom": "Boundary layer, 06–17 LT", "dly_tcc_00utc_anom": "Total cloud, 06 LT",
    "dly_lowcloud1000_frac_00utc_anom": "Cloud base <1 km, 06 LT", "dly_fog300_frac_00utc_anom": "Cloud base <300 m, 06 LT",
    "dly_dpd_00utc_anom": "Dewpoint depression, 06 LT", "win_strd_day_tend": "ΔDownward longwave, 06–17 LT",
    "win_td2m_day_tend": "Δ2-m dewpoint, 06–17 LT", "win_tcc_day_tend": "ΔTotal cloud, 06–17 LT",
    "win_strd_night_anom": "Downward longwave, 18–05 LT", "win_tcc_night_anom": "Total cloud, 18–05 LT",
    "win_td2m_night_anom": "2-m dewpoint, 18–05 LT", "win_low1000_night_anom": "Cloud base <1 km, 18–05 LT",
    "win_fog300_night_anom": "Cloud base <300 m, 18–05 LT",
}


def lab(p: str) -> str:
    return EXTRA_LABELS.get(p, G.label(p))


def forest(ax, df: pd.DataFrame, qcol: str, noncausal: pd.Series | None = None) -> None:
    df = df.iloc[::-1].reset_index(drop=True)
    if noncausal is not None:
        noncausal = pd.Series(noncausal).iloc[::-1].reset_index(drop=True)
    for i, r in df.iterrows():
        nc = bool(noncausal[i]) if noncausal is not None else False
        sig = (not nc) and bool(r[qcol] < 0.05)
        col = G.OK["red"] if sig else (G.OK["grey"] if nc else "k")
        ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=col, lw=1)
        ax.plot(r["odds_ratio_per_sd"], i, "o", ms=3.5, color=col, mfc="white" if nc else col)
    ax.axvline(1, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.set_xticks([0.3, 0.5, 1, 2, 4])
    ax.set_xticklabels(["0.3", "0.5", "1", "2", "4"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(0.25, 9)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([lab(p) for p in df["predictor"]], fontsize=6.3)
    ax.set_ylim(-0.7, len(df) - 0.3)


def forest_open(ax, df: pd.DataFrame, qcol: str, open_mask: pd.Series | None) -> None:
    """Forest plot where open markers denote concurrent (night) predictors; red = q < 0.05 within family."""
    df = df.iloc[::-1].reset_index(drop=True)
    om = pd.Series(open_mask).iloc[::-1].reset_index(drop=True) if open_mask is not None else pd.Series(False, index=df.index)
    for i, r in df.iterrows():
        col = G.OK["red"] if r[qcol] < 0.05 else "k"
        ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=col, lw=1)
        ax.plot(r["odds_ratio_per_sd"], i, "s" if om[i] else "o", ms=3.5, color=col, mfc="white" if om[i] else col)
    if om.any() and (~om).any():
        ax.axhline(om[om].index.max() + 0.5, color="0.6", lw=0.6, ls="--")
    ax.axvline(1, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.set_xticks([0.3, 0.5, 1, 2, 4, 8])
    ax.set_xticklabels(["0.3", "0.5", "1", "2", "4", "8"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(0.25, 14)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([lab(p) for p in df["predictor"]], fontsize=6.3)
    ax.set_ylim(-0.7, len(df) - 0.3)


def fig6() -> list[Path]:
    a = pd.read_csv(T / "table_91_termination_hazard_odds_ratios.csv")
    b = pd.read_csv(T / "table_106_mechanism_extension_hazard.csv")
    t114 = T / "table_114_prenight_vs_night_hazard.csv"
    if t114.exists():  # pre-night family (filled) and terminating-night family (open, concurrent)
        c = pd.read_csv(t114)
        c = pd.concat([c[c["family"] == "prenight"], c[c["family"] == "night_overlapping"]], ignore_index=True)
        c_open = c["family"] == "night_overlapping"
        c_title = "(c) Cloud, humidity, radiation: before (filled) / during (open) night"
    else:
        c = pd.read_csv(T / "table_111_cloud_radiation_hazard.csv")
        c_open = None
        c_title = "(c) Cloud, humidity and radiation"
    fig = plt.figure(figsize=(180 * MM, 165 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[len(b) + 3, len(c) + 3],
                          wspace=0.95, hspace=0.25)
    ax = fig.add_subplot(gs[:, 0])
    forest(ax, a.rename(columns={"q_fdr_causal": "q"}).assign(q=lambda d: d["q"].fillna(1.0)), "q",
           noncausal=(a["family"] == "noncausal_diagnostic"))
    ax.set_title("(a) Primary family (39 causal + 4 diagnostic)", loc="left")
    ax.set_xlabel("Odds ratio of termination per SD")
    ax = fig.add_subplot(gs[0, 1])
    forest(ax, b, "q_fdr_family")
    ax.set_title("(b) Advection and jet", loc="left")
    ax = fig.add_subplot(gs[1, 1])
    forest_open(ax, c, "q_fdr_family", c_open)
    ax.set_title(c_title, loc="left", fontsize=7)
    ax.set_xlabel("Odds ratio of termination per SD")
    return G.save(fig, "fig_final_06_hazard_forest_three_families")


def fig8() -> list[Path]:
    dyn = pd.read_csv(T / "table_102_end_aligned_composites_with_ci.csv")
    cld = pd.read_csv(T / "table_112_end_aligned_cloud_radiation.csv")
    panels = [
        (dyn, "nat_tmin_anom", "Tmin anomaly", "°C"), (dyn, "nat_tmax_anom", "Tmax anomaly", "°C"),
        (dyn, "nat_dtr_anom", "DTR anomaly", "°C"), (dyn, "era5_bd_wspd10_anom", "10-m wind, BD", "m s⁻¹"),
        (dyn, "era5_bd_v850_anom", "v850, BD", "m s⁻¹"), (dyn, "era5_bd_t850_anom", "T850, BD", "K"),
        (dyn, "era5_bd_mslp_anom", "MSLP, BD", "hPa"), (dyn, "era5_sa_z500_anom", "Z500, South Asia", "m"),
        (cld, "td2m_mean", "2-m dewpoint, BD", "°C"), (cld, "strd_wm2", "Downward longwave", "W m⁻²"),
        (cld, "ssrd_wm2", "Downward solar", "W m⁻²"), (cld, "tcc_mean", "Total cloud", "fraction"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(180 * MM, 120 * MM), sharex=True)
    for ax, (tab, var, name, unit), letter in zip(axes.flat, panels, "abcdefghijkl"):
        t = tab[tab["variable"] == var].sort_values("lag")
        ax.axvspan(-0.5, 0.5, color=G.OK["grey"], alpha=0.2, lw=0)
        ax.fill_between(t["lag"], t["ci_low"], t["ci_high"], color=G.OK["blue"], alpha=0.2, lw=0)
        ax.plot(t["lag"], t["mean"], color="k", marker="o", ms=2.3)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"({letter}) {name} ({unit})", loc="left", fontsize=7)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    for ax in axes[-1]:
        ax.set_xlabel("Days from last event day")
    fig.tight_layout()
    return G.save(fig, "fig_final_08_termination_sequence")


def main() -> None:
    C.ensure_dirs()
    paths = fig6() + fig8()
    C.write_checksums("step16u_output_sha256.txt", paths)
    print(f"{len(paths)} files written to {C.TERM_FIGURES.relative_to(C.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
