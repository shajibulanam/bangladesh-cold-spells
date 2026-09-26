"""
Step 16G - Manuscript figures for the termination and cold-day analyses.

Draws six figures from the Step 16C-16F output tables only (no recomputation):
    fig_m2_spell_climatology        cold-night vs cold-day spells per winter; durations; duration by decade
    fig_m4_leakage_free_composites  onset- and end-aligned Tmin/Tmax/DTR for short vs persistent events
    fig_m5_hazard_forest            termination odds ratios for all predictors (FDR highlighted)
    fig_m6_hazard_robustness        odds ratios across catalogues/detrending/lead; cross-validated skill
    fig_m7_termination_sequence     end-aligned all-event composites of the key variables
    fig_m8_cold_day_vs_cold_night   January 2024 case and pure cold-day minus cold-night contrasts

Style: 180 mm wide (double column), sans-serif 8 pt, Okabe-Ito colour-blind-safe palette,
PNG (300 dpi) and PDF. No figure titles inside the image; captions belong in the manuscript.

Run from the project root:
    python3 07_scripts/15g_plot_termination_figures.py 2>&1 | tee 09_logs/step16g_figures.log
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.lines  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "term_common", Path(__file__).resolve().parent / "15_termination_common.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

T = C.TERM_TABLES
OUT = C.TERM_FIGURES
MM = 1 / 25.4
W2 = 180 * MM
OK = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "red": "#D55E00", "purple": "#CC79A7", "grey": "#999999"}
COL_NIGHT, COL_DAY = OK["blue"], OK["orange"]
COL_SHORT, COL_PERS = OK["sky"], OK["red"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "pdf.fonttype": 42,
})

LABELS = {
    "nat_tmin_anom": "Tmin anomaly (°C)", "nat_tmax_anom": "Tmax anomaly (°C)", "nat_dtr_anom": "DTR anomaly (°C)",
    "p10_cold_national_area_fraction": "Cold-area fraction",
    "era5_bd_t2m_anom": "BD T2m", "era5_bd_t850_anom": "BD T850", "era5_bd_stab_t2m_minus_t850_anom": "BD T2m−T850",
    "era5_bd_thick_1000_500_anom": "BD 1000–500 hPa thickness", "era5_bd_z500_anom": "BD Z500",
    "era5_bd_mslp_anom": "BD MSLP", "era5_bd_u850_anom": "BD u850", "era5_bd_v850_anom": "BD v850",
    "era5_bd_wspd10_anom": "BD 10-m wind", "era5_sa_t850_anom": "SA T850", "era5_sa_z500_anom": "SA Z500",
    "era5_sa_mslp_anom": "SA MSLP", "era5_sa_u850_anom": "SA u850", "era5_sa_v850_anom": "SA v850",
    "siberian_high_std_anom": "Siberian High", "ural_blocking_fraction": "Ural blocking",
    "nam_proxy_10": "NAM 10 hPa", "nam_proxy_100": "NAM 100 hPa",
}


def label(v: str) -> str:
    if v in LABELS:
        return LABELS[v]
    if v.endswith("_tend"):
        base = v.replace("_tend", "_anom")
        return "Δ" + LABELS.get(base, base)
    if v.startswith("rrwp_"):
        s = v.replace("rrwp_", "")
        for a, b in (("eurasian_upstream", "Eurasian upstream"), ("central_asia", "Central Asia"),
                     ("bangladesh_sector", "Bangladesh sector"), ("asian_corridor", "Asian corridor")):
            s = s.replace(a, b)
        s = (s.replace("_R_lag7", " R (t−7)").replace("_envelope_z_trailing3", " envelope (t−2…t)")
              .replace("_R_centred_noncausal", " R centred (t±7)"))
        return "RRWP " + s
    return v


def panel(ax, letter: str) -> None:
    ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes, fontweight="bold", ha="right", va="bottom")


def save(fig, name: str) -> list[Path]:
    paths = [OUT / f"{name}.png", OUT / f"{name}.pdf"]
    for p in paths:
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)
    return paths


# ----------------------------------------------------------------------------- figures
DECADE_LABELS = ["1985–\n1994", "1995–\n2004", "2005–\n2014", "2015–\n2024"]


def fig_spell_climatology() -> list[Path]:
    win = pd.read_csv(T / "table_99_cold_day_cold_night_by_winter.csv")
    dec = pd.read_csv(T / "table_94_duration_by_decade.csv")
    tn = C.load_events()
    tx = pd.read_csv(C.PROJECT_ROOT / "06_events" / "step16f_cold_day_spell_catalogue.csv")
    fig = plt.figure(figsize=(W2, 95 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.55, wspace=0.35)
    ax = fig.add_subplot(gs[0, :])
    x = win["winter_start_year"].to_numpy()
    ax.bar(x - 0.2, win["cold_night_spells"], 0.4, color=COL_NIGHT, label="Cold-night (Tmin) spells")
    ax.bar(x + 0.2, win["cold_day_spells"], 0.4, color=COL_DAY, label="Cold-day (Tmax) spells")
    ticks = [y for y in x if y % 5 == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{y}/{str(y + 1)[-2:]}" for y in ticks])
    ax.set_ylabel("Spells per winter")
    ax.set_ylim(0, 6)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    panel(ax, "a")

    ax = fig.add_subplot(gs[1, 0])
    bins = np.arange(2.5, 18.5, 1)
    ax.hist(tn["duration_days"], bins=bins, color=COL_NIGHT, alpha=0.8, label="Cold-night")
    ax.hist(tx["duration_days"], bins=bins, histtype="step", color=COL_DAY, linewidth=1.5, label="Cold-day")
    ax.set_xlabel("Duration (days)")
    ax.set_ylabel("Spells")
    ax.legend(frameon=False)
    panel(ax, "b")

    ax = fig.add_subplot(gs[1, 1])
    xs = np.arange(len(dec))
    ax.bar(xs, dec["mean_duration"], color=COL_NIGHT, width=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(DECADE_LABELS, fontsize=6.5)
    ax.set_ylabel("Mean duration (days)")
    ax.set_title("Cold-night spells", fontsize=7)
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 2])
    ax.bar(xs - 0.18, dec["persistent_events"], 0.36, color=COL_PERS, label="Persistent (≥6 d)")
    ax.bar(xs + 0.18, dec["short_events"], 0.36, color=COL_SHORT, label="Short (3 d)")
    ax.set_xticks(xs)
    ax.set_xticklabels(DECADE_LABELS, fontsize=6.5)
    ax.set_ylabel("Cold-night spells")
    ax.set_ylim(0, 15)
    ax.legend(frameon=False, fontsize=6, ncol=1, loc="upper right")
    panel(ax, "d")
    return save(fig, "fig_m2_spell_climatology")


def fig_leakage_free() -> list[Path]:
    L = pd.read_csv(T / "table_90_onset_end_aligned_lag_composites.csv")
    vars_ = ["nat_tmin_anom", "nat_tmax_anom", "nat_dtr_anom"]
    fig, axes = plt.subplots(2, 3, figsize=(W2, 105 * MM), sharey="col")
    windows = {"onset": [(-3, -1, "pre-onset"), (0, 2, "days 0–2")],
               "end": [(0, 0, "last day"), (1, 2, "after end")]}
    letters = iter("abcdef")
    for r, align in enumerate(["onset", "end"]):
        for c, v in enumerate(vars_):
            ax = axes[r, c]
            for k, (lo, hi, _) in enumerate(windows[align]):
                ax.axvspan(lo - 0.5, hi + 0.5, color=OK["grey"], alpha=0.12 if k == 0 else 0.25, lw=0)
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5 if align == "onset" else 2))
            for g, col in (("short", COL_SHORT), ("persistent", COL_PERS)):
                d = L[(L["align"] == align) & (L["group"] == g)].sort_values("lag")
                ax.plot(d["lag"], d[f"{v}_mean"], marker="o", ms=2.5, color=col,
                        label="Short (3 d)" if g == "short" else "Persistent (≥6 d)")
            ax.axhline(0, color="k", lw=0.5)
            ax.axvline(0, color="k", lw=0.5, ls=":")
            if align == "onset":
                ax.axvline(2.5, color=COL_SHORT, lw=0.8, ls="--")
            ax.set_xlabel("Days from onset" if align == "onset" else "Days from last event day")
            if c == 0:
                ax.set_ylabel(("Onset-aligned\n" if align == "onset" else "End-aligned\n") + LABELS[v])
            else:
                ax.set_ylabel(LABELS[v])
            panel(ax, next(letters))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    return save(fig, "fig_m4_leakage_free_composites")


def fig_hazard_forest() -> list[Path]:
    H = pd.read_csv(T / "table_91_termination_hazard_odds_ratios.csv")
    fam_names = {"station_temperature": "Station temperature", "era5_state": "ERA5 state (day k)",
                 "era5_tendency": "ERA5 day-to-day change", "remote_causal": "Remote drivers (past only)",
                 "noncausal_diagnostic": "Non-causal diagnostic"}
    H = H.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(W2 * 0.62, 150 * MM))
    y = np.arange(len(H))
    for i, r in H.iterrows():
        noncausal = r["family"] == "noncausal_diagnostic"
        sig = bool(r["fdr_significant"]) if not noncausal else False
        col = OK["red"] if sig else (OK["grey"] if noncausal else OK["black"])
        ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=col, lw=1)
        ax.plot(r["odds_ratio_per_sd"], i, "o", ms=4, color=col, mfc="white" if noncausal else col)
    ax.axvline(1, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.set_xticks([0.3, 0.5, 1, 2, 3])
    ax.set_xticklabels(["0.3", "0.5", "1", "2", "3"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(y)
    ax.set_yticklabels([label(p) for p in H["predictor"]], fontsize=6.5)
    ax.set_xlabel("Odds ratio of termination per 1 SD (95% CI)")
    fam = H["family"].to_numpy()
    for f in dict.fromkeys(fam):
        idx = np.flatnonzero(fam == f)
        ax.axhspan(idx.min() - 0.5, idx.max() + 0.5, color=OK["grey"],
                   alpha=0.08 if list(dict.fromkeys(fam)).index(f) % 2 == 0 else 0.0, lw=0)
        ax.text(1.02, idx.mean(), fam_names[f], transform=ax.get_yaxis_transform(), fontsize=6.5,
                va="center", ha="left", color=OK["grey"] if f == "noncausal_diagnostic" else "k")
    ax.set_ylim(-0.7, len(H) - 0.3)
    return save(fig, "fig_m5_hazard_forest")


def fig_hazard_robustness() -> list[Path]:
    """Supplementary Fig. S1: robustness matrix (a) and discrimination skill under three validation schemes (b)."""
    R = pd.read_csv(T / "table_95_hazard_robustness.csv")
    S = pd.read_csv(T / "table_98_hazard_cross_validated_skill.csv").set_index("model")
    split_file = T / "table_117_split_period_validation.csv"
    V = pd.read_csv(split_file) if split_file.exists() else None
    order_a = ["primary", "detrended_covariates", "catalogue:p10_area20_duration3_gap1",
               "catalogue:p10_area10_duration3", "catalogue:p10_area30_duration3",
               "catalogue:p10_area20_duration4", "catalogue:p05_area20_duration3",
               "catalogue:bmd_area20_duration3", "lead_1_day"]
    names_a = ["Primary", "Detrended", "1-day gap merge", "Area ≥10%", "Area ≥30%", "Duration ≥4 d",
               "p05 threshold", "BMD ≤10 °C", "1-day lead"]
    preds = list(dict.fromkeys(R.loc[R["analysis"] == "primary", "predictor"]))
    fig = plt.figure(figsize=(W2, 95 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.45], wspace=0.62)
    ax = fig.add_subplot(gs[0])
    cmap = plt.get_cmap("RdBu_r")
    norm = matplotlib.colors.TwoSlopeNorm(vmin=np.log(0.3), vcenter=0, vmax=np.log(3.3))
    for j, a_ in enumerate(order_a):
        for i, p in enumerate(preds):
            r = R[(R["analysis"] == a_) & (R["predictor"] == p)]
            if r.empty:
                continue
            r = r.iloc[0]
            col = cmap(norm(np.log(r["odds_ratio_per_sd"])))
            if r["ci_excludes_one"]:
                ax.scatter(j, i, s=80, color=col, edgecolor="k", linewidth=1.0, marker="o", zorder=3)
            else:
                ax.scatter(j, i, s=70, color=col, edgecolor="0.55", linewidth=0.6, marker="s", zorder=3)
    ax.axvline(len(order_a) - 1.5, color="0.6", lw=0.6, ls="--")
    ax.set_xticks(range(len(order_a)))
    ax.set_xticklabels(names_a, rotation=40, ha="right")
    ax.set_yticks(range(len(preds)))
    ax.set_yticklabels([label(p) for p in preds])
    ax.invert_yaxis()
    ax.set_xlim(-0.6, len(order_a) - 0.4)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_ticks(np.log([0.3, 0.5, 1, 2, 3]))
    cb.set_ticklabels(["0.3", "0.5", "1", "2", "3"])
    cb.set_label("Odds ratio of termination per SD")
    key = [matplotlib.lines.Line2D([], [], marker="o", ls="", markerfacecolor="0.8", markeredgecolor="k",
                                   markersize=6, label="95% CI excludes 1"),
           matplotlib.lines.Line2D([], [], marker="s", ls="", markerfacecolor="0.8", markeredgecolor="0.55",
                                   markersize=6, label="95% CI includes 1")]
    ax.legend(handles=key, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, frameon=False, fontsize=6.5)
    panel(ax, "a")

    ax = fig.add_subplot(gs[1])
    schemes = [("Leave-one-winter-out", S.loc["adjustment_only", "cv_auc"], S.loc["adjustment_plus_synoptic", "cv_auc"],
                S.loc["auc_difference", "cv_auc"], S.loc["auc_difference", "ci_low"], S.loc["auc_difference", "ci_high"])]
    if V is not None:
        for r in V.itertuples():
            tr = r.train.replace("1985/86-2004/05", "1985–2004").replace("2005/06-2024/25", "2005–2024")
            te = r.test.replace("1985/86-2004/05", "1985–2004").replace("2005/06-2024/25", "2005–2024")
            schemes.append((f"Train {tr} → test {te}", r.auc_adjustment_only, r.auc_plus_synoptic,
                            r.auc_difference, r.auc_diff_ci_low, r.auc_diff_ci_high))
    colors = [OK["black"], OK["blue"], OK["orange"]]
    fmt = lambda x: f"{x:.3f}" if abs(x) < 0.01 else f"{x:.2f}"
    for k, (name, base, full, d, lo, hi) in enumerate(schemes):
        ax.plot([0, 1], [base, full], color=colors[k], marker="o", ms=5, lw=1.3,
                label=f"{name}\nΔAUC {fmt(d)} [{fmt(lo)}, {fmt(hi)}]")
    ax.axhline(0.5, color="0.6", lw=0.6, ls="--")
    ax.text(1.22, 0.505, "no skill", fontsize=6, color="0.4", va="bottom", ha="right")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), frameon=False, fontsize=6, handlelength=1.4,
              labelspacing=0.6)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0.45, 0.95)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Adjustment\nonly", "+ synoptic\npredictors"])
    ax.set_ylabel("AUC of termination (held-out data)")
    panel(ax, "b")
    return save(fig, "fig_m6_hazard_robustness")


def fig_termination_sequence() -> list[Path]:
    """End-aligned all-event composites; draws 95% bootstrap bands if Step 16H table_102 exists."""
    band_file = T / "table_102_end_aligned_composites_with_ci.csv"
    bands = pd.read_csv(band_file) if band_file.exists() else None
    L = pd.read_csv(T / "table_90_onset_end_aligned_lag_composites.csv")
    L = L[L["align"] == "end"]
    vars_ = ["nat_tmin_anom", "nat_tmax_anom", "nat_dtr_anom", "era5_bd_wspd10_anom",
             "era5_bd_t850_anom", "era5_bd_v850_anom", "era5_bd_mslp_anom", "era5_sa_z500_anom"]
    units = {"era5_bd_wspd10_anom": "m s⁻¹", "era5_bd_t850_anom": "K", "era5_bd_v850_anom": "m s⁻¹",
             "era5_bd_mslp_anom": "hPa", "era5_sa_z500_anom": "m"}
    fig, axes = plt.subplots(2, 4, figsize=(W2, 85 * MM), sharex=True)
    letters = iter("abcdefgh")
    for ax, v in zip(axes.flat, vars_):
        ax.axvspan(-0.5, 0.5, color=OK["grey"], alpha=0.2, lw=0)
        if bands is not None:
            b = bands[bands["variable"] == v].sort_values("lag")
            ax.fill_between(b["lag"], b["ci_low"], b["ci_high"], color=OK["blue"], alpha=0.2, lw=0)
            ax.plot(b["lag"], b["mean"], color="k", marker="o", ms=2.5)
        else:
            g = L.groupby("lag").apply(lambda d: np.average(d[f"{v}_mean"], weights=d[f"{v}_count"]),
                                       include_groups=False)
            ax.plot(g.index, g.values, color="k", marker="o", ms=2.5)
        ax.axhline(0, color="k", lw=0.5)
        name = LABELS[v] if v.startswith("nat_") else f"{label(v)} ({units[v]})"
        ax.set_title(name, fontsize=7)
        panel(ax, next(letters))
    for ax in axes[1]:
        ax.set_xlabel("Days from last event day")
    for ax in axes.flat:
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    fig.tight_layout()
    return save(fig, "fig_m7_termination_sequence")


def fig_cold_day_vs_night() -> list[Path]:
    tn_daily = C.read_table(C.DAILY_AREA_DIAG, columns=["date", "p10_cold_national_area_fraction"])
    tn_daily["date"] = pd.to_datetime(tn_daily["date"])
    tx_daily = pd.read_csv(C.TERM_INTERMEDIATE / "step16f_daily_cold_day_area.csv", parse_dates=["date"])
    tx = pd.read_csv(C.PROJECT_ROOT / "06_events" / "step16f_cold_day_spell_catalogue.csv",
                     parse_dates=["start_date", "end_date"])
    tn = C.load_events()
    K = pd.read_csv(T / "table_101_cold_day_vs_cold_night_contrast.csv")
    fig = plt.figure(figsize=(W2, 100 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.95)
    ax = fig.add_subplot(gs[0])
    lo, hi = pd.Timestamp("2023-12-01"), pd.Timestamp("2024-02-29")
    a = tn_daily[(tn_daily["date"] >= lo) & (tn_daily["date"] <= hi)]
    b = tx_daily[(tx_daily["date"] >= lo) & (tx_daily["date"] <= hi)]
    for r in tx[(tx["end_date"] >= lo) & (tx["start_date"] <= hi)].itertuples():
        ax.axvspan(r.start_date - pd.Timedelta(hours=12), r.end_date + pd.Timedelta(hours=12),
                   color=COL_DAY, alpha=0.15, lw=0)
    for r in tn[(tn["end_date"] >= lo) & (tn["start_date"] <= hi)].itertuples():
        ax.axvspan(r.start_date - pd.Timedelta(hours=12), r.end_date + pd.Timedelta(hours=12),
                   color=COL_NIGHT, alpha=0.15, lw=0)
    ax.plot(a["date"], a["p10_cold_national_area_fraction"], color=COL_NIGHT, label="Tmin < p10 (cold night)")
    ax.plot(b["date"], b["cold_area"], color=COL_DAY, label="Tmax < p10 (cold day)")
    ax.axhline(0.2, color="k", lw=0.6, ls="--")
    ax.set_ylabel("Fraction of national area")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper right")
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    ax.set_xlabel("Winter 2023/24")
    panel(ax, "a")

    ax = fig.add_subplot(gs[1])
    rows = []
    groups = [("Station temperature", K, [("nat_tmin_anom", "Tmin", "°C"), ("nat_tmax_anom", "Tmax", "°C"),
                                          ("nat_dtr_anom", "DTR", "°C")]),
              ("ERA5 dynamics, 06 LT", K, [("era5_bd_t2m_anom", "BD T2m", "°C"),
                                           ("era5_bd_stab_t2m_minus_t850_anom", "BD T2m − T850", "K"),
                                           ("era5_bd_t850_anom", "BD T850", "K"), ("era5_bd_v850_anom", "BD v850", "m s⁻¹"),
                                           ("era5_bd_wspd10_anom", "BD 10-m wind", "m s⁻¹")])]
    f113 = T / "table_113_cold_day_vs_cold_night_cloud_radiation.csv"
    if f113.exists():
        groups.append(("Cloud, humidity, radiation", pd.read_csv(f113), [
            ("tcc_mean", "Total cloud", ""), ("lowcloud1000_frac_00utc", "Cloud base <1 km, 06 LT", ""),
            ("fog300_frac_00utc", "Cloud base <300 m, 06 LT", ""), ("strd_wm2", "Downward longwave", "W m⁻²"),
            ("ssrd_wm2", "Downward solar", "W m⁻²"), ("td2m_mean", "2-m dewpoint", "°C"),
            ("dpd_00utc", "Dewpoint depression, 06 LT", "K"), ("blh_06utc", "Boundary layer, 12 LT", "m")]))
    f115 = T / "table_115_precipitation_cold_day_vs_night.csv"
    if f115.exists():
        groups.append(("Precipitation", pd.read_csv(f115), [
            ("tp_utcday_mm", "Precipitation", "mm d⁻¹"), ("wet_day_fraction", "Days with ≥1 mm", "")]))
    for gname, tab, items in groups:
        tab = tab.set_index("variable")
        for var, name, unit in items:
            if var not in tab.index:
                continue
            r = tab.loc[var]
            se = (r["ci_high"] - r["ci_low"]) / 3.92
            z, zlo, zhi = r["difference"] / se, r["ci_low"] / se, r["ci_high"] / se
            rows.append((gname, name, unit, r["difference"], z, zlo, zhi, r["q_fdr"]))
    # y positions: one extra row above each group for its header
    ypos, headers, cur, prev = [], [], 0.0, None
    for r in rows:
        if r[0] != prev:
            headers.append((r[0], cur))
            cur -= 1.0
            prev = r[0]
        ypos.append(cur)
        cur -= 1.0
    y = np.array(ypos)
    for yi, (gname, name, unit, d, z, zlo, zhi, q) in zip(y, rows):
        col = OK["red"] if q < 0.05 else OK["grey"]
        ax.plot([zlo, zhi], [yi, yi], color=col, lw=1)
        ax.plot(z, yi, "o", color=col, ms=3.5)
        dtxt = (f"{d:+.2f}" if abs(d) < 10 else f"{d:+.0f}") + (f" {unit}" if unit else "")
        ax.text(1.02, yi, dtxt.replace("-", "−"), transform=ax.get_yaxis_transform(), fontsize=5.8, va="center")
    for k, (gname, yh) in enumerate(headers):
        ax.text(0.0, yh, gname, transform=ax.get_yaxis_transform(), fontsize=6.3, ha="right", va="center",
                fontweight="bold", color="0.25")
        if k > 0:
            ax.axhline(yh + 0.5, color="0.85", lw=0.5)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([r[1] for r in rows], fontsize=6.3)
    ax.set_ylim(y.min() - 0.7, 0.7)
    ax.set_xlabel("Cold-day − cold-night difference / bootstrap SE")
    panel(ax, "b")
    return save(fig, "fig_m8_cold_day_vs_cold_night")


def main() -> None:
    C.ensure_dirs()
    C.write_policy("step16g_figure_policy.json", {
        "step": "Step 16G", "title": "Manuscript figures for Step 16",
        "inputs": "Step 16C-16F tables and catalogues only; no statistics are recomputed",
        "style": {"width_mm": 180, "font_pt": 8, "palette": "Okabe-Ito", "formats": ["png 300 dpi", "pdf"]},
    })
    paths = []
    for f in (fig_spell_climatology, fig_leakage_free, fig_hazard_forest, fig_hazard_robustness,
              fig_termination_sequence, fig_cold_day_vs_night):
        paths += f()
    manifest = pd.DataFrame({"file": [str(p.relative_to(C.PROJECT_ROOT)) for p in paths]})
    manifest.to_csv(C.TERM_QC / "step16g_figure_manifest.csv", index=False)
    C.write_checksums("step16g_output_sha256.txt", paths)
    print(f"{len(paths)} files written to {OUT.relative_to(C.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
