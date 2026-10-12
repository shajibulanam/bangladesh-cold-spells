"""
16g_final_advection_pulse_figure.py - final figure: cold- and warm-advection passages along the Indo-Gangetic corridor.

(a) Onset-aligned and (b) end-aligned composites (days -5..+2) of 850-hPa horizontal temperature advection
    -(u dT/dx + v dT/dy) anomalies over the northwestern Indo-Gangetic Plain (NW IGP, 25-32N, 75-88E) and Bangladesh
    (BD, 20.5-26.8N, 88-93E), all 85 cold-night spells, 95% winter-block bootstrap intervals (table_R11b).
(c) Odds ratio of termination per SD of NW IGP advection taken 1 or 2 days earlier, and the robustness checks
    (season-end censoring, stable 25-station catalogue, same-day advection controls, same-day core predictors),
    95% winter-block bootstrap intervals (table_R11).
Run from the project root:  python3 07_scripts/16g_final_advection_pulse_figure.py
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
G = _load("step16g", "15g_plot_termination_figures.py")
REV = C.PROJECT_ROOT / "08_outputs" / "tables" / "revision"
MM = 1 / 25.4
NW = "mech_nigp_adv850_k_day_anom"
ROWS = [  # (analysis, predictor, label)
    ("lead 1 day(s)", NW, "k − 1, primary"),
    ("lead 2 day(s)", NW, "k − 2, primary"),
    ("lead 1, season-end spells not censored", NW, "k − 1, uncensored"),
    ("lead 1, stable 25-station catalogue", NW, "k − 1, 25 stations"),
    ("lead 1 + same-day NW IGP and BD advection", "nigp_adv_lead1", "k − 1, + adv. day k"),
    ("lead 1 + same-day core predictors", "nigp_adv_lead1", "k − 1, + core day k"),
]


def main() -> None:
    comp = pd.read_csv(REV / "table_R11b_aligned_advection_composites.csv")
    rob = pd.read_csv(REV / "table_R11_lead_advection_robustness.csv")
    fig = plt.figure(figsize=(174 * MM, 62 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.95], wspace=0.95)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharey=ax_a)
    style = {"NW IGP": (G.OK["blue"], "NW IGP"), "Bangladesh": (G.OK["red"], "Bangladesh")}
    for ax, align, title, xl in ((ax_a, "onset", "(a) Onset", "Days from onset"),
                                 (ax_b, "end", "(b) Termination", "Days from last event day")):
        for box, (col, lab) in style.items():
            d = comp[(comp["alignment"] == align) & (comp["box"] == box)].sort_values("day")
            ax.fill_between(d["day"], d["ci_low"], d["ci_high"], color=col, alpha=0.18, lw=0)
            ax.plot(d["day"], d["mean"], "-o", color=col, ms=2.8, label=lab)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.axvline(0, color="0.5", lw=0.6, ls="--")
        ax.set_title(title, loc="left")
        ax.set_xlabel(xl)
        ax.set_xticks(range(-5, 3))
    ax_a.set_ylabel("Adv850 anomaly (K day$^{-1}$; > 0 warm)")
    ax_a.legend(frameon=False, loc="lower left")
    plt.setp(ax_b.get_yticklabels(), visible=False)

    ax = fig.add_subplot(gs[2])
    y = np.arange(len(ROWS))[::-1]
    for yi, (an, pred, lab) in zip(y, ROWS):
        r = rob[(rob["analysis"] == an) & (rob["predictor"] == pred)]
        if r.empty:
            continue
        r = r.iloc[0]
        sig = (r["ci_low"] > 1) or (r["ci_high"] < 1)
        col = "k" if sig else "0.45"
        ax.plot([r["ci_low"], r["ci_high"]], [yi, yi], color=col, lw=1.2)
        ax.plot(r["odds_ratio_per_sd"], yi, "o", color=col, ms=3.5, mfc=col if sig else "white")
    ax.axvline(1, color="0.5", lw=0.6)
    ax.set_xscale("log")
    ax.set_xticks([0.7, 1, 1.5, 2])
    ax.set_xticklabels(["0.7", "1", "1.5", "2"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, _, lab in ROWS], fontsize=6.5)
    ax.set_xlabel("Odds ratio of termination per SD")
    ax.set_title("(c) NW IGP Adv850 before the end", loc="left")
    paths = G.save(fig, "fig_final_R_advection_pulse")
    C.write_checksums("step17g_output_sha256.txt", paths)


if __name__ == "__main__":
    main()
