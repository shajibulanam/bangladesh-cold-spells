#!/usr/bin/env python3
"""
16i_plot_lifecycle_schematic.py  (Step 24, October 2026)
========================================================

Draws the conceptual schematic of the cold-night spell life cycle (main-text
Fig. 11): (a) onset, (b) maintenance, (c) termination. The figure is a
synthesis drawing, not a data plot: geography is stylized, and the numbers
printed on it are the composite means reported in the Results (Sects.
3.2-3.4), taken from the archived result tables:

  onset       jet +3.4 m/s and SA Z500 -16.5 m at onset (Fig. 5); Z500 about
              -30 m along 30-40N on day -5 (Fig. 4); NW IGP Adv850 anomaly
              -0.81 K/day on day -2, BD -1.20 K/day on day 0; Tmin -2.4 C
  maintenance BD dewpoint -2.8 C, total cloud -0.11, downward longwave
              -21 W/m2, solar +12 W/m2 two days before the end (Fig. 6);
              BD Adv850 anomaly -1.2 to -1.4 K/day before the end (Fig. 7b)
  termination NW IGP Adv850 anomaly +0.30 / +0.62 K/day two / one day before
              the last event day (Fig. 7b); lag-1 odds ratio 1.83 (Fig. 7c);
              next day: dewpoint -0.9 C, longwave -4 W/m2, Tmin +1.2 C (Fig. 6)

No input files are read. Output: 08_outputs/figures/fig_lifecycle_schematic.{png,pdf}

Usage (project root):
    python 07_scripts/16i_plot_lifecycle_schematic.py [--out 08_outputs/figures]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle, Ellipse

# Okabe-Ito colours (as in the other figures)
BLUE, SKY, VERM, ORANGE, GREY = "#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#999999"
TERRAIN = "#d9d4c7"
FS = 6.4

LON0, LON1, LAT0, LAT1 = 58, 104, 14, 42
NWIGP = (75, 25, 88, 32)          # lon0, lat0, lon1, lat1
BD = (88, 20.5, 93, 26.8)
# stylized Himalaya-Tibetan Plateau (terrain above about 1000 m)
TIBET = [(70, 37.5), (74, 35.5), (78, 32.8), (81, 30.6), (85, 28.6), (88.5, 27.6), (92, 27.6),
         (95.5, 28.4), (97.5, 26.5), (99, 25.5), (101, 27.5), (104, 31), (104, 42), (70, 42)]


def base_map(ax, title):
    ax.set_xlim(LON0, LON1)
    ax.set_ylim(LAT0, LAT1)
    ax.set_aspect(1.13)              # about 1/cos(28N)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color("0.4")
    ax.add_patch(Polygon(TIBET, closed=True, fc=TERRAIN, ec="none", zorder=0))
    ax.text(84, 37.5, "Tibetan Plateau", fontsize=FS - 1.0, color="0.5", ha="center", va="center",
            style="italic", zorder=1)
    for (x0, y0, x1, y1) in (NWIGP, BD):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="0.3", lw=0.6, ls="--", zorder=2))
    ax.text(NWIGP[0] + 0.4, NWIGP[1] - 0.4, "NW IGP", fontsize=FS - 1.0, color="0.3", va="top", zorder=3)
    ax.text(BD[0] + 0.3, BD[1] - 0.4, "BD", fontsize=FS - 1.0, color="0.3", va="top", zorder=3)
    ax.text(89.5, 16.0, "Bay of Bengal", fontsize=FS - 1.2, color="0.55", ha="center", style="italic")
    ax.text(65.0, 16.0, "Arabian Sea", fontsize=FS - 1.2, color="0.55", ha="center", style="italic")
    ax.set_title(title, loc="left", fontsize=FS + 1.0, fontweight="bold", pad=2.5)


def arrow(ax, p0, p1, color, lw=1.6, ms=8, ls="-", rad=0.0, z=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=z))


def jet(ax, strength="strong"):
    lw = {"strong": 3.0, "weak": 1.3}[strength]
    al = {"strong": 0.9, "weak": 0.6}[strength]
    ax.add_patch(FancyArrowPatch((LON0 + 1, 29.5), (LON1 - 1, 31.5), arrowstyle="-|>", mutation_scale=10,
                                 lw=lw, color=SKY, alpha=al, connectionstyle="arc3,rad=-0.12", zorder=3))
    ax.text(LON0 + 1.2, 31.6 if strength == "strong" else 31.2, "jet", fontsize=FS - 0.8, color="#1f78b4", va="bottom")


def trough(ax, x, y, label, strength=1.0, ls="-", color=BLUE):
    for w, h in ((8, 5.0), (4.8, 3.0)):
        ax.add_patch(Ellipse((x, y), w * strength, h * strength, fill=False, ec=color, lw=0.9, ls=ls, zorder=4))
    ax.text(x, y, "L", fontsize=FS + 1.5, color=color, ha="center", va="center", fontweight="bold", zorder=5)
    if label:
        ax.text(x, y + 2.8 * strength, label, fontsize=FS - 0.8, color=color, ha="center", va="bottom", zorder=5)


def cloud(ax, x, y):
    for dx, dy in ((0, 0), (1.5, 0.6), (3.0, 0)):
        ax.add_patch(Ellipse((x + dx, y + dy), 2.6, 1.6, fc="white", ec="0.4", lw=0.6, zorder=6))


def panel_onset(ax):
    base_map(ax, "(a) Onset: days −15 to 0")
    jet(ax, "strong")
    ax.text(62.0, 25.0, "H", fontsize=FS + 1.5, color=VERM, ha="center", va="center", fontweight="bold")
    ax.text(62.0, 23.4, "ridge", fontsize=FS - 1.0, color=VERM, ha="center", va="top")
    ax.text(70.5, 35.2, "wave train", fontsize=FS - 1.0, color="0.4", ha="center", rotation=22, style="italic")
    trough(ax, 98.0, 34.5, "trough")
    arrow(ax, (64.0, 26.5), (93.0, 35.5), "0.4", lw=0.7, ms=5, ls="--", rad=-0.25)
    arrow(ax, (77.0, 31.0), (85.5, 27.0), BLUE, lw=2.0, ms=8)
    ax.text(80.5, 27.3, "day −2", fontsize=FS - 0.8, color=BLUE, ha="center", va="top")
    arrow(ax, (89.2, 26.3), (91.2, 22.0), BLUE, lw=2.0, ms=8)
    ax.text(93.6, 22.0, "day 0", fontsize=FS - 0.8, color=BLUE, ha="left", va="center")


def panel_maintenance(ax):
    base_map(ax, "(b) Maintenance")
    jet(ax, "strong")
    trough(ax, 97.0, 34.5, "trough persists", strength=0.85)
    arrow(ax, (82.0, 29.0), (88.5, 25.2), BLUE, lw=2.0, ms=8)
    arrow(ax, (89.2, 26.0), (91.2, 21.8), BLUE, lw=2.0, ms=8)
    ax.text(87.4, 23.2, "cold advection", fontsize=FS - 0.8, color=BLUE, ha="right", va="top")
    arrow(ax, (95.5, 21.0), (95.5, 27.5), VERM, lw=0.9, ms=6, ls=(0, (2, 1.5)), z=6)
    ax.text(96.3, 24.0, "LW\nloss", fontsize=FS - 1.0, color=VERM, va="center")
    ax.text(67.0, 22.5, "clear, dry", fontsize=FS - 0.8, color="0.3", ha="center", style="italic")


def panel_termination(ax):
    base_map(ax, "(c) Termination")
    jet(ax, "weak")
    trough(ax, 64.5, 37.0, None, strength=0.6, ls="--")
    ax.text(68.0, 39.2, "next disturbance?", fontsize=FS - 0.8, color=BLUE, ha="left", va="center")
    arrow(ax, (72.0, 26.5), (81.5, 30.2), VERM, lw=2.2, ms=9)
    ax.text(60.5, 23.2, "warm advection", fontsize=FS - 0.8, color=VERM, ha="left", va="top")
    cloud(ax, 88.2, 28.4)
    arrow(ax, (90.4, 24.8), (91.6, 22.6), SKY, lw=0.9, ms=6, z=6)
    ax.text(93.6, 22.0, "relaxes", fontsize=FS - 0.8, color="#1f78b4", ha="left", va="center")


TEXT = {
    0: ["Rossby wave train along the subtropical",
        "jet; trough over South Asia builds from",
        "day −13; jet +3.4 m s⁻¹ at onset",
        "cold advection: NW IGP on day −2",
        "(−0.8 K d⁻¹), then BD on day 0 (−1.2 K d⁻¹)",
        "Tmin −2.4 °C at onset",
        "Siberian High, Ural blocking not",
        "anomalous on average"],
    1: ["sustained cold advection over BD",
        "(−1.2 to −1.4 K d⁻¹)",
        "dry, clear air: dewpoint −2.8 °C,",
        "cloud −0.11, longwave −21 W m⁻²",
        "→ strong night-time cooling",
        "solar +12 W m⁻² → days recover first;",
        "DTR peaks on the last day",
        "duration largely undetected at onset"],
    2: ["falling heights, weaker jet:",
        "higher termination odds",
        "warm advection returns over NW IGP",
        "2 and 1 days before the last day",
        "(+0.3, +0.6 K d⁻¹; lag-1 OR 1.83)",
        "next day over BD: cold advection relaxes",
        "to climatology; moisture and cloud",
        "return; Tmin rises by about 1.2 °C"],
}
BULLET_AT = {0: (0, 3, 5, 6), 1: (0, 2, 5, 7), 2: (0, 2, 5)}


def text_block(ax, k):
    ax.axis("off")
    y = 1.0
    for i, line in enumerate(TEXT[k]):
        bullet = i in BULLET_AT[k]
        ax.text(0.03, y, ("• " if bullet else "   ") + line, fontsize=FS - 0.3, va="top", ha="left",
                transform=ax.transAxes)
        y -= 0.122


def timeline(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.add_patch(FancyArrowPatch((0.5, 4), (99.5, 4), arrowstyle="-|>", mutation_scale=8, lw=0.8, color="0.3"))
    segs = [(1, 33, "trough and jet build; cold advection NW IGP → BD", BLUE),
            (34, 66, "spell of 3–17 days", "0.3"),
            (67, 99, "warm advection upstream → relaxation", VERM)]
    for x0, x1, txt, c in segs:
        ax.add_patch(Rectangle((x0, 2.6), x1 - x0, 2.8, fc=c, alpha=0.15, ec="none"))
        ax.text((x0 + x1) / 2, 6.4, txt, fontsize=FS - 0.6, ha="center", va="bottom", color=c)
    for x, lab in ((33.5, "onset (day 0)"), (66.5, "last event day")):
        ax.plot([x, x], [2.0, 6.0], color="0.3", lw=0.8)
        ax.text(x, 1.6, lab, fontsize=FS - 0.8, ha="center", va="top", color="0.3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="08_outputs/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": FS, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.08, 3.9))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.78, 0.22], left=0.01, right=0.99, top=0.95,
                          bottom=0.01, wspace=0.05, hspace=0.05)
    for k, f in enumerate((panel_onset, panel_maintenance, panel_termination)):
        f(fig.add_subplot(gs[0, k]))
        text_block(fig.add_subplot(gs[1, k]), k)
    timeline(fig.add_subplot(gs[2, :]))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_lifecycle_schematic.{ext}", dpi=600 if ext == "png" else None)
    print("written", out / "fig_lifecycle_schematic.png", "and .pdf")


if __name__ == "__main__":
    main()
