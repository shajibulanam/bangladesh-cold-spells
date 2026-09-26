"""
Step 16L - Onset timeline: when does each driver change relative to cold-spell onset?

All 85 primary cold-night spells, onset-aligned composites from day -15 to +5 of every driver,
with 95% winter-block bootstrap intervals (5,000 resamples of winters). For each driver the table
also reports the earliest lead before onset from which the 95% interval excludes zero on every day
up to onset ("sustained lead"), a descriptive measure of how early the driver departs from normal.

Drivers: national Tmin (reference), Siberian High, Ural-Siberian blocking fraction, 850-hPa cold-air
advection over the NW Indo-Gangetic Plain and Bangladesh, Bangladesh 850-hPa meridional wind,
South Asia Z500, 200-hPa jet-core latitude and speed (70-100E), the Asian-corridor Rossby wave-packet
envelope (trailing 3-day mean, causal), and NAM at 100 and 10 hPa.

Outputs: table_103_onset_timeline.csv, table_104_onset_timeline_leads.csv, fig_m3b_onset_timeline.

Run from the project root:
    python3 07_scripts/15l_onset_timeline.py 2>&1 | tee 09_logs/step16l_onset_timeline.log
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "term_common", Path(__file__).resolve().parent / "15_termination_common.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

T_TIMELINE = C.TERM_TABLES / "table_103_onset_timeline.csv"
T_LEADS = C.TERM_TABLES / "table_104_onset_timeline_leads.csv"
REPORT = C.TERM_QC / "step16l_onset_timeline_report.txt"
FIG = "fig_m3b_onset_timeline"
N_BOOT, SEED = 5000, 20260923
LAGS = range(-15, 6)

DRIVERS = [
    ("nat_tmin_anom", "National Tmin", "°C"),
    ("siberian_high_std_anom", "Siberian High", "SD"),
    ("ural_blocking_fraction_anom", "Ural blocking", "fraction"),
    ("mech_nigp_adv850_k_day_anom", "Adv850, NW IGP", "K d⁻¹"),
    ("mech_bd_adv850_k_day_anom", "Adv850, Bangladesh", "K d⁻¹"),
    ("era5_bd_v850_anom", "v850, Bangladesh", "m s⁻¹"),
    ("era5_sa_z500_anom", "Z500, South Asia", "m"),
    ("mech_jet200_core_lat_70_100e_anom", "Jet latitude", "°"),
    ("mech_jet200_core_speed_70_100e_anom", "Jet speed", "m s⁻¹"),
    ("rrwp_asian_corridor_envelope_z_trailing3", "Wave envelope", "SD"),
    ("nam_proxy_100", "NAM 100 hPa", "SD"),
    ("nam_proxy_10", "NAM 10 hPa", "SD"),
]

MM = 1 / 25.4
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.titlesize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "pdf.fonttype": 42})


def main() -> None:
    C.ensure_dirs()
    C.write_policy("step16l_onset_timeline_policy.json", {
        "step": "Step 16L", "title": "Onset timeline of drivers",
        "events": "all 85 primary cold-night spells", "lags": [min(LAGS), max(LAGS)],
        "anomalies": "1991-2020 calendar-day mean, +/-7-day window (blocking fraction converted to anomaly here)",
        "bootstrap": {"unit": "winter", "repetitions": N_BOOT, "seed": SEED},
        "sustained_lead": "earliest lag L<=0 such that the 95% CI excludes zero with the same sign on every day L..0",
        "note": "descriptive; no multiple-testing correction across lags",
        "drivers": [d[0] for d in DRIVERS]})
    cov = C.load_covariates_with_mechanisms()
    blk = cov[["ural_blocking_fraction"]].copy()
    cov["ural_blocking_fraction_anom"] = C.calendar_anomalies(blk, 7)["ural_blocking_fraction"]

    ev = C.load_events()
    winters = ev["winter_start_year"].to_numpy()
    uniq, counts = C.winter_bootstrap_counts(winters, N_BOOT, SEED)
    W = counts[:, np.searchsorted(uniq, winters)].astype(float)

    rows = []
    for lag in LAGS:
        vals = cov.reindex(ev["start_date"] + pd.Timedelta(days=lag))
        for var, name, unit in DRIVERS:
            x = vals[var].to_numpy(float)
            ok = ~np.isnan(x)
            w = W[:, ok]
            with np.errstate(invalid="ignore", divide="ignore"):
                b = (w @ x[ok]) / w.sum(axis=1)
            b = b[np.isfinite(b)]
            rows.append({"variable": var, "label": name, "unit": unit, "lag": lag, "n_events": int(ok.sum()),
                         "mean": x[ok].mean(), "ci_low": np.percentile(b, 2.5), "ci_high": np.percentile(b, 97.5)})
    T = pd.DataFrame(rows)
    T.to_csv(T_TIMELINE, index=False, float_format="%.4f")

    leads = []
    for var, name, unit in DRIVERS:
        t = T[(T["variable"] == var) & (T["lag"] <= 0)].sort_values("lag", ascending=False)
        sig = np.where(t["ci_low"] > 0, 1, np.where(t["ci_high"] < 0, -1, 0))
        lead = None
        if sig[0] != 0:
            k = 0
            while k < len(sig) and sig[k] == sig[0]:
                k += 1
            lead = int(t["lag"].iloc[k - 1])
        leads.append({"variable": var, "label": name, "onset_mean": t["mean"].iloc[0],
                      "onset_ci_low": t["ci_low"].iloc[0], "onset_ci_high": t["ci_high"].iloc[0],
                      "sustained_lead_day": lead, "sign_at_onset": int(sig[0])})
    Ld = pd.DataFrame(leads)
    Ld.to_csv(T_LEADS, index=False, float_format="%.4f")

    fig, axes = plt.subplots(3, 4, figsize=(180 * MM, 120 * MM), sharex=True)
    for ax, (var, name, unit), letter in zip(axes.flat, DRIVERS, "abcdefghijkl"):
        t = T[T["variable"] == var]
        ax.axvspan(-0.5, 0.5, color="#999999", alpha=0.2, lw=0)
        ax.fill_between(t["lag"], t["ci_low"], t["ci_high"], color="#0072B2", alpha=0.2, lw=0)
        ax.plot(t["lag"], t["mean"], color="k", lw=1.1)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"({letter}) {name} ({unit})", loc="left")
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    for ax in axes[-1]:
        ax.set_xlabel("Days from onset")
    fig.tight_layout()
    paths = [C.TERM_FIGURES / f"{FIG}.png", C.TERM_FIGURES / f"{FIG}.pdf"]
    for p in paths:
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    C.write_checksums("step16l_output_sha256.txt", [T_TIMELINE, T_LEADS] + paths)
    text = "Step 16L - Onset timeline\n" + "=" * 25 + "\n" + Ld.round(3).to_string(index=False) + "\n"
    REPORT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
