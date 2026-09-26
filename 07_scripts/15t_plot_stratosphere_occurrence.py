"""
Step 16T - Figure: the stratosphere controls cold-spell occurrence, not persistence.

(a) SSW-centred test: null distribution of unique cold-spell onsets within 0-45 days after the 25
    major SSWs (calendar-matched controls, as Step 16N; 10,000 replicates) with the observed count.
(b) Onset-day NAM at 100 hPa: distribution of means of 85 calendar-matched random dates (10,000 sets)
    with the observed onset-day mean (as Step 16N).
(c) Winter level: winter-mean NAM at 100 hPa vs cold-spell days per winter, coloured by DJF ENSO
    phase (Step 16O table_110), with the Spearman and ENSO-partial correlations from Step 16O.
(d) Duration: NAM at 100 hPa averaged over days -10..-1 vs spell duration (85 spells) - no relation
    (Step 16M), shown for contrast with (c).
Numbers printed in the panels are read from the Step 16N/16O/16M tables; the Monte Carlo
distributions are regenerated with the same seed and procedure as Step 16N.

Run from the project root:
    python3 07_scripts/15t_plot_stratosphere_occurrence.py 2>&1 | tee 09_logs/step16t_stratosphere_fig.log
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
N = _load("step16n", "15n_stratosphere_ssw_power_nam.py")
G = _load("step16g", "15g_plot_termination_figures.py")
MM = 1 / 25.4
N_MC = 10000


def main() -> None:
    C.ensure_dirs()
    T = C.TERM_TABLES
    ev = C.load_events()
    cov = pd.read_csv(C.DAILY_COVARIATES, parse_dates=["date"]).set_index("date")
    ssw_tab = pd.read_csv(T / "table_107_ssw_test_and_power.csv").set_index("statistic")["value"]
    nam_tab = pd.read_csv(T / "table_108_nam_background_tests.csv")
    enso_tab = pd.read_csv(T / "table_109_enso_controls.csv").set_index("test")
    wdf = pd.read_csv(T / "table_110_winter_enso_nam_coldspells.csv")
    rng = np.random.default_rng(N.SEED)

    # (a) SSW null, same procedure as Step 16N
    ssw = pd.read_csv(N.SSW_TABLE, parse_dates=["central_date"])["central_date"]
    ssw = ssw[ssw.apply(N.winter_of).between(1985, 2024)].reset_index(drop=True)
    onsets = ev["start_date"].to_numpy(dtype="datetime64[ns]")
    observed = N.unique_onsets_in_windows(onsets, ssw.to_numpy(dtype="datetime64[ns]"))
    null = np.empty(N_MC, dtype=int)
    for i in range(N_MC):
        ctrl = []
        for d in ssw:
            src = N.winter_of(d)
            for _ in range(100):
                cd = N.shift_to_winter(d, int(rng.choice(N.WINTERS[N.WINTERS != src])))
                if np.all(np.abs((ssw - cd).dt.days) > N.EXCLUSION):
                    break
            ctrl.append(cd)
        null[i] = N.unique_onsets_in_windows(onsets, np.array(ctrl, dtype="datetime64[ns]"))

    # (b) onset-day NAM100 vs calendar-matched dates
    nam = cov["nam_proxy_100"]
    cd = pd.Series(cov.index.strftime("%m-%d"), index=cov.index)
    by_cd = {k: v.index.to_numpy() for k, v in cd.groupby(cd)}
    keys = ev["start_date"].dt.strftime("%m-%d").to_numpy()
    sims = np.array([nam.loc[[rng.choice(by_cd[k]) for k in keys]].mean() for _ in range(N_MC)])
    obs_nam = nam.loc[ev["start_date"]].mean()

    # (d) pre-onset NAM vs duration
    pre = [nam.reindex(pd.date_range(s - pd.Timedelta(days=10), s - pd.Timedelta(days=1))).mean()
           for s in ev["start_date"]]

    fig, axes = plt.subplots(2, 2, figsize=(150 * MM, 115 * MM))
    axes = axes.flat
    ax = axes[0]
    bins = np.arange(null.min() - 0.5, null.max() + 1.5, 1)
    ax.hist(null, bins=bins, color=G.OK["grey"], edgecolor="white", lw=0.3)
    ax.axvline(observed, color=G.OK["red"], lw=1.6)
    ax.set_xlabel("Onsets within 45 d after SSWs")
    ax.set_ylabel("Control replicates")
    ax.set_title("(a) After major SSWs", loc="left")
    ax.text(0.97, 0.95, f"observed {observed}\nexpected {null.mean():.1f}\nratio {observed / null.mean():.2f}\n"
            f"p = {ssw_tab['p_lower_one_sided']:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=6.5)

    ax = axes[1]
    ax.hist(sims, bins=40, color=G.OK["grey"], edgecolor="white", lw=0.3)
    ax.axvline(obs_nam, color=G.OK["red"], lw=1.6)
    ax.set_xlabel("Mean NAM100 (SD)")
    ax.set_title("(b) Onset days vs matched", loc="left")
    within = nam_tab[(nam_tab["test"] == "within_winter_onset_minus_quiet_days") & (nam_tab["index"] == "nam_proxy_100")].iloc[0]
    ax.text(0.03, 0.95, f"onset − matched\n+{obs_nam - sims.mean():.2f} SD\nwithin-winter\n+{within['estimate']:.2f} SD",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5)

    ax = axes[2]
    colors = {"el_nino": G.OK["red"], "la_nina": G.OK["blue"], "neutral": G.OK["grey"]}
    names = {"el_nino": "El Niño", "la_nina": "La Niña", "neutral": "Neutral"}
    for ph, g in wdf.groupby("enso_phase"):
        ax.scatter(g["mean_nam_proxy_100"], g["cold_spell_days"], s=14, color=colors[ph], label=names[ph],
                   edgecolor="k", linewidth=0.3)
    ax.set_xlabel("Winter-mean NAM100 (SD)")
    ax.set_ylabel("Cold-spell days per winter")
    ax.set_title("(c) Winters", loc="left")
    part = enso_tab.loc["partial_spearman_cold_spell_days_vs_nam_proxy_100_given_oni"]
    rho = nam_tab[(nam_tab["test"] == "winter_mean_nam_vs_cold_spell_days_spearman") & (nam_tab["index"] == "nam_proxy_100")].iloc[0]
    ax.text(0.03, 0.97, f"ρ = {rho['estimate']:.2f} (p = {rho['p_two_sided']:.3f})\npartial on ONI {part['estimate']:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=6, loc="lower right")

    ax = axes[3]
    ax.scatter(pre, ev["duration_days"], s=12, color=G.OK["black"], alpha=0.7, linewidth=0)
    ax.set_xlabel("NAM100, days −10…−1 (SD)")
    ax.set_ylabel("Spell duration (days)")
    ax.set_title("(d) Duration", loc="left")
    dur = pd.read_csv(T / "table_105_remote_driver_duration_tests.csv")
    r = dur[(dur["window"] == "pre10") & (dur["driver"] == "nam_proxy_100")].iloc[0]
    ax.text(0.97, 0.97, f"ρ = {r['spearman_rho']:.2f}\n[{r['rho_ci_low']:.2f}, {r['rho_ci_high']:.2f}]",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
    fig.tight_layout(h_pad=1.2, w_pad=1.5)
    paths = G.save(fig, "fig_final_09_stratosphere_occurrence")
    C.write_checksums("step16t_output_sha256.txt", paths)
    print(f"observed {observed}, null mean {null.mean():.2f} (Step 16N table: {ssw_tab['null_mean']:.2f}); "
          f"onset NAM100 minus matched {obs_nam - sims.mean():.3f}")


if __name__ == "__main__":
    main()
