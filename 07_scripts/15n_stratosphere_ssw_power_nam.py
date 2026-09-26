"""
Step 16N - Stratospheric influence: SSW test with power analysis, and the NAM background state.

Part A - SSW-centred onset test (re-implementation of Step 12 with one consistent statistic).
    Statistic: number of UNIQUE cold-spell onsets falling within day 0..+45 of any major SSW.
    Null: each SSW date is moved to the same calendar date in a randomly chosen other winter;
    a control date within +/-45 days of any real SSW is redrawn (not skipped). 10,000 replicates.
    (Step 12 counted unique onsets for the observed value but a non-unique sum for controls and
    skipped, rather than redrew, invalid control dates; both are corrected here.)
    Power: the alternative count is modelled as negative binomial with mean r x null mean and the
    null's dispersion index; power(r) = P(count >= one-sided 5% critical value). Reported: power at
    r = 1.5, 2, 3 and the minimum detectable rate ratio at 80% power (MDR).

Part B - Is the NAM a precursor or a background state?
    (i) Across winters: mean NAM on onset days minus the mean of calendar-matched random dates
        from all 40 winters (10,000 Monte Carlo sets of 85 dates).
    (ii) Within winters: NAM on the onset day minus the mean NAM of DJF days in the same winter
        that lie more than 10 days from any cold-spell day; winter-block bootstrap CI.
    (iii) Winter level: Spearman correlation between winter-mean DJF NAM and cold-spell days.

Run from the project root:
    python3 07_scripts/15n_stratosphere_ssw_power_nam.py 2>&1 | tee 09_logs/step16n_stratosphere.log
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson, spearmanr

_spec = importlib.util.spec_from_file_location(
    "term_common", Path(__file__).resolve().parent / "15_termination_common.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

SSW_TABLE = C.PROJECT_ROOT / "08_outputs" / "tables" / "stratosphere" / "table_60_major_ssw_catalogue.csv"
T_SSW = C.TERM_TABLES / "table_107_ssw_test_and_power.csv"
T_NAM = C.TERM_TABLES / "table_108_nam_background_tests.csv"
REPORT = C.TERM_QC / "step16n_stratosphere_report.txt"
N_MC, N_BOOT, SEED = 10000, 5000, 20260923
HORIZON, EXCLUSION, QUIET_GAP = 45, 45, 10
WINTERS = np.arange(1985, 2025)


def winter_of(d: pd.Timestamp) -> int:
    return d.year if d.month >= 7 else d.year - 1


def shift_to_winter(d: pd.Timestamp, ws: int) -> pd.Timestamp:
    year = ws if d.month >= 7 else ws + 1
    day = min(d.day, pd.Period(f"{year}-{d.month:02d}").days_in_month)
    return pd.Timestamp(year=year, month=d.month, day=day)


def unique_onsets_in_windows(onsets: np.ndarray, starts: np.ndarray) -> int:
    lo = starts[None, :]
    hi = starts[None, :] + np.timedelta64(HORIZON, "D")
    return int(((onsets[:, None] >= lo) & (onsets[:, None] <= hi)).any(axis=1).sum())


def part_a(ev: pd.DataFrame) -> pd.DataFrame:
    ssw = pd.read_csv(SSW_TABLE, parse_dates=["central_date"])["central_date"]
    ssw = ssw[ssw.apply(winter_of).between(WINTERS.min(), WINTERS.max())].reset_index(drop=True)
    onsets = ev["start_date"].to_numpy(dtype="datetime64[ns]")
    ssw_np = ssw.to_numpy(dtype="datetime64[ns]")
    observed = unique_onsets_in_windows(onsets, ssw_np)
    rng = np.random.default_rng(SEED)
    null = np.empty(N_MC, dtype=int)
    for i in range(N_MC):
        ctrl = []
        for d in ssw:
            src = winter_of(d)
            for _ in range(100):
                cd = shift_to_winter(d, int(rng.choice(WINTERS[WINTERS != src])))
                if np.all(np.abs((ssw - cd).dt.days) > EXCLUSION):
                    break
            ctrl.append(cd)
        null[i] = unique_onsets_in_windows(onsets, np.array(ctrl, dtype="datetime64[ns]"))
    mu, var = null.mean(), null.var()
    disp = var / mu if mu > 0 else np.nan
    crit = int(np.min([k for k in range(null.max() + 2) if (null >= k).mean() <= 0.05]))

    def power(r: float) -> float:
        m = r * mu
        if disp <= 1.0001:
            return float(poisson.sf(crit - 1, m))
        n = m / (disp - 1)
        return float(nbinom.sf(crit - 1, n, 1 / disp))

    grid = np.round(np.arange(1.0, 5.01, 0.05), 2)
    pw = np.array([power(r) for r in grid])
    mdr = float(grid[np.argmax(pw >= 0.8)]) if (pw >= 0.8).any() else np.nan
    rows = [{"statistic": "ssw_count", "value": len(ssw)},
            {"statistic": "observed_unique_onsets_day0_45", "value": observed},
            {"statistic": "null_mean", "value": mu},
            {"statistic": "null_2.5pct", "value": np.percentile(null, 2.5)},
            {"statistic": "null_97.5pct", "value": np.percentile(null, 97.5)},
            {"statistic": "observed_to_null_rate_ratio", "value": observed / mu},
            {"statistic": "p_upper_one_sided", "value": (1 + (null >= observed).sum()) / (N_MC + 1)},
            {"statistic": "p_lower_one_sided", "value": (1 + (null <= observed).sum()) / (N_MC + 1)},
            {"statistic": "null_dispersion_index", "value": disp},
            {"statistic": "critical_count_5pct_upper", "value": crit},
            {"statistic": "power_rate_ratio_1.5", "value": power(1.5)},
            {"statistic": "power_rate_ratio_2", "value": power(2.0)},
            {"statistic": "power_rate_ratio_3", "value": power(3.0)},
            {"statistic": "minimum_detectable_rate_ratio_80pct", "value": mdr}]
    return pd.DataFrame(rows)


def part_b(ev: pd.DataFrame) -> pd.DataFrame:
    cov = pd.read_csv(C.DAILY_COVARIATES, parse_dates=["date"]).set_index("date")
    nam = cov[["nam_proxy_10", "nam_proxy_100"]]
    djf = nam.index
    cd = pd.Series(djf.strftime("%m-%d"), index=djf)
    by_cd = {k: v.index.to_numpy() for k, v in cd.groupby(cd)}
    event_days = set(d for e in ev.itertuples() for d in pd.date_range(e.start_date, e.end_date))
    quiet = pd.Series(True, index=djf)
    for d in event_days:
        quiet.loc[(quiet.index >= d - pd.Timedelta(days=QUIET_GAP)) &
                  (quiet.index <= d + pd.Timedelta(days=QUIET_GAP))] = False
    rng = np.random.default_rng(SEED)
    winters = ev["winter_start_year"].to_numpy()
    rows = []
    for col in ("nam_proxy_10", "nam_proxy_100"):
        onset_vals = nam.loc[ev["start_date"], col].to_numpy()
        obs = onset_vals.mean()
        keys = ev["start_date"].dt.strftime("%m-%d").to_numpy()
        sims = np.empty(N_MC)
        for i in range(N_MC):
            picks = [rng.choice(by_cd[k]) for k in keys]
            sims[i] = nam.loc[picks, col].mean()
        rows.append({"test": "across_winters_onset_vs_calendar_matched", "index": col,
                     "estimate": obs - sims.mean(), "ci_low": obs - np.percentile(sims, 97.5),
                     "ci_high": obs - np.percentile(sims, 2.5),
                     "p_two_sided": min(1.0, 2 * min((sims >= obs).mean(), (sims <= obs).mean()))})
        within = []
        for e, v in zip(ev.itertuples(), onset_vals):
            ref = nam.loc[(cov["winter_start_year"] == e.winter_start_year) & quiet, col]
            within.append(v - ref.mean() if len(ref) >= 10 else np.nan)
        within = np.array(within)
        ok = ~np.isnan(within)
        uniq, counts = C.winter_bootstrap_counts(winters[ok], N_BOOT, SEED)
        W = counts[:, np.searchsorted(uniq, winters[ok])].astype(float)
        boot = (W @ within[ok]) / W.sum(axis=1)
        rows.append({"test": "within_winter_onset_minus_quiet_days", "index": col,
                     "estimate": within[ok].mean(), "ci_low": np.percentile(boot, 2.5),
                     "ci_high": np.percentile(boot, 97.5),
                     "p_two_sided": min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())),
                     "n_events": int(ok.sum())})
        wmean = nam.groupby(cov["winter_start_year"])[col].mean()
        cold_days = ev.groupby("winter_start_year")["duration_days"].sum().reindex(wmean.index, fill_value=0)
        r = spearmanr(wmean, cold_days)
        rows.append({"test": "winter_mean_nam_vs_cold_spell_days_spearman", "index": col,
                     "estimate": r[0], "p_two_sided": r[1], "n_winters": len(wmean)})
    return pd.DataFrame(rows)


def main() -> None:
    C.ensure_dirs()
    C.write_policy("step16n_stratosphere_policy.json", {
        "step": "Step 16N", "title": "SSW test with power analysis and NAM background tests",
        "ssw_source": str(SSW_TABLE.relative_to(C.PROJECT_ROOT)), "horizon_days": HORIZON,
        "control_exclusion_days": EXCLUSION, "monte_carlo": N_MC, "bootstrap": N_BOOT, "seed": SEED,
        "power_model": "negative binomial with null dispersion index, mean = rate ratio x null mean",
        "quiet_day_gap": QUIET_GAP})
    ev = C.load_events()
    A = part_a(ev)
    A.to_csv(T_SSW, index=False, float_format="%.4f")
    B = part_b(ev)
    B.to_csv(T_NAM, index=False, float_format="%.4f")
    C.write_checksums("step16n_output_sha256.txt", [T_SSW, T_NAM])
    text = ("Step 16N - Stratosphere\n" + "=" * 23 + "\nA. SSW test and power\n" + A.round(4).to_string(index=False) +
            "\n\nB. NAM background tests\n" + B.round(4).to_string(index=False) + "\n")
    REPORT.write_text(text)
    (C.TERM_QC / "step16n_stratosphere_summary.json").write_text(json.dumps({"completed_utc": C.now_utc()}))
    print(text)


if __name__ == "__main__":
    main()
