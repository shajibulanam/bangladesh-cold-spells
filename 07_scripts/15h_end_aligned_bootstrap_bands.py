"""
Step 16H - Winter-block bootstrap confidence bands for the end-aligned termination composites.

For every primary cold-night event, the Step 16B daily covariates are taken on days -5..+5
relative to the event's last day. The all-event mean and a 95% winter-block bootstrap
interval (5,000 resamples of winters) are computed for each lag and variable. Figure M7
(Step 16G) draws these bands when this table exists.

Run from the project root:
    python3 07_scripts/15h_end_aligned_bootstrap_bands.py 2>&1 | tee 09_logs/step16h_end_aligned_bands.log
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "term_common", Path(__file__).resolve().parent / "15_termination_common.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

TABLE = C.TERM_TABLES / "table_102_end_aligned_composites_with_ci.csv"
REPORT = C.TERM_QC / "step16h_end_aligned_bands_report.txt"
N_BOOT, SEED = 5000, 20260923
LAGS = range(-5, 6)
VARIABLES = ["nat_tmin_anom", "nat_tmax_anom", "nat_dtr_anom", "era5_bd_wspd10_anom",
             "era5_bd_t850_anom", "era5_bd_v850_anom", "era5_bd_mslp_anom", "era5_sa_z500_anom"]


def main() -> None:
    C.ensure_dirs()
    C.write_policy("step16h_end_aligned_bands_policy.json", {
        "step": "Step 16H", "title": "Bootstrap bands for end-aligned all-event composites",
        "alignment": "days -5..+5 relative to the last event day", "events": "all 85 primary events",
        "bootstrap": {"unit": "winter", "repetitions": N_BOOT, "seed": SEED, "interval": "2.5-97.5 percentile"},
        "variables": VARIABLES, "input": "Step 16B daily covariate dataset"})
    cov = pd.read_csv(C.DAILY_COVARIATES, parse_dates=["date"]).set_index("date")
    ev = C.load_events()
    winters = ev["winter_start_year"].to_numpy()
    uniq, counts = C.winter_bootstrap_counts(winters, N_BOOT, SEED)
    widx = np.searchsorted(uniq, winters)
    W = counts[:, widx].astype(float)  # B x events

    rows = []
    for lag in LAGS:
        dates = ev["end_date"] + pd.Timedelta(days=lag)
        vals = cov.reindex(dates)[VARIABLES].to_numpy(dtype=float)  # events x vars
        for j, v in enumerate(VARIABLES):
            x = vals[:, j]
            ok = ~np.isnan(x)
            w = W[:, ok]
            with np.errstate(invalid="ignore", divide="ignore"):
                boot = (w @ x[ok]) / w.sum(axis=1)
            boot = boot[np.isfinite(boot)]
            rows.append({"lag": lag, "variable": v, "n_events": int(ok.sum()), "mean": x[ok].mean(),
                         "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5)})
    T = pd.DataFrame(rows)
    T.to_csv(TABLE, index=False, float_format="%.4f")
    C.write_checksums("step16h_output_sha256.txt", [TABLE])
    wide = T.pivot(index="lag", columns="variable", values="mean").round(2)
    text = "Step 16H - End-aligned all-event composites with 95% bootstrap CI\n" + "=" * 66 + "\n" + \
           wide.to_string() + "\n"
    REPORT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
