"""
Step 16A2 - Agreement between the two BMD Tmax sources before they are merged.

Sources
    old archive : 01_raw_data/MaxT_MinT_1981_2024_raw.xlsx               (1981-2024)
    new file    : 01_raw_data/Daily_Maximum_Temperature_2022_2025_raw.csv (Jan 2022-Dec 2025)
Merge rule (15_termination_common.raw_tmax_merged, same as Step 2C for Tmin): before 2022 the old
archive; from 2022 the new file where it has a value (also where the two disagree), the old archive
filling its gaps. Exact 0.0 degC in the new file is a missing value entered as zero.

Outputs
    05_qc_reports/termination/step16a2_tmax_source_overlap_report.txt
    05_qc_reports/termination/step16a2_tmax_source_overlap_summary.json
    05_qc_reports/termination/step16a2_tmax_source_conflicts_gt1C.csv
    05_qc_reports/termination/step16a2_tmax_zero_missing_codes.csv

Run from the project root (before 15b):
    python3 07_scripts/15a2_check_tmax_source_overlap.py 2>&1 | tee 09_logs/step16a2_tmax_overlap.log
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "term_common", Path(__file__).resolve().parent / "15_termination_common.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

OUT = C.TERM_QC
DJF = [12, 1, 2]


def stats(x: pd.DataFrame) -> dict:
    d = x["diff"]
    return {"pairs": int(len(x)), "identical_pct": round(float((d.abs() < 0.05).mean() * 100), 3),
            "within_0p5C_pct": round(float((d.abs() <= 0.5).mean() * 100), 3),
            "mean_diff_new_minus_old": round(float(d.mean()), 4), "max_abs_diff": round(float(d.abs().max()), 2)}


def main() -> None:
    C.ensure_dirs()
    prim = set(pd.read_csv(C.STATION_WEIGHTS)["station_uid"])
    merged, m = C.raw_tmax_merged(return_parts=True)
    new = C._new_tmax()
    zeros = new[new["missing_code"]].merge(pd.read_csv(C.STATION_MASTER)[["station_uid", "official_station_name"]],
                                           on="station_uid", how="left")
    zeros.to_csv(OUT / "step16a2_tmax_zero_missing_codes.csv", index=False)

    ov = m[(m["date"] >= C.NEW_SOURCE_START) & m["tmax_old"].notna() & m["tmax_new"].notna()].copy()
    ov["diff"] = ov["tmax_new"] - ov["tmax_old"]
    djf = ov[ov["date"].dt.month.isin(DJF)]
    summary = {
        "overlap_period": [str(ov["date"].min().date()), str(ov["date"].max().date())],
        "all_stations_all_months": stats(ov), "network26_all_months": stats(ov[ov["station_uid"].isin(prim)]),
        "all_stations_djf": stats(djf), "network26_djf": stats(djf[djf["station_uid"].isin(prim)]),
        "zero_values_treated_as_missing": int(len(zeros)),
        "zero_values_in_djf_study_period": int(zeros[zeros["date"].dt.month.isin(DJF)
                                                     & (zeros["date"] >= "1985-12-01") & (zeros["date"] <= "2025-02-28")].shape[0]),
        "merged_values_by_source": merged.loc[merged["tmax"].notna(), "tmax_source"].value_counts().to_dict(),
        "network26_djf_2024_25_completeness_pct": round(float(
            merged[merged["station_uid"].isin(prim) & merged["date"].between("2024-12-01", "2025-02-28")]["tmax"]
            .notna().mean() * 100), 2),
        "completed_utc": C.now_utc(),
    }
    big = ov[ov["diff"].abs() > 1].sort_values("diff", key=abs, ascending=False)
    big[["station_uid", "date", "tmax_old", "tmax_new", "diff"]].to_csv(OUT / "step16a2_tmax_source_conflicts_gt1C.csv", index=False)
    (OUT / "step16a2_tmax_source_overlap_summary.json").write_text(json.dumps(summary, indent=2))
    text = "Step 16A2 - Tmax source overlap\n" + json.dumps(summary, indent=2) + \
           f"\n\nPairs differing by more than 1 degC: {len(big)} (DJF: {int(big['date'].dt.month.isin(DJF).sum())})\n"
    (OUT / "step16a2_tmax_source_overlap_report.txt").write_text(text)
    C.write_checksums("step16a2_output_sha256.txt", [OUT / "step16a2_tmax_source_overlap_summary.json",
                                                     OUT / "step16a2_tmax_source_conflicts_gt1C.csv",
                                                     OUT / "step16a2_tmax_zero_missing_codes.csv"])
    print(text)


if __name__ == "__main__":
    main()
