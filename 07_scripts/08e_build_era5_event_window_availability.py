from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PL_STD_DIR = PROJECT_ROOT / "03_intermediate" / "era5_preprocessed" / "pressure_level_standardized"
SL_INSTANT_STD_DIR = PROJECT_ROOT / "03_intermediate" / "era5_preprocessed" / "single_level_instant_standardized"
SL_ACCUM_STD_DIR = PROJECT_ROOT / "03_intermediate" / "era5_preprocessed" / "single_level_accum_standardized"

CATALOGUE_PATH = PROJECT_ROOT / "06_events" / "step7e_frozen_primary_event_catalogue.csv"

OUT_DIR = PROJECT_ROOT / "03_intermediate" / "era5_preprocessed" / "event_windows"
REPORT_DIR = PROJECT_ROOT / "05_qc_reports" / "era5"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

AVAILABLE_DATES_CSV = OUT_DIR / "era5_available_dates_by_group.csv"
EVENT_WINDOW_CSV = OUT_DIR / "era5_event_window_availability.csv"
SUMMARY_CSV = REPORT_DIR / "step9b_era5_event_window_availability_summary.csv"
SUMMARY_JSON = REPORT_DIR / "step9b_era5_event_window_availability_summary.json"
REPORT_TXT = REPORT_DIR / "step9b_era5_event_window_availability_report.txt"


def available_dates_from_files(folder: Path, pattern: str, group: str) -> pd.DataFrame:
    records = []

    for path in sorted(folder.glob(pattern)):
        try:
            with xr.open_dataset(path, decode_times=True) as ds:
                times = pd.to_datetime(ds["time"].values)
                for value in times:
                    records.append(
                        {
                            "data_group": group,
                            "date": value.date().isoformat(),
                            "source_file": path.name,
                        }
                    )
        except Exception as error:
            records.append(
                {
                    "data_group": group,
                    "date": "",
                    "source_file": path.name,
                    "error": repr(error),
                }
            )

    return pd.DataFrame(records)


def classify_lag(lag: int, event_duration: int) -> str:
    if lag < -10:
        return "stratospheric_precursor_minus45_to_minus11"
    if -10 <= lag <= -1:
        return "tropospheric_precursor_minus10_to_minus1"
    if 0 <= lag <= event_duration - 1:
        return "event_period"
    if lag >= event_duration:
        return "post_event"
    return "other"


def main() -> None:
    available = pd.concat(
        [
            available_dates_from_files(
                PL_STD_DIR,
                "ERA5_PL_STD_*.nc",
                "pressure_level",
            ),
            available_dates_from_files(
                SL_INSTANT_STD_DIR,
                "ERA5_SL_INSTANT_STD_*.nc",
                "single_level_instant",
            ),
            available_dates_from_files(
                SL_ACCUM_STD_DIR,
                "ERA5_SL_ACCUM_STD_*.nc",
                "single_level_accum",
            ),
        ],
        ignore_index=True,
    )

    available = available.loc[available["date"].ne("")].copy()
    available["date"] = pd.to_datetime(available["date"]).dt.date.astype(str)
    available.to_csv(AVAILABLE_DATES_CSV, index=False)

    availability_sets = {
        group: set(subset["date"])
        for group, subset in available.groupby("data_group")
    }

    common_dates = set.intersection(*availability_sets.values())

    catalogue = pd.read_csv(CATALOGUE_PATH)
    catalogue["start_date"] = pd.to_datetime(catalogue["start_date"])
    catalogue["end_date"] = pd.to_datetime(catalogue["end_date"])

    rows = []

    for _, event in catalogue.iterrows():
        event_id = str(event["event_id"])
        start_date = event["start_date"]
        end_date = event["end_date"]
        duration = int((end_date - start_date).days + 1)

        for lag in range(-45, 11):
            date = start_date + pd.Timedelta(days=lag)
            date_text = date.date().isoformat()

            rows.append(
                {
                    "event_id": event_id,
                    "winter_label": event.get("winter_label", ""),
                    "start_date": start_date.date().isoformat(),
                    "end_date": end_date.date().isoformat(),
                    "event_duration_days": duration,
                    "lag_day_from_onset": lag,
                    "era5_date": date_text,
                    "window_class": classify_lag(lag, duration),
                    "pressure_level_available": date_text in availability_sets.get("pressure_level", set()),
                    "single_level_instant_available": date_text in availability_sets.get("single_level_instant", set()),
                    "single_level_accum_available": date_text in availability_sets.get("single_level_accum", set()),
                    "all_groups_available": date_text in common_dates,
                }
            )

    windows = pd.DataFrame(rows)
    windows.to_csv(EVENT_WINDOW_CSV, index=False)

    summary = (
        windows.groupby(["event_id", "window_class"])
        .agg(
            required_days=("era5_date", "count"),
            all_groups_available_days=("all_groups_available", "sum"),
            pressure_available_days=("pressure_level_available", "sum"),
            single_instant_available_days=("single_level_instant_available", "sum"),
            single_accum_available_days=("single_level_accum_available", "sum"),
        )
        .reset_index()
    )

    summary["all_groups_coverage_percent"] = (
        summary["all_groups_available_days"]
        / summary["required_days"]
        * 100.0
    ).round(2)

    event_summary = (
        windows.groupby("event_id")
        .agg(
            required_total_days=("era5_date", "count"),
            all_groups_available_total_days=("all_groups_available", "sum"),
            start_date=("start_date", "first"),
            end_date=("end_date", "first"),
            winter_label=("winter_label", "first"),
            event_duration_days=("event_duration_days", "first"),
        )
        .reset_index()
    )

    event_summary["total_window_coverage_percent"] = (
        event_summary["all_groups_available_total_days"]
        / event_summary["required_total_days"]
        * 100.0
    ).round(2)

    event_summary["complete_minus10_to_plus10"] = False
    event_summary["complete_minus45_to_plus10"] = False

    for index, row in event_summary.iterrows():
        event_id = row["event_id"]
        subset = windows.loc[windows["event_id"].eq(event_id)]

        minus10_plus10 = subset.loc[
            subset["lag_day_from_onset"].between(-10, 10)
        ]

        minus45_plus10 = subset.loc[
            subset["lag_day_from_onset"].between(-45, 10)
        ]

        event_summary.loc[index, "complete_minus10_to_plus10"] = bool(
            minus10_plus10["all_groups_available"].all()
        )

        event_summary.loc[index, "complete_minus45_to_plus10"] = bool(
            minus45_plus10["all_groups_available"].all()
        )

    final_summary = event_summary.merge(
        summary.pivot_table(
            index="event_id",
            columns="window_class",
            values="all_groups_coverage_percent",
            aggfunc="first",
        ).reset_index(),
        on="event_id",
        how="left",
    )

    final_summary.to_csv(SUMMARY_CSV, index=False)

    json_summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "available_date_count_pressure_level": len(availability_sets.get("pressure_level", set())),
        "available_date_count_single_level_instant": len(availability_sets.get("single_level_instant", set())),
        "available_date_count_single_level_accum": len(availability_sets.get("single_level_accum", set())),
        "common_available_date_count": len(common_dates),
        "event_count": int(event_summary["event_id"].nunique()),
        "events_complete_minus10_to_plus10": int(event_summary["complete_minus10_to_plus10"].sum()),
        "events_complete_minus45_to_plus10": int(event_summary["complete_minus45_to_plus10"].sum()),
        "event_window_csv": str(EVENT_WINDOW_CSV),
        "summary_csv": str(SUMMARY_CSV),
        "important_note": "Because only November-February ERA5 data are downloaded, some -45 to +10 windows may be incomplete.",
    }

    SUMMARY_JSON.write_text(
        json.dumps(json_summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "STEP 9B: ERA5 EVENT-WINDOW AVAILABILITY",
        "=" * 50,
        f"Common ERA5 available dates: {len(common_dates)}",
        f"Frozen event count: {json_summary['event_count']}",
        f"Events complete for -10 to +10 days: {json_summary['events_complete_minus10_to_plus10']}",
        f"Events complete for -45 to +10 days: {json_summary['events_complete_minus45_to_plus10']}",
        "",
        "Interpretation:",
        "  -10 to +10 is the main synoptic circulation window.",
        "  -45 to +10 is the extended precursor window.",
        "  Incomplete windows are expected if an event starts near early December or late February.",
        "",
        f"Available dates CSV: {AVAILABLE_DATES_CSV}",
        f"Event-window CSV: {EVENT_WINDOW_CSV}",
        f"Summary CSV: {SUMMARY_CSV}",
    ]

    REPORT_TXT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))
    print("\nSTEP 9B-2 PASSED. ERA5 EVENT-WINDOW AVAILABILITY TABLE CREATED.")


if __name__ == "__main__":
    main()
