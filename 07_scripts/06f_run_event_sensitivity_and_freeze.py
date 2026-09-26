from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DAILY_INPUT_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "step7d_area_weighted_daily_cold_diagnostics.parquet"
)

STEP7D_CATALOGUE_FILE = (
    PROJECT_ROOT
    / "06_events"
    / "step7d_primary_area_weighted_cold_spell_catalogue.csv"
)

STEP7D_EVENT_DAYS_FILE = (
    PROJECT_ROOT
    / "06_events"
    / "step7d_primary_area_weighted_cold_spell_days.csv"
)

STEP7C_EVENT_DAYS_FILE = (
    PROJECT_ROOT
    / "06_events"
    / "step7c_primary_cold_spell_event_days.csv"
)

POLICY_FILE = (
    PROJECT_ROOT
    / "00_admin"
    / "step7e_sensitivity_and_freeze_policy.yaml"
)

SENSITIVITY_DAILY_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "step7e_sensitivity_daily_diagnostics.parquet"
)

SENSITIVITY_CATALOGUE_FILE = (
    PROJECT_ROOT
    / "06_events"
    / "step7e_sensitivity_event_catalogue.csv"
)

SENSITIVITY_EVENT_DAYS_FILE = (
    PROJECT_ROOT
    / "06_events"
    / "step7e_sensitivity_event_days.csv"
)

PRIMARY_ROBUSTNESS_FILE = (
    PROJECT_ROOT
    / "06_events"
    / "step7e_primary_event_robustness.csv"
)

FROZEN_CATALOGUE_FILE = (
    PROJECT_ROOT
    / "06_events"
    / "step7e_frozen_primary_event_catalogue.csv"
)

FROZEN_EVENT_DAYS_FILE = (
    PROJECT_ROOT
    / "06_events"
    / "step7e_frozen_primary_event_days.csv"
)

FROZEN_WINDOWS_FILE = (
    PROJECT_ROOT
    / "06_events"
    / "step7e_frozen_event_onsets_and_windows.csv"
)

REPORT_DIRECTORY = PROJECT_ROOT / "05_qc_reports"

SCENARIO_SUMMARY_FILE = (
    REPORT_DIRECTORY
    / "step7e_sensitivity_scenario_summary.csv"
)

MATCH_DETAIL_FILE = (
    REPORT_DIRECTORY
    / "step7e_primary_sensitivity_event_matches.csv"
)

STEP7C_STEP7D_MATCH_FILE = (
    REPORT_DIRECTORY
    / "step7e_step7c_step7d_event_overlap.csv"
)

VALIDATION_FILE = (
    REPORT_DIRECTORY
    / "step7e_validation_issues.csv"
)

SUMMARY_FILE = (
    REPORT_DIRECTORY
    / "step7e_sensitivity_and_freeze_summary.json"
)

REPORT_FILE = (
    REPORT_DIRECTORY
    / "step7e_sensitivity_and_freeze_report.txt"
)

ADMIN_DIRECTORY = PROJECT_ROOT / "00_admin"

SCHEMA_FILE = (
    ADMIN_DIRECTORY
    / "step7e_parquet_schema.json"
)

MANIFEST_FILE = (
    ADMIN_DIRECTORY
    / "step7e_frozen_event_manifest.json"
)

TABLE_DIRECTORY = (
    PROJECT_ROOT
    / "08_outputs"
    / "tables"
)

DEFINITION_TABLE_FILE = (
    TABLE_DIRECTORY
    / "table_01_cold_spell_definitions.csv"
)

SENSITIVITY_TABLE_FILE = (
    TABLE_DIRECTORY
    / "table_02_event_sensitivity_summary.csv"
)

PRIMARY_EVENT_TABLE_FILE = (
    TABLE_DIRECTORY
    / "table_03_frozen_primary_events.csv"
)

TOP_EVENT_TABLE_FILE = (
    TABLE_DIRECTORY
    / "table_04_top_20_primary_events.csv"
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        return yaml.safe_load(file_handle)


def parse_boolean(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)

    normalized = (
        values.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
        "": False,
    }

    parsed = normalized.map(mapping)

    if parsed.isna().any():
        unknown = sorted(
            normalized.loc[
                parsed.isna()
            ].unique().tolist()
        )

        raise ValueError(
            "Unrecognized Boolean values: "
            f"{unknown}"
        )

    return parsed.astype(bool)


def scenario_token(value: str) -> str:
    token = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        value,
    )

    return token.strip("_").upper()


def assign_scenario_events(
    daily_input: pd.DataFrame,
    scenario_id: str,
    scenario: dict[str, Any],
    eligibility_column: str,
) -> pd.DataFrame:
    result = daily_input.copy()

    area_column = str(
        scenario["area_column"]
    )

    threshold = float(
        scenario["minimum_area_fraction"]
    )

    minimum_duration = int(
        scenario["minimum_consecutive_days"]
    )

    merge_gap = bool(
        scenario.get(
            "merge_one_day_gap",
            False,
        )
    )

    if area_column not in result.columns:
        raise ValueError(
            f"Scenario area column is missing: "
            f"{area_column}"
        )

    result["scenario_id"] = scenario_id
    result["scenario_label"] = str(
        scenario["label"]
    )
    result["scenario_area_column"] = area_column
    result[
        "scenario_cold_area_fraction"
    ] = pd.to_numeric(
        result[area_column],
        errors="coerce",
    )

    result[
        "scenario_minimum_area_fraction"
    ] = threshold

    result[
        "scenario_minimum_consecutive_days"
    ] = minimum_duration

    result[
        "scenario_merge_one_day_gap"
    ] = merge_gap

    result["primary_scenario"] = bool(
        scenario.get(
            "primary",
            False,
        )
    )

    result["base_candidate_day"] = (
        result[eligibility_column]
        & result[
            "scenario_cold_area_fraction"
        ].ge(threshold)
    )

    result["marginal_gap_bridge_day"] = False

    if merge_gap:
        marginal_threshold = float(
            scenario[
                "marginal_gap_minimum_area_fraction"
            ]
        )

        result[
            "marginal_gap_minimum_area_fraction"
        ] = marginal_threshold

        for _, winter in result.groupby(
            "winter_start_year",
            sort=True,
        ):
            winter = winter.sort_values("date")
            indices = winter.index

            base = winter[
                "base_candidate_day"
            ].astype(bool)

            previous_base = base.shift(
                1,
                fill_value=False,
            )

            next_base = base.shift(
                -1,
                fill_value=False,
            )

            previous_date_contiguous = (
                winter["date"]
                .diff()
                .dt.days
                .eq(1)
                .fillna(False)
            )

            next_date_contiguous = (
                winter["date"]
                .shift(-1)
                .sub(winter["date"])
                .dt.days
                .eq(1)
                .fillna(False)
            )

            bridge = (
                winter[eligibility_column]
                & ~base
                & winter[
                    "scenario_cold_area_fraction"
                ].ge(
                    marginal_threshold
                )
                & previous_base
                & next_base
                & previous_date_contiguous
                & next_date_contiguous
            )

            result.loc[
                indices,
                "marginal_gap_bridge_day",
            ] = bridge.to_numpy(dtype=bool)
    else:
        result[
            "marginal_gap_minimum_area_fraction"
        ] = np.nan

    result["candidate_event_day"] = (
        result["base_candidate_day"]
        | result["marginal_gap_bridge_day"]
    )

    result["candidate_run_length"] = 0
    result["qualifying_event_day"] = False

    result["event_id"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )

    result["event_day_index"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int16",
    )

    event_token = scenario_token(
        scenario_id
    )

    for winter_start_year, winter in result.groupby(
        "winter_start_year",
        sort=True,
    ):
        winter = winter.sort_values("date")

        candidate = winter[
            "candidate_event_day"
        ].astype(bool)

        previous_candidate = candidate.shift(
            fill_value=False
        )

        consecutive_date = (
            winter["date"]
            .diff()
            .dt.days
            .eq(1)
            .fillna(False)
        )

        run_start = (
            candidate
            & (
                ~previous_candidate
                | ~consecutive_date
            )
        )

        run_number = run_start.cumsum()
        event_sequence = 0

        candidate_rows = winter.loc[
            candidate
        ]

        if candidate_rows.empty:
            continue

        for _, run in candidate_rows.groupby(
            run_number.loc[candidate]
        ):
            run_indices = run.index
            run_length = len(run_indices)

            result.loc[
                run_indices,
                "candidate_run_length",
            ] = run_length

            if run_length < minimum_duration:
                continue

            event_sequence += 1

            event_id = (
                f"S7E_{event_token}_"
                f"{int(winter_start_year)}_"
                f"{event_sequence:02d}"
            )

            result.loc[
                run_indices,
                "qualifying_event_day",
            ] = True

            result.loc[
                run_indices,
                "event_id",
            ] = event_id

            result.loc[
                run_indices,
                "event_day_index",
            ] = np.arange(
                1,
                run_length + 1,
            )

    result[
        "candidate_run_length"
    ] = result[
        "candidate_run_length"
    ].astype("int16")

    return result


def build_event_catalogue(
    event_days: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "event_id",
        "scenario_id",
        "scenario_label",
        "primary_scenario",
        "winter_start_year",
        "winter_label",
        "start_date",
        "end_date",
        "duration_days",
        "minimum_observed_area_fraction",
        "minimum_scenario_cold_area_fraction",
        "mean_scenario_cold_area_fraction",
        "peak_scenario_cold_area_fraction",
        "cumulative_scenario_cold_area_days",
        "peak_coverage_date",
        "peak_p10_cold_area_fraction",
        "peak_p05_severe_area_fraction",
        "peak_bmd_total_cold_area_fraction",
        "minimum_area_weighted_mean_tmin",
        "minimum_cold_station_tmin",
        "total_area_weighted_p10_deficit",
        "maximum_bmd_severity_rank",
        "most_severe_bmd_category",
        "marginal_gap_day_count",
    ]

    if event_days.empty:
        return pd.DataFrame(
            columns=columns
        )

    rows: list[dict[str, Any]] = []

    for event_id, event in event_days.groupby(
        "event_id",
        sort=True,
    ):
        event = event.sort_values("date")

        peak_row = (
            event.sort_values(
                [
                    "scenario_cold_area_fraction",
                    "date",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .iloc[0]
        )

        rows.append(
            {
                "event_id":
                    event_id,

                "scenario_id":
                    event[
                        "scenario_id"
                    ].iloc[0],

                "scenario_label":
                    event[
                        "scenario_label"
                    ].iloc[0],

                "primary_scenario":
                    bool(
                        event[
                            "primary_scenario"
                        ].iloc[0]
                    ),

                "winter_start_year":
                    int(
                        event[
                            "winter_start_year"
                        ].iloc[0]
                    ),

                "winter_label":
                    event[
                        "winter_label"
                    ].iloc[0],

                "start_date":
                    event["date"].min(),

                "end_date":
                    event["date"].max(),

                "duration_days":
                    len(event),

                "minimum_observed_area_fraction":
                    float(
                        event[
                            "observed_national_area_fraction"
                        ].min()
                    ),

                "minimum_scenario_cold_area_fraction":
                    float(
                        event[
                            "scenario_cold_area_fraction"
                        ].min()
                    ),

                "mean_scenario_cold_area_fraction":
                    float(
                        event[
                            "scenario_cold_area_fraction"
                        ].mean()
                    ),

                "peak_scenario_cold_area_fraction":
                    float(
                        event[
                            "scenario_cold_area_fraction"
                        ].max()
                    ),

                "cumulative_scenario_cold_area_days":
                    float(
                        event[
                            "scenario_cold_area_fraction"
                        ].sum()
                    ),

                "peak_coverage_date":
                    peak_row["date"],

                "peak_p10_cold_area_fraction":
                    float(
                        event[
                            "p10_cold_national_area_fraction"
                        ].max()
                    ),

                "peak_p05_severe_area_fraction":
                    float(
                        event[
                            "p05_severe_national_area_fraction"
                        ].max()
                    ),

                "peak_bmd_total_cold_area_fraction":
                    float(
                        event[
                            "bmd_total_cold_area_fraction"
                        ].max()
                    ),

                "minimum_area_weighted_mean_tmin":
                    float(
                        event[
                            "area_weighted_mean_tmin"
                        ].min()
                    ),

                "minimum_cold_station_tmin":
                    float(
                        event[
                            "cold_station_minimum_tmin"
                        ].min()
                    ),

                "total_area_weighted_p10_deficit":
                    float(
                        event[
                            "area_weighted_p10_deficit"
                        ].sum()
                    ),

                "maximum_bmd_severity_rank":
                    int(
                        event[
                            "maximum_bmd_severity_rank"
                        ].max()
                    ),

                "most_severe_bmd_category":
                    event.sort_values(
                        "maximum_bmd_severity_rank",
                        ascending=False,
                    )[
                        "most_severe_bmd_category"
                    ].iloc[0],

                "marginal_gap_day_count":
                    int(
                        event[
                            "marginal_gap_bridge_day"
                        ].sum()
                    ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "scenario_id",
                "start_date",
                "event_id",
            ]
        )
        .reset_index(drop=True)
    )


def event_signatures(
    event_days: pd.DataFrame,
    id_column: str,
) -> list[tuple[str, str, int]]:
    signatures = []

    for _, event in event_days.groupby(
        id_column
    ):
        event = event.sort_values("date")

        signatures.append(
            (
                str(event["date"].min().date()),
                str(event["date"].max().date()),
                len(event),
            )
        )

    return sorted(signatures)


def compare_event_sets(
    primary_days: pd.DataFrame,
    other_days: pd.DataFrame,
    primary_id_column: str,
    other_id_column: str,
    comparison_id: str,
    minimum_reproduced_fraction: float,
) -> pd.DataFrame:
    rows = []

    for primary_event_id, primary_event in (
        primary_days.groupby(
            primary_id_column,
            sort=True,
        )
    ):
        primary_dates = set(
            pd.to_datetime(
                primary_event["date"]
            ).dt.normalize()
        )

        winter_start_year = int(
            primary_event[
                "winter_start_year"
            ].iloc[0]
        )

        possible = other_days.loc[
            other_days[
                "winter_start_year"
            ].eq(
                winter_start_year
            )
        ]

        matches = []

        for other_event_id, other_event in (
            possible.groupby(
                other_id_column,
                sort=True,
            )
        ):
            other_dates = set(
                pd.to_datetime(
                    other_event["date"]
                ).dt.normalize()
            )

            intersection = len(
                primary_dates & other_dates
            )

            if intersection == 0:
                continue

            union = len(
                primary_dates | other_dates
            )

            primary_overlap = (
                intersection
                / len(primary_dates)
            )

            other_overlap = (
                intersection
                / len(other_dates)
            )

            jaccard = (
                intersection / union
                if union > 0
                else 0.0
            )

            matches.append(
                {
                    "other_event_id":
                        other_event_id,
                    "intersection_days":
                        intersection,
                    "union_days":
                        union,
                    "primary_overlap_fraction":
                        primary_overlap,
                    "other_overlap_fraction":
                        other_overlap,
                    "jaccard_similarity":
                        jaccard,
                    "exact_date_match":
                        primary_dates
                        == other_dates,
                }
            )

        if matches:
            best = sorted(
                matches,
                key=lambda item: (
                    item[
                        "intersection_days"
                    ],
                    item[
                        "jaccard_similarity"
                    ],
                    item[
                        "primary_overlap_fraction"
                    ],
                ),
                reverse=True,
            )[0]

            rows.append(
                {
                    "primary_event_id":
                        primary_event_id,
                    "comparison_id":
                        comparison_id,
                    "matching_event_id":
                        best[
                            "other_event_id"
                        ],
                    "overlapping_event_count":
                        len(matches),
                    "intersection_days":
                        best[
                            "intersection_days"
                        ],
                    "primary_event_duration":
                        len(primary_dates),
                    "matching_event_duration":
                        len(
                            set(
                                pd.to_datetime(
                                    possible.loc[
                                        possible[
                                            other_id_column
                                        ].eq(
                                            best[
                                                "other_event_id"
                                            ]
                                        ),
                                        "date",
                                    ]
                                ).dt.normalize()
                            )
                        ),
                    "primary_overlap_fraction":
                        best[
                            "primary_overlap_fraction"
                        ],
                    "matching_overlap_fraction":
                        best[
                            "other_overlap_fraction"
                        ],
                    "jaccard_similarity":
                        best[
                            "jaccard_similarity"
                        ],
                    "exact_date_match":
                        best[
                            "exact_date_match"
                        ],
                    "primary_event_reproduced":
                        best[
                            "primary_overlap_fraction"
                        ]
                        >= minimum_reproduced_fraction,
                }
            )
        else:
            rows.append(
                {
                    "primary_event_id":
                        primary_event_id,
                    "comparison_id":
                        comparison_id,
                    "matching_event_id":
                        "",
                    "overlapping_event_count":
                        0,
                    "intersection_days":
                        0,
                    "primary_event_duration":
                        len(primary_dates),
                    "matching_event_duration":
                        0,
                    "primary_overlap_fraction":
                        0.0,
                    "matching_overlap_fraction":
                        0.0,
                    "jaccard_similarity":
                        0.0,
                    "exact_date_match":
                        False,
                    "primary_event_reproduced":
                        False,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    required_files = [
        DAILY_INPUT_FILE,
        STEP7D_CATALOGUE_FILE,
        STEP7D_EVENT_DAYS_FILE,
        STEP7C_EVENT_DAYS_FILE,
        POLICY_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    for directory in [
        SENSITIVITY_DAILY_FILE.parent,
        SENSITIVITY_CATALOGUE_FILE.parent,
        REPORT_DIRECTORY,
        ADMIN_DIRECTORY,
        TABLE_DIRECTORY,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    policy = load_yaml(POLICY_FILE)

    primary_scenario_id = str(
        policy["primary_scenario_id"]
    )

    scenarios = policy["scenarios"]

    eligibility_column = str(
        policy[
            "daily_availability"
        ]["eligibility_column"]
    )

    matching_rules = policy[
        "event_matching"
    ]

    minimum_reproduced_fraction = float(
        matching_rules[
            "reproduced_primary_event_minimum_overlap_fraction"
        ]
    )

    high_robustness_fraction = float(
        matching_rules[
            "high_robustness_fraction"
        ]
    )

    moderate_robustness_fraction = float(
        matching_rules[
            "moderate_robustness_fraction"
        ]
    )

    freeze_rules = policy[
        "final_event_freeze"
    ]

    daily = pd.read_parquet(
        DAILY_INPUT_FILE
    )

    daily["date"] = pd.to_datetime(
        daily["date"]
    )

    daily = daily.sort_values(
        [
            "winter_start_year",
            "date",
        ]
    ).reset_index(drop=True)

    if eligibility_column not in daily.columns:
        raise ValueError(
            f"Missing eligibility column: "
            f"{eligibility_column}"
        )

    daily[eligibility_column] = (
        parse_boolean(
            daily[eligibility_column]
        )
    )

    bmd_columns = [
        "bmd_mild_area_fraction",
        "bmd_moderate_area_fraction",
        "bmd_severe_area_fraction",
        "bmd_very_severe_area_fraction",
    ]

    for column in bmd_columns:
        if column not in daily.columns:
            raise ValueError(
                f"Missing BMD area column: "
                f"{column}"
            )

    daily[
        "bmd_total_cold_area_fraction"
    ] = (
        daily[bmd_columns]
        .sum(axis=1)
    )

    if (
        daily[
            "bmd_total_cold_area_fraction"
        ]
        .gt(
            daily[
                "observed_national_area_fraction"
            ]
            + 1e-8
        )
        .any()
    ):
        raise ValueError(
            "Total BMD cold area exceeds "
            "observed national area."
        )

    scenario_daily_frames = []
    scenario_catalogue_frames = []

    for scenario_id, scenario in (
        scenarios.items()
    ):
        scenario_daily = (
            assign_scenario_events(
                daily_input=daily,
                scenario_id=str(
                    scenario_id
                ),
                scenario=scenario,
                eligibility_column=(
                    eligibility_column
                ),
            )
        )

        scenario_daily_frames.append(
            scenario_daily
        )

        scenario_event_days = (
            scenario_daily.loc[
                scenario_daily[
                    "qualifying_event_day"
                ]
            ]
        )

        scenario_catalogue_frames.append(
            build_event_catalogue(
                scenario_event_days
            )
        )

    sensitivity_daily = (
        pd.concat(
            scenario_daily_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "scenario_id",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    sensitivity_catalogue = (
        pd.concat(
            scenario_catalogue_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "scenario_id",
                "start_date",
                "event_id",
            ]
        )
        .reset_index(drop=True)
    )

    sensitivity_event_days = (
        sensitivity_daily.loc[
            sensitivity_daily[
                "qualifying_event_day"
            ]
        ]
        .copy()
        .sort_values(
            [
                "scenario_id",
                "date",
                "event_id",
            ]
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------
    # Validate exact reproduction of Step 7D primary
    # -------------------------------------------------

    step7d_catalogue = pd.read_csv(
        STEP7D_CATALOGUE_FILE,
        parse_dates=[
            "start_date",
            "end_date",
        ],
    )

    step7d_days = pd.read_csv(
        STEP7D_EVENT_DAYS_FILE,
        parse_dates=["date"],
    )

    if (
        "area_weighted_event_id"
        not in step7d_days.columns
    ):
        raise ValueError(
            "Step 7D event-day ID column "
            "was not found."
        )

    generated_primary_days = (
        sensitivity_event_days.loc[
            sensitivity_event_days[
                "scenario_id"
            ].eq(
                primary_scenario_id
            )
        ]
    )

    existing_dates = set(
        step7d_days["date"].dt.normalize()
    )

    generated_dates = set(
        generated_primary_days[
            "date"
        ].dt.normalize()
    )

    if existing_dates != generated_dates:
        raise ValueError(
            "Generated Step 7E primary event dates "
            "do not exactly reproduce Step 7D."
        )

    existing_signatures = event_signatures(
        step7d_days,
        "area_weighted_event_id",
    )

    generated_signatures = event_signatures(
        generated_primary_days,
        "event_id",
    )

    if (
        existing_signatures
        != generated_signatures
    ):
        raise ValueError(
            "Generated primary event boundaries "
            "do not reproduce Step 7D."
        )

    # -------------------------------------------------
    # Sensitivity matching
    # -------------------------------------------------

    primary_days_standard = (
        step7d_days.rename(
            columns={
                "area_weighted_event_id":
                    "event_id"
            }
        )
    )

    sensitivity_matches = []

    sensitivity_scenario_ids = [
        scenario_id
        for scenario_id
        in scenarios
        if scenario_id
        != primary_scenario_id
    ]

    for scenario_id in (
        sensitivity_scenario_ids
    ):
        other_days = (
            sensitivity_event_days.loc[
                sensitivity_event_days[
                    "scenario_id"
                ].eq(
                    scenario_id
                )
            ]
        )

        match = compare_event_sets(
            primary_days=(
                primary_days_standard
            ),
            other_days=other_days,
            primary_id_column="event_id",
            other_id_column="event_id",
            comparison_id=scenario_id,
            minimum_reproduced_fraction=(
                minimum_reproduced_fraction
            ),
        )

        sensitivity_matches.append(
            match
        )

    match_detail = pd.concat(
        sensitivity_matches,
        ignore_index=True,
    )

    match_detail.to_csv(
        MATCH_DETAIL_FILE,
        index=False,
    )

    robustness = (
        match_detail.groupby(
            "primary_event_id",
            as_index=False,
        )
        .agg(
            sensitivity_scenarios_tested=(
                "comparison_id",
                "nunique",
            ),
            sensitivity_scenarios_reproduced=(
                "primary_event_reproduced",
                "sum",
            ),
            exact_date_match_count=(
                "exact_date_match",
                "sum",
            ),
            mean_primary_overlap_fraction=(
                "primary_overlap_fraction",
                "mean",
            ),
            minimum_primary_overlap_fraction=(
                "primary_overlap_fraction",
                "min",
            ),
            mean_jaccard_similarity=(
                "jaccard_similarity",
                "mean",
            ),
        )
    )

    robustness[
        "sensitivity_robustness_fraction"
    ] = (
        robustness[
            "sensitivity_scenarios_reproduced"
        ]
        / robustness[
            "sensitivity_scenarios_tested"
        ]
    )

    robustness[
        "robustness_class"
    ] = np.select(
        [
            robustness[
                "sensitivity_robustness_fraction"
            ].ge(
                high_robustness_fraction
            ),
            robustness[
                "sensitivity_robustness_fraction"
            ].ge(
                moderate_robustness_fraction
            ),
        ],
        [
            "high",
            "moderate",
        ],
        default="low",
    )

    robustness.to_csv(
        PRIMARY_ROBUSTNESS_FILE,
        index=False,
    )

    # -------------------------------------------------
    # Step 7C versus Step 7D comparison
    # -------------------------------------------------

    step7c_days = pd.read_csv(
        STEP7C_EVENT_DAYS_FILE,
        parse_dates=["date"],
    )

    if "primary_event_id" in step7c_days.columns:
        step7c_id_column = (
            "primary_event_id"
        )
    elif "event_id" in step7c_days.columns:
        step7c_id_column = "event_id"
    else:
        raise ValueError(
            "No Step 7C event-ID column found."
        )

    step7c_step7d_match = compare_event_sets(
        primary_days=primary_days_standard,
        other_days=step7c_days,
        primary_id_column="event_id",
        other_id_column=step7c_id_column,
        comparison_id=(
            "step7c_compound_station_count"
        ),
        minimum_reproduced_fraction=(
            minimum_reproduced_fraction
        ),
    )

    step7c_step7d_match.to_csv(
        STEP7C_STEP7D_MATCH_FILE,
        index=False,
    )

    # -------------------------------------------------
    # Freeze the final primary catalogue
    # -------------------------------------------------

    frozen_utc = datetime.now(
        timezone.utc
    ).isoformat()

    frozen_catalogue = (
        step7d_catalogue.merge(
            robustness,
            left_on="event_id",
            right_on="primary_event_id",
            how="left",
            validate="one_to_one",
        )
        .drop(
            columns=[
                "primary_event_id",
            ],
            errors="ignore",
        )
    )

    frozen_catalogue[
        "final_primary_definition_id"
    ] = "p10_area20_duration3"

    frozen_catalogue[
        "frozen_for_atmospheric_analysis"
    ] = True

    frozen_catalogue[
        "frozen_utc"
    ] = frozen_utc

    ranking_columns = [
        (
            "duration_days",
            "duration_rank",
        ),
        (
            "peak_p10_cold_area_fraction",
            "peak_area_rank",
        ),
        (
            "cumulative_cold_area_days",
            "cumulative_area_rank",
        ),
        (
            "total_area_weighted_p10_deficit",
            "cold_deficit_rank",
        ),
    ]

    for value_column, rank_column in (
        ranking_columns
    ):
        frozen_catalogue[
            rank_column
        ] = (
            pd.to_numeric(
                frozen_catalogue[
                    value_column
                ],
                errors="coerce",
            )
            .rank(
                method="min",
                ascending=False,
            )
            .astype("Int16")
        )

    frozen_catalogue[
        "multimetric_rank_sum"
    ] = (
        frozen_catalogue[
            [
                rank_column
                for _, rank_column
                in ranking_columns
            ]
        ]
        .sum(axis=1)
    )

    frozen_catalogue[
        "event_strength_rank"
    ] = (
        frozen_catalogue[
            "multimetric_rank_sum"
        ]
        .rank(
            method="min",
            ascending=True,
        )
        .astype("Int16")
    )

    frozen_catalogue = (
        frozen_catalogue.sort_values(
            [
                "start_date",
                "event_id",
            ]
        )
        .reset_index(drop=True)
    )

    frozen_days = (
        primary_days_standard.copy()
    )

    onset_lookup = (
        frozen_catalogue.set_index(
            "event_id"
        )["start_date"]
    )

    frozen_days[
        "event_onset_date"
    ] = (
        frozen_days[
            "event_id"
        ].map(onset_lookup)
    )

    frozen_days[
        "event_onset_date"
    ] = pd.to_datetime(
        frozen_days[
            "event_onset_date"
        ]
    )

    frozen_days[
        "day_relative_to_onset"
    ] = (
        frozen_days["date"]
        - frozen_days[
            "event_onset_date"
        ]
    ).dt.days

    frozen_days[
        "final_primary_definition_id"
    ] = "p10_area20_duration3"

    frozen_days[
        "frozen_for_atmospheric_analysis"
    ] = True

    frozen_days[
        "frozen_utc"
    ] = frozen_utc

    windows = frozen_catalogue[
        [
            "event_id",
            "winter_label",
            "start_date",
            "end_date",
            "duration_days",
            "peak_coverage_date",
            "event_strength_rank",
            "robustness_class",
        ]
    ].copy()

    windows["onset_date"] = pd.to_datetime(
        windows["start_date"]
    )

    windows[
        "tropospheric_window_start"
    ] = (
        windows["onset_date"]
        - pd.to_timedelta(
            int(
                freeze_rules[
                    "tropospheric_window_days_before_onset"
                ]
            ),
            unit="D",
        )
    )

    windows[
        "tropospheric_window_end"
    ] = (
        windows["onset_date"]
        + pd.to_timedelta(
            int(
                freeze_rules[
                    "tropospheric_window_days_after_onset"
                ]
            ),
            unit="D",
        )
    )

    windows[
        "stratospheric_window_start"
    ] = (
        windows["onset_date"]
        - pd.to_timedelta(
            int(
                freeze_rules[
                    "stratospheric_window_days_before_onset"
                ]
            ),
            unit="D",
        )
    )

    windows[
        "stratospheric_window_end"
    ] = (
        windows["onset_date"]
        + pd.to_timedelta(
            int(
                freeze_rules[
                    "stratospheric_window_days_after_onset"
                ]
            ),
            unit="D",
        )
    )

    # -------------------------------------------------
    # Scenario summary
    # -------------------------------------------------

    scenario_rows = []

    for scenario_id, scenario in (
        scenarios.items()
    ):
        scenario_daily = (
            sensitivity_daily.loc[
                sensitivity_daily[
                    "scenario_id"
                ].eq(
                    scenario_id
                )
            ]
        )

        scenario_events = (
            sensitivity_catalogue.loc[
                sensitivity_catalogue[
                    "scenario_id"
                ].eq(
                    scenario_id
                )
            ]
        )

        scenario_rows.append(
            {
                "scenario_id":
                    scenario_id,
                "scenario_label":
                    scenario["label"],
                "primary_scenario":
                    bool(
                        scenario.get(
                            "primary",
                            False,
                        )
                    ),
                "area_column":
                    scenario[
                        "area_column"
                    ],
                "minimum_area_fraction":
                    scenario[
                        "minimum_area_fraction"
                    ],
                "minimum_consecutive_days":
                    scenario[
                        "minimum_consecutive_days"
                    ],
                "merge_one_day_gap":
                    bool(
                        scenario.get(
                            "merge_one_day_gap",
                            False,
                        )
                    ),
                "candidate_day_count":
                    int(
                        scenario_daily[
                            "candidate_event_day"
                        ].sum()
                    ),
                "event_day_count":
                    int(
                        scenario_daily[
                            "qualifying_event_day"
                        ].sum()
                    ),
                "event_count":
                    len(
                        scenario_events
                    ),
                "winters_with_events":
                    int(
                        scenario_events[
                            "winter_start_year"
                        ].nunique()
                    ),
                "median_event_duration_days":
                    (
                        float(
                            scenario_events[
                                "duration_days"
                            ].median()
                        )
                        if not scenario_events.empty
                        else np.nan
                    ),
                "maximum_event_duration_days":
                    (
                        int(
                            scenario_events[
                                "duration_days"
                            ].max()
                        )
                        if not scenario_events.empty
                        else 0
                    ),
                "maximum_event_area_fraction":
                    (
                        float(
                            scenario_events[
                                "peak_scenario_cold_area_fraction"
                            ].max()
                        )
                        if not scenario_events.empty
                        else np.nan
                    ),
            }
        )

    scenario_summary = (
        pd.DataFrame(
            scenario_rows
        )
    )

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    issues = []

    for scenario_id, scenario in (
        scenarios.items()
    ):
        scenario_days = (
            sensitivity_event_days.loc[
                sensitivity_event_days[
                    "scenario_id"
                ].eq(
                    scenario_id
                )
            ]
        )

        minimum_duration = int(
            scenario[
                "minimum_consecutive_days"
            ]
        )

        threshold = float(
            scenario[
                "minimum_area_fraction"
            ]
        )

        for event_id, event in (
            scenario_days.groupby(
                "event_id"
            )
        ):
            event = event.sort_values(
                "date"
            )

            if len(event) < minimum_duration:
                issues.append(
                    {
                        "scenario_id":
                            scenario_id,
                        "event_id":
                            event_id,
                        "issue":
                            "event_too_short",
                    }
                )

            if (
                event["date"]
                .diff()
                .dropna()
                .dt.days
                .ne(1)
                .any()
            ):
                issues.append(
                    {
                        "scenario_id":
                            scenario_id,
                        "event_id":
                            event_id,
                        "issue":
                            "nonconsecutive_dates",
                    }
                )

            if (
                event[
                    "winter_start_year"
                ].nunique()
                != 1
            ):
                issues.append(
                    {
                        "scenario_id":
                            scenario_id,
                        "event_id":
                            event_id,
                        "issue":
                            "event_crosses_winter",
                    }
                )

            if not event[
                eligibility_column
            ].all():
                issues.append(
                    {
                        "scenario_id":
                            scenario_id,
                        "event_id":
                            event_id,
                        "issue":
                            "contains_ineligible_day",
                    }
                )

            regular_days = event.loc[
                ~event[
                    "marginal_gap_bridge_day"
                ]
            ]

            if (
                regular_days[
                    "scenario_cold_area_fraction"
                ].lt(threshold)
                .any()
            ):
                issues.append(
                    {
                        "scenario_id":
                            scenario_id,
                        "event_id":
                            event_id,
                        "issue":
                            "regular_day_below_area_threshold",
                    }
                )

            bridge_days = event.loc[
                event[
                    "marginal_gap_bridge_day"
                ]
            ]

            if not bridge_days.empty:
                marginal_threshold = float(
                    scenario[
                        "marginal_gap_minimum_area_fraction"
                    ]
                )

                if (
                    bridge_days[
                        "scenario_cold_area_fraction"
                    ].lt(
                        marginal_threshold
                    )
                    .any()
                ):
                    issues.append(
                        {
                            "scenario_id":
                                scenario_id,
                            "event_id":
                                event_id,
                            "issue":
                                "bridge_day_below_marginal_threshold",
                        }
                    )

    validation = pd.DataFrame(
        issues,
        columns=[
            "scenario_id",
            "event_id",
            "issue",
        ],
    )

    validation.to_csv(
        VALIDATION_FILE,
        index=False,
    )

    if not validation.empty:
        raise ValueError(
            "Step 7E validation failed. "
            f"Review {VALIDATION_FILE}"
        )

    # -------------------------------------------------
    # Save outputs
    # -------------------------------------------------

    sensitivity_daily.to_parquet(
        SENSITIVITY_DAILY_FILE,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    sensitivity_catalogue.to_csv(
        SENSITIVITY_CATALOGUE_FILE,
        index=False,
    )

    sensitivity_event_days.to_csv(
        SENSITIVITY_EVENT_DAYS_FILE,
        index=False,
    )

    frozen_catalogue.to_csv(
        FROZEN_CATALOGUE_FILE,
        index=False,
    )

    frozen_days.to_csv(
        FROZEN_EVENT_DAYS_FILE,
        index=False,
    )

    windows.to_csv(
        FROZEN_WINDOWS_FILE,
        index=False,
    )

    scenario_summary.to_csv(
        SCENARIO_SUMMARY_FILE,
        index=False,
    )

    # First manuscript-ready tables.
    definition_rows = []

    for scenario_id, scenario in (
        scenarios.items()
    ):
        definition_rows.append(
            {
                "definition_id":
                    scenario_id,
                "definition_label":
                    scenario["label"],
                "station_condition":
                    scenario[
                        "station_condition"
                    ],
                "national_area_threshold_percent":
                    100.0
                    * float(
                        scenario[
                            "minimum_area_fraction"
                        ]
                    ),
                "minimum_duration_days":
                    int(
                        scenario[
                            "minimum_consecutive_days"
                        ]
                    ),
                "one_day_gap_merging":
                    bool(
                        scenario.get(
                            "merge_one_day_gap",
                            False,
                        )
                    ),
                "primary_definition":
                    bool(
                        scenario.get(
                            "primary",
                            False,
                        )
                    ),
            }
        )

    pd.DataFrame(
        definition_rows
    ).to_csv(
        DEFINITION_TABLE_FILE,
        index=False,
    )

    scenario_summary.to_csv(
        SENSITIVITY_TABLE_FILE,
        index=False,
    )

    manuscript_columns = [
        "event_id",
        "winter_label",
        "start_date",
        "end_date",
        "duration_days",
        "peak_p10_cold_area_fraction",
        "mean_p10_cold_area_fraction",
        "cumulative_cold_area_days",
        "minimum_station_tmin",
        "most_severe_bmd_category",
        "event_centre_longitude",
        "event_centre_latitude",
        "robustness_class",
        "sensitivity_robustness_fraction",
        "event_strength_rank",
    ]

    available_manuscript_columns = [
        column
        for column in manuscript_columns
        if column
        in frozen_catalogue.columns
    ]

    frozen_catalogue[
        available_manuscript_columns
    ].to_csv(
        PRIMARY_EVENT_TABLE_FILE,
        index=False,
    )

    frozen_catalogue[
        available_manuscript_columns
    ].sort_values(
        [
            "event_strength_rank",
            "start_date",
        ]
    ).head(20).to_csv(
        TOP_EVENT_TABLE_FILE,
        index=False,
    )

    parquet_file = pq.ParquetFile(
        SENSITIVITY_DAILY_FILE
    )

    SCHEMA_FILE.write_text(
        json.dumps(
            {
                field.name: str(field.type)
                for field
                in parquet_file.schema_arrow
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "created_utc":
            frozen_utc,
        "authoritative_primary_source":
            str(
                STEP7D_CATALOGUE_FILE
            ),
        "final_definition_id":
            "p10_area20_duration3",
        "final_definition":
            (
                "Station calendar-day p10, "
                "at least 20 percent Bangladesh "
                "area, at least 3 consecutive days"
            ),
        "primary_event_count":
            len(
                frozen_catalogue
            ),
        "primary_event_day_count":
            len(
                frozen_days
            ),
        "primary_events_removed_by_sensitivity":
            0,
        "frozen_catalogue":
            str(
                FROZEN_CATALOGUE_FILE
            ),
        "frozen_event_days":
            str(
                FROZEN_EVENT_DAYS_FILE
            ),
        "atmospheric_analysis_windows":
            str(
                FROZEN_WINDOWS_FILE
            ),
    }

    MANIFEST_FILE.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = {
        "created_utc":
            frozen_utc,
        "scenario_count":
            len(scenarios),
        "primary_event_count":
            len(
                frozen_catalogue
            ),
        "primary_event_day_count":
            len(
                frozen_days
            ),
        "primary_events_high_robustness":
            int(
                frozen_catalogue[
                    "robustness_class"
                ].eq("high").sum()
            ),
        "primary_events_moderate_robustness":
            int(
                frozen_catalogue[
                    "robustness_class"
                ].eq("moderate").sum()
            ),
        "primary_events_low_robustness":
            int(
                frozen_catalogue[
                    "robustness_class"
                ].eq("low").sum()
            ),
        "step7c_events_reproducing_primary_events":
            int(
                step7c_step7d_match[
                    "primary_event_reproduced"
                ].sum()
            ),
        "validation_issue_count":
            len(validation),
        "primary_step7d_exactly_reproduced":
            True,
        "primary_events_removed":
            0,
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "STEP 7E: SENSITIVITY ANALYSIS AND EVENT FREEZE",
        "=" * 54,
        (
            "Sensitivity scenarios: "
            f"{len(scenarios)}"
        ),
        (
            "Frozen primary events: "
            f"{len(frozen_catalogue)}"
        ),
        (
            "Frozen primary event days: "
            f"{len(frozen_days)}"
        ),
        (
            "High-robustness events: "
            f"{summary['primary_events_high_robustness']}"
        ),
        (
            "Moderate-robustness events: "
            f"{summary['primary_events_moderate_robustness']}"
        ),
        (
            "Low-robustness events: "
            f"{summary['primary_events_low_robustness']}"
        ),
        (
            "Primary Step 7D dates reproduced exactly: Yes"
        ),
        (
            "Primary events removed by sensitivity: 0"
        ),
        (
            "Validation issues: "
            f"{len(validation)}"
        ),
        "",
        (
            "Frozen catalogue: "
            f"{FROZEN_CATALOGUE_FILE}"
        ),
        (
            "Frozen event days: "
            f"{FROZEN_EVENT_DAYS_FILE}"
        ),
        (
            "Atmospheric windows: "
            f"{FROZEN_WINDOWS_FILE}"
        ),
    ]

    REPORT_FILE.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(report_lines))
    print("\nSTEP 7E PASSED.")


if __name__ == "__main__":
    main()
