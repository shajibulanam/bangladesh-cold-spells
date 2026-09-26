from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OLD_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "old_xlsx_daily_standardized.parquet"
)

NEW_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "new_tmin_2022_2025_daily_standardized.parquet"
)

RECONCILIATION_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "tmin_overlap_2022_2024_reconciliation.parquet"
)

STATION_MASTER_FILE = (
    PROJECT_ROOT
    / "02_metadata"
    / "station_master_v2.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "temperature_daily_unified_preliminary.parquet"
)

REPORT_DIRECTORY = PROJECT_ROOT / "05_qc_reports"
ADMIN_DIRECTORY = PROJECT_ROOT / "00_admin"

PRE_OVERLAP_END = pd.Timestamp("2021-12-31")
OVERLAP_START = pd.Timestamp("2022-01-01")
OVERLAP_END = pd.Timestamp("2024-12-31")
NEW_ONLY_START = pd.Timestamp("2025-01-01")
NEW_ONLY_END = pd.Timestamp("2025-12-31")

EXPECTED_STATIONS_WITH_DATA = 48
EXPECTED_COMMON_STATIONS = 42
EXPECTED_PENDING_METADATA_STATIONS = 5
EXPECTED_INVALID_OLD_DATE_ROWS = 54

MIN_EXPECTED_UNIFIED_ROWS = 550_000
MAX_EXPECTED_UNIFIED_ROWS = 590_000


SEGMENT_COLUMNS = [
    "station_uid",
    "date",
    "record_origin",

    "tmin_preliminary",
    "tmin_source",
    "tmin_selection_status",
    "tmin_review_required",
    "tmin_zero_flag",

    "tmax_preliminary",
    "tmax_source",
    "tmax_selection_status",
    "tmax_review_required",

    "old_record_present",
    "old_source_row_count",
    "old_source_rows",
    "old_record_duplicate_status",

    "tmin_old_xlsx",
    "old_tmin_raw_values",
    "old_tmin_parse_statuses",
    "old_tmin_duplicate_status",

    "old_tmax_raw_values",
    "old_tmax_parse_statuses",
    "old_tmax_duplicate_status",

    "new_record_present",
    "new_source_row",
    "new_source_day_column",
    "new_source_cell",
    "new_tmin_raw",
    "tmin_new_csv",

    "reconciliation_status",
    "tmin_difference_new_minus_old",
]


STRING_COLUMNS = [
    "record_origin",
    "tmin_source",
    "tmin_selection_status",
    "tmax_source",
    "tmax_selection_status",
    "old_source_rows",
    "old_record_duplicate_status",
    "old_tmin_raw_values",
    "old_tmin_parse_statuses",
    "old_tmin_duplicate_status",
    "old_tmax_raw_values",
    "old_tmax_parse_statuses",
    "old_tmax_duplicate_status",
    "new_source_day_column",
    "new_source_cell",
    "new_tmin_raw",
    "reconciliation_status",
]


BOOLEAN_COLUMNS = [
    "tmin_review_required",
    "tmin_zero_flag",
    "tmax_review_required",
    "old_record_present",
    "new_record_present",
]


FLOAT_COLUMNS = [
    "tmin_preliminary",
    "tmax_preliminary",
    "tmin_old_xlsx",
    "tmin_new_csv",
    "tmin_difference_new_minus_old",
]


def parse_bool(value: Any) -> bool:
    """Convert common text representations to Boolean."""
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def join_unique_text(
    values: pd.Series,
) -> str:
    """Join distinct nonblank values in their source order."""
    output: list[str] = []

    for value in values:
        if pd.isna(value):
            continue

        text = str(value).strip()

        if text and text not in output:
            output.append(text)

    return "; ".join(output)


def summarize_duplicate_variable(
    group: pd.DataFrame,
    numeric_column: str,
    raw_column: str,
    parse_column: str,
) -> dict[str, Any]:
    """
    Summarize one variable across duplicate old-XLSX rows.

    Multiple different numeric values are never averaged or
    selected automatically.
    """
    numeric_values = sorted(
        {
            float(value)
            for value in group[
                numeric_column
            ].dropna()
        }
    )

    parse_values = (
        group[parse_column]
        .fillna("")
        .astype(str)
    )

    raw_values = join_unique_text(
        group[raw_column]
    )

    parse_statuses = join_unique_text(
        parse_values
    )

    contains_non_numeric = (
        parse_values.eq("non_numeric").any()
    )

    if len(numeric_values) == 0:
        candidate = np.nan

        if parse_values.isin(
            [
                "missing_cell",
                "missing_code",
            ]
        ).all():
            status = "duplicate_all_missing"
            review_required = False
        else:
            status = "duplicate_no_numeric_review"
            review_required = True

    elif len(numeric_values) == 1:
        candidate = numeric_values[0]
        status = "duplicate_consistent_numeric"
        review_required = bool(
            contains_non_numeric
        )

    else:
        candidate = np.nan
        status = "duplicate_conflicting_numeric"
        review_required = True

    return {
        "candidate": candidate,
        "raw_values": raw_values,
        "parse_statuses": parse_statuses,
        "duplicate_status": status,
        "review_required": review_required,
    }


def unique_variable_status(
    parse_status: Any,
    numeric_value: Any,
) -> tuple[str, bool]:
    """Assign a variable status for a nonduplicated row."""
    parse_text = str(
        parse_status or ""
    ).strip()

    if (
        parse_text == "numeric"
        and pd.notna(numeric_value)
    ):
        return "unique_numeric", False

    if parse_text in {
        "missing_cell",
        "missing_code",
    }:
        return "unique_missing", False

    return "unique_non_numeric", True


def load_station_master() -> pd.DataFrame:
    """Load the authoritative Step 2A station identity table."""
    master = pd.read_csv(
        STATION_MASTER_FILE,
        dtype=str,
        keep_default_na=False,
    )

    if len(master) != 51:
        raise ValueError(
            "Expected 51 rows in station_master_v2.csv; "
            f"found {len(master)}."
        )

    master[
        "has_any_temperature_data"
    ] = master[
        "has_any_temperature_data"
    ].map(parse_bool)

    master[
        "usable_for_spatial_analysis"
    ] = master[
        "usable_for_spatial_analysis"
    ].map(parse_bool)

    master = master.loc[
        master["has_any_temperature_data"]
    ].copy()

    if len(master) != EXPECTED_STATIONS_WITH_DATA:
        raise ValueError(
            "Expected "
            f"{EXPECTED_STATIONS_WITH_DATA} stations "
            "with temperature data; found "
            f"{len(master)}."
        )

    if master["station_uid"].duplicated().any():
        raise ValueError(
            "Duplicate station_uid values exist "
            "in station_master_v2.csv."
        )

    master["latitude"] = pd.to_numeric(
        master["latitude"],
        errors="coerce",
    )

    master["longitude"] = pd.to_numeric(
        master["longitude"],
        errors="coerce",
    )

    master["station_name_display"] = np.where(
        master[
            "official_station_name"
        ].str.strip().ne(""),
        master["official_station_name"],
        master["new_tmin_station_name"],
    )

    columns = [
        "station_uid",
        "station_id",
        "official_station_name",
        "station_name_display",
        "latitude",
        "longitude",
        "metadata_status",
        "usable_for_spatial_analysis",
    ]

    return master[columns]


def aggregate_old_source() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Convert the old XLSX source into one record per valid
    station-date while preserving duplicate information.
    """
    columns = [
        "source_row",
        "station_uid",
        "date",
        "date_parse_status",
        "tmax_raw",
        "tmax_old_xlsx",
        "tmax_parse_status",
        "tmin_raw",
        "tmin_old_xlsx",
        "tmin_parse_status",
    ]

    old = pd.read_parquet(
        OLD_FILE,
        columns=columns,
    )

    invalid_date_rows = old.loc[
        ~old[
            "date_parse_status"
        ].eq("valid")
    ].copy()

    valid = old.loc[
        old[
            "date_parse_status"
        ].eq("valid")
    ].copy()

    valid["date"] = pd.to_datetime(
        valid["date"]
    )

    duplicate_mask = valid.duplicated(
        subset=[
            "station_uid",
            "date",
        ],
        keep=False,
    )

    unique_rows = valid.loc[
        ~duplicate_mask
    ].copy()

    duplicate_source_rows = valid.loc[
        duplicate_mask
    ].sort_values(
        [
            "station_uid",
            "date",
            "source_row",
        ]
    )

    unique_output = pd.DataFrame(
        {
            "station_uid":
                unique_rows["station_uid"],
            "date":
                unique_rows["date"],
            "old_source_row_count":
                1,
            "old_source_rows":
                unique_rows[
                    "source_row"
                ].astype("int64").astype(str),
            "old_record_duplicate_status":
                "unique",

            "old_tmin_candidate":
                unique_rows[
                    "tmin_old_xlsx"
                ],
            "old_tmin_raw_values":
                unique_rows[
                    "tmin_raw"
                ].fillna("").astype(str),
            "old_tmin_parse_statuses":
                unique_rows[
                    "tmin_parse_status"
                ].fillna("").astype(str),

            "old_tmax_candidate":
                unique_rows[
                    "tmax_old_xlsx"
                ],
            "old_tmax_raw_values":
                unique_rows[
                    "tmax_raw"
                ].fillna("").astype(str),
            "old_tmax_parse_statuses":
                unique_rows[
                    "tmax_parse_status"
                ].fillna("").astype(str),
        }
    )

    tmin_statuses = [
        unique_variable_status(
            parse_status,
            value,
        )
        for parse_status, value in zip(
            unique_rows[
                "tmin_parse_status"
            ],
            unique_rows[
                "tmin_old_xlsx"
            ],
            strict=True,
        )
    ]

    tmax_statuses = [
        unique_variable_status(
            parse_status,
            value,
        )
        for parse_status, value in zip(
            unique_rows[
                "tmax_parse_status"
            ],
            unique_rows[
                "tmax_old_xlsx"
            ],
            strict=True,
        )
    ]

    unique_output[
        "old_tmin_duplicate_status"
    ] = [
        status
        for status, _ in tmin_statuses
    ]

    unique_output[
        "old_tmin_review_required"
    ] = [
        review
        for _, review in tmin_statuses
    ]

    unique_output[
        "old_tmax_duplicate_status"
    ] = [
        status
        for status, _ in tmax_statuses
    ]

    unique_output[
        "old_tmax_review_required"
    ] = [
        review
        for _, review in tmax_statuses
    ]

    duplicate_summaries: list[
        dict[str, Any]
    ] = []

    for (
        station_uid,
        observation_date,
    ), group in duplicate_source_rows.groupby(
        [
            "station_uid",
            "date",
        ],
        sort=True,
    ):
        tmin_summary = (
            summarize_duplicate_variable(
                group,
                "tmin_old_xlsx",
                "tmin_raw",
                "tmin_parse_status",
            )
        )

        tmax_summary = (
            summarize_duplicate_variable(
                group,
                "tmax_old_xlsx",
                "tmax_raw",
                "tmax_parse_status",
            )
        )

        if (
            tmin_summary[
                "duplicate_status"
            ] == "duplicate_conflicting_numeric"
            or tmax_summary[
                "duplicate_status"
            ] == "duplicate_conflicting_numeric"
        ):
            record_duplicate_status = (
                "duplicate_conflicting"
            )

        elif (
            tmin_summary["review_required"]
            or tmax_summary[
                "review_required"
            ]
        ):
            record_duplicate_status = (
                "duplicate_review_required"
            )

        else:
            record_duplicate_status = (
                "duplicate_consistent"
            )

        duplicate_summaries.append(
            {
                "station_uid":
                    station_uid,
                "date":
                    observation_date,
                "old_source_row_count":
                    len(group),
                "old_source_rows":
                    ";".join(
                        str(int(value))
                        for value in group[
                            "source_row"
                        ]
                    ),
                "old_record_duplicate_status":
                    record_duplicate_status,

                "old_tmin_candidate":
                    tmin_summary[
                        "candidate"
                    ],
                "old_tmin_raw_values":
                    tmin_summary[
                        "raw_values"
                    ],
                "old_tmin_parse_statuses":
                    tmin_summary[
                        "parse_statuses"
                    ],
                "old_tmin_duplicate_status":
                    tmin_summary[
                        "duplicate_status"
                    ],
                "old_tmin_review_required":
                    tmin_summary[
                        "review_required"
                    ],

                "old_tmax_candidate":
                    tmax_summary[
                        "candidate"
                    ],
                "old_tmax_raw_values":
                    tmax_summary[
                        "raw_values"
                    ],
                "old_tmax_parse_statuses":
                    tmax_summary[
                        "parse_statuses"
                    ],
                "old_tmax_duplicate_status":
                    tmax_summary[
                        "duplicate_status"
                    ],
                "old_tmax_review_required":
                    tmax_summary[
                        "review_required"
                    ],
            }
        )

    duplicate_output = pd.DataFrame(
        duplicate_summaries
    )

    old_daily = pd.concat(
        [
            unique_output,
            duplicate_output,
        ],
        ignore_index=True,
        sort=False,
    )

    old_daily = old_daily.sort_values(
        [
            "station_uid",
            "date",
        ]
    ).reset_index(drop=True)

    duplicate_final = old_daily.duplicated(
        subset=[
            "station_uid",
            "date",
        ],
        keep=False,
    )

    if duplicate_final.any():
        raise ValueError(
            "The aggregated old source still contains "
            "duplicate station-date keys."
        )

    return (
        old_daily,
        invalid_date_rows,
        duplicate_source_rows,
    )


def normalize_segment(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure every integration segment has the same columns."""
    string_defaults = {
        column: ""
        for column in STRING_COLUMNS
    }

    boolean_defaults = {
        column: False
        for column in BOOLEAN_COLUMNS
    }

    float_defaults = {
        column: np.nan
        for column in FLOAT_COLUMNS
    }

    defaults: dict[str, Any] = {
        **string_defaults,
        **boolean_defaults,
        **float_defaults,
        "old_source_row_count": pd.NA,
        "new_source_row": pd.NA,
    }

    for column in SEGMENT_COLUMNS:
        if column not in data.columns:
            data[column] = defaults.get(
                column,
                pd.NA,
            )

    return data[SEGMENT_COLUMNS].copy()


def build_pre2022_segment(
    old_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Create the 1981–2021 portion from the old XLSX."""
    source = old_daily.loc[
        old_daily["date"].le(
            PRE_OVERLAP_END
        )
    ].copy()

    tmin_zero = (
        source[
            "old_tmin_candidate"
        ].eq(0.0)
    )

    tmin_review = (
        source[
            "old_tmin_review_required"
        ].fillna(False).astype(bool)
        | tmin_zero
    )

    tmin_status = source[
        "old_tmin_duplicate_status"
    ].fillna("").astype(str)

    tmin_status = tmin_status.mask(
        tmin_zero,
        "old_zero_requires_review",
    )

    output = pd.DataFrame(
        {
            "station_uid":
                source["station_uid"],
            "date":
                source["date"],
            "record_origin":
                "old_xlsx_pre2022",

            "tmin_preliminary":
                source[
                    "old_tmin_candidate"
                ],
            "tmin_source":
                np.where(
                    source[
                        "old_tmin_candidate"
                    ].notna(),
                    "old_xlsx",
                    "none",
                ),
            "tmin_selection_status":
                tmin_status,
            "tmin_review_required":
                tmin_review,
            "tmin_zero_flag":
                tmin_zero,

            "tmax_preliminary":
                source[
                    "old_tmax_candidate"
                ],
            "tmax_source":
                np.where(
                    source[
                        "old_tmax_candidate"
                    ].notna(),
                    "old_xlsx",
                    "none",
                ),
            "tmax_selection_status":
                source[
                    "old_tmax_duplicate_status"
                ],
            "tmax_review_required":
                source[
                    "old_tmax_review_required"
                ],

            "old_record_present":
                True,
            "old_source_row_count":
                source[
                    "old_source_row_count"
                ],
            "old_source_rows":
                source[
                    "old_source_rows"
                ],
            "old_record_duplicate_status":
                source[
                    "old_record_duplicate_status"
                ],

            "tmin_old_xlsx":
                source[
                    "old_tmin_candidate"
                ],
            "old_tmin_raw_values":
                source[
                    "old_tmin_raw_values"
                ],
            "old_tmin_parse_statuses":
                source[
                    "old_tmin_parse_statuses"
                ],
            "old_tmin_duplicate_status":
                source[
                    "old_tmin_duplicate_status"
                ],

            "old_tmax_raw_values":
                source[
                    "old_tmax_raw_values"
                ],
            "old_tmax_parse_statuses":
                source[
                    "old_tmax_parse_statuses"
                ],
            "old_tmax_duplicate_status":
                source[
                    "old_tmax_duplicate_status"
                ],

            "new_record_present":
                False,
            "reconciliation_status":
                "not_applicable_pre2022",
        }
    )

    return normalize_segment(output)


def build_overlap_segment(
    reconciliation: pd.DataFrame,
    old_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Create the common-station 2022–2024 segment."""
    old_tmax = old_daily.loc[
        old_daily["date"].between(
            OVERLAP_START,
            OVERLAP_END,
        ),
        [
            "station_uid",
            "date",
            "old_tmax_candidate",
            "old_tmax_raw_values",
            "old_tmax_parse_statuses",
            "old_tmax_duplicate_status",
            "old_tmax_review_required",
            "old_record_duplicate_status",
        ],
    ].copy()

    source = reconciliation.merge(
        old_tmax,
        on=[
            "station_uid",
            "date",
        ],
        how="left",
        validate="one_to_one",
    )

    selected_zero = (
        source[
            "tmin_selected_candidate"
        ].eq(0.0)
    )

    tmin_review = (
        source[
            "reconciliation_review_required"
        ].fillna(False).astype(bool)
        | selected_zero
    )

    output = pd.DataFrame(
        {
            "station_uid":
                source["station_uid"],
            "date":
                pd.to_datetime(
                    source["date"]
                ),
            "record_origin":
                "reconciled_common_2022_2024",

            "tmin_preliminary":
                source[
                    "tmin_selected_candidate"
                ],
            "tmin_source":
                source[
                    "tmin_selected_source"
                ].fillna("none"),
            "tmin_selection_status":
                source[
                    "reconciliation_status"
                ],
            "tmin_review_required":
                tmin_review,
            "tmin_zero_flag":
                selected_zero,

            "tmax_preliminary":
                source[
                    "old_tmax_candidate"
                ],
            "tmax_source":
                np.where(
                    source[
                        "old_tmax_candidate"
                    ].notna(),
                    "old_xlsx",
                    "none",
                ),
            "tmax_selection_status":
                source[
                    "old_tmax_duplicate_status"
                ].fillna(
                    "old_record_absent"
                ),
            "tmax_review_required":
                source[
                    "old_tmax_review_required"
                ].fillna(False),

            "old_record_present":
                source[
                    "old_record_present"
                ],
            "old_source_row_count":
                source[
                    "old_source_row_count"
                ],
            "old_source_rows":
                source[
                    "old_source_rows"
                ],
            "old_record_duplicate_status":
                source[
                    "old_record_duplicate_status"
                ].fillna(
                    source[
                        "old_duplicate_status"
                    ]
                ),

            "tmin_old_xlsx":
                source[
                    "tmin_old_xlsx_compare"
                ],
            "old_tmin_raw_values":
                source[
                    "old_tmin_raw_values"
                ],
            "old_tmin_parse_statuses":
                source[
                    "old_tmin_parse_statuses"
                ],
            "old_tmin_duplicate_status":
                source[
                    "old_duplicate_status"
                ],

            "old_tmax_raw_values":
                source[
                    "old_tmax_raw_values"
                ],
            "old_tmax_parse_statuses":
                source[
                    "old_tmax_parse_statuses"
                ],
            "old_tmax_duplicate_status":
                source[
                    "old_tmax_duplicate_status"
                ],

            "new_record_present":
                source[
                    "new_record_present"
                ],
            "new_source_row":
                source[
                    "new_source_row"
                ],
            "new_source_day_column":
                source[
                    "new_source_day_column"
                ],
            "new_source_cell":
                source[
                    "new_source_cell"
                ],
            "new_tmin_raw":
                source[
                    "new_tmin_raw"
                ],
            "tmin_new_csv":
                source[
                    "tmin_new_csv"
                ],

            "reconciliation_status":
                source[
                    "reconciliation_status"
                ],
            "tmin_difference_new_minus_old":
                source[
                    "tmin_difference_new_minus_old"
                ],
        }
    )

    return normalize_segment(output)


def build_new_source_segment(
    source: pd.DataFrame,
    record_origin: str,
    status_prefix: str,
) -> pd.DataFrame:
    """Create a segment sourced only from the updated Tmin CSV."""
    numeric = source[
        "tmin_parse_status"
    ].eq("numeric")

    missing = source[
        "tmin_parse_status"
    ].isin(
        [
            "missing_cell",
            "missing_code",
        ]
    )

    zero = (
        source["tmin_zero_flag"]
        .fillna(False)
        .astype(bool)
    )

    status = np.select(
        [
            zero,
            numeric,
            missing,
        ],
        [
            f"{status_prefix}_zero_requires_review",
            f"{status_prefix}_numeric",
            f"{status_prefix}_missing",
        ],
        default=(
            f"{status_prefix}_non_numeric"
        ),
    )

    review_required = (
        zero
        | ~(numeric | missing)
    )

    output = pd.DataFrame(
        {
            "station_uid":
                source["station_uid"],
            "date":
                pd.to_datetime(
                    source["date"]
                ),
            "record_origin":
                record_origin,

            "tmin_preliminary":
                source[
                    "tmin_new_csv"
                ],
            "tmin_source":
                np.where(
                    numeric,
                    "new_csv",
                    "none",
                ),
            "tmin_selection_status":
                status,
            "tmin_review_required":
                review_required,
            "tmin_zero_flag":
                zero,

            "tmax_preliminary":
                np.nan,
            "tmax_source":
                "none",
            "tmax_selection_status":
                "no_tmax_source",
            "tmax_review_required":
                False,

            "old_record_present":
                False,
            "new_record_present":
                True,

            "new_source_row":
                source[
                    "source_row"
                ],
            "new_source_day_column":
                source[
                    "source_day_column"
                ],
            "new_source_cell":
                source[
                    "source_cell"
                ],
            "new_tmin_raw":
                source[
                    "tmin_raw"
                ],
            "tmin_new_csv":
                source[
                    "tmin_new_csv"
                ],

            "reconciliation_status":
                "not_applicable",
        }
    )

    return normalize_segment(output)


def main() -> None:
    required_files = [
        OLD_FILE,
        NEW_FILE,
        RECONCILIATION_FILE,
        STATION_MASTER_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    ADMIN_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    station_master = (
        load_station_master()
    )

    (
        old_daily,
        invalid_old_date_rows,
        duplicate_old_source_rows,
    ) = aggregate_old_source()

    reconciliation = pd.read_parquet(
        RECONCILIATION_FILE
    )

    reconciliation["date"] = pd.to_datetime(
        reconciliation["date"]
    )

    new = pd.read_parquet(
        NEW_FILE
    )

    new["date"] = pd.to_datetime(
        new["date"]
    )

    common_station_uids = set(
        reconciliation[
            "station_uid"
        ].dropna().unique()
    )

    if (
        len(common_station_uids)
        != EXPECTED_COMMON_STATIONS
    ):
        raise ValueError(
            "Expected "
            f"{EXPECTED_COMMON_STATIONS} common "
            "stations in reconciliation; found "
            f"{len(common_station_uids)}."
        )

    pre2022_segment = (
        build_pre2022_segment(
            old_daily
        )
    )

    overlap_segment = (
        build_overlap_segment(
            reconciliation,
            old_daily,
        )
    )

    additional_station_overlap = (
        new.loc[
            new["date"].between(
                OVERLAP_START,
                OVERLAP_END,
            )
            & ~new[
                "station_uid"
            ].isin(
                common_station_uids
            )
        ]
        .copy()
    )

    additional_station_segment = (
        build_new_source_segment(
            additional_station_overlap,
            (
                "new_additional_station_"
                "2022_2024"
            ),
            "new_additional_station",
        )
    )

    new_2025 = new.loc[
        new["date"].between(
            NEW_ONLY_START,
            NEW_ONLY_END,
        )
    ].copy()

    new_2025_segment = (
        build_new_source_segment(
            new_2025,
            "new_csv_2025",
            "new_2025",
        )
    )

    expected_concatenated_rows = sum(
        [
            len(pre2022_segment),
            len(overlap_segment),
            len(
                additional_station_segment
            ),
            len(new_2025_segment),
        ]
    )

    unified = pd.concat(
        [
            pre2022_segment,
            overlap_segment,
            additional_station_segment,
            new_2025_segment,
        ],
        ignore_index=True,
        sort=False,
    )

    if (
        len(unified)
        != expected_concatenated_rows
    ):
        raise ValueError(
            "Row loss occurred while concatenating "
            "the integration segments."
        )

    unified = unified.merge(
        station_master,
        on="station_uid",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched_metadata = unified.loc[
        unified["_merge"].ne("both")
    ]

    if not unmatched_metadata.empty:
        unmatched_metadata.to_csv(
            REPORT_DIRECTORY
            / "step3d_unmatched_station_metadata.csv",
            index=False,
        )

        raise ValueError(
            "Some unified records do not match "
            "station_master_v2.csv."
        )

    unified = unified.drop(
        columns=["_merge"]
    )

    unified["date"] = pd.to_datetime(
        unified["date"]
    )

    unified["year"] = (
        unified["date"]
        .dt.year.astype("int16")
    )

    unified["month"] = (
        unified["date"]
        .dt.month.astype("int8")
    )

    unified["day"] = (
        unified["date"]
        .dt.day.astype("int8")
    )

    for column in BOOLEAN_COLUMNS:
        unified[column] = (
            unified[column]
            .fillna(False)
            .astype(bool)
        )

    unified[
        "usable_for_spatial_analysis"
    ] = (
        unified[
            "usable_for_spatial_analysis"
        ]
        .fillna(False)
        .astype(bool)
    )

    unified[
        "metadata_pending_flag"
    ] = ~unified[
        "metadata_status"
    ].eq(
        "official_metadata_available"
    )

    unified[
        "record_review_required"
    ] = (
        unified[
            "tmin_review_required"
        ]
        | unified[
            "tmax_review_required"
        ]
    )

    unified[
        "integration_status"
    ] = np.select(
        [
            unified[
                "record_review_required"
            ],
            unified[
                "metadata_pending_flag"
            ],
        ],
        [
            "value_review_required",
            "metadata_pending",
        ],
        default="ready_for_step4_qc",
    )

    for column in STRING_COLUMNS:
        unified[column] = (
            unified[column]
            .fillna("")
            .astype(str)
        )

    unified["station_id"] = (
        unified["station_id"]
        .fillna("")
        .astype(str)
    )

    unified[
        "official_station_name"
    ] = (
        unified[
            "official_station_name"
        ]
        .fillna("")
        .astype(str)
    )

    unified[
        "station_name_display"
    ] = (
        unified[
            "station_name_display"
        ]
        .fillna("")
        .astype(str)
    )

    unified[
        "old_source_row_count"
    ] = pd.to_numeric(
        unified[
            "old_source_row_count"
        ],
        errors="coerce",
    ).astype("Int16")

    unified[
        "new_source_row"
    ] = pd.to_numeric(
        unified["new_source_row"],
        errors="coerce",
    ).astype("Int32")

    for column in FLOAT_COLUMNS:
        unified[column] = pd.to_numeric(
            unified[column],
            errors="coerce",
        ).astype("float32")

    unified["latitude"] = pd.to_numeric(
        unified["latitude"],
        errors="coerce",
    )

    unified["longitude"] = pd.to_numeric(
        unified["longitude"],
        errors="coerce",
    )

    final_columns = [
        "station_uid",
        "station_id",
        "official_station_name",
        "station_name_display",
        "latitude",
        "longitude",
        "metadata_status",
        "metadata_pending_flag",
        "usable_for_spatial_analysis",

        "date",
        "year",
        "month",
        "day",
        "record_origin",
        "integration_status",

        "tmin_preliminary",
        "tmin_source",
        "tmin_selection_status",
        "tmin_review_required",
        "tmin_zero_flag",

        "tmax_preliminary",
        "tmax_source",
        "tmax_selection_status",
        "tmax_review_required",

        "record_review_required",

        "old_record_present",
        "old_source_row_count",
        "old_source_rows",
        "old_record_duplicate_status",

        "tmin_old_xlsx",
        "old_tmin_raw_values",
        "old_tmin_parse_statuses",
        "old_tmin_duplicate_status",

        "old_tmax_raw_values",
        "old_tmax_parse_statuses",
        "old_tmax_duplicate_status",

        "new_record_present",
        "new_source_row",
        "new_source_day_column",
        "new_source_cell",
        "new_tmin_raw",
        "tmin_new_csv",

        "reconciliation_status",
        "tmin_difference_new_minus_old",
    ]

    unified = (
        unified[final_columns]
        .sort_values(
            [
                "station_uid",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    duplicate_keys = unified.duplicated(
        subset=[
            "station_uid",
            "date",
        ],
        keep=False,
    )

    duplicate_key_rows = unified.loc[
        duplicate_keys
    ].copy()

    duplicate_key_rows.to_csv(
        REPORT_DIRECTORY
        / "step3d_duplicate_final_keys.csv",
        index=False,
    )

    old_duplicate_keys = (
        old_daily.loc[
            old_daily[
                "old_source_row_count"
            ].gt(1)
        ]
        .copy()
    )

    old_duplicate_keys.to_csv(
        REPORT_DIRECTORY
        / "step3d_old_duplicate_keys_all.csv",
        index=False,
    )

    pre2022_duplicate_review = (
        old_duplicate_keys.loc[
            old_duplicate_keys[
                "date"
            ].le(PRE_OVERLAP_END)
            & (
                old_duplicate_keys[
                    "old_tmin_review_required"
                ].fillna(False)
                | old_duplicate_keys[
                    "old_tmax_review_required"
                ].fillna(False)
            )
        ]
        .copy()
    )

    pre2022_duplicate_review.to_csv(
        REPORT_DIRECTORY
        / "step3d_pre2022_duplicate_review.csv",
        index=False,
    )

    invalid_old_date_rows.to_csv(
        REPORT_DIRECTORY
        / "step3d_excluded_invalid_old_dates.csv",
        index=False,
    )

    review_records = unified.loc[
        unified[
            "record_review_required"
        ]
    ].copy()

    review_records.to_csv(
        REPORT_DIRECTORY
        / "step3d_value_review_required_records.csv",
        index=False,
    )

    pending_metadata_stations = (
        unified.loc[
            unified[
                "metadata_pending_flag"
            ],
            [
                "station_uid",
                "station_name_display",
                "metadata_status",
                "usable_for_spatial_analysis",
            ],
        ]
        .drop_duplicates()
        .sort_values("station_uid")
    )

    pending_metadata_stations.to_csv(
        REPORT_DIRECTORY
        / "step3d_pending_metadata_stations.csv",
        index=False,
    )

    station_coverage = (
        unified.groupby(
            [
                "station_uid",
                "station_id",
                "station_name_display",
                "metadata_status",
            ],
            dropna=False,
        )
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            total_rows=(
                "date",
                "size",
            ),
            numeric_tmin=(
                "tmin_preliminary",
                "count",
            ),
            numeric_tmax=(
                "tmax_preliminary",
                "count",
            ),
            tmin_review_records=(
                "tmin_review_required",
                "sum",
            ),
            tmax_review_records=(
                "tmax_review_required",
                "sum",
            ),
        )
        .reset_index()
        .sort_values("station_uid")
    )

    station_coverage.to_csv(
        REPORT_DIRECTORY
        / "step3d_station_coverage.csv",
        index=False,
    )

    origin_counts = (
        unified["record_origin"]
        .value_counts()
        .rename_axis("record_origin")
        .reset_index(name="record_count")
    )

    origin_counts.to_csv(
        REPORT_DIRECTORY
        / "step3d_record_origin_counts.csv",
        index=False,
    )

    tmin_source_counts = (
        unified["tmin_source"]
        .value_counts()
        .rename_axis("tmin_source")
        .reset_index(name="record_count")
    )

    tmin_source_counts.to_csv(
        REPORT_DIRECTORY
        / "step3d_tmin_source_counts.csv",
        index=False,
    )

    tmin_status_counts = (
        unified[
            "tmin_selection_status"
        ]
        .value_counts()
        .rename_axis(
            "tmin_selection_status"
        )
        .reset_index(name="record_count")
    )

    tmin_status_counts.to_csv(
        REPORT_DIRECTORY
        / "step3d_tmin_selection_status_counts.csv",
        index=False,
    )

    unified.to_parquet(
        OUTPUT_FILE,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    minimum_date = unified["date"].min()
    maximum_date = unified["date"].max()

    unique_station_count = (
        unified[
            "station_uid"
        ].nunique()
    )

    pending_station_count = (
        pending_metadata_stations[
            "station_uid"
        ].nunique()
    )

    summary = {
        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "pre2022_rows":
            len(pre2022_segment),
        "common_overlap_rows":
            len(overlap_segment),
        "additional_station_overlap_rows":
            len(
                additional_station_segment
            ),
        "new_2025_rows":
            len(new_2025_segment),
        "unified_rows":
            len(unified),

        "unique_stations":
            unique_station_count,
        "common_overlap_stations":
            len(common_station_uids),
        "pending_metadata_stations":
            pending_station_count,

        "minimum_date":
            minimum_date.date().isoformat(),
        "maximum_date":
            maximum_date.date().isoformat(),

        "duplicate_final_keys":
            int(duplicate_keys.sum()),
        "invalid_old_date_rows_excluded":
            len(invalid_old_date_rows),
        "old_duplicate_keys":
            len(old_duplicate_keys),
        "pre2022_duplicate_review_keys":
            len(
                pre2022_duplicate_review
            ),
        "value_review_required_records":
            len(review_records),

        "numeric_tmin_records":
            int(
                unified[
                    "tmin_preliminary"
                ].notna().sum()
            ),
        "missing_tmin_records":
            int(
                unified[
                    "tmin_preliminary"
                ].isna().sum()
            ),
        "numeric_tmax_records":
            int(
                unified[
                    "tmax_preliminary"
                ].notna().sum()
            ),
        "zero_tmin_records":
            int(
                unified[
                    "tmin_zero_flag"
                ].sum()
            ),

        "record_origin_counts":
            {
                str(row.record_origin):
                    int(row.record_count)
                for row in origin_counts.itertuples(
                    index=False
                )
            },

        "tmin_source_counts":
            {
                str(row.tmin_source):
                    int(row.record_count)
                for row in tmin_source_counts.itertuples(
                    index=False
                )
            },

        "output_file":
            str(OUTPUT_FILE),
        "output_size_bytes":
            OUTPUT_FILE.stat().st_size,
    }

    (
        REPORT_DIRECTORY
        / "step3d_integration_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    parquet_schema = pq.ParquetFile(
        OUTPUT_FILE
    ).schema_arrow

    (
        ADMIN_DIRECTORY
        / "step3d_parquet_schema.json"
    ).write_text(
        json.dumps(
            {
                field.name: str(field.type)
                for field in parquet_schema
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "STEP 3D: UNIFIED TEMPERATURE DATASET",
        "=" * 41,
        (
            "Pre-2022 old-XLSX rows: "
            f"{len(pre2022_segment):,}"
        ),
        (
            "Common-station overlap rows: "
            f"{len(overlap_segment):,}"
        ),
        (
            "Additional-station 2022-2024 rows: "
            f"{len(additional_station_segment):,}"
        ),
        (
            "Updated CSV 2025 rows: "
            f"{len(new_2025_segment):,}"
        ),
        (
            "Unified station-date rows: "
            f"{len(unified):,}"
        ),
        (
            "Unique stations: "
            f"{unique_station_count}"
        ),
        (
            "Pending-metadata stations: "
            f"{pending_station_count}"
        ),
        (
            "Date range: "
            f"{minimum_date.date()} to "
            f"{maximum_date.date()}"
        ),
        (
            "Duplicate final keys: "
            f"{int(duplicate_keys.sum()):,}"
        ),
        (
            "Invalid old-date rows excluded: "
            f"{len(invalid_old_date_rows):,}"
        ),
        (
            "Old duplicate station-date keys: "
            f"{len(old_duplicate_keys):,}"
        ),
        (
            "Pre-2022 duplicate keys requiring review: "
            f"{len(pre2022_duplicate_review):,}"
        ),
        (
            "Value-review records: "
            f"{len(review_records):,}"
        ),
        (
            "Numeric Tmin records: "
            f"{unified['tmin_preliminary'].notna().sum():,}"
        ),
        (
            "Missing Tmin records: "
            f"{unified['tmin_preliminary'].isna().sum():,}"
        ),
        (
            "Numeric Tmax records: "
            f"{unified['tmax_preliminary'].notna().sum():,}"
        ),
        (
            "Zero Tmin records flagged: "
            f"{unified['tmin_zero_flag'].sum():,}"
        ),
        "",
        "Record origins:",
        *[
            (
                f"- {row.record_origin}: "
                f"{row.record_count:,}"
            )
            for row in origin_counts.itertuples(
                index=False
            )
        ],
        "",
        "Tmin selected sources:",
        *[
            (
                f"- {row.tmin_source}: "
                f"{row.record_count:,}"
            )
            for row in tmin_source_counts.itertuples(
                index=False
            )
        ],
        "",
        f"Output: {OUTPUT_FILE}",
    ]

    (
        REPORT_DIRECTORY
        / "step3d_integration_report.txt"
    ).write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(report_lines))

    failures: list[str] = []

    if duplicate_keys.any():
        failures.append(
            "Duplicate station-date keys exist "
            "in the unified dataset."
        )

    if not (
        MIN_EXPECTED_UNIFIED_ROWS
        <= len(unified)
        <= MAX_EXPECTED_UNIFIED_ROWS
    ):
        failures.append(
            "Unexpected unified row count: "
            f"{len(unified):,}."
        )

    if (
        unique_station_count
        != EXPECTED_STATIONS_WITH_DATA
    ):
        failures.append(
            "Expected "
            f"{EXPECTED_STATIONS_WITH_DATA} stations; "
            f"found {unique_station_count}."
        )

    if (
        pending_station_count
        != EXPECTED_PENDING_METADATA_STATIONS
    ):
        failures.append(
            "Expected "
            f"{EXPECTED_PENDING_METADATA_STATIONS} "
            "pending-metadata stations; found "
            f"{pending_station_count}."
        )

    if (
        len(invalid_old_date_rows)
        != EXPECTED_INVALID_OLD_DATE_ROWS
    ):
        failures.append(
            "Expected "
            f"{EXPECTED_INVALID_OLD_DATE_ROWS} "
            "invalid old-date rows; found "
            f"{len(invalid_old_date_rows)}."
        )

    if minimum_date != pd.Timestamp(
        "1981-01-01"
    ):
        failures.append(
            "Unexpected minimum date: "
            f"{minimum_date}."
        )

    if maximum_date != pd.Timestamp(
        "2025-12-31"
    ):
        failures.append(
            "Unexpected maximum date: "
            f"{maximum_date}."
        )

    pre2022_invalid_sources = unified.loc[
        unified["date"].le(
            PRE_OVERLAP_END
        )
        & ~unified[
            "tmin_source"
        ].isin(
            [
                "old_xlsx",
                "none",
            ]
        )
    ]

    if not pre2022_invalid_sources.empty:
        failures.append(
            "Pre-2022 Tmin records contain an "
            "unexpected selected source."
        )

    tmax_2025_present = unified.loc[
        unified["year"].eq(2025),
        "tmax_preliminary",
    ].notna().any()

    if tmax_2025_present:
        failures.append(
            "Tmax values unexpectedly exist for 2025."
        )

    pending_usable = unified.loc[
        unified[
            "metadata_pending_flag"
        ]
        & unified[
            "usable_for_spatial_analysis"
        ]
    ]

    if not pending_usable.empty:
        failures.append(
            "Metadata-pending stations are incorrectly "
            "marked usable for spatial analysis."
        )

    if failures:
        raise SystemExit(
            "\nSTEP 3D FAILED:\n- "
            + "\n- ".join(failures)
        )

    print("\nSTEP 3D PASSED.")


if __name__ == "__main__":
    main()
