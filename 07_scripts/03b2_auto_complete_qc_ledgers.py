from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QC_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "temperature_daily_qc_stage1.parquet"
)

OLD_FILE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "old_xlsx_daily_standardized.parquet"
)

VALUE_LEDGER = (
    PROJECT_ROOT
    / "04_clean_data"
    / "qc_decisions"
    / "value_decisions.csv"
)

INVALID_DATE_LEDGER = (
    PROJECT_ROOT
    / "04_clean_data"
    / "qc_decisions"
    / "invalid_date_decisions.csv"
)

POLICY_FILE = (
    PROJECT_ROOT
    / "00_admin"
    / "step4b_automatic_decision_policy.yaml"
)

REPORT_DIRECTORY = PROJECT_ROOT / "05_qc_reports"

REVIEWER = "automated_rule_v1"
REVIEW_DATE = date.today().isoformat()

VERIFICATION_SOURCE = (
    "Official BMD datasets; deterministic Step 4B "
    "automatic decision policy v1"
)


def optional_float(value: Any) -> float | None:
    """Convert a value to finite float or None."""
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()

    if text == "":
        return None

    try:
        number = float(text)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def split_flags(value: Any) -> set[str]:
    """Convert semicolon-delimited QC codes to a set."""
    if value is None or pd.isna(value):
        return set()

    return {
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    }


def values_differ(
    first: float | None,
    second: float | None,
) -> bool:
    if first is None or second is None:
        return first is not second

    return not math.isclose(
        first,
        second,
        abs_tol=1e-6,
        rel_tol=0.0,
    )


def main() -> None:
    required_files = [
        QC_FILE,
        OLD_FILE,
        VALUE_LEDGER,
        INVALID_DATE_LEDGER,
        POLICY_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with POLICY_FILE.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        policy = yaml.safe_load(file_handle)

    limits = policy["screening_limits"]

    tmin_min = float(
        limits["tmin_minimum"]
    )

    tmin_max = float(
        limits["tmin_maximum"]
    )

    tmax_min = float(
        limits["tmax_minimum"]
    )

    tmax_max = float(
        limits["tmax_maximum"]
    )

    minimum_dtr = float(
        limits["minimum_positive_dtr"]
    )

    high_dtr = float(
        limits["high_dtr_threshold"]
    )

    qc = pd.read_parquet(QC_FILE)
    qc["date"] = pd.to_datetime(qc["date"])

    qc["date_key"] = qc[
        "date"
    ].dt.strftime("%Y-%m-%d")

    qc_lookup = qc.set_index(
        [
            "station_uid",
            "date_key",
        ],
        drop=False,
    )

    value_ledger = pd.read_csv(
        VALUE_LEDGER,
        dtype=str,
        keep_default_na=False,
    )

    if value_ledger["review_id"].duplicated().any():
        raise ValueError(
            "Duplicate review IDs in value ledger."
        )

    automatic_value_log: list[
        dict[str, Any]
    ] = []

    decision_counts: dict[str, int] = {}

    for index, ledger_row in value_ledger.iterrows():
        key = (
            ledger_row["station_uid"],
            str(ledger_row["date"])[:10],
        )

        if key not in qc_lookup.index:
            raise ValueError(
                f"Review record not found in QC dataset: {key}"
            )

        record = qc_lookup.loc[key]

        if isinstance(record, pd.DataFrame):
            raise ValueError(
                f"Duplicate QC keys found for {key}"
            )

        flags = split_flags(
            record["qc_flag_codes"]
        )

        current_tmin = optional_float(
            record["tmin_qc_stage1"]
        )

        current_tmax = optional_float(
            record["tmax_qc_stage1"]
        )

        old_tmin = optional_float(
            record["tmin_old_xlsx"]
        )

        new_tmin = optional_float(
            record["tmin_new_csv"]
        )

        tmin_status = str(
            record["tmin_selection_status"]
        ).lower()

        tmax_status = str(
            record["tmax_selection_status"]
        ).lower()

        def plausible_tmin(
            candidate: float | None,
        ) -> bool:
            if candidate is None:
                return False

            if not (
                tmin_min
                <= candidate
                <= tmin_max
            ):
                return False

            # Zero values remain suspicious without
            # independent verification.
            if math.isclose(
                candidate,
                0.0,
                abs_tol=1e-9,
            ):
                return False

            if current_tmax is not None:
                dtr = current_tmax - candidate

                if dtr < minimum_dtr:
                    return False

                if dtr > high_dtr:
                    return False

            return True

        def preferred_alternate() -> (
            tuple[str, float] | None
        ):
            candidates = [
                ("new_csv", new_tmin),
                ("old_xlsx", old_tmin),
            ]

            for source, candidate in candidates:
                if not plausible_tmin(candidate):
                    continue

                if values_differ(
                    candidate,
                    current_tmin,
                ):
                    return source, float(candidate)

            return None

        decision = "keep"
        reason = (
            "Soft QC flag retained because there is no "
            "deterministic evidence that the official "
            "BMD value is incorrect."
        )

        corrected_tmin = ""
        corrected_tmax = ""

        alternate = preferred_alternate()

        tmin_physical = (
            "TMIN_PHYSICAL_RANGE" in flags
        )

        tmax_physical = (
            "TMAX_PHYSICAL_RANGE" in flags
        )

        tmax_below_tmin = (
            "TMAX_LT_TMIN" in flags
        )

        dtr_zero = "DTR_ZERO" in flags
        dtr_high_flag = "DTR_HIGH" in flags

        tmin_zero = (
            "TMIN_ZERO" in flags
        )

        source_conflict = (
            "SOURCE_NUMERIC_CONFLICT" in flags
            or "SOURCE_DUPLICATE_CONFLICT" in flags
        )

        old_duplicate_conflict = (
            "OLD_DUPLICATE_CONFLICT" in flags
        )

        if (
            tmin_physical
            and tmax_physical
        ):
            decision = "set_both_missing"
            reason = (
                "Both Tmin and Tmax failed broad physical "
                "screening; both were set missing."
            )

        elif tmax_below_tmin:
            if alternate is not None:
                source, _ = alternate

                decision = (
                    "select_new_tmin"
                    if source == "new_csv"
                    else "select_old_tmin"
                )

                reason = (
                    "Tmax was below selected Tmin. A plausible "
                    f"independent {source} Tmin resolves the "
                    "internal inconsistency."
                )

            else:
                decision = "set_both_missing"
                reason = (
                    "Tmax was below Tmin and no independent "
                    "plausible Tmin resolved the inconsistency. "
                    "Both values were set missing."
                )

        elif dtr_zero:
            if alternate is not None:
                source, _ = alternate

                decision = (
                    "select_new_tmin"
                    if source == "new_csv"
                    else "select_old_tmin"
                )

                reason = (
                    "Tmax equalled Tmin. A plausible independent "
                    f"{source} Tmin produces a positive diurnal "
                    "temperature range."
                )

            else:
                decision = "set_both_missing"
                reason = (
                    "Tmax equalled Tmin and no independent "
                    "plausible source resolved the record."
                )

        elif tmin_physical or tmin_zero:
            if alternate is not None:
                source, _ = alternate

                decision = (
                    "select_new_tmin"
                    if source == "new_csv"
                    else "select_old_tmin"
                )

                issue = (
                    "zero Tmin"
                    if tmin_zero
                    else "Tmin physical-range failure"
                )

                reason = (
                    f"The selected value had {issue}. "
                    f"A plausible independent {source} "
                    "value was selected."
                )

            else:
                decision = "set_tmin_missing"
                reason = (
                    "The selected Tmin was zero or failed "
                    "physical screening and no plausible "
                    "independent source was available."
                )

        elif tmax_physical:
            decision = "set_tmax_missing"
            reason = (
                "Tmax failed broad physical screening and "
                "no independent Tmax source is available."
            )

        elif old_duplicate_conflict:
            tmin_duplicate_conflict = (
                "conflict" in tmin_status
            )

            tmax_duplicate_conflict = (
                "conflict" in tmax_status
            )

            if (
                tmin_duplicate_conflict
                and tmax_duplicate_conflict
            ):
                decision = "set_both_missing"
                reason = (
                    "Both Tmin and Tmax have unresolved "
                    "conflicting duplicate source values."
                )

            elif tmax_duplicate_conflict:
                decision = "set_tmax_missing"
                reason = (
                    "Tmax has unresolved conflicting duplicate "
                    "source values and no independent Tmax source."
                )

            elif tmin_duplicate_conflict:
                if plausible_tmin(new_tmin):
                    decision = "select_new_tmin"
                    reason = (
                        "Old Tmin duplicate values conflict; "
                        "the independent updated BMD CSV Tmin "
                        "was selected."
                    )
                else:
                    decision = "set_tmin_missing"
                    reason = (
                        "Old Tmin duplicate values conflict and "
                        "no plausible independent Tmin exists."
                    )

            else:
                decision = "set_both_missing"
                reason = (
                    "A duplicate conflict was inherited but "
                    "the affected variable could not be "
                    "identified reliably."
                )

        elif source_conflict:
            if plausible_tmin(new_tmin):
                decision = "select_new_tmin"
                reason = (
                    "Official BMD sources disagree. The updated "
                    "2022-2025 BMD CSV value is plausible and "
                    "has documented source priority."
                )

            elif plausible_tmin(old_tmin):
                decision = "select_old_tmin"
                reason = (
                    "The updated CSV value was not plausible, "
                    "while the old official XLSX value passed "
                    "the deterministic checks."
                )

            else:
                decision = "set_tmin_missing"
                reason = (
                    "Official Tmin sources disagree and neither "
                    "candidate passed deterministic plausibility "
                    "checks."
                )

        elif dtr_high_flag:
            # High DTR alone is a soft flag. Removing it without
            # independent evidence could suppress real extremes.
            decision = "keep"
            reason = (
                "High diurnal temperature range is a soft "
                "screening flag only. The official observation "
                "was retained for later temporal and spatial QC."
            )

        else:
            decision = "keep"
            reason = (
                "No hard physical, duplicate, zero, or source "
                "selection rule required changing the official "
                "BMD observation."
            )

        value_ledger.at[
            index,
            "review_status",
        ] = "completed"

        value_ledger.at[
            index,
            "decision",
        ] = decision

        value_ledger.at[
            index,
            "corrected_tmin",
        ] = corrected_tmin

        value_ledger.at[
            index,
            "corrected_tmax",
        ] = corrected_tmax

        value_ledger.at[
            index,
            "decision_reason",
        ] = reason

        value_ledger.at[
            index,
            "verification_source",
        ] = VERIFICATION_SOURCE

        value_ledger.at[
            index,
            "reviewer",
        ] = REVIEWER

        value_ledger.at[
            index,
            "review_date",
        ] = REVIEW_DATE

        decision_counts[decision] = (
            decision_counts.get(
                decision,
                0,
            )
            + 1
        )

        automatic_value_log.append(
            {
                "review_id":
                    ledger_row["review_id"],
                "station_uid":
                    ledger_row["station_uid"],
                "date":
                    str(
                        ledger_row["date"]
                    )[:10],
                "qc_flags":
                    ";".join(
                        sorted(flags)
                    ),
                "decision": decision,
                "decision_reason": reason,
                "current_tmin": current_tmin,
                "old_tmin": old_tmin,
                "new_tmin": new_tmin,
                "current_tmax": current_tmax,
            }
        )

    value_ledger.to_csv(
        VALUE_LEDGER,
        index=False,
    )

    # ---------------------------------------------------------
    # Invalid-date decisions
    # ---------------------------------------------------------

    invalid_ledger = pd.read_csv(
        INVALID_DATE_LEDGER,
        dtype=str,
        keep_default_na=False,
    )

    if invalid_ledger["review_id"].duplicated().any():
        raise ValueError(
            "Duplicate review IDs in invalid-date ledger."
        )

    old = pd.read_parquet(
        OLD_FILE,
        columns=[
            "source_row",
            "station_uid",
            "date",
            "date_parse_status",
        ],
    )

    old["source_row"] = pd.to_numeric(
        old["source_row"],
        errors="raise",
    ).astype(int)

    old["date"] = pd.to_datetime(
        old["date"]
    )

    valid_old = old.loc[
        old["date_parse_status"].eq("valid")
        & old["station_uid"].ne("")
    ].copy()

    valid_by_station = {
        station_uid: group.sort_values(
            "source_row"
        ).reset_index(drop=True)
        for station_uid, group
        in valid_old.groupby(
            "station_uid"
        )
    }

    existing_keys = {
        (
            row.station_uid,
            row.date.strftime(
                "%Y-%m-%d"
            ),
        )
        for row in qc[
            [
                "station_uid",
                "date",
            ]
        ].itertuples(index=False)
    }

    invalid_counts = {
        "insert_corrected_date": 0,
        "reject_source_row": 0,
    }

    automatic_date_log: list[
        dict[str, Any]
    ] = []

    for index, ledger_row in (
        invalid_ledger.iterrows()
    ):
        station_uid = ledger_row[
            "station_uid"
        ]

        source_row = int(
            ledger_row["source_row"]
        )

        station_valid = (
            valid_by_station.get(
                station_uid
            )
        )

        candidate_date: pd.Timestamp | None = None
        previous_date: pd.Timestamp | None = None
        next_date: pd.Timestamp | None = None

        if (
            station_valid is not None
            and not station_valid.empty
        ):
            previous_rows = station_valid.loc[
                station_valid[
                    "source_row"
                ].lt(source_row)
            ]

            next_rows = station_valid.loc[
                station_valid[
                    "source_row"
                ].gt(source_row)
            ]

            if (
                not previous_rows.empty
                and not next_rows.empty
            ):
                previous_date = (
                    previous_rows.iloc[-1][
                        "date"
                    ]
                )

                next_date = (
                    next_rows.iloc[0][
                        "date"
                    ]
                )

                if (
                    next_date
                    - previous_date
                    == pd.Timedelta(days=2)
                ):
                    candidate_date = (
                        previous_date
                        + pd.Timedelta(days=1)
                    )

        if candidate_date is not None:
            candidate_text = (
                candidate_date.strftime(
                    "%Y-%m-%d"
                )
            )

            candidate_key = (
                station_uid,
                candidate_text,
            )

            if candidate_key not in existing_keys:
                decision = (
                    "insert_corrected_date"
                )

                corrected_date = (
                    candidate_text
                )

                reason = (
                    "The immediately preceding and following "
                    "valid source rows define exactly one "
                    "missing calendar date: "
                    f"{previous_date.date()} -> "
                    f"{candidate_text} -> "
                    f"{next_date.date()}."
                )

                existing_keys.add(
                    candidate_key
                )

            else:
                decision = (
                    "reject_source_row"
                )

                corrected_date = ""

                reason = (
                    "Source-row sequence suggested "
                    f"{candidate_text}, but that station-date "
                    "already exists. The invalid source row "
                    "was rejected to avoid duplication or "
                    "unsupported replacement."
                )

        else:
            decision = "reject_source_row"
            corrected_date = ""

            if (
                previous_date is not None
                and next_date is not None
            ):
                context = (
                    f"Previous valid date={previous_date.date()}, "
                    f"next valid date={next_date.date()}."
                )
            else:
                context = (
                    "Two valid bounding source dates were "
                    "not available."
                )

            reason = (
                "No unique date could be inferred from the "
                "official source-row sequence. "
                + context
            )

        invalid_ledger.at[
            index,
            "review_status",
        ] = "completed"

        invalid_ledger.at[
            index,
            "decision",
        ] = decision

        invalid_ledger.at[
            index,
            "corrected_date",
        ] = corrected_date

        invalid_ledger.at[
            index,
            "corrected_tmin",
        ] = ""

        invalid_ledger.at[
            index,
            "corrected_tmax",
        ] = ""

        invalid_ledger.at[
            index,
            "decision_reason",
        ] = reason

        invalid_ledger.at[
            index,
            "verification_source",
        ] = (
            "Deterministic source-row sequence in "
            "official BMD XLSX; automatic policy v1"
        )

        invalid_ledger.at[
            index,
            "reviewer",
        ] = REVIEWER

        invalid_ledger.at[
            index,
            "review_date",
        ] = REVIEW_DATE

        invalid_counts[decision] += 1

        automatic_date_log.append(
            {
                "review_id":
                    ledger_row["review_id"],
                "station_uid":
                    station_uid,
                "source_row":
                    source_row,
                "previous_valid_date":
                    (
                        previous_date.strftime(
                            "%Y-%m-%d"
                        )
                        if previous_date is not None
                        else ""
                    ),
                "next_valid_date":
                    (
                        next_date.strftime(
                            "%Y-%m-%d"
                        )
                        if next_date is not None
                        else ""
                    ),
                "decision":
                    decision,
                "corrected_date":
                    corrected_date,
                "decision_reason":
                    reason,
            }
        )

    invalid_ledger.to_csv(
        INVALID_DATE_LEDGER,
        index=False,
    )

    pd.DataFrame(
        automatic_value_log
    ).to_csv(
        REPORT_DIRECTORY
        / "step4b_automatic_value_decisions.csv",
        index=False,
    )

    pd.DataFrame(
        automatic_date_log
    ).to_csv(
        REPORT_DIRECTORY
        / "step4b_automatic_invalid_date_decisions.csv",
        index=False,
    )

    summary = {
        "review_date":
            REVIEW_DATE,
        "value_decision_rows":
            len(value_ledger),
        "invalid_date_rows":
            len(invalid_ledger),
        "value_decision_counts":
            decision_counts,
        "invalid_date_decision_counts":
            invalid_counts,
        "pending_value_decisions":
            int(
                value_ledger[
                    "review_status"
                ].str.lower().ne(
                    "completed"
                ).sum()
            ),
        "pending_invalid_date_decisions":
            int(
                invalid_ledger[
                    "review_status"
                ].str.lower().ne(
                    "completed"
                ).sum()
            ),
    }

    (
        REPORT_DIRECTORY
        / "step4b_automatic_decision_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "STEP 4B: AUTOMATIC DECISION COMPLETION",
        "=" * 43,
        (
            "Value decision rows completed: "
            f"{len(value_ledger):,}"
        ),
        (
            "Invalid-date rows completed: "
            f"{len(invalid_ledger):,}"
        ),
        "",
        "Value decisions:",
        *[
            f"- {key}: {value:,}"
            for key, value
            in sorted(
                decision_counts.items()
            )
        ],
        "",
        "Invalid-date decisions:",
        *[
            f"- {key}: {value:,}"
            for key, value
            in sorted(
                invalid_counts.items()
            )
        ],
        "",
        "No temporal interpolation was applied.",
    ]

    (
        REPORT_DIRECTORY
        / "step4b_automatic_decision_report.txt"
    ).write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(report_lines))

    if summary[
        "pending_value_decisions"
    ] != 0:
        raise SystemExit(
            "Automatic value decisions remain pending."
        )

    if summary[
        "pending_invalid_date_decisions"
    ] != 0:
        raise SystemExit(
            "Automatic invalid-date decisions remain pending."
        )

    print(
        "\nSTEP 4B AUTOMATIC DECISIONS PASSED."
    )


if __name__ == "__main__":
    main()
