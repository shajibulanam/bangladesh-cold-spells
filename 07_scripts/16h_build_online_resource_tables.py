#!/usr/bin/env python3
"""
16h_build_online_resource_tables.py  (v4, Step 19)
==================================================

Builds the new material of Online Resource 1 (ESM_1) from the Step 17
revision outputs. Nothing is recomputed: every number is read from the CSV
files written by 16a-16g, so the supplement and the main text cannot drift.

Outputs (08_outputs/online_resource_1/):
  online_resource_1_tables_S1_S9.md      Tables S1-S9 with captions (v3: from result tables 60-119)
  online_resource_1_tables_S10_S17.md    Tables S10-S17 with captions
  FigS1_hazard_robustness.{pdf,png,tif}  (a) odds-ratio robustness,
                                         (b) nested validation
  FigS2_gridded_leakage_free.{pdf,png,tif}  copy of the Step 17D figure
  online_resource_1_build_report.txt     sources, unlabelled names,
                                         rounding ties, consistency checks

Usage (project root):
    python 07_scripts/16h_build_online_resource_tables.py --inspect
    python 07_scripts/16h_build_online_resource_tables.py
"""
from __future__ import annotations

import argparse
from decimal import ROUND_HALF_UP, Decimal
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MINUS = "−"
DASH = "—"
NDASH = "–"

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

def find_root(cli_root):
    if cli_root:
        return Path(cli_root).resolve()
    here = Path(__file__).resolve().parent
    for cand in (here.parent, Path.cwd(), Path.cwd().parent):
        if (cand / "08_outputs").is_dir():
            return cand
    sys.exit("Cannot find the project root (a folder containing 08_outputs). Pass --root.")


def locate(root: Path, name: str):
    pref = root / "08_outputs" / "tables" / "revision" / name
    if pref.exists():
        return pref
    hits = sorted((root / "08_outputs").rglob(name)) or sorted((root / "02_metadata").glob(name))
    return hits[0] if hits else None


# --------------------------------------------------------------------------
# Report bookkeeping
# --------------------------------------------------------------------------

class Report:
    def __init__(self):
        self.lines, self.unknown, self.ties, self.checks = [], {}, [], []
        self.ctx = ""

    def add(self, s=""):
        self.lines.append(s)

    def flag(self, raw, lab):
        self.unknown.setdefault(self.ctx, set()).add(f"{raw!r} -> {lab!r}")


REP = Report()


# --------------------------------------------------------------------------
# Number formatting
# --------------------------------------------------------------------------

def _isnan(x):
    return x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x)


def num(x, d=2):
    """Format with d decimals and a true minus sign; record rounding ties
    (a value stored with d+1 decimals ending in 5 cannot be rounded to d
    decimals without the full-precision value)."""
    if _isnan(x):
        return DASH
    x = float(x)
    scaled = abs(x) * 10 ** (d + 1)
    if abs(scaled - round(scaled)) < 1e-6 and int(round(scaled)) % 10 == 5:
        REP.ties.append(f"{REP.ctx}: {x!r} shown with {d} decimals")
    # round half up on the stored decimal value (Python's binary rounding
    # would turn a stored 1.555 into 1.55 but 0.0495 into 0.050)
    s = str(Decimal(repr(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", MINUS)


def pval(x):
    if _isnan(x):
        return DASH
    x = float(x)
    return "< 0.001" if x < 0.001 else f"{x:.3f}"


def ci(lo, hi, d=2, ratio=False):
    if _isnan(lo) or _isnan(hi):
        return DASH
    if ratio:
        return f"{num(lo, d)}{NDASH}{num(hi, d)}"
    return f"[{num(lo, d)}, {num(hi, d)}]"


def integer(x):
    return DASH if _isnan(x) else str(int(round(float(x))))


def excludes(lo, hi, ref):
    return (not _isnan(lo)) and (not _isnan(hi)) and (lo > ref or hi < ref)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/") for c in r) + " |")
    return "\n".join(out)


def need(df, cols, name):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"{name}: missing columns {miss}; available {list(df.columns)}")


# --------------------------------------------------------------------------
# Labels (same wording as Tables S2-S4)
# --------------------------------------------------------------------------

EXACT = {
    "nat_tmax_anom": "Tmax anomaly", "nat_dtr_anom": "DTR anomaly",
    "nat_tmin_anom": "Tmin anomaly",
    "p10_cold_national_area_fraction": "Cold-area fraction",
    "siberian_high_std_anom": "Siberian High", "ural_blocking_fraction": "Ural blocking",
    "nam_proxy_10": "NAM 10 hPa", "nam_proxy_100": "NAM 100 hPa",
    "cold_spell_days": "Cold-night spell days per winter",
    "nigp_adv_lead1": "Adv850, NW IGP (day k − 1)",
    "night_rh_ge95": "Nights with box-mean RH ≥ 95% (fraction)",
}
RRWP = {"eurasian_upstream": "Eurasian upstream", "central_asia": "Central Asia",
        "bangladesh_sector": "Bangladesh sector", "asian_corridor": "Asian corridor"}
ERA5 = {"t850": "T850", "t2m": "T2m", "stab_t2m_minus_t850": "T2m − T850",
        "z500": "Z500", "mslp": "MSLP", "msl": "MSLP", "u850": "u850", "v850": "v850",
        "wspd10": "10-m wind"}
WIN = {"strd": "Downward longwave", "ssrd": "Downward solar", "tcc": "Total cloud",
       "td2m": "2-m dewpoint", "low1000": "Cloud base < 1 km", "fog300": "Cloud base < 300 m",
       "blh": "Boundary-layer height"}
DLY = {"tcc": "Total cloud", "lowcloud1000_frac": "Cloud base < 1 km",
       "fog300_frac": "Cloud base < 300 m", "dpd": "Dewpoint depression"}
SS = {"skt": "Skin temperature", "skt_minus_t2m": "Skin − 2-m air temperature"}
TIME = {"day": "06–17 LT", "night": "18–05 LT", "00utc": "06 LT"}
# hourly accumulations are labelled by the end of their hour: timestamps 00-11 UTC cover 23-11 UTC
FLUX_TIME = {"day": "05–17 LT", "night": "17–05 LT"}


def var_label(raw) -> str:
    """Manuscript label of an internal predictor name; unknown names are
    flagged in the report and returned unchanged."""
    s = str(raw).strip()
    if ";" in s:
        return "; ".join(var_label(x) for x in s.split(";") if x.strip())
    if s in EXACT:
        return EXACT[s]
    m = re.fullmatch(r"rrwp_([a-z_]+?)_(R_lag7|envelope_z_trailing3)", s)
    if m and m.group(1) in RRWP:
        kind = "R (t − 7)" if m.group(2) == "R_lag7" else "envelope (t − 2 … t)"
        return f"RRWP {RRWP[m.group(1)]} {kind}"
    m = re.fullmatch(r"era5_(bd|sa)_(.+)_(anom|tend)", s)
    if m:
        v = m.group(2)
        lab = "1000–500 hPa thickness" if v.startswith("thick") else ERA5.get(v)
        if lab:
            d = "Δ" if m.group(3) == "tend" else ""
            return f"{d}{m.group(1).upper()} {lab}"
    m = re.fullmatch(r"mech_(bd|nigp)_adv850_k_day_(anom|tend)", s)
    if m:
        d = "Δ" if m.group(2) == "tend" else ""
        return f"{d}Adv850, {'Bangladesh' if m.group(1) == 'bd' else 'NW IGP'}"
    m = re.fullmatch(r"mech_jet200_core_(lat|speed)_70_100e_(anom|tend)", s)
    if m:
        d = "Δ" if m.group(2) == "tend" else ""
        return f"{d}Jet {'latitude' if m.group(1) == 'lat' else 'speed'}"
    m = re.fullmatch(r"win_([a-z0-9]+)_(day|night)_(anom|tend)", s)
    if m and m.group(1) in WIN:
        d = "Δ" if m.group(3) == "tend" else ""
        tm = FLUX_TIME[m.group(2)] if m.group(1) in ("strd", "ssrd") else TIME[m.group(2)]
        return f"{d}{WIN[m.group(1)]}, {tm}"
    m = re.fullmatch(r"dly_([a-z0-9_]+?)_00utc_(anom|tend)", s)
    if m and m.group(1) in DLY:
        d = "Δ" if m.group(2) == "tend" else ""
        return f"{d}{DLY[m.group(1)]}, 06 LT"
    m = re.fullmatch(r"ss_(skt_minus_t2m|skt)_(00utc|day|night)_(anom|tend)", s)
    if m:
        d = "Δ" if m.group(3) == "tend" else ""
        return f"{d}{SS[m.group(1)]}, {TIME[m.group(2)]}"
    REP.flag(s, s)
    return s


def metric_label(raw) -> str:
    """Pre-onset stratospheric metrics (Table S14)."""
    s = str(raw).lower()
    if "ssw" in s:
        return "SSW within the 45 days before onset (0/1)"
    if "displacement" in s:
        return "Mean vortex displacement, 45 days before onset"
    if "weak_vortex" in s:
        return "Fraction of days with weak vortex (NAM 10 hPa ≤ −1 SD), 20 days before onset"
    if "neg" in s:
        return "Fraction of days with negative NAM at both 10 and 100 hPa, 20 days before onset"
    m = re.search(r"pre(\d+)_.*?nam_?(100|10)(?!\d)", s)
    if m:
        return f"Mean NAM {m.group(2)} hPa, {m.group(1)} days before onset"
    REP.flag(raw, raw)
    return str(raw)


def cap(s: str) -> str:
    s = str(s)
    return s[:1].upper() + s[1:] if s else s


def scheme_label(raw) -> str:
    s = str(raw).lower()
    if "leave-one-winter-out" in s or s in {"lowo", "leave_one_winter_out", "nested_lowo"}:
        return "Leave-one-winter-out"
    if re.search(r"train 1985/86", s) or s == "split_train_early":
        return "Train 1985/86–2004/05, test 2005/06–2024/25"
    if re.search(r"train 2005/06", s) or s == "split_train_late":
        return "Train 2005/06–2024/25, test 1985/86–2004/05"
    REP.flag(raw, raw)
    return str(raw)


FAMILY = {"primary": "Primary", "advection_jet": "Advection and jet",
          "adv_jet": "Advection and jet", "prenight": "Before terminating night",
          "pre_night": "Before terminating night", "night": "Terminating night (concurrent)"}

FIELD = {"z500": ("Z500", "m"), "u200": ("U200", "m s⁻¹"), "t850": ("T850", "K"),
         "msl": ("MSLP", "hPa"), "mslp": ("MSLP", "hPa"), "t2m": ("T2m", "K")}


def field_label(raw, units=True):
    k = str(raw).lower()
    if k in FIELD:
        lab, u = FIELD[k]
        return f"{lab} ({u})" if units else lab
    REP.flag(raw, raw)
    return str(raw)


def window_label(raw) -> str:
    s = str(raw).strip().replace("-", MINUS)
    s = re.sub(r"(\d) to (\d)", rf"\1{NDASH}\2", s) if "−" not in s.split("to")[0] else s
    return cap(s)


# --------------------------------------------------------------------------
# Table builders. Each returns a list of (subtitle or None, markdown table).
# --------------------------------------------------------------------------

def t_s10(D):
    r1, r1b = D["table_R1_nested_validation.csv"], D["table_R1b_nested_selection.csv"]
    need(r1, ["scheme", "test_rows", "test_terminations", "auc_adjustment_only",
              "auc_nested_model", "auc_difference", "auc_diff_ci_low", "auc_diff_ci_high",
              "brier_adjustment_only", "brier_nested_model", "brier_training_climatology",
              "brier_diff_full_minus_adj", "brier_diff_ci_low", "brier_diff_ci_high",
              "brier_skill_vs_training_climatology", "calibration_slope",
              "observed_over_expected", "selected_predictors"], "table_R1")
    # transposed (v4): one column per validation scheme, so the table fits a portrait page
    metrics = [
        ("Test rows (terminations)", lambda r: f"{integer(r.test_rows)} ({integer(r.test_terminations)})"),
        ("AUC, adjustment only", lambda r: num(r.auc_adjustment_only, 3)),
        ("AUC, + selected predictors", lambda r: num(r.auc_nested_model, 3)),
        ("ΔAUC [95% CI]", lambda r: f"{num(r.auc_difference, 3)} {ci(r.auc_diff_ci_low, r.auc_diff_ci_high, 3)}"),
        ("Brier score, adjustment only", lambda r: num(r.brier_adjustment_only, 3)),
        ("Brier score, + selected predictors", lambda r: num(r.brier_nested_model, 3)),
        ("Brier score, training climatology", lambda r: num(r.brier_training_climatology, 3)),
        ("ΔBrier [95% CI]", lambda r: f"{num(r.brier_diff_full_minus_adj, 3)} "
                                      f"{ci(r.brier_diff_ci_low, r.brier_diff_ci_high, 3)}"),
        ("Brier skill", lambda r: num(r.brier_skill_vs_training_climatology, 2)),
        ("Calibration slope", lambda r: num(r.calibration_slope, 2)),
        ("Observed/expected terminations", lambda r: num(r.observed_over_expected, 2)),
    ]
    heads = [scheme_label(sch) for sch in r1.scheme]
    rows = [[lab] + [f(r) for _, r in r1.iterrows()] for lab, f in metrics]
    a = md_table(["Score"] + heads, rows)

    need(r1b, ["scheme", "predictor", "folds_selected"], "table_R1b")
    sc = r1b["scheme"].astype(str)
    freq = r1b[sc == "selection_frequency"]
    n_lowo = int(((~sc.str.startswith("split")) & (sc != "selection_frequency")).sum())
    split = {}
    for _, r in r1.iterrows():
        lab = scheme_label(r.scheme)
        if lab.startswith("Train") and isinstance(r.selected_predictors, str):
            split[lab] = set(x.strip() for x in r.selected_predictors.split(";"))
    preds = list(freq["predictor"].astype(str))
    for s in split.values():
        preds += [p for p in sorted(s) if p not in preds]
    early = "Train 1985/86–2004/05, test 2005/06–2024/25"
    late = "Train 2005/06–2024/25, test 1985/86–2004/05"
    rows = []
    fdict = dict(zip(freq["predictor"].astype(str), freq["folds_selected"]))
    for p in preds:
        rows.append([var_label(p), integer(fdict.get(p, 0)),
                     "yes" if p in split.get(early, set()) else DASH,
                     "yes" if p in split.get(late, set()) else DASH])
    b = md_table(["Predictor", f"Leave-one-winter-out folds selected (of {n_lowo})",
                  "Selected, trained on 1985/86–2004/05",
                  "Selected, trained on 2005/06–2024/25"], rows)
    REP.checks.append(f"S10: {n_lowo} leave-one-winter-out folds")
    return [("(a) Out-of-sample scores", a), ("(b) Predictors selected in the training folds", b)]


def _or_cell(r):
    star = "\\*" if excludes(r.ci_low, r.ci_high, 1) else ""
    return f"{num(r.odds_ratio_per_sd)} ({ci(r.ci_low, r.ci_high, ratio=True)}){star}"


def t_s11(D):
    r10, r10b, r2 = (D["table_R10_stable_network_catalogue.csv"],
                     D["table_R10b_stable_network_hazard.csv"],
                     D["table_R2_censoring_sensitivity.csv"])
    need(r10, ["catalogue", "spells", "spell_days", "short_3d", "intermediate_4_5d",
               "persistent_ge6d", "winters_with_spells"], "table_R10")
    rows = [[cap(r.catalogue), integer(r.spells), integer(r.spell_days), integer(r.short_3d),
             integer(r.intermediate_4_5d), integer(r.persistent_ge6d),
             integer(r.winters_with_spells)] for _, r in r10.iterrows()]
    a = md_table(["Catalogue", "Spells", "Spell days", "Short (3 d)", "Intermediate (4–5 d)",
                  "Persistent (≥ 6 d)", "Winters with spells"], rows)

    need(r2, ["analysis", "predictor", "odds_ratio_per_sd", "ci_low", "ci_high",
              "n_rows", "n_terminations"], "table_R2")
    need(r10b, ["predictor", "odds_ratio_per_sd", "ci_low", "ci_high", "n_rows",
                "n_terminations"], "table_R10b")
    an = list(dict.fromkeys(r2["analysis"].astype(str)))
    prim = next(a_ for a_ in an if a_.lower().startswith("primary"))
    cens = next(a_ for a_ in an if a_.startswith("A"))
    trunc = next(a_ for a_ in an if a_.startswith("B"))
    cols = [("Primary (26 stations)", r2[r2.analysis == prim]),
            ("25 stations", r10b),
            ("Season-end spells not censored", r2[r2.analysis == cens]),
            ("1 December starts removed", r2[r2.analysis == trunc])]
    r18 = D.get("table_R18_season_end_final_row_dropped.csv")
    if r18 is not None:   # Step 20: final rows of the two censored spells omitted instead of coded 0
        cols.append(("Final censored rows omitted",
                     r18[r18.analysis.astype(str).str.startswith("same day")]))
    idx = [c[1].set_index("predictor") for c in cols]
    rows = []
    for p in idx[0].index:
        rows.append([var_label(p)] + [_or_cell(t.loc[p]) if p in t.index else DASH for t in idx])
    rows.append(["Risk-set rows (terminations)"] +
                [f"{integer(t.n_rows.iloc[0])} ({integer(t.n_terminations.iloc[0])})" for t in idx])
    b = md_table(["Predictor"] + [c[0] for c in cols], rows)
    return [("(a) Cold-night spell catalogues", a),
            ("(b) Odds ratios of termination per SD (95% CI)", b)]


def t_s12(D):
    r3, r11 = D["table_R3_one_day_lead_all_families.csv"], D["table_R11_lead_advection_robustness.csv"]
    need(r3, ["family", "predictor", "odds_ratio_per_sd", "ci_low", "ci_high", "p_bootstrap",
              "q_fdr_family", "q_fdr_all"], "table_R3")
    rows = []
    for _, r in r3.iterrows():
        fam = FAMILY.get(str(r.family).lower())
        if fam is None:
            REP.flag(r.family, r.family)
            fam = cap(r.family)
        rows.append([fam, var_label(r.predictor), num(r.odds_ratio_per_sd),
                     ci(r.ci_low, r.ci_high, ratio=True), pval(r.p_bootstrap),
                     pval(r.q_fdr_family), pval(r.q_fdr_all)])
    a = md_table(["Family", "Predictor (day k − 1)", "OR per SD", "95% CI", "p",
                  "q (within family)", f"q (all {len(r3)})"], rows)
    REP.checks.append(f"S12: {len(r3)} one-day-lead predictors")

    need(r11, ["analysis", "predictor", "odds_ratio_per_sd", "ci_low", "ci_high",
               "p_bootstrap", "n_rows", "n_terminations"], "table_R11")
    rows = []
    known = {}  # rows without a termination count reuse the primary (censored) risk set of the same size
    for _, r in r11.iterrows():
        if not _isnan(r.n_rows) and not _isnan(r.n_terminations):
            known.setdefault(int(r.n_rows), int(r.n_terminations))
    for _, r in r11.iterrows():
        s = str(r.analysis)
        s = re.sub(r"^lead (\d) day\(s\)$", lambda m: f"predictor lag {m.group(1)} day" + ("s" if m.group(1) != "1" else ""), s)
        s = re.sub(r"^lead 1\b", "predictor lag 1 day", s)
        nt = r.n_terminations if not _isnan(r.n_terminations) else known.get(int(r.n_rows), np.nan)
        rs = integer(r.n_rows) + (f" ({integer(nt)})" if not _isnan(nt) else "")
        rows.append([cap(s), var_label(r.predictor), num(r.odds_ratio_per_sd),
                     ci(r.ci_low, r.ci_high, ratio=True), pval(r.p_bootstrap), rs])
    r18 = D.get("table_R18_season_end_final_row_dropped.csv")
    if r18 is not None:
        for _, r in r18[r18.predictor == "mech_nigp_adv850_k_day_anom"].iterrows():
            lag = "1 day" if str(r.analysis).startswith("predictor lag 1") else "2 days"
            rows.append([f"Predictor lag {lag}, final censored rows omitted", var_label(r.predictor),
                         num(r.odds_ratio_per_sd), ci(r.ci_low, r.ci_high, ratio=True), pval(r.p_bootstrap),
                         f"{integer(r.n_rows)} ({integer(r.n_terminations)})"])
    b = md_table(["Analysis", "Predictor", "OR per SD", "95% CI", "p",
                  "Risk-set rows (terminations)"], rows)
    out = [("(a) All predictors one day earlier", a),
           ("(b) 850-hPa advection and jet speed one and two days earlier", b)]
    r15 = D.get("table_R15_upstream_advection_alternative_catalogues.csv")
    if r15 is not None:
        need(r15, ["catalogue", "spells", "predictor_lag_days", "n_rows", "n_terminations",
                   "odds_ratio_per_sd", "ci_low", "ci_high", "p_bootstrap"], "table_R15")
        rows = []
        for lag in (1, 2):
            for _, r in r15[r15.predictor_lag_days == lag].iterrows():
                rows.append([CAT_LABEL.get(str(r.catalogue), cap(str(r.catalogue))), integer(r.spells),
                             f"{lag} day" + ("s" if lag > 1 else ""), num(r.odds_ratio_per_sd),
                             ci(r.ci_low, r.ci_high, ratio=True), pval(r.p_bootstrap),
                             f"{integer(r.n_rows)} ({integer(r.n_terminations)})"])
        REP.checks.append("S12c: upstream OR (lag 1) by catalogue: " + ", ".join(
            f"{r.catalogue} {r.odds_ratio_per_sd:.2f}" for _, r in r15[r15.predictor_lag_days == 1].iterrows()))
        out.append(("(c) Adv850, NW IGP, one and two days earlier, in the alternative event catalogues",
                    md_table(["Catalogue", "Spells", "Predictor lag", "OR per SD", "95% CI", "p",
                              "Risk-set rows (terminations)"], rows)))
    return out


CAT_LABEL = {"primary": "Primary (p10, area ≥ 20%, ≥ 3 days)",
             "p10_area20_duration3_gap1": "One-day gap merge",
             "p10_area10_duration3": "Area ≥ 10%", "p10_area30_duration3": "Area ≥ 30%",
             "p10_area20_duration4": "Minimum duration 4 days", "p05_area20_duration3": "p05 threshold",
             "bmd_area20_duration3": "BMD absolute 10 °C threshold"}


def null_label(raw):
    s = str(raw)
    if s.startswith("calendar"):
        return "Calendar-matched null"
    if s.startswith("decade"):
        return "Decade-matched null (±5 winters)"
    if "all other" in s:
        return "Onset rate vs all other at-risk DJF days"
    m = re.search(r"within-winter \((\d+) SSW winters\)", s)
    if m:
        return f"Onset rate vs rest of the same winters ({m.group(1)} SSW winters)"
    REP.flag(raw, raw)
    return cap(s)


def t_s13(D):
    r4 = D["table_R4_ssw_robustness.csv"]
    need(r4, ["window_days", "null", "observed_onsets", "null_mean", "null_2.5", "null_97.5",
              "rate_ratio_vs_null", "p_two_sided", "observed_at_risk_days",
              "exposure_adjusted_ratio", "rr_ci_low", "rr_ci_high",
              "onsets_rest_of_same_winters", "at_risk_days_rest_of_same_winters"], "table_R4")
    cnt, rate = [], []
    for _, r in r4.iterrows():
        w = str(r.window_days).replace("-", NDASH) + " d"
        rest = (f"{integer(r.onsets_rest_of_same_winters)} / "
                f"{integer(r.at_risk_days_rest_of_same_winters)}"
                if not _isnan(r.onsets_rest_of_same_winters) else DASH)
        if not _isnan(r.null_mean):
            cnt.append([w, null_label(r.null), integer(r.observed_onsets), num(r.null_mean, 1),
                        f"{integer(r['null_2.5'])}{NDASH}{integer(r['null_97.5'])}",
                        num(r.rate_ratio_vs_null), pval(r.p_two_sided), num(r.exposure_adjusted_ratio)])
        else:
            rate.append([w, null_label(r.null), integer(r.observed_onsets), integer(r.observed_at_risk_days),
                         num(r.rate_ratio_vs_null), ci(r.rr_ci_low, r.rr_ci_high, ratio=True), rest])
    a = md_table(["Window after SSW", "Null", "Observed onsets", "Expected onsets",
                  "Null 2.5–97.5% range", "Count ratio", "p (two-sided)", "Ratio per at-risk day"], cnt)
    b = md_table(["Window after SSW", "Comparison", "Onsets after SSWs", "At-risk days after SSWs",
                  "Rate ratio", "95% CI", "Onsets / at-risk days, rest of same winters"], rate)
    return [("(a) Count comparisons with Monte Carlo nulls", a), ("(b) Onset-rate comparisons", b)]


def t_s14(D):
    r8, r8b = D["table_R8_stratosphere_duration.csv"], D["table_R8b_stratosphere_duration_extra.csv"]
    need(r8, ["metric", "events_with_nonzero", "spearman_rho", "rho_ci_low", "rho_ci_high",
              "q_fdr", "partial_rho_season_year", "partial_ci_low", "partial_ci_high",
              "partial_q_fdr", "rho_metric_vs_onset_day"], "table_R8")
    rows = [[metric_label(r.metric), integer(r.events_with_nonzero), num(r.spearman_rho),
             ci(r.rho_ci_low, r.rho_ci_high), pval(r.q_fdr), num(r.partial_rho_season_year),
             ci(r.partial_ci_low, r.partial_ci_high), pval(r.partial_q_fdr),
             num(r.rho_metric_vs_onset_day)] for _, r in r8.iterrows()]
    a = md_table(["Pre-onset metric", "Spells with non-zero value", "Spearman ρ", "95% CI",
                  "q", "Partial ρ", "95% CI", "q (partial)", "ρ with onset day"], rows)
    for _, r in r8.iterrows():
        if not _isnan(r.partial_q_fdr) and r.partial_q_fdr < 0.05:
            REP.checks.append(f"S14: partial q < 0.05 for {r.metric} "
                              f"(partial rho {r.partial_rho_season_year:.3f}, q {r.partial_q_fdr:.4f}, "
                              f"spells non-zero {r.events_with_nonzero})")
    need(r8b, ["metric", "difference", "ci_low", "ci_high", "p_bootstrap", "n_a", "n_b",
               "odds_ratio_per_sd", "n_rows", "n_terminations"], "table_R8b")
    rows = []
    for _, r in r8b.iterrows():
        s = str(r.metric)
        if "persistent minus short" in s:
            rows.append(["Weak-vortex fraction, 20 days before onset: persistent minus short",
                         num(r.difference, 3), ci(r.ci_low, r.ci_high, 3), pval(r.p_bootstrap),
                         f"{integer(r.n_a)} persistent, {integer(r.n_b)} short spells"])
        elif s.startswith("hazard"):
            what = ("Weak-vortex fraction" if "weak_vortex" in s else
                    "Mean NAM 10 hPa" if "nam10" in s else s)
            rows.append([f"{what}, trailing 20 days: odds ratio of termination per SD",
                         num(r.odds_ratio_per_sd), ci(r.ci_low, r.ci_high, ratio=True),
                         pval(r.p_bootstrap),
                         f"{integer(r.n_rows)} rows, {integer(r.n_terminations)} terminations"])
        else:
            REP.flag(s, s)
            rows.append([s, num(r.difference), ci(r.ci_low, r.ci_high), pval(r.p_bootstrap), DASH])
    b = md_table(["Test", "Estimate", "95% CI", "p", "Sample"], rows)
    return [("(a) Pre-onset metrics and duration (85 spells)", a),
            ("(b) Further tests of the weak-vortex fraction", b)]


def stat_label(raw):
    s = str(raw)
    if "no_enso" in s:
        return "Regression on NAM 100 hPa (per SD)"
    if "enso_removed" in s:
        return "Regression on NAM 100 hPa, ONI as second predictor (per SD)"
    if s.startswith("post-SSW"):
        return "Post-SSW (0–45 d) minus rest of winter"
    REP.flag(raw, raw)
    return cap(s)


IDX_UNITS = {"Jet speed": "m s⁻¹", "Jet latitude": "°", "SA Z500": "m",
             "Adv850, NW IGP": "K day⁻¹", "BD v850": "m s⁻¹",
             "Tmin anomaly": "°C", "Cold-night spell days per winter": "days"}


def t_s15(D):
    r6b, r6 = D["table_R6b_enso_controlled_pathway.csv"], D["table_R6_pathway_box_indices.csv"]
    need(r6b, ["field", "statistic", "south_asia_box", "bangladesh_box", "fraction_cells_q05"],
         "table_R6b")
    order = {"no_enso": 0, "enso_removed": 1, "post": 2}
    r6b = r6b.assign(_o=r6b.statistic.astype(str).map(
        lambda s: next((v for k, v in order.items() if k in s), 9)))
    fo = list(dict.fromkeys(r6b.field.astype(str)))
    r6b = r6b.assign(_f=r6b.field.astype(str).map(fo.index)).sort_values(["_f", "_o"])
    rows = [[field_label(r.field), stat_label(r.statistic), num(r.south_asia_box),
             num(r.bangladesh_box), num(r.fraction_cells_q05)] for _, r in r6b.iterrows()]
    a = md_table(["Field", "Statistic", "South Asia box", "Bangladesh box",
                  "Fraction of grid points with q < 0.05"], rows)

    need(r6, ["analysis", "index", "spearman_rho", "p_value", "partial_rho_given_oni",
              "mean_difference", "ci_low", "ci_high"], "table_R6")
    corr = r6[r6.analysis.astype(str).str.startswith("winter")].set_index("index")
    post = r6[r6.analysis.astype(str).str.startswith("post")].set_index("index")
    rows = []
    for ix in dict.fromkeys(list(corr.index) + list(post.index)):
        lab = var_label(ix)
        u = IDX_UNITS.get(lab, "")
        c = corr.loc[ix] if ix in corr.index else None
        p_ = post.loc[ix] if ix in post.index else None
        sf = (p_["p_sign_flip_exact"] if (p_ is not None and "p_sign_flip_exact" in p_.index) else np.nan)
        rows.append([f"{lab} ({u})" if u else lab,
                     num(c.spearman_rho) if c is not None else DASH,
                     pval(c.p_value) if c is not None else DASH,
                     num(c.partial_rho_given_oni) if c is not None else DASH,
                     num(p_.mean_difference) if p_ is not None else DASH,
                     ci(p_.ci_low, p_.ci_high) if p_ is not None else DASH,
                     pval(sf) if not _isnan(sf) else DASH])
    b = md_table(["Index (units of the difference)", "ρ with winter-mean NAM 100 hPa", "p",
                  "Partial ρ given ONI", "Post-SSW (0–45 d) minus rest of winter",
                  "95% CI", "Sign-flip p (exact)"], rows)
    return [("(a) Box means of the gridded regressions and composites", a),
            ("(b) Onset-pathway indices", b)]


R5VAR = {"t2m": "T2m (°C)", "td2m": "2-m dewpoint (°C)",
         "dpd": "Dewpoint depression (K)", "rh": "Relative humidity (%)"}


def t_s16(D):
    r5, r9c, r9 = (D["table_R5_absolute_humidity.csv"], D["table_R9c_skin_soil_regimes.csv"],
                   D["table_R9_skin_soil_hazard.csv"])
    need(r5, ["variable", "pure_cold_day_mean", "pure_cold_night_mean", "djf_climatology",
              "difference", "ci_low", "ci_high", "p_bootstrap", "n_pure_cold_day",
              "n_pure_cold_night"], "table_R5")
    rows = []
    for _, r in r5.iterrows():
        s = str(r.variable)
        m = re.fullmatch(r"(t2m|td2m|dpd|rh)_(day|night)", s)
        if m:
            lab, win = R5VAR[m.group(1)], ("Night (18–05 LT)" if m.group(2) == "night"
                                          else "Day (06–17 LT)")
            d = 1 if m.group(1) == "rh" else 2
        elif s in EXACT:
            lab, win, d = EXACT[s], "Night (18–05 LT)", 2
        else:
            REP.flag(s, s)
            lab, win, d = s, DASH, 2
        rows.append([lab, win, num(r.pure_cold_day_mean, d), num(r.pure_cold_night_mean, d),
                     num(r.djf_climatology, d), num(r.difference, d),
                     ci(r.ci_low, r.ci_high, d), pval(r.p_bootstrap)])
    n_day, n_night = integer(r5.n_pure_cold_day.iloc[0]), integer(r5.n_pure_cold_night.iloc[0])
    a = md_table(["Variable", "Window", f"Pure cold-day spells (n = {n_day})",
                  f"Pure cold-night spells (n = {n_night})", "DJF climatology",
                  "Difference (cold-day minus cold-night)", "95% CI", "p"], rows)

    need(r9c, ["variable", "difference", "ci_low", "ci_high", "p_bootstrap"], "table_R9c")
    rows = [[var_label(r.variable) + " (K)", num(r.difference), ci(r.ci_low, r.ci_high),
             pval(r.p_bootstrap)] for _, r in r9c.iterrows()]
    b = md_table(["Anomaly", "Difference (cold-day minus cold-night)", "95% CI", "p"], rows)

    need(r9, ["predictor", "window", "odds_ratio_per_sd", "ci_low", "ci_high", "p_bootstrap",
              "q_fdr_window"], "table_R9")
    rows = [[cap(r.window), var_label(r.predictor), num(r.odds_ratio_per_sd),
             ci(r.ci_low, r.ci_high, ratio=True), pval(r.p_bootstrap), pval(r.q_fdr_window)]
            for _, r in r9.iterrows()]
    c = md_table(["Window", "Predictor", "OR per SD", "95% CI", "p", "q (within window)"], rows)
    return [("(a) Humidity of pure cold-day and cold-night spells", a),
            ("(b) Skin temperature, cold-day minus cold-night spells", b),
            ("(c) Skin temperature in the termination hazard model", c)]


def t_s17(D):
    r7 = D["table_R7_gridded_leakage_free.csv"]
    need(r7, ["window", "field", "cells_fdr05", "cells_total", "fraction_fdr05", "min_p",
              "upstream_box_difference", "upstream_box_ci_low", "upstream_box_ci_high"], "table_R7")
    rows = [[window_label(r.window), field_label(r.field), integer(r.cells_fdr05),
             integer(r.cells_total), num(r.fraction_fdr05, 3), pval(r.min_p),
             num(r.upstream_box_difference), ci(r.upstream_box_ci_low, r.upstream_box_ci_high)]
            for _, r in r7.iterrows()]
    REP.checks.append(f"S17: max fraction of cells with q < 0.05 = {r7.fraction_fdr05.max():.4f}")
    t = md_table(["Window", "Field (units)", "Grid points with q < 0.05", "Grid points",
                  "Fraction", "Smallest p", "Upstream box difference", "95% CI"], rows)
    return [(None, t)]


UNIT = {"m s-1": "m s⁻¹", "K day-1": "K day⁻¹", "m day-1": "m day⁻¹", "degC": "°C"}


def sd_sentence(D, which):
    """Standard deviations in physical units (table R17, Step 19) for the per-SD odds ratios."""
    r17 = D.get("table_R17_predictor_sd_original_units.csv")
    if r17 is None:
        return ""
    r17 = r17.set_index("predictor")
    parts = [f"{lab} {num(r17.loc[p, 'sd'], 2)} {UNIT.get(str(r17.loc[p, 'unit']), r17.loc[p, 'unit'])}"
             for p, lab in which if p in r17.index and not _isnan(r17.loc[p, "sd"])]
    return (" One standard deviation corresponds to: " + "; ".join(parts) + ".") if parts else ""


def t_s18(D):
    a = D["step6c_analytical_station_network.csv"]
    w = D["step7d_primary_station_area_weights.csv"]
    need(a, ["station_id", "official_station_name", "latitude", "longitude",
             "combined_homogeneity_status", "member_sensitivity_stable_only"], "step6c network")
    need(w, ["station_id", "national_area_weight"], "step7d weights")
    m = a.merge(w[["station_id", "national_area_weight"]], on="station_id", how="left").sort_values("station_id")
    rows = []
    for i, r in enumerate(m.itertuples(), 1):
        hom = ("source-transition variance signal" if "moderate" in str(r.combined_homogeneity_status)
               else "no signal")
        rows.append([i, integer(r.station_id), str(r.official_station_name).replace("Coxs_Bazar", "Cox’s Bazar").replace("_", " "),
                     num(r.latitude, 2), num(r.longitude, 2), num(100 * r.national_area_weight, 1), hom,
                     "yes" if bool(r.member_sensitivity_stable_only) else "no"])
    REP.checks.append(f"S18: {len(rows)} stations, weights sum {100 * m.national_area_weight.sum():.1f}%")
    return [(None, md_table(["#", "WMO/BMD ID", "Station", "Latitude (°N)", "Longitude (°E)",
                             "Voronoi area weight (%)", "Tmin homogeneity screen", "25-station network"], rows))]


# --------------------------------------------------------------------------
# Table specifications and captions
# --------------------------------------------------------------------------

ADJ = ("adjusted for log event age, day of the season, the same-day national Tmin anomaly "
       "and winter start year")

TABLES = [
    dict(id="S10", title="Nested validation of the termination hazard model",
         files=["table_R1_nested_validation.csv", "table_R1b_nested_selection.csv"], build=t_s10,
         caption=(
             "(a) Out-of-sample performance of the adjustment-only model (log event age, day of "
             "the season, same-day national Tmin anomaly, winter start year) and of the model that "
             "adds the predictors selected within each training fold. In every fold, predictors "
             "were standardized with training data only, each of the 39 pre-outcome primary-family "
             "candidates was tested with a likelihood-ratio test, Benjamini–Hochberg q < 0.05 "
             "was applied, the diurnal temperature range was excluded (it is algebraically "
             "redundant with Tmax once Tmin is adjusted for), and candidates correlated at "
             "|r| > 0.8 with a stronger selected predictor were skipped. Folds are single winters "
             "with at least one spell (leave-one-winter-out) or the two halves of the record. "
             "ΔAUC and ΔBrier are the selected minus the adjustment-only model, with 95% "
             "winter-block bootstrap intervals; Brier skill is relative to the training-fold event "
             "frequency; the calibration slope and the ratio of observed to expected terminations "
             "refer to the held-out predictions. In the split-period folds the winter-start-year "
             "term is extrapolated beyond the training period; trained on the later half, even the "
             "adjustment-only model scores worse than the training-period climatology (compare the "
             "Brier columns), and calibration is poor in both directions. The risk set treats the two "
             "season-end spells as right-censored (83 terminations). (b) Predictors selected "
             "in the training folds. This table supersedes Table S7, in which the predictors had "
             "been chosen on the full record. The AUC values are plotted in Fig. S1b.")),
    dict(id="S11",
         title="Sensitivity of the termination hazard model to the station network and to season boundaries",
         files=["table_R10_stable_network_catalogue.csv", "table_R10b_stable_network_hazard.csv",
                "table_R2_censoring_sensitivity.csv", "table_R18_season_end_final_row_dropped.csv"],
         build=t_s11,
         caption=(
             "(a) The primary cold-night spell catalogue (26 stations) and the catalogue rebuilt "
             "on the 25 stations other than Satkhira, the station flagged by the homogeneity "
             "screen, with Voronoi weights recomputed. The rebuild code was first validated by "
             "reproducing the 26-station weights and all 85 primary spells exactly; the 84 rebuilt "
             "spells overlap 83 of the 85 primary spells (79 with identical dates). (b) Odds ratios of termination per "
             "standard deviation (95% winter-block bootstrap interval) for the 39 pre-outcome "
             "primary-family predictors, " + ADJ + ": as in Table S2 (primary), on the 25-station "
             "catalogue, with the two spells that end on the last day of the season counted as "
             "terminations instead of being right-censored, "
             "with the two spells that start on 1 December (whose age at the start of the season "
             "is unknown) removed, and with the final rows of the two censored spells omitted instead of "
             "coded as non-terminations (their outcome is decided by the Tmin of 1 March, outside the "
             "analysed season). An asterisk marks an "
             "interval that excludes 1, judged on the unrounded limits (a limit shown as 1.00 can still "
             "exclude 1). The last row gives the number of risk-set rows and "
             "terminations.")),
    dict(id="S12", title="Termination hazard model with predictors taken one and two days earlier",
         files=["table_R3_one_day_lead_all_families.csv", "table_R11_lead_advection_robustness.csv",
                "table_R15_upstream_advection_alternative_catalogues.csv",
                "table_R17_predictor_sd_original_units.csv", "table_R18_season_end_final_row_dropped.csv"],
         build=t_s12,
         caption=lambda D: (
             "(a) Odds ratios of termination per standard deviation with every predictor of the "
             "four families (primary; advection and jet; before the terminating night; terminating "
             "night) taken on day k − 1 instead of day k (predictor lag 1 day), " + ADJ + ". The "
             "adjustment still uses the Tmin of day k and the spell is ongoing on day k, so a lag "
             "describes older information, not a longer forecast lead. q is given within each "
             "family and for a single Benjamini–Hochberg correction across all 66 predictors. "
             "(b) 850-hPa advection and jet speed one and two days earlier, and the robustness of "
             "the one-day lag of advection over the northwestern Indo-Gangetic Plain (NW IGP) to "
             "counting the season-end spells as terminations, to the 25-station catalogue, to adding "
             "the same-day core predictors (10-m wind, South Asian Z500 tendency, T2m − T850 "
             "tendency, Tmax) and to adding same-day advection over both regions; only the one-day "
             "lag of panel (a) is part of the 66-predictor correction, and the intervals and p-values of "
             "panels (b) and (c) are nominal sensitivity results. At a one-day lag the terminating-night "
             "family describes the night before day k and is no longer concurrent with the outcome. (c) The NW IGP advection "
             "one and two days earlier refitted in each of the six alternative event catalogues "
             "(added after the pre-submission review); the minimum event age of the risk set is the "
             "shortest spell of each catalogue. Adv850 anomalies are relative to the calendar-day "
             "climatology; a positive anomaly means more warm or less cold advection than normal. "
             "Intervals are 95% winter-block bootstrap intervals (10,000 resamples). The NW IGP rows of "
             "panels (a) and (b) are plotted in Fig. 7c; panel (c) is not.")
             + sd_sentence(D, [("mech_nigp_adv850_k_day_anom", "Adv850, NW IGP")])),
    dict(id="S13",
         title="Robustness of the deficit of cold-spell onsets after major sudden stratospheric warmings",
         files=["table_R4_ssw_robustness.csv"], build=t_s13,
         caption=(
             "Cold-spell onsets within each window after the 25 major SSWs of 1985/86–2024/25 "
             "(Table S1). Null rows compare the observed count with control dates moved to the "
             "same calendar date in randomly chosen other winters (calendar-matched) or in winters "
             "within five years of the SSW winter (decade-matched), 10,000 replicates each, "
             "redrawing any control date within 45 days of a real SSW (each SSW independently; the "
             "100-attempt limit was never reached, Text S1). In these rows the ratio is a count "
             "ratio, observed over expected onsets, and the range is the 2.5–97.5 percentile range "
             "of the null distribution, not a confidence interval; two-sided Monte Carlo p-values "
             "are min{1, 2 min(p₋, p₊)}, with p₋ and p₊ the plus-one proportions of replicates at or "
             "below and at or above the observed count. The ratio per at-risk day divides onsets by "
             "at-risk winter days (days not inside an ongoing spell after its first day) in both "
             "the observed and the control windows. Onset-rate rows compare onsets per at-risk day "
             "after SSWs with all other at-risk DJF days, or with the rest of the winters that "
             "contain the SSWs; their ratio is a rate ratio with an exact conditional binomial 95% "
             "interval, which conditions on the counts and ignores dependence between onsets of "
             "the same winter. SSWs in March have "
             "no December–February follow-up. The ENSO-phase-matched null (6 onsets against "
             "15.9 expected) is reported in the main text.")),
    dict(id="S14", title="Stratospheric state before onset and spell duration",
         files=["table_R8_stratosphere_duration.csv", "table_R8b_stratosphere_duration_extra.csv"],
         build=t_s14,
         caption=(
             "(a) Spearman correlation with duration of the eight pre-specified pre-onset metrics "
             "(defined in Text S3; vortex displacement is 90° minus the latitude of the minimum "
             "10-hPa geopotential height north of 50°N), "
             "and partial correlation controlling for onset date and winter start year. Intervals "
             "are 95% winter-block bootstrap intervals; q is the false discovery rate across the "
             "eight metrics. The last column gives the correlation of each metric with the onset "
             "date. (b) Persistent-minus-short difference of the weak-vortex fraction, and odds "
             "ratios of termination per standard deviation for trailing 20-day means of the "
             "weak-vortex fraction and of NAM at 10 hPa in the hazard model, " + ADJ + ".")),
    dict(id="S15", title="Stratosphere–troposphere pathway: box indices and ENSO control",
         files=["table_R6b_enso_controlled_pathway.csv", "table_R6_pathway_box_indices.csv"],
         build=t_s15,
         caption=(
             "(a) Box means of the gridded fields of Fig. S4: the regression of winter-mean "
             "(DJF) anomalies on winter-mean NAM at 100 hPa (40 winters), alone and with DJF ONI "
             "as a second predictor, and the mean over the 45 days after an SSW minus the mean "
             "over the rest of the same winter (19 winters with DJF days after an SSW). Boxes: "
             "South Asia (15–35°N, 70–100°E) and Bangladesh (20.5–26.8°N, "
             "88–93°E). The last column gives the fraction of grid points over "
             "10–70°N, 30–120°E passing Benjamini–Hochberg control "
             "(q < 0.05; t-tests for the regressions, exact sign-flip tests over the 19 winters for the "
             "post-SSW differences); T850 and MSLP exclude grid points above 1000 m. (b) Onset-pathway indices: Spearman correlation of winter means with "
             "winter-mean NAM at 100 hPa (40 winters; partial correlation given DJF ONI), and the "
             "post-SSW minus rest-of-winter difference with a bootstrap interval over winters (2000 "
             "resamples) and an exact two-sided sign-flip p-value over the 19 winters. Jet indices are the latitude and speed of the 200-hPa zonal wind maximum "
             "over 70–100°E. The South Asian Z500 post-SSW difference differs slightly between (a) "
             "and (b) because (a) averages the 2° composite field over the box and (b) the daily "
             "0.25° box index. The p-values of (b) are nominal, without correction across the indices.")),
    dict(id="S16", title="Humidity and surface temperature of cold-day and cold-night spells",
         files=["table_R5_absolute_humidity.csv", "table_R9c_skin_soil_regimes.csv",
                "table_R9_skin_soil_hazard.csv"], build=t_s16,
         caption=(
             "Bangladesh box (20.5–26.8°N, 88–93°E), ERA5 hourly fields from the "
             "ARCO archive; night = 12–23 UTC (18–05 LT), day = 00–11 UTC "
             "(06–17 LT). Pure spells share no day with a spell of the other type. (a) Event "
             "means of box-mean 2-m temperature, dewpoint, dewpoint depression and relative "
             "humidity (Magnus formula over water; Alduchov and Eskridge 1996; computed from the box-mean "
             "temperature and dewpoint, so it describes the box-mean state rather than the area mean "
             "of grid-point humidity, and the absence of box-mean saturation does not exclude local "
             "fog), the "
             "December–February climatology, and the cold-day minus cold-night difference "
             "with a 95% winter-block bootstrap interval (10,000 resamples). (b) Cold-day minus "
             "cold-night difference of the anomalies of skin temperature and of skin minus 2-m air "
             "temperature (10,000 resamples). (c) Skin-temperature predictors in the termination "
             "hazard model, " + ADJ + "; q is controlled within each window. Night-time values "
             "overlap the night that decides termination and are concurrent with the outcome. "
             "Soil temperature is not available in the ARCO archive.")),
    dict(id="S17", title="Field significance of persistent-minus-short differences in gridded fields",
         files=["table_R7_gridded_leakage_free.csv"], build=t_s17,
         caption=(
             "Event-mean anomaly fields, persistent (≥ 6 days, n = 26) minus short (3 days, "
             "n = 30) cold-night spells, in the two windows of identical event status, on 2° "
             "means of the 0.25° ERA5 fields over 10–70°N, 30–120°E. Grid "
             "points with q < 0.05: two-sided plus-one winter-block bootstrap p-value at each grid "
             "point (10,000 resamples) with Benjamini–Hochberg control across the grid points of each field and window "
             "(Wilks 2016); T850 and MSLP are tested only where the mean elevation of the 2° cell is "
             "below 1000 m. The last two columns give the difference of the means over the "
             "upstream box (40–60°N, 30–60°E) with its 95% bootstrap interval; the "
             "box was examined because earlier fixed-lag composites suggested it, so this "
             "comparison is exploratory. The fields are mapped in Fig. S2.")),
    dict(id="S18", title="Station network",
         files=["step6c_analytical_station_network.csv", "step7d_primary_station_area_weights.csv"],
         build=t_s18,
         caption=(
             "The 26 Bangladesh Meteorological Department stations of the fixed analysis network, each "
             "with at least 95% of December–February days observed in at least 38 of the 40 winters. "
             "Coordinates are those of the BMD station metadata. The area weight is the share of the "
             "national area in the station’s Voronoi (Thiessen) cell on a Bangladesh-centred Albers "
             "equal-area projection, clipped to the national boundary (geoBoundaries ADM0); the exact "
             "weights sum to 100% (the rounded values shown sum to 99.8%) and are archived in "
             "02_metadata/step7d_primary_station_area_weights.csv. The homogeneity screen compares each station’s monthly Tmin residuals from "
             "the median of all other stations (Sect. 2.1); Satkhira, the only station with a signal "
             "(a 2.5-fold increase in residual variance after the 2022 source change), is excluded "
             "from the 25-station sensitivity catalogue (Table S11). Station data are not "
             "redistributed; the weights and network files are archived with the code.")),
]


# --------------------------------------------------------------------------
# Tables S1-S9 (v4, Step 19): generated from the Step 11-16 result tables, so
# that no number in Online Resource 1 is typed by hand.
# --------------------------------------------------------------------------

EXACT.update({
    "ural_blocking_fraction_anom": "Ural blocking",
})
CLD = {"strd_wm2": "Downward longwave (UTC-day mean)", "ssrd_wm2": "Downward solar (UTC-day mean)",
       "tcc_mean": "Total cloud (UTC-day mean)", "tcc_00utc": "Total cloud, 06 LT",
       "lowcloud1000_frac_00utc": "Cloud base < 1 km, 06 LT",
       "fog300_frac_00utc": "Cloud base < 300 m, 06 LT",
       "td2m_mean": "2-m dewpoint (UTC-day mean)", "dpd_00utc": "Dewpoint depression, 06 LT",
       "blh_06utc": "Boundary-layer height, 12 LT"}
GROUP = {"station_temperature": "Station temperature", "era5_state": "ERA5 state",
         "era5_tendency": "ERA5 tendency", "remote_causal": "Remote (pre-outcome)",
         "noncausal_diagnostic": "Non-causal diagnostic"}
FAMILY.update({"night_overlapping": "Terminating night (concurrent)"})
PWIN = {"pre_onset": "Days −3 to −1", "onset_0_2": "Days 0–2", "last_day": "Last event day",
        "post_end": "Days +1 to +2", "pre10": "Days −10 to −1", "pre5": "Days −5 to −1"}


def var_label2(raw) -> str:
    """var_label plus the UTC-day cloud names (Table S5) and centred R (Table S2)."""
    s = str(raw)
    m = re.fullmatch(r"cld_(.+)_(anom|tend)", s)
    if m and m.group(1) in CLD:
        return ("Δ" if m.group(2) == "tend" else "") + CLD[m.group(1)]
    m = re.fullmatch(r"rrwp_([a-z_]+?)_R_centred_noncausal", s)
    if m and m.group(1) in RRWP:
        return f"RRWP {RRWP[m.group(1)]} R centred (t ± 7)"
    return var_label(s)


def short_label(raw) -> str:
    return var_label2(raw).replace(" anomaly", "")


def _or_row(r, q_col):
    return [num(r.odds_ratio_per_sd), ci(r.ci_low, r.ci_high, ratio=True), pval(r.p_bootstrap),
            pval(r[q_col]) if q_col and q_col in r and not _isnan(r[q_col]) else DASH]


def t_s1(D):
    c, v = D["table_60_major_ssw_catalogue.csv"], D["table_60b_major_ssw_validation.csv"]
    need(c, ["central_date", "winter_label", "u60_10_central_ms", "easterly_run_days"], "table_60")
    exact = set(pd.to_datetime(v.loc[v["exact_match"].astype(bool), "nearest_derived_date"]).dt.strftime("%Y-%m-%d"))
    rows = []
    for i, r in enumerate(c.sort_values("central_date").itertuples(), 1):
        d = str(r.central_date)[:10]
        ref = "exact match" if d in exact else ("after 2023, not in reference list" if d >= "2024" else "no match")
        rows.append([i, d, r.winter_label, num(r.u60_10_central_ms, 1), integer(r.easterly_run_days), ref])
    REP.checks.append(f"S1: {len(rows)} major SSWs, {len(exact)} exact matches with the NOAA compendium")
    return [(None, md_table(["#", "Central date", "Winter", "U(60°N, 10 hPa) at central date (m s⁻¹)",
                             "Easterly run (days)", "NOAA compendium"], rows))]


def t_s2(D):
    t = D["table_91_termination_hazard_odds_ratios.csv"]
    need(t, ["family", "predictor", "odds_ratio_per_sd", "ci_low", "ci_high", "p_bootstrap",
             "q_fdr_causal", "n_rows", "n_terminations"], "table_91")
    rows = [[GROUP.get(r.family, r.family), var_label2(r.predictor)] + _or_row(r, "q_fdr_causal")
            for _, r in t.iterrows()]
    REP.checks.append(f"S2: {int(t.n_rows.max())} risk-set rows, {int(t.n_terminations.max())} terminations; "
                      f"q < 0.05: {[var_label2(p) for p in t.loc[t.q_fdr_causal < 0.05, 'predictor']]}")
    return [(None, md_table(["Group", "Predictor", "OR per SD", "95% CI", "p", "q (FDR)"], rows))]


def t_s3(D):
    t = D["table_106_mechanism_extension_hazard.csv"]
    order = ["mech_bd_adv850_k_day_anom", "mech_nigp_adv850_k_day_anom", "mech_jet200_core_lat_70_100e_anom",
             "mech_jet200_core_speed_70_100e_anom", "mech_bd_adv850_k_day_tend", "mech_nigp_adv850_k_day_tend",
             "mech_jet200_core_lat_70_100e_tend", "mech_jet200_core_speed_70_100e_tend"]
    t = t.set_index("predictor").reindex([p for p in order if p in set(t.predictor)] +
                                         [p for p in t.predictor if p not in order]).reset_index()
    rows = [[var_label2(r.predictor)] + _or_row(r, "q_fdr_family") for _, r in t.iterrows()]
    return [(None, md_table(["Predictor", "OR per SD", "95% CI", "p", "q (FDR)"], rows))]


def t_s4(D):
    t = D["table_114_prenight_vs_night_hazard.csv"]
    rows = [[FAMILY.get(r.family, r.family), var_label2(r.predictor)] + _or_row(r, "q_fdr_family")
            for _, r in t.iterrows()]
    return [(None, md_table(["Window", "Predictor", "OR per SD", "95% CI", "p", "q (FDR within window)"], rows))]


def t_s5(D):
    t = D["table_111_cloud_radiation_hazard.csv"]
    rows = [[var_label2(r.predictor)] + _or_row(r, "q_fdr_family") for _, r in t.iterrows()]
    return [(None, md_table(["Predictor", "OR per SD", "95% CI", "p", "q (FDR)"], rows))]


def t_s6(D):
    t = D["table_95_hazard_robustness.csv"]
    cols = ROB_COLS + [("lead_1_day", "Predictor lag 1 day")]
    preds = list(t.loc[t.analysis == "primary", "predictor"])
    grid = {k: t[t.analysis == k].set_index("predictor") for k, _ in cols}
    rows = []
    for p in preds:
        row = [short_label(p)]
        for k, _ in cols:
            g = grid[k]
            if p in g.index:
                r = g.loc[p]
                row.append(num(r.odds_ratio_per_sd) + ("\\*" if excludes(r.ci_low, r.ci_high, 1) else ""))
            else:
                row.append(DASH)
        rows.append(row)
    REP.checks.append(f"S6: {len(preds)} primary-family predictors with q < 0.10")
    return [(None, md_table(["Predictor"] + [lab for _, lab in cols], rows))]


def t_s7(D):
    t = D["table_117_split_period_validation.csv"]
    rows = []
    for _, r in t.iterrows():
        rows.append([str(r.train).replace("-", NDASH), str(r.test).replace("-", NDASH),
                     f"{integer(r.n_test_rows)} ({integer(r.test_terminations)})",
                     num(r.auc_adjustment_only, 3), num(r.auc_plus_synoptic, 3),
                     f"{num(r.auc_difference, 3)} {ci(r.auc_diff_ci_low, r.auc_diff_ci_high, 3)}",
                     num(r.brier_adjustment_only, 3), num(r.brier_plus_synoptic, 3),
                     num(r.brier_climatology, 3)])
    preds = "; ".join(var_label2(x) for x in str(t.synoptic_predictors.iloc[0]).split(";"))
    REP.checks.append(f"S7: synoptic predictors chosen on the full record: {preds}")
    return [(None, md_table(["Trained on", "Tested on", "Test rows (terminations)", "AUC, adjustment only",
                             "AUC, + synoptic", "ΔAUC [95% CI]", "Brier, adjustment only",
                             "Brier, + synoptic", "Brier, climatology"], rows))]


def t_s8(D):
    a, b = D["table_89_leakage_free_persistence_tests.csv"], D["table_118_minimum_detectable_differences.csv"]
    m = a.merge(b[["window", "variable", "mdd_80pct", "mdd_80pct_standardized"]], on=["window", "variable"], how="left")
    REP.checks.append(f"S8: {len(m)} contrasts, min q {m.q_fdr_within_window.min():.3f}")
    out = []
    for k, (w, g) in enumerate(m.groupby("window", sort=False)):
        rows = [[short_label(r.variable), num(r.difference), ci(r.ci_low, r.ci_high),
                 pval(r.q_fdr_within_window), num(r.mdd_80pct), num(r.mdd_80pct_standardized)]
                for _, r in g.iterrows()]
        out.append((f"({chr(97 + k)}) {PWIN.get(w, w)}",
                    md_table(["Variable", "Persistent − short", "95% CI", "q", "MDD (80% power)",
                              "MDD / between-event SD"], rows)))
    return out


def t_s9(D):
    a, b = D["table_105_remote_driver_duration_tests.csv"], D["table_119_minimum_detectable_correlations.csv"]
    m = a.merge(b[["window", "driver", "min_detectable_abs_rho_80pct"]], on=["window", "driver"], how="left")
    rows = [[PWIN.get(r.window, r.window), short_label(r.driver), num(r.persistent_minus_short),
             pval(r.diff_q), num(r.spearman_rho), ci(r.rho_ci_low, r.rho_ci_high), pval(r.rho_q),
             num(r.min_detectable_abs_rho_80pct)] for _, r in m.iterrows()]
    REP.checks.append(f"S9: min q (difference) {a.diff_q.min():.3f}, min q (rho) {a.rho_q.min():.3f}")
    return [(None, md_table(["Window", "Driver", "Persistent − short", "q (difference)", "Spearman ρ", "95% CI",
                             "q (correlation)", "Minimum detectable ρ"], rows))]


INF = ("Intervals are 95% winter-block bootstrap intervals and p-values two-sided plus-one bootstrap "
       "p-values, min{1, 2[min(n≤0, n≥0) + 1]/(B + 1)}, computed on the log-odds coefficient (null value 0) "
       "from B = 10,000 resamples")
TABLES_A = [
    dict(id="S1", title="Major sudden stratospheric warmings, 1985/86–2024/25",
         files=["table_60_major_ssw_catalogue.csv", "table_60b_major_ssw_validation.csv"], build=t_s1,
         caption=("Major SSWs identified from the first reversal of the daily-mean zonal-mean zonal wind at "
                  "60°N and 10 hPa in ERA5, with at least 20 consecutive westerly days between events and "
                  "final warmings excluded unless westerlies return for at least 10 consecutive days before "
                  "30 April (Charlton and Polvani 2007). All events up to 2023 coincide exactly (zero-day "
                  "offset) with the ERA5 central dates of the NOAA SSW Compendium (Butler et al. 2017; table of major "
                  "mid-winter SSWs at https://csl.noaa.gov/groups/csl8/sswcompendium/majorevents.html, ERA5 "
                  "column through February 2023, accessed 12 October 2026; the copy used is archived with the "
                  "code); the two events of 2024 postdate that list and were detected here.")),
    dict(id="S2", title="Termination hazard model: primary predictor family",
         files=["table_91_termination_hazard_odds_ratios.csv", "table_R17_predictor_sd_original_units.csv"],
         build=t_s2,
         caption=lambda D: ("39 pre-outcome predictors, measured on day k or earlier (station Tmin and Tmax of day k; ERA5 at "
                  "00 UTC of day k; RRWP R lagged by 7 days or trailing envelopes), " + ADJ + ". The risk set "
                  "has 260 rows (spell days k ≥ 3) and 83 terminations: the two spells whose last day is the last day of "
                  "February are right-censored. ERA5 box means of 1000- and 850-hPa fields and MSLP exclude "
                  "grid points above 1000 m. q is controlled across the 39 pre-outcome predictors. The four "
                  "same-day centred R values (±7-day window) use information from after day k; they are "
                  "shown as non-causal diagnostics and are excluded from the false-discovery-rate correction. "
                  + INF + "." + sd_sentence(D, [("era5_bd_wspd10_anom", "BD 10-m wind"),
                                                 ("era5_bd_stab_t2m_minus_t850_tend", "ΔBD T2m − T850"),
                                                 ("era5_sa_z500_tend", "ΔSA Z500"),
                                                 ("nat_tmax_anom", "Tmax")]))),
    dict(id="S3", title="Termination hazard model: advection and jet family",
         files=["table_106_mechanism_extension_hazard.csv"], build=t_s3,
         caption=("Daily 850-hPa temperature advection (positive = warm advection; grid points above 1000 m and "
                  "their neighbours excluded) and 200-hPa jet diagnostics at 00 UTC, as anomalies and "
                  "day-to-day tendencies, " + ADJ + "; q is controlled within this family of eight "
                  "predictors. " + INF + ".")),
    dict(id="S4", title="Termination hazard model: cloud, humidity and radiation before and during the terminating night",
         files=["table_114_prenight_vs_night_hazard.csv"], build=t_s4,
         caption=("ERA5 hourly fields from the ARCO archive, Bangladesh box. “Before terminating night”: means "
                  "over 00–11 UTC of day k (06–17 LT), 00 UTC (06 LT) dawn values, and day-to-day changes of "
                  "the daytime means; these can carry precursor information. “Terminating night”: means over "
                  "12–23 UTC (18–05 LT), which overlap the night that decides whether the spell continues; "
                  "these are concurrent with the outcome. Radiation accumulations (J m⁻²) were converted to "
                  "mean fluxes by dividing by the accumulation time (3600 s per hourly field); ERA5 labels each "
                  "accumulation with the end of its hour, so the 00–11 UTC window covers 23–11 UTC and still "
                  "ends before the terminating night. Flux labels therefore give the accumulation intervals (05–17 "
                  "and 17–05 LT), whereas cloud, dewpoint and boundary-layer labels give the sampling hours "
                  "(06–17 and 18–05 LT). Cloud-base fractions are shares of the whole box area; "
                  "cloud-free points have no cloud base and count as not low cloud. q is controlled "
                  "separately within each window; the global correction across all 61 pre-outcome predictors is "
                  "given in the main text. " + INF + ".")),
    dict(id="S5", title="Superseded analysis: cloud, humidity and radiation as UTC-day means",
         files=["table_111_cloud_radiation_hazard.csv"], build=t_s5,
         caption=("Retained for transparency (see Text S2). UTC-day means run from 06 LT on day k to 06 LT on "
                  "day k + 1 and therefore include the terminating night; they are superseded by Table S4. "
                  "q is controlled within this family of 14 predictors. " + INF + ".")),
    dict(id="S6", title="Robustness of the termination odds ratios",
         files=["table_95_hazard_robustness.csv"], build=t_s6,
         caption=("Odds ratios per SD for the primary-family predictors with q < 0.10, re-estimated with linearly "
                  "detrended covariates, with each alternative event catalogue (number of spells: one-day gap "
                  "merge 86; area ≥ 10% 110; area ≥ 30% 63; minimum duration 4 days 55; p05 threshold 35; BMD "
                  "absolute 10 °C threshold 93), and with every predictor taken one day earlier (predictor lag 1 day; "
                  "the adjustment still uses the Tmin of day k). These ten predictors are a sensitivity subset; "
                  "the upstream advection predictor is tested in the alternative catalogues in Table S12c. "
                  "An asterisk marks a 95% winter-block bootstrap interval excluding 1 (10,000 resamples). The "
                  "same values are plotted in Fig. S1a.")),
    dict(id="S7", title="Superseded analysis: split-period validation with predictors chosen on the full record",
         files=["table_117_split_period_validation.csv"], build=t_s7,
         caption=("Retained for transparency; superseded by the nested validation of Table S10. The synoptic "
                  "predictors here were chosen using all 40 winters, including the test half, so these AUC "
                  "differences are optimistic; when selection is repeated inside each training fold, neither "
                  "split improves on the adjustment-only model (Table S10). Each model is fitted on one half of "
                  "the record and scored on the other; the interval of the AUC difference is a winter-block "
                  "bootstrap of the test rows (2000 resamples).")),
    dict(id="S8", title="Leakage-free comparisons of persistent and short cold-night spells, with minimum detectable differences",
         files=["table_89_leakage_free_persistence_tests.csv", "table_118_minimum_detectable_differences.csv"],
         build=t_s8,
         caption=("Event-mean anomalies, persistent (≥ 6 days, n = 26) minus short (3 days, n = 30), in windows "
                  "where the event status of both groups is identical. Units: °C for Tmin, Tmax, DTR and T2m; K "
                  "for T850 and T2m − T850; m for thickness and Z500; hPa for MSLP; m s⁻¹ for winds. SA box "
                  "means of 1000- and 850-hPa fields and MSLP exclude grid points above 1000 m. CI: winter-block "
                  "bootstrap (10,000 resamples); q: false discovery rate within each window. MDD: minimum "
                  "difference detectable with 80% power at a two-sided 5% level, 2.80 × the bootstrap standard "
                  "error, also given in units of the pooled between-event standard deviation; these MDDs are "
                  "nominal single-comparison values that do not account for the false-discovery-rate correction.")),
    dict(id="S9", title="Large-scale and remote drivers versus spell duration, with minimum detectable correlations",
         files=["table_105_remote_driver_duration_tests.csv", "table_119_minimum_detectable_correlations.csv"],
         build=t_s9,
         caption=("Event-window means of 14 drivers in three windows of identical event status, related to "
                  "duration in two ways: persistent-minus-short difference (units as in Table S8; SD for the "
                  "Siberian High, NAM and RRWP measures; fraction for Ural blocking; K day⁻¹ for 850-hPa "
                  "advection; degrees and m s⁻¹ for jet latitude and speed), and Spearman correlation with "
                  "duration across all 85 spells. Rossby-wave measures enter only in forms that use no information "
                  "after the target day (R lagged by 7 days; trailing 3-day envelope). q: false discovery rate within each window and statistic. "
                  "Minimum detectable ρ: the smallest absolute correlation detectable with 80% power at a "
                  "two-sided 5% level (2.80 × bootstrap standard error; nominal, single comparison).")),
]


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def mpl_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.dpi": 600})
    return plt


def save_fig(fig, stem: Path):
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        Image.open(stem.with_suffix(".png")).convert("RGB").save(
            stem.with_suffix(".tif"), dpi=(600, 600), compression="tiff_lzw")
    except Exception as exc:  # pragma: no cover
        REP.add(f"  TIFF not written: {exc}")


ROB_COLS = [("primary", "Primary"), ("detrended_covariates", "Detrended"),
            ("catalogue:p10_area20_duration3_gap1", "Gap merge"),
            ("catalogue:p10_area10_duration3", "Area ≥ 10%"),
            ("catalogue:p10_area30_duration3", "Area ≥ 30%"),
            ("catalogue:p10_area20_duration4", "Duration ≥ 4 d"),
            ("catalogue:p05_area20_duration3", "p05"),
            ("catalogue:bmd_area20_duration3", "BMD ≤ 10 °C")]


def build_fig_s1(root: Path, outdir: Path, D: dict):
    plt = mpl_setup()
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    REP.ctx = "Fig. S1"
    t95 = locate(root, "table_95_hazard_robustness.csv")
    if t95 is None:
        REP.add("Fig. S1: table_95_hazard_robustness.csv not found - skipped")
        return
    rob = pd.read_csv(t95)
    preds = list(rob.loc[rob.analysis == "primary", "predictor"])
    grids = [(lab, rob[rob.analysis == k].set_index("predictor")) for k, lab in ROB_COLS
             if k in set(rob.analysis)]
    if "table_R10b_stable_network_hazard.csv" in D:
        grids.append(("25 stations", D["table_R10b_stable_network_hazard.csv"].set_index("predictor")))
    if "table_R2_censoring_sensitivity.csv" in D:
        r2 = D["table_R2_censoring_sensitivity.csv"]
        a = next((x for x in r2.analysis.astype(str).unique() if x.startswith("A")), None)
        if a:
            grids.append(("Not censored", r2[r2.analysis == a].set_index("predictor")))
    if "lead_1_day" in set(rob.analysis):
        grids.append(("Lag 1 day", rob[rob.analysis == "lead_1_day"].set_index("predictor")))
    labels = [g[0] for g in grids]
    M = np.full((len(preds), len(grids)), np.nan)
    S = np.zeros_like(M, dtype=bool)
    for j, (_, g) in enumerate(grids):
        for i, p in enumerate(preds):
            if p in g.index:
                r = g.loc[p]
                M[i, j], S[i, j] = r.odds_ratio_per_sd, excludes(r.ci_low, r.ci_high, 1)

    r1 = D.get("table_R1_nested_validation.csv")
    fig = plt.figure(figsize=(174 / 25.4, (150 if r1 is not None else 95) / 25.4))
    gs = fig.add_gridspec(2 if r1 is not None else 1, 1,
                          height_ratios=[1.75, 1] if r1 is not None else [1], hspace=0.55)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(np.clip(M, 0.25, 4), cmap="RdBu_r", norm=LogNorm(0.25, 4), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(j, i, DASH, ha="center", va="center", fontsize=6)
                continue
            dark = abs(np.log(M[i, j])) > 0.75
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    fontweight="bold" if S[i, j] else "normal",
                    color="white" if dark else "black")
            if S[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=0.9, ec="black"))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(range(len(preds)))
    ax.set_yticklabels([var_label(p) for p in preds])
    if "Lag 1 day" in labels:
        ax.axvline(labels.index("Lag 1 day") - 0.5, color="black", lw=0.8, ls="--")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015)
    cb.ax.minorticks_off()
    cb.ax.set_yticks([0.25, 0.5, 1, 2, 4])
    cb.ax.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
    cb.set_label("Odds ratio of termination per SD")
    cb.outline.set_linewidth(0.5)
    ax.set_title("(a) Robustness of the termination odds ratios", loc="left", fontweight="bold")

    if r1 is not None:
        bx = fig.add_subplot(gs[1])
        y = np.arange(len(r1))[::-1]
        for k, (yi, (_, r)) in enumerate(zip(y, r1.iterrows())):
            bx.plot([r.auc_adjustment_only, r.auc_nested_model], [yi, yi], color="0.6", lw=1.2, zorder=1)
            bx.scatter(r.auc_adjustment_only, yi, s=26, color="0.55", zorder=2,
                       label="Adjustment only" if k == 0 else None)
            bx.scatter(r.auc_nested_model, yi, s=26, color="#2166ac", zorder=3,
                       label="+ predictors selected in training fold" if k == 0 else None)
            bx.text(1.005, yi, f"ΔAUC {num(r.auc_difference, 3)} "
                    f"{ci(r.auc_diff_ci_low, r.auc_diff_ci_high, 3)}",
                    transform=bx.get_yaxis_transform(), va="center", fontsize=6.3)
        bx.set_yticks(y)
        bx.set_yticklabels([scheme_label(s).replace(", test", ",\ntest") for s in r1.scheme])
        v = r1[["auc_adjustment_only", "auc_nested_model"]].values
        lo = math.floor((np.nanmin(v) - 0.04) * 20) / 20
        hi = math.ceil((np.nanmax(v) + 0.04) * 20) / 20
        bx.set_xlim(lo, hi)
        bx.set_ylim(-0.7, len(r1) - 0.3)
        bx.set_xlabel("AUC of termination on held-out winters")
        bx.spines[["top", "right"]].set_visible(False)
        bx.legend(loc="lower left", frameon=False, ncol=2, bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
        bx.set_title("(b) Nested validation", loc="left", fontweight="bold", pad=16)
        fig.subplots_adjust(right=0.80)
    save_fig(fig, outdir / "FigS1_hazard_robustness")
    plt.close(fig)
    REP.add(f"Fig. S1: columns {labels}")


def copy_fig_s2(root: Path, outdir: Path):
    hits = sorted((root / "08_outputs").rglob("fig_R7_gridded_leakage_free.*"))
    if not hits:
        REP.add("Fig. S2: fig_R7_gridded_leakage_free.* not found")
        return
    for h in hits:
        if h.suffix.lower() in {".pdf", ".png", ".tif", ".tiff", ".svg"}:
            shutil.copy2(h, outdir / f"FigS2_gridded_leakage_free{h.suffix.lower()}")
    pngs = [h for h in hits if h.suffix.lower() == ".png"]
    if pngs and not any(h.suffix.lower() in {".tif", ".tiff"} for h in hits):
        try:
            from PIL import Image
            Image.open(pngs[0]).convert("RGB").save(outdir / "FigS2_gridded_leakage_free.tif",
                                                    dpi=(600, 600), compression="tiff_lzw")
        except Exception as exc:  # pragma: no cover
            REP.add(f"  Fig. S2 TIFF not written: {exc}")
    REP.add(f"Fig. S2: copied from {hits[0].parent.relative_to(root)}")


# --------------------------------------------------------------------------
# Inspect and main
# --------------------------------------------------------------------------

def inspect(root: Path):
    for name in [f for t in TABLES for f in t["files"]] + ["table_95_hazard_robustness.csv"]:
        p = locate(root, name)
        print("=" * 78)
        if p is None:
            print(f"{name}: NOT FOUND")
            continue
        df = pd.read_csv(p)
        print(f"{p.relative_to(root)}  ({len(df)} rows)\ncolumns: {list(df.columns)}")
        with pd.option_context("display.width", 220, "display.max_columns", 60):
            print(df.head(4).to_string(index=False))
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                vals = sorted({str(x) for x in df[c].unique()})
                if len(vals) <= 70:
                    print(f"  {c}: {vals}")


def main():
    ap = argparse.ArgumentParser(description="Build Online Resource 1 tables S1-S18 and Figs S1-S2")
    ap.add_argument("--root")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    root = find_root(args.root)
    if args.inspect:
        inspect(root)
        return
    outdir = Path(args.out) if args.out else root / "08_outputs" / "online_resource_1"
    outdir.mkdir(parents=True, exist_ok=True)
    REP.add("Online Resource 1 build report (16h v4)")
    REP.add(f"project root: {root}\n")

    D = {}
    for t in TABLES_A + TABLES:
        for f in t["files"]:
            p = locate(root, f)
            if p is not None:
                D[f] = pd.read_csv(p)
                REP.add(f"source: {p.relative_to(root)} ({len(D[f])} rows)")
            else:
                REP.add(f"MISSING: {f}")
    REP.add("")

    for name, tabs in (("online_resource_1_tables_S1_S9.md", TABLES_A),
                       ("online_resource_1_tables_S10_S18.md", TABLES)):
        write_tables(outdir / name, tabs, D)

    try:
        build_fig_s1(root, outdir, D)
    except Exception as exc:
        REP.add(f"Fig. S1 NOT BUILT: {type(exc).__name__}: {exc}")
    copy_fig_s2(root, outdir)
    finish_report(outdir)


def write_tables(path: Path, tables: list, D: dict):
    md = ["<!-- Generated by 07_scripts/16h_build_online_resource_tables.py (v4). "
          "Do not edit numbers by hand; re-run the script. -->", ""]
    for t in tables:
        REP.ctx = f"Table {t['id']}"
        caption = t["caption"](D) if callable(t["caption"]) else t["caption"]
        md += [f"## Table {t['id']}. {t['title']}", "", caption, ""]
        try:
            for sub, tab in t["build"](D):
                if sub:
                    md += [f"**{sub}**", ""]
                md += [tab, ""]
        except Exception as exc:
            msg = f"Table {t['id']} NOT BUILT: {type(exc).__name__}: {exc}"
            REP.add(msg)
            md += [f"*[{msg}]*", ""]
    path.write_text("\n".join(md), encoding="utf-8")


def finish_report(outdir: Path):
    REP.add("\nConsistency checks (compare with the main text):")
    REP.lines += [f"  {c}" for c in REP.checks] or ["  none"]
    REP.add("\nRounding ties (stored value ends in 5 one decimal beyond the display; rounded "
            "half up, so quote the same rounding in the text):")
    REP.lines += [f"  {t}" for t in dict.fromkeys(REP.ties)] or ["  none"]
    REP.add("\nNames without a label (shown unchanged):")
    if not REP.unknown:
        REP.add("  none")
    for k, v in REP.unknown.items():
        REP.add(f"  {k}: " + "; ".join(sorted(v)))
    (outdir / "online_resource_1_build_report.txt").write_text("\n".join(REP.lines), encoding="utf-8")
    print("\n".join(REP.lines))
    print(f"\nWritten to {outdir}")


if __name__ == "__main__":
    main()
