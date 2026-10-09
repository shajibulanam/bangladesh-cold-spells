"""
16b_revision_stable_network_catalogue.py - revision analysis R10: cold-night catalogue on the stable network.

The Step 6B screen flagged one station (Satkhira, BMD 41946: larger residual variance after the 2022 source
change). Step 6C defined a 'sensitivity_stable_only' network of the other 25 stations, but no catalogue was built
from it. This script builds that catalogue with exactly the Step 7D rules and re-estimates the termination model.

Steps
  1. Self-validation: rebuild the Voronoi weights and the catalogue for the 26-station primary network with the
     same code. The script STOPS unless (a) the weights match 02_metadata/step7d_primary_station_area_weights.csv
     within 1e-6 and (b) the 85 primary spells are reproduced date for date.
  2. Build Voronoi weights for the 25 stable stations (Bangladesh-centred Albers equal-area projection, polygons
     extended to the boundary envelope and clipped to the ADM0 boundary; weights sum to one).
  3. Daily national diagnostics and spells with the Step 7D rules: observed area >= 0.80 and >= 15 observed
     stations; cold area (station Tmin < station p10, full national denominator) >= 0.20; >= 3 consecutive days;
     no gap merging; no cross-winter spells.
  4. Compare with the primary catalogue (counts, duration classes, matched spells) and re-estimate all 39 primary
     termination predictors (winter-block bootstrap, BH q).

Inputs (unchanged project files): 04_clean_data/temperature_daily_djf_with_compound_cold_indicators.parquet,
02_metadata/step7d_bangladesh_boundary.gpkg, step7d_primary_station_area_weights.csv,
step6c_sensitivity_network_membership.csv, 06_events/step7e_frozen_primary_event_catalogue.csv
Outputs: 06_events/step17b_stable_network_event_catalogue.csv, 02_metadata/step17b_stable_network_area_weights.csv,
08_outputs/tables/revision/table_R10*.csv, 05_qc_reports/revision/step17b_report.txt
Run from the project root:  python3 07_scripts/16b_revision_stable_network_catalogue.py 2>&1 | tee 09_logs/step17b.log
"""
from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from shapely import MultiPoint, get_parts, voronoi_polygons
from shapely.geometry import Point

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("term_common", "15_termination_common.py")
D = _load("step16d", "15d_discrete_time_hazard_model.py")
E = _load("step16e", "15e_hazard_robustness_and_trend.py")

ROOT = C.PROJECT_ROOT
DATA = ROOT / "04_clean_data" / "temperature_daily_djf_with_compound_cold_indicators.parquet"
POLICY_7D = ROOT / "00_admin" / "step7d_area_weighted_event_policy.yaml"
BOUNDARY = ROOT / "02_metadata" / "step7d_bangladesh_boundary.gpkg"
WEIGHTS_26 = ROOT / "02_metadata" / "step7d_primary_station_area_weights.csv"
NETWORKS = ROOT / "02_metadata" / "step6c_sensitivity_network_membership.csv"
PRIMARY_CAT = ROOT / "06_events" / "step7e_frozen_primary_event_catalogue.csv"
OUT_CAT = ROOT / "06_events" / "step17b_stable_network_event_catalogue.csv"
OUT_W = ROOT / "02_metadata" / "step17b_stable_network_area_weights.csv"
OUT_T = ROOT / "08_outputs" / "tables" / "revision"
OUT_Q = ROOT / "05_qc_reports" / "revision"
STABLE_NETWORK = "sensitivity_stable_only"
N_BOOT = 1000
PRIMARY = [p for fam, ps in D.FAMILIES.items() if fam != "noncausal_diagnostic" for p in ps]


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def first_col(df: pd.DataFrame, names: list[str]) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise ValueError(f"none of {names} in data")


def build_weights(stations: pd.DataFrame, boundary: gpd.GeoDataFrame, proj: str) -> pd.DataFrame:
    """Voronoi weights exactly as Step 06D: projected points, polygons extended to the boundary envelope, clipped."""
    b = boundary.to_crs(proj)
    geom = b.geometry.union_all() if hasattr(b.geometry, "union_all") else b.geometry.unary_union
    pts = gpd.GeoSeries([Point(xy) for xy in zip(stations["lon"], stations["lat"])], crs="EPSG:4326").to_crs(proj)
    coll = voronoi_polygons(MultiPoint(list(pts)), extend_to=geom.envelope, ordered=True)
    parts = list(get_parts(coll))
    if len(parts) != len(pts):
        raise RuntimeError("Voronoi polygon count does not match station count")
    areas = np.array([p.intersection(geom).area for p in parts])
    return pd.DataFrame({"station_uid": stations["station_uid"].values, "cell_area_km2": areas / 1e6,
                         "national_area_weight": areas / geom.area})


def daily_and_spells(data: pd.DataFrame, weights: pd.DataFrame, cols: dict, rules: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = data[data["station_uid"].isin(weights["station_uid"])].merge(weights[["station_uid", "national_area_weight"]],
                                                                       on="station_uid", how="left")
    d["_obs"] = d[cols["elig"]] & d[cols["temp"]].notna()
    d["_cold"] = d["_obs"] & d[cols["cold"]]
    g = d.groupby("date")
    daily = pd.DataFrame({
        "winter_start_year": g["winter_start_year"].first(),
        "observed_area_fraction": g.apply(lambda x: x.loc[x["_obs"], "national_area_weight"].sum()),
        "observed_station_count": g["_obs"].sum(),
        "cold_area_fraction": g.apply(lambda x: x.loc[x["_cold"], "national_area_weight"].sum())}).reset_index()
    daily["usable"] = (daily["observed_area_fraction"] >= rules["min_obs_area"]) & \
                      (daily["observed_station_count"] >= rules["min_obs_stations"])
    daily["national_cold_day"] = daily["usable"] & (daily["cold_area_fraction"] >= rules["min_cold_area"])
    daily = daily.sort_values("date").reset_index(drop=True)
    spells, run = [], []
    for row in daily.itertuples(index=False):
        contiguous = run and (row.date - run[-1].date).days == 1 and row.winter_start_year == run[-1].winter_start_year
        if row.national_cold_day and (not run or contiguous):
            run.append(row)
            continue
        if len(run) >= rules["min_days"]:
            spells.append(run)
        run = [row] if row.national_cold_day else []
    if len(run) >= rules["min_days"]:
        spells.append(run)
    cat = pd.DataFrame([{"start_date": r[0].date, "end_date": r[-1].date, "duration_days": len(r),
                         "winter_start_year": int(r[0].winter_start_year),
                         "peak_cold_area_fraction": max(x.cold_area_fraction for x in r)} for r in spells])
    cat.insert(0, "event_id", [f"STABLE25_{w}_{i + 1:02d}" for w, i in
                               zip(cat["winter_start_year"], cat.groupby("winter_start_year").cumcount())])
    return daily, cat


def main() -> None:
    OUT_T.mkdir(parents=True, exist_ok=True)
    OUT_Q.mkdir(parents=True, exist_ok=True)
    pol = yaml.safe_load(POLICY_7D.read_text())
    net = pol["network"]
    cols = {"member": net["membership_column"], "elig": net["percentile_eligibility_column"],
            "cold": pol["primary_station_condition"]["column"], "temp": "analysis_tmin_unadjusted"}
    rules = {"min_obs_area": pol["daily_data_availability"]["minimum_observed_national_area_fraction"],
             "min_obs_stations": pol["daily_data_availability"]["minimum_observed_station_count"],
             "min_cold_area": pol["regional_cold_day"]["minimum_cold_national_area_fraction"],
             "min_days": pol["cold_spell_event"]["minimum_consecutive_days"]}
    proj = " ".join(pol["spatial_projection"]["proj_string"].split())
    data = pd.read_parquet(DATA)
    data["station_uid"] = data["station_uid"].astype(str)
    data["date"] = pd.to_datetime(data["date"])
    for c in (cols["member"], cols["elig"], cols["cold"]):
        data[c] = as_bool(data[c])
    lat, lon = first_col(data, ["latitude", "station_latitude", "official_latitude", "latitude_dd"]), \
        first_col(data, ["longitude", "station_longitude", "official_longitude", "longitude_dd"])
    st = data.loc[data[cols["member"]], ["station_uid", lat, lon]].drop_duplicates("station_uid")
    st = st.rename(columns={lat: "lat", lon: "lon"}).sort_values("station_uid").reset_index(drop=True)
    boundary = gpd.read_file(BOUNDARY)

    # 1. self-validation on the 26-station network
    w26 = build_weights(st, boundary, proj)
    ref = pd.read_csv(WEIGHTS_26, dtype={"station_uid": str})[["station_uid", "national_area_weight"]]
    chk = w26.merge(ref, on="station_uid", suffixes=("", "_ref"))
    max_dw = float((chk["national_area_weight"] - chk["national_area_weight_ref"]).abs().max())
    _, cat26 = daily_and_spells(data, w26, cols, rules)
    prim = pd.read_csv(PRIMARY_CAT, parse_dates=["start_date", "end_date"])
    same = (len(cat26) == len(prim)) and \
        set(zip(cat26["start_date"], cat26["end_date"])) == set(zip(prim["start_date"], prim["end_date"]))
    print(f"Validation: {len(chk)} stations, max weight difference {max_dw:.2e}; "
          f"rebuilt {len(cat26)} spells vs primary {len(prim)}; identical dates: {same}", flush=True)
    if len(chk) != 26 or max_dw > 1e-6 or not same:
        raise SystemExit("STOP: the 26-station rebuild does not reproduce Step 7D. Nothing written.")

    # 2-3. stable network
    nets = pd.read_csv(NETWORKS, dtype={"station_uid": str})
    keep = nets.loc[(nets["network_id"] == STABLE_NETWORK) & as_bool(nets["included"]), "station_uid"]
    excluded = nets.loc[(nets["network_id"] == STABLE_NETWORK) & ~as_bool(nets["included"]), "station_name_display"].tolist()
    st25 = st[st["station_uid"].isin(keep)].reset_index(drop=True)
    w25 = build_weights(st25, boundary, proj)
    w25.to_csv(OUT_W, index=False, float_format="%.8f")
    daily25, cat25 = daily_and_spells(data, w25, cols, rules)
    cat25.to_csv(OUT_CAT, index=False, date_format="%Y-%m-%d")

    # 4a. comparison with the primary catalogue
    def classes(c):
        return {"spells": len(c), "spell_days": int(c["duration_days"].sum()),
                "short_3d": int((c["duration_days"] == 3).sum()),
                "intermediate_4_5d": int(c["duration_days"].between(4, 5).sum()),
                "persistent_ge6d": int((c["duration_days"] >= 6).sum()),
                "winters_with_spells": int(c["winter_start_year"].nunique())}
    days = lambda c: {d for r in c.itertuples() for d in pd.date_range(r.start_date, r.end_date)}
    dp, ds = days(prim), days(cat25)
    matched = sum(any(not (r.end_date < s.start_date or r.start_date > s.end_date) for s in cat25.itertuples())
                  for r in prim.itertuples())
    comp = pd.DataFrame([{"catalogue": "primary (26 stations)", **classes(prim)},
                         {"catalogue": "stable (25 stations)", **classes(cat25)}])
    overlap = {"excluded_station": excluded, "shared_spell_days": len(dp & ds),
               "primary_only_days": len(dp - ds), "stable_only_days": len(ds - dp),
               "primary_spells_overlapping_a_stable_spell": int(matched)}
    comp.to_csv(OUT_T / "table_R10_stable_network_catalogue.csv", index=False)

    # 4b. termination model on the stable catalogue
    cov = pd.read_csv(C.DAILY_COVARIATES, parse_dates=["date"]).set_index("date")
    rs = E.risk_set(cov, cat25, 3)
    rows = [{"predictor": p, **E.odds_ratio(rs, p, N_BOOT)} for p in PRIMARY if rs[p].notna().sum() >= 30]
    H = pd.DataFrame(rows)
    H["q_fdr"] = C.benjamini_hochberg(H["p_bootstrap"].to_numpy())
    H.to_csv(OUT_T / "table_R10b_stable_network_hazard.csv", index=False, float_format="%.4f")

    lines = ["Step 17B - stable-network (25-station) cold-night catalogue", "=" * 55,
             f"Validation: 26-station rebuild reproduces Step 7D weights (max diff {max_dw:.1e}) and all {len(prim)} spells",
             f"Excluded station(s): {excluded}", "", comp.to_string(index=False), "",
             json.dumps(overlap, indent=1), "", "Termination model on the stable catalogue (q < 0.10):",
             H[H["q_fdr"] < 0.10][["predictor", "odds_ratio_per_sd", "ci_low", "ci_high", "q_fdr"]].round(3).to_string(index=False)
             or "none", "Note: national covariates (e.g. Tmin anomaly) are the 26-station series."]
    (OUT_Q / "step17b_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
