#!/usr/bin/env bash
# 100_make_paper_repository.sh
# Build a LEAN repository containing only what reproduces the manuscript and its supplement.
# It COPIES an explicit whitelist; nothing in the working project is changed. BMD station data are excluded.
#
# Run from the project root:   bash 07_scripts/100_make_paper_repository.sh
# Output: ~/bangladesh-cold-spells
set -euo pipefail
PROJECT="$(pwd)"
RELEASE="${HOME}/bangladesh-cold-spells"
[ -d "${PROJECT}/07_scripts" ] || { echo "Run from the project root."; exit 1; }
[ -e "${RELEASE}" ] && { echo "${RELEASE} exists. Remove it first: rm -rf ${RELEASE}"; exit 1; }
mkdir -p "${RELEASE}"
MISSING=()
copy() { local f="$1"
  if [ -f "${PROJECT}/${f}" ]; then mkdir -p "${RELEASE}/$(dirname "${f}")"; cp -p "${PROJECT}/${f}" "${RELEASE}/${f}"
  else MISSING+=("${f}"); fi; }

# ---------------------------------------------------------------- scripts used by the paper
SCRIPTS=(
 00_verify_step1.py 01_build_station_master.py 01b_extend_station_identity_for_new_tmin.py
 02a_ingest_old_xlsx.py 02b_ingest_new_tmin_csv.py 02c_reconcile_tmin_overlap.py 02d_build_unified_temperature_dataset.py
 03a_first_pass_quality_control.py 03b_prepare_qc_decision_ledgers.py 03b2_auto_complete_qc_ledgers.py 03c_apply_qc_decisions.py
 04a_climatological_temporal_qc.py 04b_spatial_consistency_qc.py 04c_apply_spatial_qc_decisions.py
 05a_djf_completeness_fixed_network.py 05b_source_transition_homogeneity.py 05c_freeze_analytical_network.py
 06a_calculate_station_cold_thresholds.py 06b_apply_bmd_compound_cold_indicators.py 06c_build_primary_cold_spell_events.py
 06d_create_bangladesh_voronoi_weights.py 06e_build_area_weighted_cold_spell_catalogue.py 06f_run_event_sensitivity_and_freeze.py
 08a_inventory_era5_downloads.py 08b_extract_single_level_era5_zips.py 08c_validate_era5_analysis_ready.py
 08d_preprocess_era5_daily_fields.py 08e_build_era5_event_window_availability.py 08f_download_era5_oct_mar_extension.py
 08g_validate_era5_oct_mar_extension.py 08h_build_era5_calendar_day_climatology.py 08i_build_era5_event_lag_composites.py
 08l_compute_era5_bootstrap_significance.py download_era5_pl_2003_01_only.py
 09a_download_tropospheric_index_era5.py 09a_recover_missing_mslp_files.py 09a_recover_missing_z500_files.py
 09a_recover_z500_2024_12_nocache.py 09b_build_siberian_high_blocking_indices.py 09n_compute_tn_waf_diagnostics.py
 09s_download_rrwp_v250_global.py 09t_build_rrwp_r_metric.py
 11_stratosphere_common.py 11a_download_era5_stratosphere_daily.py 11b_build_stratospheric_indices.py 11c_detect_major_ssw_and_validate.py
 15_termination_common.py 15a_extract_era5_box_daily.py 15a2_check_tmax_source_overlap.py 15b_build_daily_covariates.py 15c_leakage_free_persistence_tests.py
 15d_discrete_time_hazard_model.py 15e_hazard_robustness_and_trend.py 15f_cold_day_spell_catalogue.py
 15g_plot_termination_figures.py 15h_end_aligned_bootstrap_bands.py 15i_plot_onset_pathway_maps.py
 15j_extract_advection_jet_daily.py 15k3_extract_cloud_radiation_arco.py 15l_onset_timeline.py
 15m_remote_driver_duration_tests.py 15n_stratosphere_ssw_power_nam.py 15o_enso_controls.py 15q_cloud_radiation_analysis.py
 15r_plot_domain_network_map.py 15s_termination_composite_maps.py 15t_plot_stratosphere_occurrence.py
 15u_plot_final_hazard_and_termination.py 15v_extract_arco_windows.py 15w_prenight_hazard_and_precipitation.py
 15x_validation_power_globalfdr.py 100_make_paper_repository.sh
)
for s in "${SCRIPTS[@]}"; do copy "07_scripts/${s}"; done

# ---------------------------------------------------------------- policies, schemas, dictionaries, checksums
(cd "${PROJECT}" && find 00_admin -maxdepth 1 -type f \( \
   -name "step1_*" -o -name "step2*" -o -name "step3*" -o -name "step4*" -o -name "step5*" -o -name "step6*" -o -name "step7*" \
   -o -name "step9a_*" -o -name "step9b_*" -o -name "step9c_*" -o -name "step9g_*" \
   -o -name "step10a_*" -o -name "step10d_*" -o -name "step10e_*" -o -name "step12_*" -o -name "step16*" \
   -o -name "project_config.yaml" -o -name "requirements-lock.txt" -o -name "temperature_source_policy.md" \
   -o -name "raw_file_manifest.txt" -o -name "raw_file_sha256*.txt" -o -name "noaa_era5_major_ssw_reference_through_2023.csv" \) -print) \
 | while read -r f; do copy "${f}"; done || true

# ---------------------------------------------------------------- metadata, public inputs, catalogues
for f in step6a_fixed_station_network.csv step6c_analytical_station_network.csv step6c_sensitivity_network_membership.csv \
         step7d_primary_station_area_weights.csv step7d_primary_station_voronoi_weights.gpkg step7d_bangladesh_boundary.gpkg; do
  copy "02_metadata/${f}"; done
for f in geospatial/geoBoundaries-BGD-ADM0.geojson geospatial/etopo1_south_asia_0p1deg.nc \
         climate_indices/oni.ascii.txt climate_indices/oni_monthly.csv; do copy "01_raw_data/${f}"; done
for f in step7e_frozen_primary_event_catalogue.csv step7e_frozen_primary_event_days.csv step7e_frozen_event_onsets_and_windows.csv \
         step7e_sensitivity_event_catalogue.csv step16f_cold_day_spell_catalogue.csv; do copy "06_events/${f}"; done

# ---------------------------------------------------------------- ERA5-derived series and gridded composites used in figures
for f in era5_box_daily/era5_box_daily_oct_mar.csv era5_box_daily/era5_mechanism_daily_oct_mar.csv \
         era5_box_daily/era5_cloud_radiation_daily_oct_mar.csv era5_box_daily/era5_cloud_radiation_windows_oct_mar.csv \
         era5_significance/netcdf/ERA5_SIG_z500_selected_lags.nc tn_waf/netcdf/ERA5_TN_WAF_250hPa_selected_lags.nc \
         termination/step16s_end_aligned_composites.nc; do copy "03_intermediate/${f}"; done

# ---------------------------------------------------------------- result tables used by the paper and supplement
(cd "${PROJECT}" && find 08_outputs/tables/termination -type f -name "*.csv" -print) | while read -r f; do copy "${f}"; done || true
for f in stratosphere/table_59_daily_stratospheric_indices.csv stratosphere/table_59b_stratospheric_calendar_day_climatology.csv \
         stratosphere/table_60_major_ssw_catalogue.csv stratosphere/table_60b_major_ssw_validation.csv \
         tropospheric_indices/table_22_daily_siberian_high_blocking_indices.csv rrwp/table_43_daily_rrwp_sector_metrics.csv; do
  copy "08_outputs/tables/${f}"; done

# ---------------------------------------------------------------- manuscript figures only
for n in fig_final_01_domain_network fig_m2_spell_climatology fig_m3_onset_pathway_maps fig_m3b_onset_timeline \
         fig_m4_leakage_free_composites fig_final_06_hazard_forest_three_families fig_final_07_termination_maps \
         fig_final_08_termination_sequence fig_final_09_stratosphere_occurrence fig_m8_cold_day_vs_cold_night fig_m6_hazard_robustness; do
  copy "08_outputs/figures/termination/${n}.png"; copy "08_outputs/figures/termination/${n}.pdf"; done

# ---------------------------------------------------------------- QC and run summaries (no station values)
(cd "${PROJECT}" && find 05_qc_reports -maxdepth 1 -type f -name "step[2-7]*_summary.json" -print) | while read -r f; do copy "${f}"; done || true
copy 05_qc_reports/step6b_homogeneity_report.txt
(cd "${PROJECT}" && find 05_qc_reports/termination -type f \( -name "*.txt" -o -name "*.json" \) -print) | while read -r f; do copy "${f}"; done || true
(cd "${PROJECT}" && find 05_qc_reports/era5 05_qc_reports/stratosphere 05_qc_reports/rrwp 05_qc_reports/tropospheric_indices 05_qc_reports/tn_waf \
   -type f -name "*summary.json" ! -path "*corrupted*" -print 2>/dev/null || true) | while read -r f; do copy "${f}"; done || true
(cd "${PROJECT}" && find 09_logs -maxdepth 1 -type f -name "step16*.log" -print) | while read -r f; do copy "${f}"; done || true
rm -f "${RELEASE}/00_admin/step16k_download_cloud_radiation_policy.json" "${RELEASE}/09_logs/step16k_download_cloud_radiation.log"

# ---------------------------------------------------------------- repository files
cp -p "${PROJECT}/00_admin/requirements-lock.txt" "${RELEASE}/requirements.txt" 2>/dev/null || true
cat > "${RELEASE}/.gitignore" <<'EOF'
__pycache__/
*.pyc
# BMD station data must never be committed
01_raw_data/*.xlsx
01_raw_data/*_raw.csv
03_intermediate/*.parquet
04_clean_data/
EOF
cat > "${RELEASE}/LICENSE" <<'EOF'
MIT License

Copyright (c) 2026 Shajibul Anam, Shabista Yildiz, Shipa Rani Singha

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
EOF
cat > "${RELEASE}/CITATION.cff" <<'EOF'
cff-version: 1.2.0
message: "If you use this code or the derived data, please cite the article and this archive."
title: "Code and derived data for: Do stratospheric and recurrent Rossby-wave precursors control persistent cold spells over Bangladesh? Evidence from 40 winters, 1985/86-2024/25"
version: 1.1.0
license: MIT
authors:
  - family-names: Anam
    given-names: Shajibul
    affiliation: "Department of Meteorology, University of Dhaka"
    orcid: "https://orcid.org/0009-0009-9681-9339"
  - family-names: Yildiz
    given-names: Shabista
    affiliation: "Department of Meteorology, University of Dhaka"
    orcid: "https://orcid.org/0009-0007-4131-5542"
  - family-names: Singha
    given-names: Shipa Rani
    affiliation: "Department of Meteorology, University of Dhaka"
    orcid: "https://orcid.org/0009-0001-0209-1490"
EOF
cat > "${RELEASE}/README.md" <<'EOF'
# Bangladesh winter cold spells: onset, persistence, termination and stratospheric modulation (1985/86-2024/25)

Code and derived data reproducing the figures and tables of the article
"Do stratospheric and recurrent Rossby-wave precursors control persistent cold spells over Bangladesh?
Evidence from 40 winters, 1985/86-2024/25" (S. Anam, S. Yildiz, S. R. Singha; Department of Meteorology,
University of Dhaka). Corresponding author: S. Yildiz, shabistayildiz@du.ac.bd.

## Data availability
BMD daily station observations were obtained from the Bangladesh Meteorological Department on request and
may not be redistributed; they are not included. Scripts `00`-`06` and `15b`/`15f` need them. All other inputs
are public (ERA5 via the Copernicus Climate Data Store and ERA5 ARCO; NOAA CPC Oceanic Nino Index; NOAA ETOPO1;
geoBoundaries; NOAA SSW Compendium dates). This repository includes the derived cold-spell catalogues, station
network metadata and Voronoi weights, ERA5-derived daily series, and every result table and figure of the paper.

## Where each figure and table comes from
| Paper item | File | Produced by |
| --- | --- | --- |
| Fig. 1 | `fig_final_01_domain_network` | `15r` |
| Fig. 2 | `fig_m2_spell_climatology` | `15g` (tables 94, 99) |
| Fig. 3 | `fig_m3_onset_pathway_maps` | `15i` (inputs from `08l`, `09n`) |
| Fig. 4 | `fig_m3b_onset_timeline` | `15l` (table 103) |
| Fig. 5 | `fig_m4_leakage_free_composites` | `15g` (table 90) |
| Fig. 6 | `fig_final_06_hazard_forest_three_families` | `15u` (tables 91, 106, 114) |
| Fig. 7 | `fig_final_07_termination_maps` | `15s` |
| Fig. 8 | `fig_final_08_termination_sequence` | `15u` (tables 102, 112) |
| Fig. 9 | `fig_final_09_stratosphere_occurrence` | `15t` (tables 107-110) |
| Fig. 10 | `fig_m8_cold_day_vs_cold_night` | `15g` (table 101) |
| Fig. S1 | `fig_m6_hazard_robustness` | `15g` (tables 95, 98) |
| Table 1 | tables 101, 113, 115 | `15f`, `15q`, `15w` |
| Table 2 | tables 107, 108, 109 | `15n`, `15o` |
| Tables S1-S9 | tables 60/60b, 91, 106, 114, 111, 95, 117, 89 + 118, 105 + 119 | `11c`, `15d`, `15m`, `15w`, `15q`, `15e`, `15x`, `15c`, `15m` |
Tables 89-120 are in `08_outputs/tables/termination`; figures in `08_outputs/figures/termination`.

## Workflow
Script prefix *n* is analysis Step *n* + 1 (e.g. `06c` = Step 7C, `15d` = Step 16D). Frozen settings for every step
are in `00_admin/*policy*`; output checksums in `00_admin/*_sha256.txt`; all random procedures use fixed seeds.
1. Station data (BMD, not included): ingestion `01`-`02`, quality control `03`-`04`, network and homogeneity `05`,
   thresholds and catalogues `06`.
2. ERA5 preparation `08a`-`08i`; onset significance `08l`; indices `09b` (Siberian High, blocking), `09n` (wave-activity
   flux), `09t` (recurrent Rossby wave metric R), `11b`/`11c` (stratospheric indices, SSWs).
3. Analysis (Step 16): `15a, 15a2, 15b, 15c, 15d, 15e, 15f, 15h, 15j, 15k3, 15l, 15m, 15n, 15o, 15q, 15v, 15w, 15x`;
   figures `15g, 15i, 15r, 15s, 15t, 15u`. Shared helpers: `15_termination_common.py`, `11_stratosphere_common.py`.

## Development note
Analysis code and parts of the text were drafted with the assistance of AI language models (ChatGPT, OpenAI, for the
earlier pipeline steps; Claude, Anthropic, for Step 16). The authors specified the analyses, reviewed and ran all code,
reproduced every reported number on their own computer with fixed random seeds, and are responsible for the results.

## Environment and licence
Python 3; exact package versions in `requirements.txt`. Maps need `cartopy`; ERA5 ARCO access needs
`zarr fsspec aiohttp requests` and a Copernicus key in `~/.cdsapirc`.
Code: MIT (`LICENSE`). Derived tables, catalogues and figures: CC BY 4.0. Third-party data keep their own terms.
EOF

echo "Repository folder: ${RELEASE}"
du -sh "${RELEASE}"; echo "Files: $(find "${RELEASE}" -type f | wc -l)"
if [ ${#MISSING[@]} -gt 0 ]; then echo "Listed but not found (check these):"; printf '  %s\n' "${MISSING[@]}"; fi
echo "Privacy check (should be 'none found'):"
grep -rlE "analysis_tmin|tmin_raw|tmax_raw|analysis_tmax" "${RELEASE}" --include="*.csv" || echo "none found"
(cd "${RELEASE}" && find . -type f | sort > "${HOME}/bangladesh-cold-spells_file_list.txt")
echo "File list: ${HOME}/bangladesh-cold-spells_file_list.txt"
