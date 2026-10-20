#!/usr/bin/env bash
# 100_make_paper_repository.sh  (version 1.5.0 of the paper repository)
# Build a LEAN repository containing only what reproduces the manuscript and its supplement.
# It COPIES an explicit whitelist; nothing in the working project is changed. BMD station data are excluded.
#
# Run from the project root:   bash 07_scripts/100_make_paper_repository.sh [output-folder]
# Default output: ~/bangladesh-cold-spells_v1.5.0  (a fresh folder; sync it into your git clone afterwards)
set -euo pipefail
VERSION="1.5.0"
# Zenodo DOI of THIS version. Zenodo assigns it only when the GitHub release is archived, so leave it empty for the
# release itself; afterwards set it here (or pass DOI=10.5281/zenodo.NNNN on the command line) and rerun to update
# README.md and CITATION.cff in a follow-up commit.
DOI="${DOI:-10.5281/zenodo.23495779}"
PREV_DOI="10.5281/zenodo.23321486"   # version 1.4.0
DOI_130="10.5281/zenodo.23282635"    # version 1.3.0
TITLE="Onset, maintenance and termination of Bangladesh winter cold spells: a subtropical trough pathway and the return of upstream warm advection"
PREV_TITLE="Onset and termination of Bangladesh winter cold spells and their association with polar vortex variability: evidence from 40 winters"
PROJECT="$(pwd)"
RELEASE="${1:-${HOME}/bangladesh-cold-spells_v${VERSION}}"
MAX_MB=50            # GitHub refuses files over 100 MB; anything above this is reported, not copied
[ -d "${PROJECT}/07_scripts" ] || { echo "Run from the project root."; exit 1; }
[ -e "${RELEASE}" ] && { echo "${RELEASE} exists. Remove it first: rm -rf ${RELEASE}"; exit 1; }
mkdir -p "${RELEASE}"
MISSING=(); TOOBIG=()
copy() { local f="$1"
  if [ -f "${PROJECT}/${f}" ]; then
    local mb; mb=$(( $(stat -c %s "${PROJECT}/${f}") / 1048576 ))
    if [ "${mb}" -gt "${MAX_MB}" ]; then TOOBIG+=("${f} (${mb} MB)"); return; fi
    mkdir -p "${RELEASE}/$(dirname "${f}")"; cp -p "${PROJECT}/${f}" "${RELEASE}/${f}"
  else MISSING+=("${f}"); fi; }
# copy every file under the project that matches a name pattern (keeps its relative path)
copy_find() { local dir="$1"; shift
  [ -d "${PROJECT}/${dir}" ] || { MISSING+=("${dir}/ ($*)"); return; }
  local n=0
  while IFS= read -r f; do copy "${f}"; n=$((n+1)); done < <(cd "${PROJECT}" && find "${dir}" -type f \( "$@" \) | sort)
  [ "${n}" -gt 0 ] || MISSING+=("${dir}/ ($*)"); }

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
 15x_validation_power_globalfdr.py
 16a_revision_hazard_validation.py 16b_revision_stable_network_catalogue.py 16c_revision_stratosphere.py
 16d_revision_humidity_gridded_lead.py 16e_revision_skin_soil.py 16f_final_stratosphere_figure.py
 16g_final_advection_pulse_figure.py 16h_build_online_resource_tables.py 16i_plot_lifecycle_schematic.py
 17a_check_step18_patches.py 17b_step19_checks.py 17c_step20_checks.py
 100_make_paper_repository.sh
)
for s in "${SCRIPTS[@]}"; do copy "07_scripts/${s}"; done

# ---------------------------------------------------------------- policies, schemas, dictionaries, checksums
(cd "${PROJECT}" && find 00_admin -maxdepth 1 -type f \( \
   -name "step1_*" -o -name "step2*" -o -name "step3*" -o -name "step4*" -o -name "step5*" -o -name "step6*" -o -name "step7*" \
   -o -name "step9a_*" -o -name "step9b_*" -o -name "step9c_*" -o -name "step9g_*" \
   -o -name "step10a_*" -o -name "step10d_*" -o -name "step10e_*" -o -name "step12_*" -o -name "step16*" -o -name "step17*" -o -name "step18*" -o -name "STEP18*" -o -name "step19*" -o -name "STEP19*" -o -name "step20*" -o -name "STEP20*" \
   -o -name "project_config.yaml" -o -name "requirements-lock.txt" -o -name "temperature_source_policy.md" \
   -o -name "raw_file_manifest.txt" -o -name "raw_file_sha256*.txt" -o -name "noaa_era5_major_ssw_reference_through_2023.csv" \) -print) \
 | while read -r f; do copy "${f}"; done || true

# ---------------------------------------------------------------- metadata, public inputs, catalogues
for f in step6a_fixed_station_network.csv step6c_analytical_station_network.csv step6c_sensitivity_network_membership.csv \
         step7d_primary_station_area_weights.csv step7d_primary_station_voronoi_weights.gpkg step7d_bangladesh_boundary.gpkg \
         step17b_stable_network_area_weights.csv; do
  copy "02_metadata/${f}"; done
for f in geospatial/geoBoundaries-BGD-ADM0.geojson geospatial/etopo1_south_asia_0p1deg.nc \
         climate_indices/oni.ascii.txt climate_indices/oni_monthly.csv; do copy "01_raw_data/${f}"; done
for f in step7e_frozen_primary_event_catalogue.csv step7e_frozen_primary_event_days.csv step7e_frozen_event_onsets_and_windows.csv \
         step7e_sensitivity_event_catalogue.csv step16f_cold_day_spell_catalogue.csv step17b_stable_network_event_catalogue.csv; do
  copy "06_events/${f}"; done

# ---------------------------------------------------------------- ERA5-derived series and gridded fields used in figures
for f in era5_box_daily/era5_box_daily_oct_mar.csv era5_box_daily/era5_mechanism_daily_oct_mar.csv \
         era5_box_daily/era5_cloud_radiation_daily_oct_mar.csv era5_box_daily/era5_cloud_radiation_windows_oct_mar.csv \
         era5_significance/netcdf/ERA5_SIG_z500_selected_lags.nc tn_waf/netcdf/ERA5_TN_WAF_250hPa_selected_lags.nc \
         termination/step16s_end_aligned_composites.nc; do copy "03_intermediate/${f}"; done
# Step 18 sensitivity series: previous (unmasked) box means and the 1500-m terrain threshold
copy_find 03_intermediate/era5_box_daily -name "*_unmasked.csv" -o -name "*_terrain1500.csv"
# Step 17 gridded caches (stratospheric pathway, winter fields, gridded leakage-free test)
copy_find 03_intermediate -name "step17*.nc"
rm -rf "${RELEASE}"/03_intermediate/*_backup*   # never ship the local v1.2.0 backups

# ---------------------------------------------------------------- result tables used by the paper and supplement
(cd "${PROJECT}" && find 08_outputs/tables/termination -type f -name "*.csv" -print) | while read -r f; do copy "${f}"; done || true
copy_find 08_outputs/tables/revision -name "*.csv"
for f in stratosphere/table_59_daily_stratospheric_indices.csv stratosphere/table_59b_stratospheric_calendar_day_climatology.csv \
         stratosphere/table_60_major_ssw_catalogue.csv stratosphere/table_60b_major_ssw_validation.csv \
         tropospheric_indices/table_22_daily_siberian_high_blocking_indices.csv rrwp/table_43_daily_rrwp_sector_metrics.csv; do
  copy "08_outputs/tables/${f}"; done

# ---------------------------------------------------------------- manuscript and Online Resource figures only
# Fig. 1-11 in manuscript order (version 1.5.0 numbering), then Figs. S2-S4 (Fig. S1 is written by 16h to
# 08_outputs/online_resource_1)
FIGS=(fig_final_01_domain_network fig_m2_spell_climatology fig_m8_cold_day_vs_cold_night fig_m3_onset_pathway_maps
      fig_m3b_onset_timeline fig_final_08_termination_sequence fig_final_R_advection_pulse fig_final_07_termination_maps
      fig_final_06_hazard_forest_three_families fig_final_R_stratosphere_occurrence fig_lifecycle_schematic
      fig_R7_gridded_leakage_free fig_m4_leakage_free_composites FigS4_stratosphere_pathway_maps)
for n in "${FIGS[@]}"; do
  for ext in png pdf; do
    f=$(cd "${PROJECT}" && find 08_outputs/figures -type f -name "${n}.${ext}" | sort | head -n 1)
    if [ -n "${f}" ]; then copy "${f}"; else MISSING+=("08_outputs/figures/**/${n}.${ext}"); fi
  done
done
for f in FigS1_hazard_robustness.png FigS1_hazard_robustness.pdf online_resource_1_tables_S1_S9.md online_resource_1_tables_S10_S18.md \
         online_resource_1_build_report.txt; do copy "08_outputs/online_resource_1/${f}"; done

# ---------------------------------------------------------------- QC and run summaries (no station values)
(cd "${PROJECT}" && find 05_qc_reports -maxdepth 1 -type f -name "step[2-7]*_summary.json" -print) | while read -r f; do copy "${f}"; done || true
copy 05_qc_reports/step6b_homogeneity_report.txt
(cd "${PROJECT}" && find 05_qc_reports/termination -type f \( -name "*.txt" -o -name "*.json" \) -print) | while read -r f; do copy "${f}"; done || true
copy_find 05_qc_reports/revision -name "*.txt" -o -name "*.json"
(cd "${PROJECT}" && find 05_qc_reports/era5 05_qc_reports/stratosphere 05_qc_reports/rrwp 05_qc_reports/tropospheric_indices 05_qc_reports/tn_waf \
   -type f -name "*summary.json" ! -path "*corrupted*" -print 2>/dev/null || true) | while read -r f; do copy "${f}"; done || true
(cd "${PROJECT}" && find 09_logs -maxdepth 1 -type f \( -name "step16*.log" -o -name "step17*.log" -o -name "step18*.log" -o -name "step19*.log" -o -name "step20*.log" -o -name "step24*.log" -o -name "step25*.log" \) -print) | while read -r f; do copy "${f}"; done || true
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
cat > "${RELEASE}/CITATION.cff" <<EOF
cff-version: 1.2.0
message: "If you use this code or the derived data, please cite the article and this archive."
title: "Code and derived data for: ${TITLE}"
version: ${VERSION}
date-released: "$(date +%Y-%m-%d)"
license: MIT
$( [ -n "${DOI}" ] && echo "doi: ${DOI}" )
repository-code: "https://github.com/shajibulanam/bangladesh-cold-spells"
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
cat > "${RELEASE}/.zenodo.json" <<EOF
{
  "title": "Code and derived data for: ${TITLE}",
  "version": "${VERSION}",
  "upload_type": "software",
  "description": "Analysis code, frozen analysis policies, cold-night and cold-day spell catalogues (including the 25-station sensitivity catalogue), ERA5-derived daily series and gridded fields, and all result tables and figures of the article and its Online Resource 1. Version 1.5.0 (Steps 24-25) follows the restructuring of the article around the cold-spell life cycle (new title): it adds the life-cycle schematic (script 16i, Fig. 11), splits the stratosphere figure into the occurrence panels (Fig. 10) and the circulation maps (Online Resource 1, Fig. S4) in script 16f, and renumbers the figures; no analysis or number changed. Earlier title of versions 1.3.0 and 1.4.0: Onset and termination of Bangladesh winter cold spells and their association with polar vortex variability: evidence from 40 winters. Version 1.4.0 (Steps 19-20, after the second and third pre-submission reviews) adds the hazard model without the final rows of the two season-end spells (table R18), checksums of the fixed external inputs, a check of pressure-level validity against ERA5 surface pressure, the upstream advection predictor in the six alternative event catalogues, SSW null redraw counts, predictor standard deviations in physical units (tables R14-R17), 10,000 resamples for the Fig. 4 stippling and the station table S18. Version 1.3.0 (Step 18): ERA5 box means exclude grid points above 1000 m (ETOPO1), where 1000/850 hPa and MSLP are extrapolated below ground (1500 m as sensitivity; unmasked series kept); resampling p-values are plus-one and every test uses at least 10,000 resamples; exact sign-flip tests for the post-SSW maps; season-end spells are right-censored in the primary hazard model; new article title. Version 1.2.0 added the Step 17 revision analyses (nested cross-validation, censoring and station-network sensitivity, lead predictors, stratospheric robustness and pathway, humidity and skin temperature, gridded field significance). BMD station observations are not included; they were obtained from the Bangladesh Meteorological Department on request and may not be redistributed.",
  "creators": [
    {"name": "Anam, Shajibul", "affiliation": "Department of Meteorology, University of Dhaka", "orcid": "0009-0009-9681-9339"},
    {"name": "Yildiz, Shabista", "affiliation": "Department of Meteorology, University of Dhaka", "orcid": "0009-0007-4131-5542"},
    {"name": "Singha, Shipa Rani", "affiliation": "Department of Meteorology, University of Dhaka", "orcid": "0009-0001-0209-1490"}
  ],
  "license": "mit",
  "keywords": ["cold spell", "Bangladesh", "subtropical jet", "temperature advection", "discrete-time hazard model",
               "sudden stratospheric warming", "Indo-Gangetic Plain", "ERA5"]
}
EOF
# checksums of the fixed external inputs
(cd "${RELEASE}" && sha256sum 01_raw_data/geospatial/geoBoundaries-BGD-ADM0.geojson 00_admin/step7d_geoboundaries_metadata.json \
   01_raw_data/climate_indices/oni.ascii.txt 01_raw_data/climate_indices/oni_monthly.csv \
   00_admin/noaa_era5_major_ssw_reference_through_2023.csv 01_raw_data/geospatial/etopo1_south_asia_0p1deg.nc \
   02_metadata/step7d_primary_station_area_weights.csv > 00_admin/fixed_inputs_sha256.txt) \
  || echo "WARNING: some fixed inputs missing for the checksum file"

cat > "${RELEASE}/README.md" <<EOF
# Bangladesh winter cold spells: onset, maintenance and termination (1985/86-2024/25)

$( [ -n "${DOI}" ] && echo "[![DOI](https://zenodo.org/badge/DOI/${DOI}.svg)](https://doi.org/${DOI})" )

Version ${VERSION}$( [ -n "${DOI}" ] && echo " (https://doi.org/${DOI})" ). Code and derived data reproducing the figures and tables of the article
"${TITLE}" (S. Anam, S. Yildiz, S. R. Singha; Department of Meteorology, University of Dhaka),
submitted to *Climate Dynamics*, and of its Online Resource 1. Corresponding author: S. Yildiz, shabistayildiz@du.ac.bd.

## Data availability
BMD daily station observations were obtained from the Bangladesh Meteorological Department on request and
may not be redistributed; they are not included. Scripts \`00\`-\`06\`, \`15b\`, \`15f\` and \`16b\` need them. All other
inputs are public (ERA5 via the Copernicus Climate Data Store and ERA5 ARCO; NOAA CPC Oceanic Nino Index; NOAA ETOPO1;
geoBoundaries; NOAA SSW Compendium dates). This repository includes the derived cold-spell catalogues, station
network metadata and Voronoi weights, ERA5-derived daily series and gridded fields, and every result table and figure
of the paper and its supplement.

## Fixed external inputs
SHA-256 checksums of the external files used are in \`00_admin/fixed_inputs_sha256.txt\`.
- National boundary: geoBoundaries ADM0 for Bangladesh (\`01_raw_data/geospatial/geoBoundaries-BGD-ADM0.geojson\`; boundary ID
  BGD-ADM0-71232402, build date 12 Dec 2023; full metadata in \`00_admin/step7d_geoboundaries_metadata.json\`). Voronoi cells are computed on a Bangladesh-centred
  Albers equal-area projection (+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=90 +datum=WGS84 +units=m; also in
  \`00_admin/step7d_area_weighted_event_policy.yaml\`); exact station
  weights (summing to 1) are in \`02_metadata/step7d_primary_station_area_weights.csv\`.
- Oceanic Nino Index: NOAA CPC \`oni.ascii.txt\` as retrieved for the analysis (\`01_raw_data/climate_indices/\`).
- Major SSW reference dates: NOAA CSL SSW Compendium, https://csl.noaa.gov/groups/csl8/sswcompendium/majorevents.html
  (ERA5 column through February 2023, accessed 12 October 2026), copied to \`00_admin/noaa_era5_major_ssw_reference_through_2023.csv\`.
- Terrain: ETOPO1 ice-surface relief, 0-55N, 45-125E, thinned to 0.1 deg (\`01_raw_data/geospatial/etopo1_south_asia_0p1deg.nc\`).
- ERA5 request settings (variables, levels, area, times) are in the \`00_admin/*policy*\` files of the download steps.

## Where each figure and table comes from
| Paper item | File | Produced by |
| --- | --- | --- |
| Fig. 1 | \`fig_final_01_domain_network\` | \`15r\` |
| Fig. 2 | \`fig_m2_spell_climatology\` | \`15g\` (tables 94, 99) |
| Fig. 3 | \`fig_m8_cold_day_vs_cold_night\` | \`15g\` (tables 101, 113, 115) |
| Fig. 4 | \`fig_m3_onset_pathway_maps\` | \`15i\` (inputs from \`08l\`, \`09n\`) |
| Fig. 5 | \`fig_m3b_onset_timeline\` | \`15l\` (table 103) |
| Fig. 6 | \`fig_final_08_termination_sequence\` | \`15u\` (tables 102, 112) |
| Fig. 7 | \`fig_final_R_advection_pulse\` | \`16g\` (tables R11, R11b) |
| Fig. 8 | \`fig_final_07_termination_maps\` | \`15s\` |
| Fig. 9 | \`fig_final_06_hazard_forest_three_families\` | \`15u\` (tables 91, 106, 114) |
| Fig. 10 | \`fig_final_R_stratosphere_occurrence\` | \`16f\` (tables 107-110, R4) |
| Fig. 11 | \`fig_lifecycle_schematic\` | \`16i\` (conceptual drawing; values quoted from the Results, no input files) |
| Terrain-mask checks | tables R12, R13 | \`17a\` |
| Step 19 checks; Table S12c, Text S3 | tables R14-R17 | \`17b\` |
| Step 20 check: final rows of the season-end spells omitted | table R18 | \`17c\` |
| Table S18 | station network and weights (\`02_metadata\`) | \`16h\` |
| Table 1 | tables 101, 113, 115 | \`15f\`, \`15q\`, \`15w\` |
| Table 2 | tables 107-109, R4 | \`15n\`, \`15o\`, \`16c\` |
| Fig. S1 | \`FigS1_hazard_robustness\` | \`16h\` (tables 95, R1, R2, R10b) |
| Fig. S2 | \`fig_R7_gridded_leakage_free\` | \`16d\` (table R7) |
| Fig. S3 | \`fig_m4_leakage_free_composites\` | \`15g\` (table 90) |
| Fig. S4 | \`FigS4_stratosphere_pathway_maps\` | \`16f\` (table R6b) |
| Tables S1-S9 | tables 60/60b, 91, 106, 114, 111, 95, 117, 89 + 118, 105 + 119 | \`11c\`, \`15d\`, \`15m\`, \`15w\`, \`15q\`, \`15e\`, \`15x\`, \`15c\`, \`15m\` |
| Table S10 | tables R1, R1b | \`16a\` |
| Table S11 | tables R10, R10b, R2, R18 | \`16b\`, \`16a\`, \`17c\` |
| Table S12 | tables R3, R11, R15, R18 | \`16a\`, \`16d\`, \`17b\`, \`17c\` |
| Table S13 | table R4 | \`16c\` |
| Table S14 | tables R8, R8b | \`16c\` |
| Table S15 | tables R6b, R6 | \`16f\`, \`16c\` |
| Table S16 | tables R5, R9c, R9 | \`16d\`, \`16e\` |
| Table S17 | table R7 | \`16d\` |

Tables 89-120 are in \`08_outputs/tables/termination\`, tables R1-R18 in \`08_outputs/tables/revision\`.
Tables S1-S18 are rendered from the revision tables by \`16h\` (\`08_outputs/online_resource_1\`), which also draws Fig. S1.

## Workflow
Script prefix *n* is analysis Step *n* + 1 (e.g. \`06c\` = Step 7C, \`15d\` = Step 16D, \`16a\` = Step 17A). Frozen settings for
every step are in \`00_admin/*policy*\`; output checksums in \`00_admin/*_sha256.txt\`; all random procedures use fixed seeds.
1. Station data (BMD, not included): ingestion \`01\`-\`02\`, quality control \`03\`-\`04\`, network and homogeneity \`05\`,
   thresholds and catalogues \`06\`.
2. ERA5 preparation \`08a\`-\`08i\`; onset significance \`08l\`; indices \`09b\` (Siberian High, blocking), \`09n\` (wave-activity
   flux), \`09t\` (recurrent Rossby wave metric R), \`11b\`/\`11c\` (stratospheric indices, SSWs).
3. Analysis (Step 16): \`15a, 15a2, 15b, 15c, 15d, 15e, 15f, 15h, 15j, 15k3, 15l, 15m, 15n, 15o, 15q, 15v, 15w, 15x\`;
   figures \`15g, 15i, 15r, 15s, 15t, 15u\`. Shared helpers: \`15_termination_common.py\`, \`11_stratosphere_common.py\`.
4. Revision analyses (Step 17): \`16a\` nested cross-validation, censoring at season boundaries and one-day leads of all
   predictors; \`16b\` catalogue rebuilt on 25 stations; \`16c\` robustness of the SSW tests, stratosphere and duration,
   pathway indices; \`16d\` absolute humidity, gridded field significance and lead advection; \`16e\` skin temperature;
   figures \`16f\` (Fig. 10 and Fig. S4) and \`16g\` (Fig. 7); Online Resource tables and Fig. S1 \`16h\`.
5. Step 18 (pre-submission review): terrain mask and plus-one p-values in the shared helpers and in \`15a\`, \`15b\`,
   \`15d\`, \`15e\`, \`15j\`, \`15m\`, \`15n\`, \`15o\`, \`15s\`, \`16a\`, \`16c\`, \`16d\`, \`16f\`, \`16g\`, \`08l\`; checks
   \`17a\` (fitter and terrain fractions, then masked versus unmasked odds ratios, tables R12-R13). Order of the rerun:
   \`00_admin/STEP18_RUNBOOK.md\`. The switch \`CENSOR_SEASON_END\` in \`15_termination_common.py\` selects the
   primary (censored) or the version 1.2.0 (not censored) risk set.
6. Steps 24-25 (article restructured around the life cycle): \`16i\` draws the schematic Fig. 11; \`16f\` writes Fig. 10 and
   Fig. S4 (and still the earlier seven-panel figure); \`16h\` table notes follow the new figure numbers.

## Changes in version ${VERSION}
The article was restructured around the cold-spell life cycle (onset, maintenance, termination) and has a new title;
the archive of versions 1.3.0 and 1.4.0 carries the earlier title "${PREV_TITLE}". New: \`16i\` (life-cycle schematic,
Fig. 11). \`16f\` also writes the occurrence panels as Fig. 10 and the circulation maps as Online Resource 1, Fig. S4, from
the same statistics and seeds; \`16h\` table notes refer to Figs. 7c and S4. Figures are renumbered (table above); former
Fig. 6 is Fig. S3. No analysis, table or number changed.

## Changes in version 1.4.0
Version 1.4.0: https://doi.org/${PREV_DOI}. Figure numbers in this section follow version 1.4.0.
Step 19 (second pre-submission review): \`17b\` checks pressure-level validity against ERA5 surface pressure (table R14,
R14b), refits the upstream advection predictor in the six alternative event catalogues (R15), counts SSW null redraws
(R16) and gives predictor standard deviations in physical units (R17); \`08l\` uses 10,000 map resamples (Fig. 4);
\`16h\` v4 adds Table S12c and the station table S18 and uses lag labels. Unchanged primary results.
Step 20 (third pre-submission review): \`17c\` refits the hazard model without the final rows of the two season-end
spells, whose outcome depends on the Tmin of 1 March (table R18; 258 rows, 83 terminations; no odds ratio changes by more
than 5%); \`16h\` v5 adds R18 to Tables S11 and S12, labels radiation accumulation intervals (05-17 and 17-05 LT) and
splits Tables S8 and S13 into panels; \`15u\` and \`16g\` relabel Figs. 7 and 11. Unchanged primary results.

## Changes in version 1.3.0
New article title. ERA5 box means of 1000/850-hPa fields and MSLP exclude grid points whose ETOPO1 cell-mean elevation
exceeds 1000 m (700 hPa: 2500 m); for 850-hPa advection the four neighbours of masked points are excluded too, and
thickness and T2m - T850 are differenced point by point. The previous series are kept as \`*_unmasked.csv\`, the
1500-m sensitivity as \`*_terrain1500.csv\`. Every bootstrap or Monte Carlo p-value is now 2(min(n-, n+) + 1)/(B + 1)
with B >= 10,000 for hypothesis tests (logistic refits are batched); the post-SSW maps use an exact sign-flip test over
19 winters. The two spells that end on the last day of February are right-censored in the primary hazard model.
Added \`17a\`, tables R12-R13 and Online Resource Tables S1-S9 as Markdown. All results were regenerated.
Version 1.3.0: https://doi.org/${DOI_130}.

## Development note
Analysis code and parts of the text were drafted with the assistance of AI language models (ChatGPT, OpenAI, for the
earlier pipeline steps; Claude, Anthropic, for Steps 16 to 25). The authors specified the analyses, reviewed and ran all
code, reproduced every reported number on their own computer with fixed random seeds, and are responsible for the results.

## Environment and licence
Python 3; exact package versions in \`requirements.txt\`. Maps need \`cartopy\`; ERA5 ARCO access needs
\`zarr fsspec aiohttp requests\` and a Copernicus key in \`~/.cdsapirc\`.
Code: MIT (\`LICENSE\`). Derived tables, catalogues and figures: CC BY 4.0. Third-party data keep their own terms.
EOF

echo "Repository folder: ${RELEASE}  (version ${VERSION})"
du -sh "${RELEASE}"; echo "Files: $(find "${RELEASE}" -type f | wc -l)"
echo "Step 18 files: series $(find "${RELEASE}/03_intermediate" -name '*_unmasked.csv' -o -name '*_terrain1500.csv' | wc -l), tables R12-R13 $(ls "${RELEASE}"/08_outputs/tables/revision/table_R1[23]_*.csv 2>/dev/null | wc -l), terrain $(ls "${RELEASE}"/01_raw_data/geospatial/etopo1_*.nc 2>/dev/null | wc -l)"
[ -n "${DOI}" ] || echo "NOTE: DOI empty (expected for the release itself). After Zenodo archives it, rerun with DOI=10.5281/zenodo.NNNN."
echo "Step 17 files: scripts $(ls "${RELEASE}"/07_scripts/16*.py 2>/dev/null | wc -l), revision tables $(ls "${RELEASE}"/08_outputs/tables/revision/*.csv 2>/dev/null | wc -l), gridded $(find "${RELEASE}/03_intermediate" -name 'step17*.nc' | wc -l)"
if [ ${#MISSING[@]} -gt 0 ]; then echo "Listed but not found (check these):"; printf '  %s\n' "${MISSING[@]}"; fi
if [ ${#TOOBIG[@]} -gt 0 ]; then echo "NOT copied, larger than ${MAX_MB} MB:"; printf '  %s\n' "${TOOBIG[@]}"; fi
echo "Privacy check (should be 'none found'):"
grep -rlE "analysis_tmin|tmin_raw|tmax_raw|analysis_tmax" "${RELEASE}" --include="*.csv" || echo "none found"
(cd "${RELEASE}" && find . -type f | sort > "${HOME}/bangladesh-cold-spells_v${VERSION}_file_list.txt")
echo "File list: ${HOME}/bangladesh-cold-spells_v${VERSION}_file_list.txt"
