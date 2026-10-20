# Bangladesh winter cold spells: onset, maintenance and termination (1985/86-2024/25)



Version 1.5.0. Code and derived data reproducing the figures and tables of the article
"Onset, maintenance and termination of Bangladesh winter cold spells: a subtropical trough pathway and the return of upstream warm advection" (S. Anam, S. Yildiz, S. R. Singha; Department of Meteorology, University of Dhaka),
submitted to *Climate Dynamics*, and of its Online Resource 1. Corresponding author: S. Yildiz, shabistayildiz@du.ac.bd.

## Data availability
BMD daily station observations were obtained from the Bangladesh Meteorological Department on request and
may not be redistributed; they are not included. Scripts `00`-`06`, `15b`, `15f` and `16b` need them. All other
inputs are public (ERA5 via the Copernicus Climate Data Store and ERA5 ARCO; NOAA CPC Oceanic Nino Index; NOAA ETOPO1;
geoBoundaries; NOAA SSW Compendium dates). This repository includes the derived cold-spell catalogues, station
network metadata and Voronoi weights, ERA5-derived daily series and gridded fields, and every result table and figure
of the paper and its supplement.

## Fixed external inputs
SHA-256 checksums of the external files used are in `00_admin/fixed_inputs_sha256.txt`.
- National boundary: geoBoundaries ADM0 for Bangladesh (`01_raw_data/geospatial/geoBoundaries-BGD-ADM0.geojson`; boundary ID
  BGD-ADM0-71232402, build date 12 Dec 2023; full metadata in `00_admin/step7d_geoboundaries_metadata.json`). Voronoi cells are computed on a Bangladesh-centred
  Albers equal-area projection (+proj=aea +lat_1=21 +lat_2=26 +lat_0=23.5 +lon_0=90 +datum=WGS84 +units=m; also in
  `00_admin/step7d_area_weighted_event_policy.yaml`); exact station
  weights (summing to 1) are in `02_metadata/step7d_primary_station_area_weights.csv`.
- Oceanic Nino Index: NOAA CPC `oni.ascii.txt` as retrieved for the analysis (`01_raw_data/climate_indices/`).
- Major SSW reference dates: NOAA CSL SSW Compendium, https://csl.noaa.gov/groups/csl8/sswcompendium/majorevents.html
  (ERA5 column through February 2023, accessed 12 October 2026), copied to `00_admin/noaa_era5_major_ssw_reference_through_2023.csv`.
- Terrain: ETOPO1 ice-surface relief, 0-55N, 45-125E, thinned to 0.1 deg (`01_raw_data/geospatial/etopo1_south_asia_0p1deg.nc`).
- ERA5 request settings (variables, levels, area, times) are in the `00_admin/*policy*` files of the download steps.

## Where each figure and table comes from
| Paper item | File | Produced by |
| --- | --- | --- |
| Fig. 1 | `fig_final_01_domain_network` | `15r` |
| Fig. 2 | `fig_m2_spell_climatology` | `15g` (tables 94, 99) |
| Fig. 3 | `fig_m8_cold_day_vs_cold_night` | `15g` (tables 101, 113, 115) |
| Fig. 4 | `fig_m3_onset_pathway_maps` | `15i` (inputs from `08l`, `09n`) |
| Fig. 5 | `fig_m3b_onset_timeline` | `15l` (table 103) |
| Fig. 6 | `fig_final_08_termination_sequence` | `15u` (tables 102, 112) |
| Fig. 7 | `fig_final_R_advection_pulse` | `16g` (tables R11, R11b) |
| Fig. 8 | `fig_final_07_termination_maps` | `15s` |
| Fig. 9 | `fig_final_06_hazard_forest_three_families` | `15u` (tables 91, 106, 114) |
| Fig. 10 | `fig_final_R_stratosphere_occurrence` | `16f` (tables 107-110, R4) |
| Fig. 11 | `fig_lifecycle_schematic` | `16i` (conceptual drawing; values quoted from the Results, no input files) |
| Terrain-mask checks | tables R12, R13 | `17a` |
| Step 19 checks; Table S12c, Text S3 | tables R14-R17 | `17b` |
| Step 20 check: final rows of the season-end spells omitted | table R18 | `17c` |
| Table S18 | station network and weights (`02_metadata`) | `16h` |
| Table 1 | tables 101, 113, 115 | `15f`, `15q`, `15w` |
| Table 2 | tables 107-109, R4 | `15n`, `15o`, `16c` |
| Fig. S1 | `FigS1_hazard_robustness` | `16h` (tables 95, R1, R2, R10b) |
| Fig. S2 | `fig_R7_gridded_leakage_free` | `16d` (table R7) |
| Fig. S3 | `fig_m4_leakage_free_composites` | `15g` (table 90) |
| Fig. S4 | `FigS4_stratosphere_pathway_maps` | `16f` (table R6b) |
| Tables S1-S9 | tables 60/60b, 91, 106, 114, 111, 95, 117, 89 + 118, 105 + 119 | `11c`, `15d`, `15m`, `15w`, `15q`, `15e`, `15x`, `15c`, `15m` |
| Table S10 | tables R1, R1b | `16a` |
| Table S11 | tables R10, R10b, R2, R18 | `16b`, `16a`, `17c` |
| Table S12 | tables R3, R11, R15, R18 | `16a`, `16d`, `17b`, `17c` |
| Table S13 | table R4 | `16c` |
| Table S14 | tables R8, R8b | `16c` |
| Table S15 | tables R6b, R6 | `16f`, `16c` |
| Table S16 | tables R5, R9c, R9 | `16d`, `16e` |
| Table S17 | table R7 | `16d` |

Tables 89-120 are in `08_outputs/tables/termination`, tables R1-R18 in `08_outputs/tables/revision`.
Tables S1-S18 are rendered from the revision tables by `16h` (`08_outputs/online_resource_1`), which also draws Fig. S1.

## Workflow
Script prefix *n* is analysis Step *n* + 1 (e.g. `06c` = Step 7C, `15d` = Step 16D, `16a` = Step 17A). Frozen settings for
every step are in `00_admin/*policy*`; output checksums in `00_admin/*_sha256.txt`; all random procedures use fixed seeds.
1. Station data (BMD, not included): ingestion `01`-`02`, quality control `03`-`04`, network and homogeneity `05`,
   thresholds and catalogues `06`.
2. ERA5 preparation `08a`-`08i`; onset significance `08l`; indices `09b` (Siberian High, blocking), `09n` (wave-activity
   flux), `09t` (recurrent Rossby wave metric R), `11b`/`11c` (stratospheric indices, SSWs).
3. Analysis (Step 16): `15a, 15a2, 15b, 15c, 15d, 15e, 15f, 15h, 15j, 15k3, 15l, 15m, 15n, 15o, 15q, 15v, 15w, 15x`;
   figures `15g, 15i, 15r, 15s, 15t, 15u`. Shared helpers: `15_termination_common.py`, `11_stratosphere_common.py`.
4. Revision analyses (Step 17): `16a` nested cross-validation, censoring at season boundaries and one-day leads of all
   predictors; `16b` catalogue rebuilt on 25 stations; `16c` robustness of the SSW tests, stratosphere and duration,
   pathway indices; `16d` absolute humidity, gridded field significance and lead advection; `16e` skin temperature;
   figures `16f` (Fig. 10 and Fig. S4) and `16g` (Fig. 7); Online Resource tables and Fig. S1 `16h`.
5. Step 18 (pre-submission review): terrain mask and plus-one p-values in the shared helpers and in `15a`, `15b`,
   `15d`, `15e`, `15j`, `15m`, `15n`, `15o`, `15s`, `16a`, `16c`, `16d`, `16f`, `16g`, `08l`; checks
   `17a` (fitter and terrain fractions, then masked versus unmasked odds ratios, tables R12-R13). Order of the rerun:
   `00_admin/STEP18_RUNBOOK.md`. The switch `CENSOR_SEASON_END` in `15_termination_common.py` selects the
   primary (censored) or the version 1.2.0 (not censored) risk set.
6. Steps 24-25 (article restructured around the life cycle): `16i` draws the schematic Fig. 11; `16f` writes Fig. 10 and
   Fig. S4 (and still the earlier seven-panel figure); `16h` table notes follow the new figure numbers.

## Changes in version 1.5.0
The article was restructured around the cold-spell life cycle (onset, maintenance, termination) and has a new title;
the archive of versions 1.3.0 and 1.4.0 carries the earlier title "Onset and termination of Bangladesh winter cold spells and their association with polar vortex variability: evidence from 40 winters". New: `16i` (life-cycle schematic,
Fig. 11). `16f` also writes the occurrence panels as Fig. 10 and the circulation maps as Online Resource 1, Fig. S4, from
the same statistics and seeds; `16h` table notes refer to Figs. 7c and S4. Figures are renumbered (table above); former
Fig. 6 is Fig. S3. No analysis, table or number changed.

## Changes in version 1.4.0
Version 1.4.0: https://doi.org/10.5281/zenodo.23321486. Figure numbers in this section follow version 1.4.0.
Step 19 (second pre-submission review): `17b` checks pressure-level validity against ERA5 surface pressure (table R14,
R14b), refits the upstream advection predictor in the six alternative event catalogues (R15), counts SSW null redraws
(R16) and gives predictor standard deviations in physical units (R17); `08l` uses 10,000 map resamples (Fig. 4);
`16h` v4 adds Table S12c and the station table S18 and uses lag labels. Unchanged primary results.
Step 20 (third pre-submission review): `17c` refits the hazard model without the final rows of the two season-end
spells, whose outcome depends on the Tmin of 1 March (table R18; 258 rows, 83 terminations; no odds ratio changes by more
than 5%); `16h` v5 adds R18 to Tables S11 and S12, labels radiation accumulation intervals (05-17 and 17-05 LT) and
splits Tables S8 and S13 into panels; `15u` and `16g` relabel Figs. 7 and 11. Unchanged primary results.

## Changes in version 1.3.0
New article title. ERA5 box means of 1000/850-hPa fields and MSLP exclude grid points whose ETOPO1 cell-mean elevation
exceeds 1000 m (700 hPa: 2500 m); for 850-hPa advection the four neighbours of masked points are excluded too, and
thickness and T2m - T850 are differenced point by point. The previous series are kept as `*_unmasked.csv`, the
1500-m sensitivity as `*_terrain1500.csv`. Every bootstrap or Monte Carlo p-value is now 2(min(n-, n+) + 1)/(B + 1)
with B >= 10,000 for hypothesis tests (logistic refits are batched); the post-SSW maps use an exact sign-flip test over
19 winters. The two spells that end on the last day of February are right-censored in the primary hazard model.
Added `17a`, tables R12-R13 and Online Resource Tables S1-S9 as Markdown. All results were regenerated.
Version 1.3.0: https://doi.org/10.5281/zenodo.23282635.

## Development note
Analysis code and parts of the text were drafted with the assistance of AI language models (ChatGPT, OpenAI, for the
earlier pipeline steps; Claude, Anthropic, for Steps 16 to 25). The authors specified the analyses, reviewed and ran all
code, reproduced every reported number on their own computer with fixed random seeds, and are responsible for the results.

## Environment and licence
Python 3; exact package versions in `requirements.txt`. Maps need `cartopy`; ERA5 ARCO access needs
`zarr fsspec aiohttp requests` and a Copernicus key in `~/.cdsapirc`.
Code: MIT (`LICENSE`). Derived tables, catalogues and figures: CC BY 4.0. Third-party data keep their own terms.
