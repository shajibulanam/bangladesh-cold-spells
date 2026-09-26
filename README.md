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
3. Analysis (Step 16): `15a, 15b, 15c, 15d, 15e, 15f, 15h, 15j, 15k3, 15l, 15m, 15n, 15o, 15q, 15v, 15w, 15x`;
   figures `15g, 15i, 15r, 15s, 15t, 15u`. Shared helpers: `15_termination_common.py`, `11_stratosphere_common.py`.

## Development note
Analysis code and parts of the text were drafted with the assistance of AI language models (ChatGPT, OpenAI, for the
earlier pipeline steps; Claude, Anthropic, for Step 16). The authors specified the analyses, reviewed and ran all code,
reproduced every reported number on their own computer with fixed random seeds, and are responsible for the results.

## Environment and licence
Python 3; exact package versions in `requirements.txt`. Maps need `cartopy`; ERA5 ARCO access needs
`zarr fsspec aiohttp requests` and a Copernicus key in `~/.cdsapirc`.
Code: MIT (`LICENSE`). Derived tables, catalogues and figures: CC BY 4.0. Third-party data keep their own terms.
