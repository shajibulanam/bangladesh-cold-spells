# Step 19a: checks for the second pre-submission review (11 October 2026)

Run from the project root. Nothing here changes the primary results; it adds checks and makes Fig. 4 use 10,000 resamples.

```
cd /media/shajibul/Porag/Bangladesh_ColdWave_Analysis/ColdWave_Project/Bangladesh_Cold_Wave
source .venv/bin/activate
unzip -o ~/Downloads/step19a_checks.zip -d .

# 1. Fig. 4 with 10,000 map resamples (about 10 times the previous run time)
python 07_scripts/08l_compute_era5_bootstrap_significance.py 2>&1 | tee 09_logs/step19_08l.log
python 07_scripts/15i_plot_onset_pathway_maps.py 2>&1 | tee 09_logs/step19_15i.log

# 2. Step 19 checks: surface pressure at each level, upstream advection in the six alternative
#    catalogues, SSW redraw counts, predictor SDs in physical units (10-30 minutes)
python 07_scripts/17b_step19_checks.py 2>&1 | tee 09_logs/step19_17b.log

# 3. Collect the results
zip -j ~/step19a_results.zip 05_qc_reports/revision/step19a_checks.txt \
    08_outputs/tables/revision/table_R1[4-7]*.csv \
    08_outputs/tables/era5/table_14_era5_gridpoint_significant_area_fraction.csv \
    08_outputs/figures/termination/fig_m3_onset_pathway_maps.png \
    09_logs/step19_08l.log 09_logs/step19_17b.log
```
Send `~/step19a_results.zip`.
