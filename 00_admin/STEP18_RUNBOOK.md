# Step 18 revision: terrain mask and plus-one p-values (October 2026)

Run everything from the project root (the folder that contains `07_scripts`, `03_intermediate`, `08_outputs`).

## What changed

| Problem (pre-submission reviews) | Fix | Files |
| --- | --- | --- |
| About 40% of the NW IGP box and 30% of the South Asia box are high terrain where 850 hPa is below ground; ERA5 values there are extrapolated, and the box means had no mask. | Grid points whose ETOPO1 cell-mean elevation exceeds **1,000 m** are excluded from 1000/850-hPa and MSLP box means (700 hPa above 2,500 m). For advection, the four neighbours used by the centred differences are excluded too. Thickness and T2m − T850 are now differenced point by point over the same valid area. The **1,500 m** threshold is the pre-stated sensitivity. The unmasked series are kept as `*_unmasked.csv`. | `15_termination_common.py`, `15a`, `15j`, `15b` |
| Bootstrap p-values were `2·min(frac≤0, frac≥0)`, which is exactly 0 when no resample crosses zero. Benjamini–Hochberg then accepted every such test (for example "q < 0.001" for the NW IGP odds ratio, and the stippling in Fig. 8g). | Every bootstrap or Monte Carlo p is now plus-one: `2·(min(#≤0, #≥0) + 1)/(B + 1)`. Every hypothesis test uses at least **10,000** resamples (`C.MIN_TEST_BOOT`). | `15_termination_common.py`, `15d`, `15e`, `15m`, `15n`, `15o`, `15s`, `16c`, `16d`, `08l` |
| 10,000 sklearn refits per predictor would take hours. | `C.logit_fit_batch` fits all replicates at once. It uses the same unpenalised estimator as sklearn and matches the exact MLE to about 1e-6. It is more accurate than the old sklearn setting (tol 1e-4), so bootstrap CIs can move in the third decimal. | `15_termination_common.py`, `15d`, `15e`, `16d` |
| The 19-winter post-SSW maps (Fig. 8f/g, R6b) used a 1,000- or 2,000-resample bootstrap of 19 values. | Exact sign-flip test over all 2^19 sign patterns, then Benjamini–Hochberg. Fig. 8 boxes and fractions use valid (above-ground) cells only. | `16f`, `16c` |
| T850 and MSLP fields over Tibet entered the gridded FDR families (Fig. 10 maps, Fig. S2 / Table S17). | These fields are masked with the same 1,000 m rule before compositing and testing. | `15s`, `16d`, `16c`, `16f` |

The random draws are unchanged: the same seeds, and the first 1,000 or 2,000 replicates are identical to before. The new p-values can be larger, never smaller, than before.

## 0. Back up

```
cp -r 07_scripts 07_scripts_v1.2.0_backup
cp -r 08_outputs 08_outputs_v1.2.0_backup
cp -r 03_intermediate/era5_box_daily 03_intermediate/era5_box_daily_v1.2.0_backup
```

Then copy the patched scripts from this package into `07_scripts/` (overwrite), and this runbook into `00_admin/`.

## 1. Check before rerunning (about 1 minute)

```
python 07_scripts/17a_check_step18_patches.py --pre 2>&1 | tee 09_logs/step18a_pre.log
```

Expect `PASS` for the fitter. The terrain section should show roughly 1% for Bangladesh, 40% for NW IGP and 30% for South Asia.

## 2. Rebuild the ERA5 series (same run time as before)

```
python 07_scripts/15a_extract_era5_box_daily.py 2>&1 | tee 09_logs/step18_15a.log
python 07_scripts/15a_extract_era5_box_daily.py --terrain-max-m 1500 2>&1 | tee 09_logs/step18_15a_1500.log
python 07_scripts/15j_extract_advection_jet_daily.py 2>&1 | tee 09_logs/step18_15j.log
python 07_scripts/15j_extract_advection_jet_daily.py --terrain-max-m 1500 2>&1 | tee 09_logs/step18_15j_1500.log
python 07_scripts/15b_build_daily_covariates.py 2>&1 | tee 09_logs/step18_15b.log
```

## 3. Decision point: the masked odds ratio (about 5 minutes)

```
python 07_scripts/17a_check_step18_patches.py --post 2>&1 | tee 09_logs/step18a_post.log
```

Send `05_qc_reports/revision/step18a_post_checks.txt` to Claude before going on. It gives the following:
- the NW IGP and Bangladesh advection odds ratios (same day, lag 1 and lag 2) for the unmasked series, the 1,000 m mask and the 1,500 m mask (table R12);
- total advection and v850 around onset and termination, with the share of events that have warm advection or southerly flow (table R13).

The paper's framing depends on these numbers.

## 4. Rerun the analyses (in this order)

```
for s in 15c 15d 15e 15f 15h 15l 15m 15n 15o 15q 15w 15x 16a 16b 16e; do
  f=$(ls 07_scripts/${s}_*.py); echo "== $f"; python "$f" 2>&1 | tee 09_logs/step18_${s}.log
done
python 07_scripts/16c_revision_stratosphere.py --maps 2>&1 | tee 09_logs/step18_16c.log
python 07_scripts/16d_revision_humidity_gridded_lead.py 2>&1 | tee 09_logs/step18_16d.log
```

On Windows PowerShell, run the same scripts one by one in the order listed.

## 5. Figures and Online Resource

```
python 07_scripts/15g_plot_termination_figures.py
python 07_scripts/15s_termination_composite_maps.py
python 07_scripts/15u_plot_final_hazard_and_termination.py
python 07_scripts/16f_final_stratosphere_figure.py
python 07_scripts/16g_final_advection_pulse_figure.py
python 07_scripts/16h_build_online_resource_tables.py
```

Fig. 4 stippling (optional but consistent): `python 07_scripts/08l_compute_era5_bootstrap_significance.py`, then `python 07_scripts/15i_plot_onset_pathway_maps.py`. This keeps 1,000 map resamples and changes only the p formula.

## Do not rerun

- `15k3`, `15v`: these download ERA5 ARCO data again, and their inputs are unchanged.
- `08a`–`08i`, `09*`, `11*`: unchanged.
- `15t`, `15i`: unless step 5 says so.

## Afterwards

1. Send the logs and tables R1–R13 to Claude for the text revision (total versus anomaly wording, Brier skill, lead time, q-values, Table S17 and Fig. 8).
2. Then rebuild the repository release (v1.3.0) and the submission files.
