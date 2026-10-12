# Step 20b: final text package after the third pre-submission review (12 October 2026)

| File | Change |
| --- | --- |
| `09_submission/source/manuscript.md` | Abstract advection wording and validation scope (249 words); Introduction names the checks for each analysis; Sect. 2.7 explains the final season-end row and gives the Step 20 numbers (258 rows), the information times of Tmin, Tmax and the lagged model, and the bootstrap fit record; flux accumulation intervals (05–17 / 17–05 LT) in Sects. 2.9–2.10, Table 1, Figs. 7 and 9; headings 3.3, 4.2, 4.3 qualified; Sect. 3.5 gives the backward split-period Brier result and the validation scope; 4.1 no longer equates percentile spells with operational cold waves; 4.2 and 4.3 conditional wording; Conclusions 1, 2 and 4; Table 2 count vs rate ratios; captions of Figs. 3, 5, 7, 9, 11 |
| `09_submission/source/online_resource_1.md` | Text S1: count multiplier, one-sided power, bootstrap and Monte Carlo assumptions, fit record; Text S2: flux intervals; tables from `16h` v5 (R18 in Tables S11 and S12) |
| `07_scripts/15u_plot_final_hazard_and_termination.py` | Fig. 7 panel (a) title on two lines (no longer overlaps panel b) |
| `07_scripts/16g_final_advection_pulse_figure.py` | Fig. 11c: rows labelled "primary"; stray "3×10⁰" tick removed; wider panel gap |
| `07_scripts/16h_build_online_resource_tables.py` | Table S11 caption grammar |
| `07_scripts/93_build_cover_letter.py` | Same wording fixes as the abstract (advection, validation scope, "association") |
| `07_scripts/100_make_paper_repository.sh` | Includes `17c`, R18 and Step 20 logs; README section "Fixed external inputs" (boundary ID, projection, NOAA URL) and `00_admin/fixed_inputs_sha256.txt` |

## Commands (from the project root)

```
cd /media/shajibul/Porag/Bangladesh_ColdWave_Analysis/ColdWave_Project/Bangladesh_Cold_Wave
source .venv/bin/activate
cp 09_submission/source/manuscript.md 09_submission/source/manuscript_step19b.md
cp 09_submission/source/online_resource_1.md 09_submission/source/online_resource_1_step19b.md
unzip -o ~/Downloads/step20b_text_revision.zip -d .
python 07_scripts/15u_plot_final_hazard_and_termination.py 2>&1 | tee 09_logs/step20_15u.log
python 07_scripts/16g_final_advection_pulse_figure.py 2>&1 | tee 09_logs/step20_16g.log
python 07_scripts/16h_build_online_resource_tables.py 2>&1 | tee 09_logs/step20_16h.log
python - <<'PY'
from pathlib import Path
ms = Path("09_submission/source/online_resource_1.md").read_text(encoding="utf-8")
bad = 0
for f in ("online_resource_1_tables_S1_S9.md", "online_resource_1_tables_S10_S18.md"):
    for line in Path("08_outputs/online_resource_1", f).read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and line not in ms:
            bad += 1; print("DIFFERS:", line[:120])
print("OK: all table rows match" if bad == 0 else f"{bad} table rows differ")
PY
python 07_scripts/92_export_figures.py 2>&1 | tee 09_logs/step20_92.log
python 07_scripts/91_build_submission_docx.py 2>&1 | tee 09_logs/step20_91.log
python 07_scripts/93_build_cover_letter.py --date "12 October 2026"
```

Send: the table-check line, `09_logs/step20_91.log`, and a quick look at Figs. 7 and 11 in the new manuscript Word file.
Then repository v1.4.0: `bash 07_scripts/100_make_paper_repository.sh`, rsync dry run, commit, push, release, DOI.
