# Step 19b: text, supplement and build scripts after the second pre-submission review (11 October 2026)

| File | Change |
| --- | --- |
| `09_submission/source/manuscript.md` | Wording fixes from the review (duration claims, count vs rate ratios, lag labels, terrain and p-value definitions, blocking definition), Alam et al. (2023), Step 19 results, verified DOIs, Park order; repository now cited as v1.4.0 with two `⟦NEW-DOI⟧` placeholders |
| `09_submission/source/online_resource_1.md` | New Text S3 (index definitions, pressure-level check), Text S1 (p-values, redraws), Table S12c (alternative catalogues), Table S18 (stations), caption fixes |
| `07_scripts/16h_build_online_resource_tables.py` | v4: S12c, S15 sign-flip column, S18, SDs in physical units, lag labels; writes `online_resource_1_tables_S10_S18.md` |
| `07_scripts/91_build_submission_docx.py` | Supplement tables: no split rows, no broken numbers, captions kept with tables |
| `07_scripts/93_build_cover_letter.py` | 18 tables, three notes, alternative definitions, v1.4.0 DOI placeholder |
| `07_scripts/94_check_reference_dois.py` | New: checks and completes reference DOIs with Crossref |
| `07_scripts/100_make_paper_repository.sh` | Version 1.4.0 |

## A. Rebuild (run from the project root)

```
cd /media/shajibul/Porag/Bangladesh_ColdWave_Analysis/ColdWave_Project/Bangladesh_Cold_Wave
source .venv/bin/activate
cp 09_submission/source/manuscript.md 09_submission/source/manuscript_v1.3.0.md
cp 09_submission/source/online_resource_1.md 09_submission/source/online_resource_1_v1.3.0.md
unzip -o ~/Downloads/step19b_text_revision.zip -d .
rm -f 08_outputs/online_resource_1/online_resource_1_tables_S10_S17.md
python 07_scripts/16h_build_online_resource_tables.py 2>&1 | tee 09_logs/step19_16h.log
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
python 07_scripts/94_check_reference_dois.py 2>&1 | tee 09_logs/step19_94.log
python 07_scripts/92_export_figures.py 2>&1 | tee 09_logs/step19_92.log
python 07_scripts/91_build_submission_docx.py 2>&1 | tee 09_logs/step19_91.log
python 07_scripts/93_build_cover_letter.py --date "11 October 2026"
```
Send `09_logs/step19_16h.log`, the table check line, `09_logs/step19_94.log` and `09_logs/step19_91.log`.

## B. Repository version 1.4.0 (after A has been checked)
Same steps as for v1.3.0: `bash 07_scripts/100_make_paper_repository.sh`, rsync dry run into the clone,
commit and push, GitHub release `v1.4.0`, then the Zenodo DOI goes into manuscript.md, 93 and 100.
