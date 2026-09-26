# Step 4A First-Pass QC Data Dictionary

## Dataset status

The Step 4A dataset contains the same preliminary temperature
values as Step 3D plus non-destructive quality-control flags.

No value has been corrected, replaced or deleted.

## Working temperature fields

- `tmin_qc_stage1`: unchanged copy of preliminary Tmin
- `tmax_qc_stage1`: unchanged copy of preliminary Tmax
- `diurnal_temperature_range`: Tmax minus Tmin

## Missingness

- `qc_tmin_missing`
- `qc_tmax_missing`
- `qc_tmax_missing_expected_2025`
- `qc_tmax_missing_unexpected`

## Zero and physical-range flags

- `qc_tmin_zero`
- `qc_tmin_zero_winter_priority`
- `qc_tmin_physical_range`
- `qc_tmax_physical_range`

Physical screening limits are broad plausibility limits and do
not define climatological extremes.

## Internal consistency

- `qc_tmax_lt_tmin`
- `qc_dtr_zero`
- `qc_dtr_high`

## Source and metadata flags

- `qc_metadata_pending`
- `qc_old_duplicate`
- `qc_old_duplicate_conflict`
- `qc_source_numeric_conflict`
- `qc_source_duplicate_conflict`
- `qc_inherited_value_review`

## Combined record fields

- `qc_stage1_value_review_required`
- `qc_stage1_priority`
- `qc_stage1_status`
- `qc_flag_codes`
- `qc_flag_count`

## Priority meanings

- `critical`: physical inconsistency or unresolved conflicting duplicate
- `high`: winter zero or source disagreement
- `review`: suspicious but not necessarily incorrect
- `metadata_pending`: station metadata remains incomplete
- `none`: no first-pass value-review flag

## Important limitation

Step 4A flags observations only. Decisions to keep, correct,
replace or reject observations must be made in Step 4B using
documented evidence.
