# Step 3C Tmin Overlap Reconciliation Data Dictionary

## Key fields

- `station_uid`: permanent project station identifier
- `station_id`: official BMD station ID
- `station_name_official`: official station name
- `date`: daily comparison date

## Old XLSX fields

- `old_record_present`: old source contains the station-date
- `old_source_row_count`: number of old source rows for the key
- `old_source_rows`: original XLSX row numbers
- `old_tmin_raw_values`: distinct raw Tmin values
- `old_tmin_parse_statuses`: old parsing statuses
- `old_duplicate_status`: unique, exact duplicate or conflicting
- `tmin_old_xlsx_compare`: old value usable for comparison
- `old_compare_parse_status`: old comparison-value status

Conflicting duplicate old records are not silently averaged or selected.

## New CSV fields

- `new_record_present`: updated CSV contains the station-date
- `new_source_row`: original CSV row number
- `new_source_day_column`: original Day1-Day31 field
- `new_source_cell`: combined source-row and day-column reference
- `new_tmin_raw`: updated CSV raw value
- `tmin_new_csv`: updated CSV numeric Tmin
- `new_tmin_parse_status`: updated parsing status
- `tmin_zero_flag`: updated value equals 0.0°C

## Reconciliation fields

- `tmin_difference_new_minus_old`: signed new minus old difference
- `absolute_tmin_difference`: absolute difference
- `reconciliation_status`: source-comparison category
- `tmin_selected_candidate`: preliminary selected value
- `tmin_selected_source`: new CSV, old XLSX or none
- `reconciliation_review_required`: whether manual or QC review is needed

## Main reconciliation categories

- `exact_agreement`
- `new_fills_old_missing`
- `old_fills_new_missing`
- `numeric_conflict_new_selected`
- `both_missing`
- `new_zero_requires_review`
- `old_conflicting_duplicate_new_selected`
- `old_conflicting_duplicate_unresolved`
- `new_only_record`
- `new_only_missing`
- `old_only_record`
- `old_only_missing`

The selected value is preliminary. Zero values, source conflicts and
other suspicious observations remain subject to later quality control.
