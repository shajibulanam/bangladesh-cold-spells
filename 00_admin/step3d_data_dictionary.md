# Step 3D Unified Preliminary Temperature Dataset

## Dataset status

This dataset combines the old BMD XLSX source, the updated
2022-2025 Tmin CSV and the Step 3C overlap reconciliation.

It is preliminary and has not yet passed full quality control.

## Station identity

- `station_uid`: permanent project station identifier
- `station_id`: official BMD station ID when available
- `official_station_name`: official BMD name
- `station_name_display`: official or provisional display name
- `latitude`, `longitude`: official coordinates when available
- `metadata_status`: official metadata available or pending
- `metadata_pending_flag`: station metadata remains incomplete
- `usable_for_spatial_analysis`: coordinates are available

## Date and source period

- `date`: valid station observation date
- `year`, `month`, `day`: date components
- `record_origin`: source-period integration category
- `integration_status`: ready for Step 4, metadata pending,
  or value review required

## Preliminary Tmin

- `tmin_preliminary`: selected preliminary Tmin value
- `tmin_source`: old_xlsx, new_csv or none
- `tmin_selection_status`: reason for source selection
- `tmin_review_required`: value requires later review
- `tmin_zero_flag`: selected Tmin equals 0.0 degrees Celsius

## Preliminary Tmax

- `tmax_preliminary`: old-XLSX Tmax through 2024
- `tmax_source`: old_xlsx or none
- `tmax_selection_status`: source or duplicate status
- `tmax_review_required`: value requires later review

No Tmax source is available for 2025.

## Old-source provenance

- `old_record_present`
- `old_source_row_count`
- `old_source_rows`
- `old_record_duplicate_status`
- `tmin_old_xlsx`
- `old_tmin_raw_values`
- `old_tmin_parse_statuses`
- `old_tmin_duplicate_status`
- `old_tmax_raw_values`
- `old_tmax_parse_statuses`
- `old_tmax_duplicate_status`

## Updated-source provenance

- `new_record_present`
- `new_source_row`
- `new_source_day_column`
- `new_source_cell`
- `new_tmin_raw`
- `tmin_new_csv`

## Reconciliation provenance

- `reconciliation_status`
- `tmin_difference_new_minus_old`

## Important limitations

- Invalid old-source calendar dates remain outside this table.
- Conflicting duplicate values have not been averaged.
- Zero Tmin observations remain flagged.
- Physical, temporal, spatial and climatological QC has not
  yet been performed.
- Metadata-pending stations cannot yet be used for spatial
  weighting or ERA5 matching.
