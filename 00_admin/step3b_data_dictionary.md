# Step 3B Updated Tmin CSV Data Dictionary

## Provenance

- `source_dataset`: new Tmin CSV source identifier
- `source_file`: original CSV filename
- `source_row`: physical CSV row number
- `source_day_column`: original Day1-Day31 column
- `source_cell`: unique source-row and day-column combination

## Station identity

- `station_name_raw`: original CSV station name
- `station_name_key`: normalized station key
- `station_uid`: permanent project identifier
- `station_id`: official BMD ID when available
- `station_name_official`: official BMD name when available
- `latitude`: official latitude when available
- `longitude`: official longitude when available
- `metadata_status`: official metadata available or missing
- `usable_for_spatial_analysis`: whether coordinates are available

## Date

- `year_raw`: source year text
- `month_raw`: source month text
- `year`: parsed year
- `month`: parsed month
- `day`: day number generated from Day1-Day31
- `date`: valid calendar date

Only valid calendar days are written. Noncalendar placeholders
such as Day31 for April are checked but not written.

## Temperature

- `tmin_raw`: original source-cell text
- `tmin_new_csv`: parsed numeric Tmin in degrees Celsius
- `tmin_parse_status`: numeric, missing_cell,
  missing_code or non_numeric
- `tmin_zero_flag`: true when the numeric source value is 0.0

A zero value is preserved and flagged. It is not automatically
treated as valid or missing during Step 3B.

## Structural status

- `ready_for_qc`: station has official metadata
- `ready_for_qc_metadata_pending`: record is retained but station
  metadata is incomplete
- `review_required`: unexpected parsing problem

Step 3B does not correct missing values, zero values or other
temperature-quality problems.
