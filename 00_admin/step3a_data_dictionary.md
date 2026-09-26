# Step 3A Old-XLSX Data Dictionary

## Provenance

- `source_dataset`: source identifier
- `source_file`: original XLSX filename
- `source_sheet`: original worksheet
- `source_row`: original Excel row number

## Station identity

- `station_name_raw`: original station name
- `station_name_key`: normalized matching key
- `station_uid`: permanent project station identifier
- `station_id`: official BMD station ID
- `station_name_official`: official BMD name
- `latitude`: decimal degrees north
- `longitude`: decimal degrees east
- `station_match_status`: matched or unmatched

## Date

- `year_raw`, `month_raw`, `day_raw`: preserved source text
- `year`, `month`, `day`: parsed integer components
- `date`: valid calendar date or missing
- `date_parse_status`: valid, invalid_calendar_date,
  invalid_date_component or missing_date_component

## Temperature

- `tmax_raw`: preserved source Tmax text
- `tmax_old_xlsx`: parsed numeric Tmax
- `tmax_parse_status`: numeric, missing_cell,
  missing_code or non_numeric
- `tmin_raw`: preserved source Tmin text
- `tmin_old_xlsx`: parsed numeric Tmin
- `tmin_parse_status`: numeric, missing_cell,
  missing_code or non_numeric

## Structural status

- `ready_for_qc`: station and date are structurally usable
- `review_required`: station, date or numeric parsing requires review

Step 3A does not correct invalid dates, duplicates,
missing values or suspicious temperatures.
