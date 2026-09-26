# Temperature Data Source Policy

## Tmin

- 1981-2021:
  Use Tmin from `MaxT_MinT_1981_2024_raw.xlsx`.

- 2022-2025:
  Use Tmin from
  `Daily_Minimum_Temperature_2022_2025_raw.csv`
  as the primary source.

## Tmin overlap, 2022-2024

Both values must be preserved:

- `tmin_old_xlsx`
- `tmin_new_csv`

The combined value must have:

- `tmin_selected`
- `tmin_selected_source`
- `tmin_reconciliation_status`

Expected reconciliation statuses include:

- `exact_agreement`
- `new_fills_old_missing`
- `numeric_conflict_new_selected`
- `both_missing`
- `new_zero_requires_review`

## Tmax

Use Tmax from `MaxT_MinT_1981_2024_raw.xlsx`
for 1981-2024.

## Five stations awaiting metadata

- Ambagan(CTG)
- Bagabari
- Koyra
- Manikganj
- Narayanganj

Their observations may undergo quality control but must not
be used in area weighting or station-to-ERA5 matching until
official metadata is obtained.
