# Step 6A DJF Completeness and Fixed Network

## Winter definition

A winter is labelled using its December start year.

Examples:

- December 1985, January 1986 and February 1986 form winter 1985/86.
- December 2024, January 2025 and February 2025 form winter 2024/25.

The study contains 40 winters from 1985/86 through 2024/25.

## Completeness calculation

The analysis variable is:

`tmin_cleaned_stage2`

For each station-winter:

- `expected_days`: 90 or 91 calendar days
- `represented_days`: station-date rows present
- `numeric_tmin_days`: nonmissing cleaned Tmin observations
- `missing_station_date_rows`: absent calendar rows
- `represented_rows_missing_tmin`: rows present with missing Tmin
- `total_missing_tmin_days`: expected minus numeric days
- `tmin_completeness_percent`: numeric days divided by expected days
- `minimum_required_numeric_days`: ceiling of 95 percent of expected days
- `winter_complete`: completeness passes the 95 percent rule

For a 90-day DJF winter, at least 86 numeric days are required.

For a 91-day DJF winter, at least 87 numeric days are required.

## Fixed-network rule

A station enters the fixed network when:

- it has official BMD metadata;
- valid latitude and longitude are available;
- it is usable for spatial analysis;
- at least 38 of 40 winters are complete;
- its complete-winter fraction is at least 0.95.

## Individual-winter use

A fixed-network station is usable in an individual winter only when:

`usable_for_winter_analysis = True`

This requires both:

- station belongs to the fixed network;
- station passes the 95 percent completeness rule for that winter.

## Winter-level network rule

A study winter is network-eligible when the number of complete
fixed-network stations reaches both:

- at least 80 percent of the fixed network;
- at least 15 stations.

The stricter of the two requirements is used.

## Important limitation

No missing observation is interpolated.

Missing values are retained as missing and are handled explicitly
during percentile threshold and event-catalogue construction.
