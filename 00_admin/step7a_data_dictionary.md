# Step 7A Station-Specific Daily Cold Thresholds

## Reference climate

The station thresholds use the calendar-date reference period:

1 January 1991 through 31 December 2020.

This is a calendar-year reference period rather than a set of
30 complete winter labels.

Therefore:

- January-February 1991 are included;
- December 2020 is included;
- threshold windows near December 1 may include late-November data;
- threshold windows near February 29 may include early-March data.

## Baseline eligibility

Only cleaned, numeric `tmin_cleaned_stage2` observations are used.

Every observation must be associated with a station-winter that
passed the Step 6A completeness rule.

The threshold network contains the Step 6C primary fixed stations.

## Calendar window

Each target DJF calendar day uses a five-day moving window:

- target day minus two days;
- target day minus one day;
- target day;
- target day plus one day;
- target day plus two days.

The window is circular across 31 December and 1 January.

February 29 receives its own threshold.

## Percentiles

Full-reference thresholds include:

- `station_p05_threshold_full`
- `station_p10_threshold_full`
- `station_p20_threshold_full`

The primary cold threshold is the station-specific p10 value.

Quantiles use NumPy's `median_unbiased` estimator.

## Leave-one-calendar-year-out threshold

For analysis dates inside 1991-2020, the p10 threshold excludes
all baseline-pool observations from the same calendar year.

This reduces in-base percentile bias.

The assigned threshold is:

`station_p10_threshold_analysis`

For dates outside 1991-2020, the full-reference p10 threshold is
used.

## Cold-day definition

A primary station percentile cold day requires:

- station belongs to the primary network;
- station-winter passed the completeness rule;
- Tmin is numeric;
- threshold is available;
- `analysis_tmin_unadjusted` is strictly less than
  `station_p10_threshold_analysis`.

The resulting Boolean field is:

`station_percentile_cold_day_p10`

## Deficit fields

`tmin_minus_station_p10`

is negative when Tmin lies below the station p10 threshold.

`station_p10_cold_deficit`

is positive only for percentile cold days and represents:

p10 threshold minus observed Tmin.

## Important limitations

The p10 flag identifies a station-relative cold day.

It does not yet constitute a cold-spell event.

The next step must combine:

- station percentile condition;
- BMD absolute-temperature categories;
- consecutive-day duration;
- spatial coverage across Bangladesh.

No temperature is modified or interpolated in Step 7A.
