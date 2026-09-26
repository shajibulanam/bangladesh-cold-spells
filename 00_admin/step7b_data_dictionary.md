# Step 7B BMD Absolute and Compound Cold Indicators

## Input temperature

All categories use:

`analysis_tmin_unadjusted`

This is the unchanged Step 5C cleaned Tmin retained in the
Step 6C analytical dataset.

## BMD absolute categories

### Mild

Greater than 8 degrees Celsius and at most 10 degrees Celsius.

### Moderate

Greater than 6 degrees Celsius and at most 8 degrees Celsius.

### Severe

Greater than 4 degrees Celsius and at most 6 degrees Celsius.

### Very severe

At most 4 degrees Celsius.

### Above the BMD cold-wave threshold

Greater than 10 degrees Celsius.

## BMD fields

- `bmd_absolute_category_raw`
- `bmd_absolute_severity_rank_raw`
- `bmd_absolute_category_primary`
- `bmd_absolute_severity_rank_primary`
- `eligible_for_bmd_absolute_test_primary`
- `station_bmd_mild_day`
- `station_bmd_moderate_day`
- `station_bmd_severe_day`
- `station_bmd_very_severe_day`
- `station_bmd_cold_wave_day`

Severity ranks are:

- -1: unavailable or ineligible
- 0: above BMD cold-wave threshold
- 1: mild
- 2: moderate
- 3: severe
- 4: very severe

## Relative percentile condition

The station-relative condition is:

`station_percentile_cold_day_p10`

It requires Tmin to be strictly below the station-calendar-day
p10 threshold.

## Primary compound condition

The primary compound station-day indicator is:

`station_compound_cold_day_primary`

It requires both:

- station p10 cold-day condition;
- BMD absolute cold-wave condition, Tmin at most 10 degrees Celsius.

## Sensitivity conditions

- `station_cold_candidate_percentile_sensitivity`
- `station_cold_candidate_bmd_sensitivity`
- `station_cold_candidate_union_sensitivity`

These allow the later event catalogue to test alternative
station-day definitions.

## Mutually exclusive condition classes

`station_day_cold_condition` may contain:

- `compound_p10_and_bmd`
- `percentile_only`
- `bmd_only`
- `neither`
- `missing_tmin`
- `station_threshold_unavailable`
- `ineligible_station_winter`

## Deficit metrics

`tmin_below_bmd_10c_deficit`

is 10 degrees Celsius minus observed Tmin for BMD cold-wave days.

`tmin_below_station_p10_deficit`

is station p10 threshold minus observed Tmin for percentile cold
days.

`compound_bmd_deficit` and `compound_p10_deficit` are populated
only on primary compound cold days.

## Important limitation

Step 7B classifies individual station-days only.

It does not yet impose:

- a minimum number of consecutive days;
- a minimum proportion of Bangladesh stations;
- national or regional event boundaries;
- event-merging rules.

Those rules are applied in Step 7C.

No temperature is changed or interpolated in Step 7B.
