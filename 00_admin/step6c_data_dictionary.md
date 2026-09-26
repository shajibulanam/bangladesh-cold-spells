# Step 6C Analytical Network Freeze

## Primary analytical network

The primary network is:

`primary_all_fixed_unadjusted`

It contains every station selected by the Step 6A fixed-network
criteria.

Step 6B homogeneity signals do not automatically remove or adjust
a station in the primary analysis.

## Sensitivity networks

### sensitivity_exclude_high

Excludes stations classified as:

`high_priority_homogeneity_review`

### sensitivity_stable_only

Excludes stations classified as:

- `high_priority_homogeneity_review`
- `moderate_priority_homogeneity_review`

### sensitivity_no_source_transition

Excludes stations classified as:

- `high_source_transition_signal`
- `moderate_source_transition_signal`

## Temperature field

The analysis temperature is:

`analysis_tmin_unadjusted`

It is an unchanged copy of:

`tmin_cleaned_stage2`

No homogenization, interpolation, mean-shift correction or
statistical adjustment is applied.

## Complete DJF calendar

The analysis dataset contains one row for every combination of:

- fixed-network station;
- expected DJF date;
- winter from 1985/86 through 2024/25.

Missing source station-date rows are represented explicitly.

## Missing-reason field

`analysis_missing_reason` may be:

- `available`
- `missing_station_date_row`
- `represented_row_missing_tmin`

## Station-winter usability

For every network, two Boolean fields are created:

- `station_winter_usable_<network_id>`
- `observation_usable_<network_id>`

Station-winter usability requires:

- station belongs to the network;
- station-winter passed Step 6A completeness.

Observation usability additionally requires numeric Tmin on that
date.

## Network winter eligibility

For every network and winter, eligibility requires the greater of:

- 80 percent of network stations;
- 15 stations.

The threshold is calculated separately for each sensitivity
network.

## Interpretation

The primary analysis should be reported using the full fixed
network without adjustment.

Sensitivity analyses should repeat the principal cold-spell
statistics using the alternative network memberships.

Results are considered robust when the principal conclusions do
not materially change across these networks.
