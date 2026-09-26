# Step 5B Spatial Consistency QC Data Dictionary

## Dataset status

The Step 5B dataset contains unchanged Step 4B cleaned
temperature values plus same-day neighbouring-station
consistency fields.

No temperature value is altered in Step 5B.

## Fixed station network

The fixed directed station network uses:

- up to five nearest stations;
- a maximum target-neighbour distance of 250 km;
- official latitude and longitude only;
- metadata-pending stations excluded.

The network is stored in:

`step5b_fixed_station_neighbours.parquet`

## Spatial anomaly metrics

For Tmin and Tmax:

- `*_spatial_neighbour_count`
- `*_spatial_neighbour_median_anomaly`
- `*_spatial_neighbour_mad`
- `*_spatial_neighbour_scale`
- `*_spatial_neighbour_minimum_anomaly`
- `*_spatial_neighbour_maximum_anomaly`
- `*_spatial_same_sign_fraction`
- `*_spatial_residual`
- `*_spatial_robust_z`

The spatial residual is:

target station anomaly minus median neighbouring-station anomaly.

The robust spatial scale is:

1.4826 multiplied by neighbour MAD,

with a minimum scale of 1 degree Celsius.

## Spatial flags

- `qc_tmin_spatial_outlier`
- `qc_tmax_spatial_outlier`
- `qc_tmin_regional_support`
- `qc_tmax_regional_support`
- `qc_tmin_regional_cold_support`
- `qc_tmin_spatial_insufficient`
- `qc_tmax_spatial_insufficient`

## Isolated spatial outlier

A candidate is flagged as an isolated spatial outlier when:

- at least three same-day neighbouring anomalies are available;
- the absolute spatial residual is at least 8 degrees Celsius;
- the absolute robust spatial z-score is at least 4.

## Regional support

Regional support requires:

- at least three same-day neighbours;
- target and neighbour median anomalies have the same sign;
- neighbour median anomaly magnitude is at least 3 degrees Celsius;
- target residual from neighbour median is no more than 4 degrees Celsius;
- at least 60 percent of available neighbours have the same anomaly sign.

## Regional cold support

Tmin regional cold support additionally requires:

- target Tmin anomaly is at most -3 degrees Celsius;
- neighbour median Tmin anomaly is at most -3 degrees Celsius.

## Combined fields

- `step5b_spatial_metadata_unavailable`
- `step5b_spatial_baseline_unavailable`
- `step5b_isolated_spatial_outlier`
- `step5b_regional_signal_supported`
- `step5b_regional_cold_signal_supported`
- `step5b_insufficient_same_day_neighbours`
- `step5b_candidate_for_cleaning_decision`
- `step5b_status`
- `step5b_priority`
- `step5b_flag_codes`
- `step5b_flag_count`

## Interpretation

A regionally supported cold anomaly should normally be retained.

An isolated spatial outlier is a candidate for conservative
automatic removal in Step 5C.

A candidate with insufficient neighbours should normally remain
unchanged because there is inadequate evidence to reject it.
