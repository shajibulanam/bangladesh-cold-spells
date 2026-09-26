# Step 5A Climatological and Temporal QC Data Dictionary

## Dataset status

The Step 5A dataset contains the unchanged Step 4B cleaned
temperatures plus station-climatology and temporal-consistency
screening fields.

No temperature value is altered in Step 5A.

## Reference climatology

The reference period is 1991-2020.

For each station and calendar day, observations within a circular
plus-or-minus 15-day calendar window are collected.

The reference statistics include:

- `tmin_clim_n`, `tmax_clim_n`
- `tmin_clim_years`, `tmax_clim_years`
- `tmin_clim_median`, `tmax_clim_median`
- `tmin_clim_mad`, `tmax_clim_mad`
- `tmin_clim_scale`, `tmax_clim_scale`
- `tmin_clim_p01`, `tmin_clim_p99`
- `tmax_clim_p01`, `tmax_clim_p99`

The robust scale is 1.4826 times MAD with a documented minimum
scale to avoid division by zero.

## Climatological fields

- `tmin_clim_anomaly`
- `tmax_clim_anomaly`
- `tmin_clim_robust_z`
- `tmax_clim_robust_z`
- `qc_tmin_climatological_outlier`
- `qc_tmax_climatological_outlier`
- `qc_tmin_climatology_unavailable`
- `qc_tmax_climatology_unavailable`

A climatological outlier requires both a large absolute anomaly
and a large robust-z value.

## Temporal fields

- `previous_date`, `next_date`
- `previous_tmin`, `next_tmin`
- `previous_tmax`, `next_tmax`
- `tmin_change_from_previous`
- `tmax_change_from_previous`
- `qc_tmin_large_daily_change`
- `qc_tmax_large_daily_change`
- `qc_tmin_isolated_spike`
- `qc_tmax_isolated_spike`

An isolated spike requires a large difference from both adjacent
days while the two adjacent days remain mutually consistent.

## Persistence fields

- `tmin_identical_run_length`
- `tmax_identical_run_length`
- `qc_tmin_persistence`
- `qc_tmax_persistence`

Persistence is calculated on values rounded to one decimal place
and requires at least seven consecutive calendar days.

## Combined Step 5A fields

- `qc_step4b_screen_carryover`
- `step5a_candidate_review_required`
- `step5a_priority`
- `step5a_status`
- `step5a_flag_codes`
- `step5a_flag_count`

## Important interpretation

A Step 5A flag does not prove that a value is wrong.

Regional cold-wave onset may legitimately produce:

- large negative Tmin anomalies;
- large day-to-day Tmin decreases;
- simultaneous anomalies at several stations.

Step 5B will compare candidate values with neighbouring stations
before any additional automatic cleaning decision is made.
