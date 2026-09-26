# Step 5C Conservative Spatial Cleaning Data Dictionary

## Dataset status

The Step 5C dataset is the observation-level QC-completed
temperature dataset before completeness screening, homogenization
and cold-spell extraction.

No interpolation is applied.

## Stage-2 cleaned temperature fields

- `tmin_cleaned_stage2`
- `tmax_cleaned_stage2`
- `tmin_cleaned_stage2_source`
- `tmax_cleaned_stage2_source`

The Step 4B values remain preserved in:

- `tmin_cleaned_stage1`
- `tmax_cleaned_stage1`

## Automatic actions

Tmin is set missing only when:

- `qc_tmin_spatial_outlier` is true.

Tmax is set missing only when:

- `qc_tmax_spatial_outlier` is true.

Both values are set missing when both variable-specific spatial
outlier flags are true.

## Retained observations

The following are retained unchanged:

- regional cold signals;
- regionally supported anomalies;
- candidates with insufficient same-day neighbours;
- stations without coordinates;
- stations without baseline climatology;
- spatially inconclusive candidates.

## Decision fields

- `step5c_tmin_set_missing`
- `step5c_tmax_set_missing`
- `step5c_any_value_changed`
- `step5c_decision`
- `step5c_decision_reason`
- `step5c_decision_method`
- `step5c_interpolation_applied`
- `step5c_status`

## Post-cleaning screening fields

- `step5c_post_dtr`
- `step5c_post_tmin_zero`
- `step5c_post_tmin_physical_range`
- `step5c_post_tmax_physical_range`
- `step5c_post_tmax_lt_tmin`
- `step5c_post_high_dtr`
- `step5c_remaining_hard_flag`
- `step5c_remaining_soft_flag`

## Important limitation

Step 5C completes observation-level QC, but the dataset is not yet
the final study network.

The next stages must assess:

- station and winter completeness;
- fixed-network eligibility;
- source-transition discontinuities;
- potential station inhomogeneities;
- baseline-period suitability.
