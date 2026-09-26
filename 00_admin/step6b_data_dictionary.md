# Step 6B Source-Transition and Homogeneity Screening

## Purpose

Step 6B screens fixed-network Tmin records for relative
discontinuities, particularly the January 2022 transition from
the original XLSX source to the updated BMD Tmin CSV.

No temperature adjustment or station exclusion occurs in Step 6B.

## Daily anomaly

For each valid station-day:

`daily anomaly = cleaned stage-2 Tmin - station calendar climatology`

The station calendar climatology was generated during Step 5A.

## Monthly anomaly

Monthly station anomalies are the median of available daily
climatological anomalies.

A month is valid when at least 80 percent of its expected calendar
days contain both:

- cleaned Tmin;
- station climatological anomaly.

## Leave-one-out network reference

For every station-month, the network reference is the median
monthly anomaly among all other valid fixed-network stations.

The target station is excluded from its own network reference.

## Station-network residual

`station residual = station monthly anomaly - leave-one-out network median`

Regional warming, cooling and cold-wave conditions should be
substantially reduced in this residual series.

## January 2022 transition test

The primary all-month comparison uses:

- pre period: January 2018 through December 2021;
- post period: January 2022 through December 2025.

A separate DJF-only comparison is also calculated.

For each comparison:

- pre-period median residual;
- post-period median residual;
- median shift;
- robust pooled scale;
- standardized shift;
- bootstrap 95 percent confidence interval;
- pre/post robust-scale ratio.

## Source-transition classification

High signal requires:

- bootstrap confidence interval excludes zero;
- absolute shift at least 1.5 degrees Celsius;
- standardized shift at least 2.5.

Moderate signal requires:

- bootstrap confidence interval excludes zero;
- absolute shift at least 1.0 degree Celsius;
- standardized shift at least 1.5.

A robust-scale ratio of at least 2.0 also creates a moderate
source-transition signal.

## General breakpoint scan

Annual station residuals are calculated from valid monthly
residuals.

A year is valid when at least eight monthly residuals are
available.

Every possible split with at least ten valid years on each side
is evaluated. The split with the largest absolute standardized
median shift is retained as the screening candidate.

This is a diagnostic scan, not proof of a documented station
break.

## Combined homogeneity status

Possible values include:

- `high_priority_homogeneity_review`
- `moderate_priority_homogeneity_review`
- `insufficient_homogeneity_data`
- `no_major_inhomogeneity_signal`

## Important limitations

A flagged source transition or breakpoint may result from:

- undocumented station relocation;
- instrument or exposure change;
- source-processing changes;
- local environmental change;
- residual data-quality problems;
- incomplete network-reference coverage;
- nonstationary local climate.

Step 6B does not automatically adjust data.

Step 6C will define the final analytical treatment and sensitivity
experiments for flagged stations.
