# Step 7E Sensitivity Analysis and Final Event Freeze

## Authoritative primary catalogue

The Step 7D area-weighted catalogue remains the authoritative
primary catalogue.

The primary definition is:

- daily Tmin below the station-specific calendar-day p10;
- stations representing at least 20 percent of Bangladesh area;
- at least three consecutive regional cold days;
- no gap merging;
- at least 80 percent of Bangladesh area observed.

## Sensitivity scenarios

Step 7E evaluates:

- p10 with 10 percent area coverage;
- p10 with 30 percent area coverage;
- p10 with four-day duration;
- p05 with 20 percent area coverage;
- BMD Tmin at or below 10 degrees Celsius;
- one marginal gap day with at least 15 percent p10-cold area.

## Event reproduction

A primary event is classified as reproduced by a sensitivity
scenario when at least 50 percent of its primary event days
overlap a sensitivity event.

This overlap rule is used only to describe robustness.

## Robustness classes

- high: reproduced in at least 75 percent of sensitivity scenarios;
- moderate: reproduced in at least 50 percent;
- low: reproduced in fewer than 50 percent.

No primary event is removed because of its robustness class.

## Step 7C comparison

The Step 7C catalogue used a compound percentile-plus-BMD
station condition and station-count spatial coverage.

Step 7E compares its dates with the Step 7D area-weighted p10
catalogue using event-day overlap and Jaccard similarity.

## Frozen outputs

The final primary event catalogue is:

`step7e_frozen_primary_event_catalogue.csv`

The final event-day table is:

`step7e_frozen_primary_event_days.csv`

The atmospheric-analysis window table is:

`step7e_frozen_event_onsets_and_windows.csv`

## Atmospheric windows

Tropospheric analysis:

- day -10 to day +10 relative to event onset.

Cold-spell-centred stratospheric analysis:

- day -45 to day +5 relative to event onset.

## Event-strength ranking

The ranking is based on the sum of ranks for:

- event duration;
- peak p10-cold area;
- cumulative cold area-days;
- cumulative area-weighted p10 deficit.

The ranking is descriptive and does not affect event inclusion.

## First manuscript tables

Step 7E creates:

- cold-spell definition table;
- sensitivity-summary table;
- frozen primary event table;
- top-20 event table.

No temperature observation is modified in Step 7E.
