# Step 7C Primary Bangladesh Cold-Spell Events

## Primary station-day condition

The station-day condition is:

`station_compound_cold_day_primary`

It requires:

- Tmin below the station-specific p10 threshold;
- Tmin at or below 10 degrees Celsius.

## Daily spatial rule

A national candidate cold day requires:

- adequate fixed-network station-winter availability;
- adequate daily numeric Tmin availability;
- at least 20 percent of observed eligible stations cold;
- at least five compound cold stations.

The required cold-station count is:

maximum of five stations and 20 percent of observed eligible stations.

## Event-duration rule

A primary cold-spell event requires:

- at least three consecutive national candidate cold days;
- no missing or nonqualifying day inside the event;
- no merging across a one-day gap;
- no crossing from one DJF winter into another.

## Main outputs

The event catalogue contains one row per event.

The event-day table contains one row per day within each event.

The station-participation table identifies stations that met the
compound condition during an event.

The event-ready daily dataset attaches the national event ID to
every original station-day row.

## Important limitation

This Step 7C version creates only the primary event catalogue.

Alternative spatial thresholds, duration thresholds and
sensitivity station networks will be tested in Step 7D.

No temperature is modified or interpolated.
