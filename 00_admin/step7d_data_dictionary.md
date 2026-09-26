# Step 7D Area-Weighted Bangladesh Cold-Spell Catalogue

## Primary station condition

A station cold day occurs when daily minimum temperature is
strictly below the station-specific calendar-day p10 threshold.

The Step 7A analysis threshold is used, including the
leave-one-calendar-year-out p10 threshold for baseline dates.

## Station geographical weights

Primary-network stations are assigned Thiessen/Voronoi polygons.

The polygons are:

- generated in a Bangladesh-centred Albers Equal Area projection;
- clipped to the Bangladesh ADM0 boundary;
- converted to square kilometres;
- divided by total Bangladesh boundary area.

The resulting station weights sum to one.

## Missing observations

A station's missing area is not reassigned to other stations.

Daily observed-area fraction is the sum of the fixed national
weights of stations with usable Tmin and p10 thresholds.

A day is eligible only when:

- observed national-area fraction is at least 0.80;
- at least 15 stations are observed.

## Regional p10 cold day

A regional cold day occurs when stations with p10 cold-day
conditions represent at least 0.20 of total Bangladesh area.

The denominator is always the complete national area, not only
the observed area.

## Primary event

A primary Bangladesh cold spell requires at least three
uninterrupted regional p10 cold days.

Events:

- cannot cross DJF winter boundaries;
- are broken by a nonqualifying day;
- do not merge across a one-day gap.

## Severe and BMD information

The station p05 condition is retained as severe climate-relative
information but does not define the primary event in Step 7D.

BMD mild, moderate, severe and very-severe categories are
retained as complementary operational severity measures.

## Spatial characteristics

Each event includes:

- affected national-area fraction;
- event footprint area;
- geographical centre;
- north-south spatial extent;
- east-west spatial extent;
- number of affected stations.

## Important

Step 7C is retained as a station-count and compound-condition
sensitivity catalogue.

Step 7D is the synopsis-consistent primary area-weighted
climatological catalogue.

No temperature is modified or interpolated.
