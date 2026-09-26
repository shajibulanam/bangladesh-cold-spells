# Step 4B QC Decision Policy

## General principles

1. Never correct an observation only because it looks unusual.
2. Prefer the original or latest official BMD record.
3. Use adjacent dates and nearby stations as supporting evidence,
   not as the sole basis for inventing a replacement.
4. Never average conflicting duplicate observations automatically.
5. Every completed decision requires:
   - decision reason;
   - verification source;
   - reviewer;
   - review date.

## Allowed daily-value decisions

- `keep`
- `set_tmin_missing`
- `set_tmax_missing`
- `set_both_missing`
- `correct_tmin`
- `correct_tmax`
- `correct_both`
- `select_old_tmin`
- `select_new_tmin`

## Allowed invalid-date decisions

- `reject_source_row`
- `insert_corrected_date`
- `replace_existing_date`

## Correction requirements

- `correct_tmin` requires `corrected_tmin`.
- `correct_tmax` requires `corrected_tmax`.
- `correct_both` requires both corrected values.
- `insert_corrected_date` requires `corrected_date`.
- `replace_existing_date` requires a corrected date that already
  exists for the station and at least one usable temperature value.

Dates must use:

`YYYY-MM-DD`

Review dates must also use:

`YYYY-MM-DD`

## Evidence examples

Suitable evidence includes:

- original BMD observation sheet;
- corrected BMD export;
- official station register;
- documented transcription correction;
- agreement with the new official BMD extract;
- adjacent-day and neighbouring-station assessment documented
  as supporting evidence.

A value may be kept after review when the evidence supports it,
even if a QC flag remains.
