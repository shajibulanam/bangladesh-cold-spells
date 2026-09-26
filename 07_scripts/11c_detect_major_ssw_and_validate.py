from __future__ import annotations

from importlib import import_module
from pathlib import Path
import numpy as np
import pandas as pd
C = import_module('11_stratosphere_common')

DAILY = C.TABLE_DIR / 'table_59_daily_stratospheric_indices.csv'
REF = C.PROJECT_ROOT / '00_admin' / 'noaa_era5_major_ssw_reference_through_2023.csv'
TABLE60 = C.TABLE_DIR / 'table_60_major_ssw_catalogue.csv'
TABLE60B = C.TABLE_DIR / 'table_60b_major_ssw_validation.csv'


def has_westerly_run(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp, length: int) -> bool:
    sub = series.loc[(series.index >= start) & (series.index <= end)] > 0
    return C.consecutive_max(sub.to_numpy()) >= length


def detect_ssw(data: pd.DataFrame) -> pd.DataFrame:
    s = data.set_index('date')['u60_10_ms'].sort_index()
    candidates = []
    for date in s.index:
        if date.month not in [11,12,1,2,3]: continue
        previous = date - pd.Timedelta(days=1)
        if previous not in s.index: continue
        if s.loc[date] < 0 and s.loc[previous] >= 0:
            candidates.append(date)
    accepted = []
    previous_event = None
    for date in candidates:
        if previous_event is not None:
            if not has_westerly_run(s, previous_event + pd.Timedelta(days=1), date - pd.Timedelta(days=1), 20):
                continue
        april30 = pd.Timestamp(year=date.year if date.month <= 4 else date.year + 1, month=4, day=30)
        if not has_westerly_run(s, date + pd.Timedelta(days=1), april30, 10):
            continue  # final warming
        accepted.append(date); previous_event = date
    rows=[]
    lookup = data.set_index('date')
    for i,date in enumerate(accepted,1):
        after = s.loc[date:]
        easterly_duration=0
        for value in after:
            if value < 0: easterly_duration += 1
            else: break
        w10 = lookup.loc[date:min(date+pd.Timedelta(days=10), lookup.index.max()), 'u60_10_ms']
        pre = lookup.loc[max(date-pd.Timedelta(days=10), lookup.index.min()):date, 'nam_proxy_10']
        rows.append({
            'ssw_id': f'SSW_{date:%Y%m%d}', 'central_date': date, 'winter_label': f'{date.year-1}/{str(date.year)[-2:]}' if date.month <= 4 else f'{date.year}/{str(date.year+1)[-2:]}',
            'u60_10_central_ms': float(lookup.loc[date,'u60_10_ms']), 'minimum_u60_first10_ms': float(w10.min()), 'easterly_run_days': easterly_duration,
            'nam10_central': float(lookup.loc[date,'nam_proxy_10']), 'nam100_central': float(lookup.loc[date,'nam_proxy_100']), 'pre10_nam10_mean': float(pre.mean()),
        })
    return pd.DataFrame(rows)


def validate(derived: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(derived['central_date']) if len(derived) else pd.DatetimeIndex([])
    rows=[]
    for _,r in reference.iterrows():
        ref=pd.Timestamp(r['era5_central_date'])
        if d.size:
            offsets=(d-ref).dt.days.to_numpy(); j=int(np.argmin(np.abs(offsets))); nearest=d.iloc[j]; offset=int((nearest-ref).days)
        else: nearest=pd.NaT; offset=999
        rows.append({'event_name':r['event_name'],'reference_date':ref,'nearest_derived_date':nearest,'offset_days':offset,'exact_match':offset==0,'within_1_day':abs(offset)<=1,'within_2_days':abs(offset)<=2})
    return pd.DataFrame(rows)


def main() -> None:
    data=pd.read_csv(DAILY, parse_dates=['date'])
    ssw=detect_ssw(data); ssw.to_csv(TABLE60,index=False)
    ref=pd.read_csv(REF, parse_dates=['era5_central_date'])
    # compare only reference dates in our analysis period and <=2023-12-31
    ref=ref.loc[(ref.era5_central_date>=data.date.min())&(ref.era5_central_date<=min(data.date.max(),pd.Timestamp('2023-12-31')))]
    val=validate(ssw.loc[pd.to_datetime(ssw.central_date)<=pd.Timestamp('2023-12-31')] if len(ssw) else ssw, ref)
    val.to_csv(TABLE60B,index=False)
    report=[
        'STEP 12 MAJOR SSW DETECTION AND VALIDATION REPORT', f'Derived SSWs: {len(ssw)}', f'Reference events compared: {len(val)}',
        f'Exact-date matches: {int(val.exact_match.sum()) if len(val) else 0}/{len(val)}', f'Within +/-1 day: {int(val.within_1_day.sum()) if len(val) else 0}/{len(val)}', f'Within +/-2 days: {int(val.within_2_days.sum()) if len(val) else 0}/{len(val)}',
        'Definition: first daily-mean 10-hPa 60N zonal-wind reversal to easterly; 20 consecutive westerly days required between events; final warmings excluded unless >=10 consecutive westerly days return before 30 April.',
    ]
    (C.QC_DIR/'step12_major_ssw_report.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
    C.write_json(C.QC_DIR/'step12_major_ssw_summary.json', {'derived_ssw_count':len(ssw),'reference_count':len(val),'exact_matches':int(val.exact_match.sum()) if len(val) else 0,'within_1_day':int(val.within_1_day.sum()) if len(val) else 0})
    print('STEP 12-SSW PASSED.')


if __name__=='__main__': main()
