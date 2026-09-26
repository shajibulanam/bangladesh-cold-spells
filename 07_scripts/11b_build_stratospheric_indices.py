from __future__ import annotations

from importlib import import_module
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
C = import_module('11_stratosphere_common')

TABLE59 = C.TABLE_DIR / 'table_59_daily_stratospheric_indices.csv'
TABLE59B = C.TABLE_DIR / 'table_59b_stratospheric_calendar_day_climatology.csv'
DAILY_NC = C.NETCDF_DIR / 'ERA5_stratospheric_indices_daily_1985_2025.nc'


def level_slice(da: xr.DataArray, level: int) -> xr.DataArray:
    return da.sel(level_hpa=level, method='nearest')


def daily_raw_from_month(path: Path) -> list[dict]:
    rows = []
    with xr.open_dataset(path) as ds:
        ds = ds.load()
        lat = ds['lat'].values.astype(float)
        polar = lat >= 65.0
        for ti, timestamp in enumerate(pd.to_datetime(ds['time'].values)):
            row = {'date': timestamp.normalize()}
            for level in C.LEVELS:
                z = level_slice(ds['geopotential_height_m'].isel(time=ti), level).values
                u = level_slice(ds['u_wind_ms'].isel(time=ti), level).values
                t = level_slice(ds['air_temperature_k'].isel(time=ti), level).values
                # full-longitude zonal mean first, then cosine-latitude polar-cap mean
                row[f'pcz_{level}_m'] = C.weighted_mean_lat(z, lat, polar)
                row[f'pct_{level}_k'] = C.weighted_mean_lat(t, lat, polar)
                uzm = np.nanmean(u, axis=-1)
                row[f'u60_{level}_ms'] = float(uzm[np.nanargmin(np.abs(lat - 60.0))])
                if level == 10:
                    z10 = np.asarray(z, dtype=float)
                    high = lat >= 50.0
                    masked = np.where(high[:, None], z10, np.nan)
                    flat = int(np.nanargmin(masked))
                    ilat, ilon = np.unravel_index(flat, masked.shape)
                    row['vortex_min_lat_10hpa'] = float(lat[ilat])
                    row['vortex_min_lon_10hpa'] = float(ds['lon'].values[ilon])
                    row['vortex_min_displacement_deg_10hpa'] = float(90.0 - lat[ilat])
            rows.append(row)
    return rows


def main() -> None:
    files = [C.RAW_DIR / f'ERA5_STRAT_DAILY_{y}_{m:02d}.nc' for y, m in C.winter_months()]
    missing = [str(p) for p in files if not p.exists()]
    if missing: raise FileNotFoundError(f'Missing {len(missing)} monthly files; first: {missing[:3]}')
    rows = []
    for i, path in enumerate(files, 1):
        rows.extend(daily_raw_from_month(path))
        if i % 20 == 0 or i == len(files): print(f'Processed {i}/{len(files)} monthly files')
    data = pd.DataFrame(rows).drop_duplicates('date').sort_values('date').reset_index(drop=True)
    data['month_day'] = data['date'].dt.strftime('%m-%d')
    baseline = data.loc[data['date'].dt.year.between(1991, 2020)].copy()
    scalar_cols = [c for c in data.columns if c not in ['date', 'month_day']]
    clim_mean = baseline.groupby('month_day')[scalar_cols].mean()
    clim_sd = baseline.groupby('month_day')[scalar_cols].std(ddof=1)
    clim_rows = []
    for md in clim_mean.index:
        row = {'month_day': md}
        for col in scalar_cols:
            row[f'{col}_mean'] = clim_mean.loc[md, col]
            row[f'{col}_sd'] = clim_sd.loc[md, col]
        clim_rows.append(row)
    pd.DataFrame(clim_rows).to_csv(TABLE59B, index=False)

    for level in C.LEVELS:
        md_mean = data['month_day'].map(clim_mean[f'pcz_{level}_m'])
        md_sd = data['month_day'].map(clim_sd[f'pcz_{level}_m']).replace(0, np.nan)
        data[f'pcz_{level}_anomaly_m'] = data[f'pcz_{level}_m'] - md_mean
        data[f'nam_proxy_{level}'] = -data[f'pcz_{level}_anomaly_m'] / md_sd
        u_mean = data['month_day'].map(clim_mean[f'u60_{level}_ms'])
        u_sd = data['month_day'].map(clim_sd[f'u60_{level}_ms']).replace(0, np.nan)
        data[f'u60_{level}_anomaly_ms'] = data[f'u60_{level}_ms'] - u_mean
        data[f'u60_{level}_zscore'] = data[f'u60_{level}_anomaly_ms'] / u_sd
    data['weak_vortex_nam10_le_m1'] = (data['nam_proxy_10'] <= -1.0).astype(int)
    data['very_weak_vortex_nam10_le_m2'] = (data['nam_proxy_10'] <= -2.0).astype(int)
    data['negative_nam10'] = (data['nam_proxy_10'] < 0).astype(int)
    data['negative_nam100'] = (data['nam_proxy_100'] < 0).astype(int)
    data['negative_nam10_and_100'] = ((data['nam_proxy_10'] < 0) & (data['nam_proxy_100'] < 0)).astype(int)
    data['stratospheric_nam_mean_10_100'] = data[[f'nam_proxy_{l}' for l in [10,30,50,100]]].mean(axis=1)
    data.to_csv(TABLE59, index=False)

    times = pd.to_datetime(data['date']).to_numpy(dtype='datetime64[ns]')
    nam = np.stack([data[f'nam_proxy_{l}'].to_numpy(float) for l in C.LEVELS], axis=1)
    u60 = np.stack([data[f'u60_{l}_ms'].to_numpy(float) for l in C.LEVELS], axis=1)
    pcz = np.stack([data[f'pcz_{l}_anomaly_m'].to_numpy(float) for l in C.LEVELS], axis=1)
    dsout = xr.Dataset(
        data_vars={
            'nam_proxy': (('time','level_hpa'), nam.astype('float32')),
            'u60_ms': (('time','level_hpa'), u60.astype('float32')),
            'polar_cap_height_anomaly_m': (('time','level_hpa'), pcz.astype('float32')),
            'vortex_min_displacement_deg_10hpa': (('time',), data['vortex_min_displacement_deg_10hpa'].to_numpy('float32')),
        },
        coords={'time': times, 'level_hpa': np.asarray(C.LEVELS, dtype='int16')},
        attrs={'nam_proxy_definition': 'negative standardized polar-cap (>=65N) geopotential-height anomaly; positive=strong vortex, negative=weak vortex', 'climatology': '1991-2020 calendar-day'},
    )
    dsout.to_netcdf(DAILY_NC)
    report = [
        'STEP 12 STRATOSPHERIC INDEX REPORT', f'Daily records: {len(data)}',
        f'Coverage: {data.date.min().date()} to {data.date.max().date()}', f'Levels: {C.LEVELS}',
        'NAM proxy: -1 x standardized polar-cap geopotential-height anomaly north of 65N.',
        'Positive NAM proxy denotes stronger vortex; negative denotes weaker vortex.',
    ]
    (C.QC_DIR / 'step12_stratospheric_index_report.txt').write_text('\n'.join(report)+'\n', encoding='utf-8')
    C.write_json(C.QC_DIR / 'step12_stratospheric_index_summary.json', {'daily_records': len(data), 'levels': C.LEVELS, 'start': str(data.date.min().date()), 'end': str(data.date.max().date())})
    print('STEP 12-INDICES PASSED.')


if __name__ == '__main__': main()
