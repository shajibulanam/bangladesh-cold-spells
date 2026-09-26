"""
Step 16K3 - Daily Bangladesh-box cloud, fog, humidity, boundary-layer and radiation series from the
ERA5 ARCO (Analysis Ready Cloud Optimised) geo-chunked Zarr store.

Why ARCO: the CDS MARS queue (Step 16K) held thousands of requests and each file took hours. The ARCO
store is read directly (no queue) with the same CDS API key, and its geo-chunking is designed for long
time series over small areas. It replaces Steps 16K/16K2 (their partial downloads are not used).

Source : https://arco.datastores.ecmwf.int/cadl-arco-geo-002/arco/reanalysis_era5_single_levels/sfc/geoChunked.zarr
         (ECMWF Confluence: "ERA5 hourly ARCO data on single levels", Table 1)
Box    : 20.5-26.8N, 88-93E, cosine-latitude weights (same box as Step 16A)
Period : 1 October - 31 March, winters 1985/86-2024/25, all 24 hours
Output : 03_intermediate/era5_box_daily/era5_cloud_radiation_daily_oct_mar.csv, one row per UTC day:
    tcc_mean, tcc_00utc, tcc_06utc           total cloud cover (fraction); 00 UTC = 06 local, 06 UTC = 12 local
    lowcloud1000_frac_mean / _00utc / _06utc fraction of box area with cloud base below 1000 m
    fog300_frac_00utc, fog300_frac_mean      fraction of box area with cloud base below 300 m (fog / very low stratus)
    dpd_00utc                                2-m dewpoint depression at 00 UTC (K; small = near saturation)
    td2m_mean                                daily-mean 2-m dewpoint (degC)
    blh_06utc, blh_mean                      boundary-layer height (m)
    ssrd_wm2, strd_wm2                       daily-mean downward solar / longwave radiation (W m-2),
                                             sum of the 24 hourly accumulations / 86400 s
Low cloud cover is not in the ARCO store; cloud-base-height fractions are used instead.

Requirements (once):  pip install zarr fsspec aiohttp requests
Run from the project root:
    python3 07_scripts/15k3_extract_cloud_radiation_arco.py 2>&1 | tee 09_logs/step16k3_cloud_radiation_arco.log
"""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

_spec = importlib.util.spec_from_file_location(
    "term_common", Path(__file__).resolve().parent / "15_termination_common.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

URL = "https://arco.datastores.ecmwf.int/cadl-arco-geo-002/arco/reanalysis_era5_single_levels/sfc/geoChunked.zarr"
OUT = C.PROJECT_ROOT / "03_intermediate" / "era5_box_daily" / "era5_cloud_radiation_daily_oct_mar.csv"
PART_DIR = C.PROJECT_ROOT / "03_intermediate" / "era5_box_daily" / "arco_winter_parts"
REPORT = C.TERM_QC / "step16k3_cloud_radiation_arco_report.txt"
BOX = (20.5, 26.8, 88.0, 93.0)
CANDIDATES = {
    "t2m": ["t2m", "2m_temperature"], "d2m": ["d2m", "2m_dewpoint_temperature"],
    "tcc": ["tcc", "total_cloud_cover"], "blh": ["blh", "boundary_layer_height"],
    "cbh": ["cbh", "cloud_base_height"], "ssrd": ["ssrd", "surface_solar_radiation_downwards"],
    "strd": ["strd", "surface_thermal_radiation_downwards"],
}


def cds_key() -> str:
    rc = Path.home() / ".cdsapirc"
    for line in rc.read_text().splitlines():
        if line.strip().startswith("key:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit("No 'key:' line found in ~/.cdsapirc")


def resolve(ds: xr.Dataset) -> tuple[dict, str, str, str]:
    names = {}
    for k, cands in CANDIDATES.items():
        hit = [c for c in cands if c in ds.data_vars]
        if not hit:
            raise SystemExit(f"Variable for {k} not found. Available: {list(ds.data_vars)}")
        names[k] = hit[0]
    lat = "latitude" if "latitude" in ds.coords else "lat"
    lon = "longitude" if "longitude" in ds.coords else "lon"
    tim = "valid_time" if "valid_time" in ds.coords else "time"
    return names, lat, lon, tim


def main() -> None:
    C.ensure_dirs()
    PART_DIR.mkdir(parents=True, exist_ok=True)
    C.write_policy("step16k3_cloud_radiation_arco_policy.json", {
        "step": "Step 16K3", "source": URL, "box": BOX, "weights": "cosine latitude",
        "period": "Oct-Mar 1985/86-2024/25, hourly", "day": "UTC day (06 local to 06 local)",
        "low_cloud_proxy": "fraction of box area with cloud base height < 1000 m; fog proxy < 300 m",
        "radiation": "sum of 24 hourly accumulations / 86400 s", "replaces": "Steps 16K and 16K2"})
    ds = xr.open_zarr(URL, consolidated=True,
                      storage_options={"headers": {"Authorization": f"Bearer {cds_key()}"}})
    names, LAT, LON, TIM = resolve(ds)
    print("Opened ARCO store. Using variables:", names, "| coords:", LAT, LON, TIM, flush=True)
    lat = ds[LAT].values
    lat_slice = slice(BOX[1], BOX[0]) if lat[0] > lat[-1] else slice(BOX[0], BOX[1])
    box = ds[list(names.values())].sel({LAT: lat_slice, LON: slice(BOX[2], BOX[3])})
    w = np.cos(np.deg2rad(box[LAT]))
    w2d = (w / w.sum()).values[:, None] * np.ones((1, box.sizes[LON])) / box.sizes[LON]

    for winter in range(1985, 2025):
        part = PART_DIR / f"arco_{winter}.csv"
        if part.exists():
            print(f"winter {winter}/{str(winter + 1)[-2:]}: already done, skipped", flush=True)
            continue
        t0 = time.time()
        sub = box.sel({TIM: slice(f"{winter}-10-01", f"{winter + 1}-03-31T23:00")}).load()
        times = pd.to_datetime(sub[TIM].values)
        H = pd.DataFrame(index=times)
        for k in ("t2m", "d2m", "tcc", "blh", "ssrd", "strd"):
            arr = sub[names[k]].transpose(TIM, LAT, LON).values
            H[k] = np.nansum(arr * w2d[None], axis=(1, 2)) / np.nansum(np.where(np.isnan(arr), 0, 1) * w2d[None],
                                                                         axis=(1, 2))
        cbh = sub[names["cbh"]].transpose(TIM, LAT, LON).values
        H["low1000"] = np.sum(np.where(np.isfinite(cbh) & (cbh < 1000.0), 1.0, 0.0) * w2d[None], axis=(1, 2))
        H["fog300"] = np.sum(np.where(np.isfinite(cbh) & (cbh < 300.0), 1.0, 0.0) * w2d[None], axis=(1, 2))
        H["t2m"] -= 273.15
        H["d2m"] -= 273.15
        day, hour = H.index.normalize(), H.index.hour
        g = H.groupby(day)
        out = pd.DataFrame({
            "tcc_mean": g["tcc"].mean(), "lowcloud1000_frac_mean": g["low1000"].mean(),
            "fog300_frac_mean": g["fog300"].mean(), "td2m_mean": g["d2m"].mean(), "blh_mean": g["blh"].mean(),
            "ssrd_wm2": g["ssrd"].sum() / 86400.0, "strd_wm2": g["strd"].sum() / 86400.0,
            "hours_present": g["tcc"].count()})

        def at(h, col):
            m = hour == h
            return H.loc[m, col].groupby(day[m]).mean()

        out["tcc_00utc"], out["tcc_06utc"] = at(0, "tcc"), at(6, "tcc")
        out["lowcloud1000_frac_00utc"], out["lowcloud1000_frac_06utc"] = at(0, "low1000"), at(6, "low1000")
        out["fog300_frac_00utc"] = at(0, "fog300")
        out["dpd_00utc"] = at(0, "t2m") - at(0, "d2m")
        out["blh_06utc"] = at(6, "blh")
        out.index.name = "date"
        out.to_csv(part, float_format="%.5f")
        print(f"winter {winter}/{str(winter + 1)[-2:]}: {len(out)} days in {(time.time() - t0) / 60:.1f} min",
              flush=True)

    D = pd.concat([pd.read_csv(p, parse_dates=["date"], index_col="date") for p in sorted(PART_DIR.glob("arco_*.csv"))])
    D = D[~D.index.duplicated()].sort_index()
    D.to_csv(OUT, float_format="%.5f")
    C.write_checksums("step16k3_output_sha256.txt", [OUT])
    text = (f"Step 16K3 - ARCO cloud/radiation daily series\nRows: {len(D)} (expected 7290)\n"
            f"Days with fewer than 24 hours: {int((D['hours_present'] < 24).sum())}\n"
            f"Period: {D.index.min().date()} to {D.index.max().date()}\n\n"
            + D.describe().T[["mean", "std", "min", "max"]].round(3).to_string() + "\n")
    REPORT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
