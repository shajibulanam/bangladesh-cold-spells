"""
Step 16V - Time-windowed ERA5 ARCO series for the Bangladesh box (analyses 7A and 7B).

Why: Step 16K3 daily means cover the UTC day k (06 LT day k to 06 LT day k+1) and therefore overlap
the night whose Tmin decides whether a cold spell continues. This step splits each UTC day into
    day window   00-11 UTC (06-17 LT)  - ends before the terminating night  -> usable as precursor
    night window 12-23 UTC (18-05 LT)  - overlaps the terminating night      -> concurrent only
and adds precipitation for the cold-day versus cold-night check.

Variables (Bangladesh box 20.5-26.8N, 88-93E, cosine-latitude weights):
    strd, ssrd  window-mean downward longwave / solar flux (W m-2) = hourly accumulation / 3600
    tcc         window-mean total cloud cover (fraction)
    td2m, t2m   window-mean 2-m dewpoint / temperature (degC)
    low1000     window-mean fraction of box area with cloud base below 1000 m
    fog300      window-mean fraction of box area with cloud base below 300 m
    blh         window-mean boundary-layer height (m)
    tp          window total precipitation (mm), box mean; also UTC-day total
Output: 03_intermediate/era5_box_daily/era5_cloud_radiation_windows_oct_mar.csv
Resumable: one part file per winter.

Requirements (installed for Step 16K3): zarr fsspec aiohttp requests
Run from the project root:
    python3 07_scripts/15v_extract_arco_windows.py 2>&1 | tee 09_logs/step16v_arco_windows.log
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
OUT = C.PROJECT_ROOT / "03_intermediate" / "era5_box_daily" / "era5_cloud_radiation_windows_oct_mar.csv"
PART_DIR = C.PROJECT_ROOT / "03_intermediate" / "era5_box_daily" / "arco_window_parts"
REPORT = C.TERM_QC / "step16v_arco_windows_report.txt"
BOX = (20.5, 26.8, 88.0, 93.0)
DAY_HOURS = range(0, 12)     # 00-11 UTC = 06-17 LT
NIGHT_HOURS = range(12, 24)  # 12-23 UTC = 18-05 LT
CANDIDATES = {
    "t2m": ["t2m", "2m_temperature"], "d2m": ["d2m", "2m_dewpoint_temperature"],
    "tcc": ["tcc", "total_cloud_cover"], "blh": ["blh", "boundary_layer_height"],
    "cbh": ["cbh", "cloud_base_height"], "ssrd": ["ssrd", "surface_solar_radiation_downwards"],
    "strd": ["strd", "surface_thermal_radiation_downwards"], "tp": ["tp", "total_precipitation"],
}


def cds_key() -> str:
    for line in (Path.home() / ".cdsapirc").read_text().splitlines():
        if line.strip().startswith("key:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit("No 'key:' line found in ~/.cdsapirc")


def main() -> None:
    C.ensure_dirs()
    PART_DIR.mkdir(parents=True, exist_ok=True)
    C.write_policy("step16v_arco_windows_policy.json", {
        "step": "Step 16V", "source": URL, "box": BOX,
        "day_window_utc": "00-11 (06-17 LT), ends before the terminating night",
        "night_window_utc": "12-23 (18-05 LT), overlaps the terminating night",
        "radiation": "hourly accumulation / 3600 s, averaged over the window",
        "precipitation": "hourly accumulation summed over the window, converted to mm",
        "cloud_base_proxies": {"low1000": "cloud base < 1000 m", "fog300": "cloud base < 300 m"}})
    ds = xr.open_zarr(URL, consolidated=True, storage_options={"headers": {"Authorization": f"Bearer {cds_key()}"}})
    names = {}
    for k, cands in CANDIDATES.items():
        hit = [c for c in cands if c in ds.data_vars]
        if hit:
            names[k] = hit[0]
        else:
            print(f"WARNING: {k} not found in the ARCO store; it will be skipped")
    if "tp" not in names:
        print("WARNING: no precipitation variable - analysis 7B cannot run from this output")
    LAT = "latitude" if "latitude" in ds.coords else "lat"
    LON = "longitude" if "longitude" in ds.coords else "lon"
    TIM = "valid_time" if "valid_time" in ds.coords else "time"
    print("Using variables:", names, flush=True)
    lat = ds[LAT].values
    ls = slice(BOX[1], BOX[0]) if lat[0] > lat[-1] else slice(BOX[0], BOX[1])
    box = ds[list(names.values())].sel({LAT: ls, LON: slice(BOX[2], BOX[3])})
    w = np.cos(np.deg2rad(box[LAT].values))
    w2d = (w[:, None] * np.ones((1, box.sizes[LON])))
    w2d = w2d / w2d.sum()

    for winter in range(1985, 2025):
        part = PART_DIR / f"arco_windows_{winter}.csv"
        if part.exists():
            print(f"winter {winter}/{str(winter + 1)[-2:]}: already done, skipped", flush=True)
            continue
        t0 = time.time()
        sub = box.sel({TIM: slice(f"{winter}-10-01", f"{winter + 1}-03-31T23:00")}).load()
        H = pd.DataFrame(index=pd.to_datetime(sub[TIM].values))
        for k in ("t2m", "d2m", "tcc", "blh", "ssrd", "strd", "tp"):
            if k not in names:
                continue
            arr = sub[names[k]].transpose(TIM, LAT, LON).values.astype("float64")
            ok = np.isfinite(arr)
            H[k] = np.nansum(np.where(ok, arr, 0) * w2d[None], axis=(1, 2)) / np.sum(ok * w2d[None], axis=(1, 2))
        if "cbh" in names:
            cbh = sub[names["cbh"]].transpose(TIM, LAT, LON).values
            H["low1000"] = np.sum(np.where(np.isfinite(cbh) & (cbh < 1000.0), 1.0, 0.0) * w2d[None], axis=(1, 2))
            H["fog300"] = np.sum(np.where(np.isfinite(cbh) & (cbh < 300.0), 1.0, 0.0) * w2d[None], axis=(1, 2))
        for k in ("t2m", "d2m"):
            if k in H:
                H[k] -= 273.15
        for k in ("ssrd", "strd"):
            if k in H:
                H[k] /= 3600.0
        if "tp" in H:
            H["tp"] *= 1000.0  # m -> mm per hour
        day, hour = H.index.normalize(), H.index.hour
        out = pd.DataFrame(index=pd.Index(day.unique(), name="date"))
        for label, hrs in (("day", DAY_HOURS), ("night", NIGHT_HOURS)):
            m = np.isin(hour, list(hrs))
            g = H[m].groupby(day[m])
            for k in H.columns:
                name = {"d2m": "td2m"}.get(k, k)
                if k == "tp":
                    out[f"tp_{label}_mm"] = g[k].sum()
                else:
                    out[f"{name}_{label}"] = g[k].mean()
            out[f"hours_{label}"] = g[H.columns[0]].count()
        if "tp" in H:
            out["tp_utcday_mm"] = H["tp"].groupby(day).sum()
        out.to_csv(part, float_format="%.5f")
        print(f"winter {winter}/{str(winter + 1)[-2:]}: {len(out)} days in {(time.time() - t0) / 60:.1f} min", flush=True)

    D = pd.concat([pd.read_csv(p, parse_dates=["date"], index_col="date") for p in sorted(PART_DIR.glob("arco_windows_*.csv"))])
    D = D[~D.index.duplicated()].sort_index()
    D.to_csv(OUT, float_format="%.5f")
    C.write_checksums("step16v_output_sha256.txt", [OUT])
    incomplete = int(((D.get("hours_day", 12) < 12) | (D.get("hours_night", 12) < 12)).sum())
    text = (f"Step 16V - ARCO windowed series\nRows: {len(D)} (expected 7290)\nDays with incomplete windows: {incomplete}\n"
            f"Period: {D.index.min().date()} to {D.index.max().date()}\n\n"
            + D.describe().T[["mean", "std", "min", "max"]].round(3).to_string() + "\n")
    REPORT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
