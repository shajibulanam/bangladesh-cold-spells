"""
Step 16S - Termination composite maps (manuscript Fig. 7).

End-aligned all-event composites (85 primary spells) at days -2, 0 (last event day) and +1:
    Row 1: Z500 anomaly (shading) and MSLP anomaly (contours, hPa)
    Row 2: T850 anomaly (shading) and 850-hPa wind anomaly (vectors)
    Row 3: T2m anomaly (shading) and 10-m wind anomaly (vectors)
Anomalies use the existing Step 9C calendar-day climatology (1991-2020). Fields are coarsened from
0.25 to 1 degree (4 x 4 block means). Stippling: winter-block bootstrap (1000 resamples of winters),
two-sided, Benjamini-Hochberg FDR q < 0.05 across grid points of each panel's shaded field.
The composites are also saved as NetCDF for the archive. Where the surface is higher than 1500 m
(NOAA ETOPO1, file from Step 16R --download-etopo), MSLP, T850 and 850-hPa winds are masked, because
those levels lie below ground there (Tibetan Plateau, Himalaya). Use --replot to redraw from the saved
NetCDF without recomputing.

Inputs (read only):
    03_intermediate/era5_preprocessed/pressure_level_standardized/ERA5_PL_STD_YYYY_MM.nc
    03_intermediate/era5_preprocessed/single_level_instant_standardized/ERA5_SL_INSTANT_STD_YYYY_MM.nc
    03_intermediate/era5_climatology/{pressure_level,single_level_instant}/ERA5_CLIM_*_calendar_day_1991_2020.nc

Run from the project root:
    python3 07_scripts/15s_termination_composite_maps.py 2>&1 | tee 09_logs/step16s_termination_maps.log
"""
from __future__ import annotations

import argparse
import importlib.util
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "term_common", Path(__file__).resolve().parent / "15_termination_common.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except ImportError:  # pragma: no cover
    HAVE_CARTOPY = False

PL_DIR = C.PROJECT_ROOT / "03_intermediate" / "era5_preprocessed" / "pressure_level_standardized"
SL_DIR = C.PROJECT_ROOT / "03_intermediate" / "era5_preprocessed" / "single_level_instant_standardized"
CLIM = C.PROJECT_ROOT / "03_intermediate" / "era5_climatology"
OUT_NC = C.TERM_INTERMEDIATE / "step16s_end_aligned_composites.nc"
ETOPO = C.PROJECT_ROOT / "01_raw_data" / "geospatial" / "etopo1_south_asia_0p1deg.nc"
HIGH_TERRAIN_M = 1500.0
FIG = "fig_final_07_termination_maps"
LAGS = [-2, 0, 1]
DOMAIN = (10.0, 45.0, 60.0, 110.0)  # lat0, lat1, lon0, lon1
COARSEN = 4
N_BOOT, SEED, Q = 1000, 20260923, 0.05
FIELDS = {  # key: (group, variable, level)
    "z500": ("pressure_level", "geopotential_height_m", 500.0),
    "t850": ("pressure_level", "air_temperature_k", 850.0),
    "u850": ("pressure_level", "u_wind_ms", 850.0),
    "v850": ("pressure_level", "v_wind_ms", 850.0),
    "t2m": ("single_level_instant", "t2m_c", None),
    "msl": ("single_level_instant", "msl_hpa", None),
    "u10": ("single_level_instant", "u10_ms", None),
    "v10": ("single_level_instant", "v10_ms", None),
}


def safe(name: str) -> str:  # same rule as Step 9C
    return name.replace("/", "_").replace(" ", "_").replace("-", "minus").replace(".", "_")


def subset(da: xr.DataArray) -> xr.DataArray:
    lat = da["lat"].values
    ls = slice(DOMAIN[1], DOMAIN[0]) if lat[0] > lat[-1] else slice(DOMAIN[0], DOMAIN[1])
    da = da.sel(lat=ls, lon=slice(DOMAIN[2], DOMAIN[3]))
    return da.coarsen(lat=COARSEN, lon=COARSEN, boundary="trim").mean()


def load_climatology() -> dict[str, xr.DataArray]:
    out = {}
    for key, (group, var, lev) in FIELDS.items():
        f = CLIM / group / f"ERA5_CLIM_{group}_{safe(var)}_calendar_day_1991_2020.nc"
        if not f.exists():
            raise SystemExit(f"Climatology file not found: {f}")
        with xr.open_dataset(f) as ds:
            da = ds[var]
            if lev is not None:
                da = da.sel(level_hpa=lev)
            out[key] = subset(da).load()
    return out


def load_anomalies(dates: list[pd.Timestamp], clim: dict[str, xr.DataArray]) -> dict[pd.Timestamp, dict[str, np.ndarray]]:
    by_month = defaultdict(list)
    for d in dates:
        by_month[(d.year, d.month)].append(d)
    result: dict[pd.Timestamp, dict[str, np.ndarray]] = defaultdict(dict)
    for (y, m), ds_dates in sorted(by_month.items()):
        for group, directory, pattern in (("pressure_level", PL_DIR, "ERA5_PL_STD"),
                                          ("single_level_instant", SL_DIR, "ERA5_SL_INSTANT_STD")):
            f = directory / f"{pattern}_{y}_{m:02d}.nc"
            if not f.exists():
                raise SystemExit(f"Missing ERA5 file: {f}")
            with xr.open_dataset(f) as ds:
                times = pd.to_datetime(ds["time"].values).normalize()
                idx = [int(np.flatnonzero(times == d)[0]) for d in ds_dates]
                for key, (grp, var, lev) in FIELDS.items():
                    if grp != group:
                        continue
                    da = ds[var].isel(time=idx)
                    if lev is not None:
                        da = da.sel(level_hpa=lev)
                    da = subset(da).load()
                    for j, d in enumerate(ds_dates):
                        c = clim[key].sel(calendar_day=d.strftime("%m-%d")).values
                        result[d][key] = da.isel(time=j).values - c
        print(f"loaded {y}-{m:02d} ({len(ds_dates)} dates)", flush=True)
    return result


def fdr_mask(x: np.ndarray, winters: np.ndarray) -> np.ndarray:
    """x: events x ny x nx. Returns boolean mask of FDR-significant cells."""
    n, ny, nx = x.shape
    flat = x.reshape(n, -1)
    uniq, counts = C.winter_bootstrap_counts(winters, N_BOOT, SEED)
    W = counts[:, np.searchsorted(uniq, winters)].astype(float)
    ok = np.isfinite(flat)
    means = (W @ np.nan_to_num(flat)) / (W @ ok.astype(float))
    p = np.minimum(1.0, 2 * np.minimum((means <= 0).mean(axis=0), (means >= 0).mean(axis=0)))
    q = C.benjamini_hochberg(p)
    return (q < Q).reshape(ny, nx)


def compute() -> xr.Dataset:
    C.write_policy("step16s_termination_maps_policy.json", {
        "step": "Step 16S", "lags_from_last_event_day": LAGS, "domain_lat_lon": DOMAIN,
        "coarsening": f"{COARSEN}x{COARSEN} block mean (0.25 -> 1 degree)", "climatology": "Step 9C calendar-day",
        "significance": {"bootstrap": "winter-block", "resamples": N_BOOT, "fdr_q": Q, "seed": SEED},
        "high_terrain_mask": f"MSLP, T850 and 850-hPa wind masked where ETOPO1 elevation > {HIGH_TERRAIN_M} m"})
    ev = C.load_events()
    winters = ev["winter_start_year"].to_numpy()
    clim = load_climatology()
    dates = sorted({d + pd.Timedelta(days=lag) for d in ev["end_date"] for lag in LAGS})
    anom = load_anomalies(dates, clim)
    lat = clim["z500"]["lat"].values
    lon = clim["z500"]["lon"].values
    ds_out = xr.Dataset(coords={"lag": LAGS, "lat": lat, "lon": lon})
    comp = {k: [] for k in FIELDS}
    sig = {k: [] for k in ("z500", "t850", "t2m")}
    for lag in LAGS:
        ds_ = [d + pd.Timedelta(days=lag) for d in ev["end_date"]]
        for key in FIELDS:
            stack = np.stack([anom[d][key] for d in ds_])
            comp[key].append(np.nanmean(stack, axis=0))
            if key in sig:
                sig[key].append(fdr_mask(stack, winters))
        print(f"composites and significance done for day {lag:+d}", flush=True)
    for key in FIELDS:
        ds_out[f"{key}_anom"] = (("lag", "lat", "lon"), np.stack(comp[key]))
    for key in sig:
        ds_out[f"{key}_fdr_significant"] = (("lag", "lat", "lon"), np.stack(sig[key]).astype("i1"))
    ds_out.attrs["description"] = "End-aligned all-event anomaly composites (85 spells), Step 16S"
    ds_out.to_netcdf(OUT_NC)
    return ds_out


def high_terrain(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    if not ETOPO.exists():
        print("WARNING: ETOPO file not found - no high-terrain mask (run 15r with --download-etopo).")
        return np.zeros((len(lat), len(lon)), dtype=bool)
    with xr.open_dataset(ETOPO) as e:
        alt = e["altitude"].astype("float32")
        # mean elevation within each 1-degree cell
        z = alt.interp(latitude=lat, longitude=lon, method="linear")
        zc = alt.coarsen(latitude=10, longitude=10, boundary="trim").mean().interp(
            latitude=lat, longitude=lon, method="nearest")
    return (np.maximum(z.values, zc.values) > HIGH_TERRAIN_M)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true", help="redraw from the saved NetCDF")
    args = ap.parse_args()
    C.ensure_dirs()
    ds_out = xr.open_dataset(OUT_NC) if (args.replot and OUT_NC.exists()) else compute()
    lat, lon = ds_out["lat"].values, ds_out["lon"].values
    mask = high_terrain(lat, lon)
    comp, sig = {}, {}
    for i, lag in enumerate(LAGS):
        for key in FIELDS:
            f = ds_out[f"{key}_anom"].values[i].copy()
            if key in ("msl", "t850", "u850", "v850"):
                f[mask] = np.nan
            comp[(lag, key)] = f
        for key in ("z500", "t850", "t2m"):
            m = ds_out[f"{key}_fdr_significant"].values[i].astype(bool)
            if key == "t850":
                m &= ~mask
            sig[(lag, key)] = m

    MM = 1 / 25.4
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.titlesize": 7.5,
                         "savefig.dpi": 300, "pdf.fonttype": 42})
    proj = {"projection": ccrs.PlateCarree()} if HAVE_CARTOPY else {}
    tr = {"transform": ccrs.PlateCarree()} if HAVE_CARTOPY else {}
    fig = plt.figure(figsize=(180 * MM, 135 * MM))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.04], hspace=0.18, wspace=0.06)
    rows = [("z500", "Z500 (m) + MSLP (hPa)", 40, "msl", None),
            ("t850", "T850 (K) + 850-hPa wind", 3, None, ("u850", "v850")),
            ("t2m", "T2m (°C) + 10-m wind", 3, None, ("u10", "v10"))]
    LON, LAT = np.meshgrid(lon, lat)
    for r, (key, title, lim, cont, vec) in enumerate(rows):
        mesh = None
        for c, lag in enumerate(LAGS):
            ax = fig.add_subplot(gs[r, c], **proj)
            mesh = ax.pcolormesh(lon, lat, comp[(lag, key)], cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto", **tr)
            m = sig[(lag, key)]
            yy, xx = np.nonzero(m[::2, ::2])
            ax.scatter(lon[::2][xx], lat[::2][yy], s=1.4, color="k", alpha=0.6, linewidths=0, **tr)
            if cont:
                ax.contour(lon, lat, comp[(lag, cont)], levels=[-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2],
                           colors="k", linewidths=0.5, **tr)  # interval 0.5 hPa; negative dashed
            if vec:
                s = (slice(None, None, 3), slice(None, None, 3))
                q = ax.quiver(LON[s], LAT[s], comp[(lag, vec[0])][s], comp[(lag, vec[1])][s],
                              scale=40, width=0.004, color="k", minlength=0, **tr)
                if c == 2:
                    ax.quiverkey(q, 0.85, 1.06, 2, "2 m s⁻¹", labelpos="E", fontproperties={"size": 6})
            if key == "t850":  # 850 hPa lies below ground here: show masked terrain as plain light grey
                ax.contourf(lon, lat, mask.astype(float), levels=[0.5, 1.5], colors=["0.85"], **tr)
            if HAVE_CARTOPY:
                ax.set_extent([DOMAIN[2], DOMAIN[3], DOMAIN[0], DOMAIN[1]], crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.4)
                ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3, edgecolor="0.4")
                gl = ax.gridlines(draw_labels=True, linewidth=0.2, color="0.7", linestyle=":")
                gl.top_labels = gl.right_labels = False
                gl.left_labels = c == 0
                gl.bottom_labels = r == 2
                gl.xlabel_style = gl.ylabel_style = {"size": 6}
            day = "last day" if lag == 0 else (f"day {lag:+d}".replace("-", "−"))
            ax.set_title(f"({'abcdefghi'[3 * r + c]}) {title.split(' (')[0]}, {day}", loc="left")
        cax = fig.add_subplot(gs[r, 3])
        cb = fig.colorbar(mesh, cax=cax)
        cb.set_label(title.split(" + ")[0], fontsize=7)
    paths = [C.TERM_FIGURES / f"{FIG}.png", C.TERM_FIGURES / f"{FIG}.pdf"]
    for p in paths:
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    C.write_checksums("step16s_output_sha256.txt", [OUT_NC] + paths)
    print(f"High-terrain mask covers {100 * mask.mean():.1f}% of the map grid")
    print("Saved", *[p.relative_to(C.PROJECT_ROOT) for p in paths], OUT_NC.relative_to(C.PROJECT_ROOT), sep="\n  ")


if __name__ == "__main__":
    main()
