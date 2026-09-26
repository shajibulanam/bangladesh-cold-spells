"""
Step 16I - Journal-quality onset-pathway map (Z500 and 250-hPa wave-activity flux).

Redraws EXISTING results only; nothing is recomputed:
    Row 1: 500-hPa geopotential-height anomaly (all 85 events) at lags -10, -5, 0.
           Stippling = Step 9G robust grid points (winter-block bootstrap FDR q < 0.05 AND
           >= 60% event sign agreement), thinned to every 3rd grid point for legibility.
           Source: 03_intermediate/era5_significance/netcdf/ERA5_SIG_z500_selected_lags.nc
    Row 2: 250-hPa geopotential-height anomaly with Takaya-Nakamura wave-activity flux at
           lags -5, -3, 0. Only robust vectors are drawn (Step 10D: component FDR q < 0.05 and
           >= 60% directional agreement), thinned for legibility.
           Source: 03_intermediate/tn_waf/netcdf/ERA5_TN_WAF_250hPa_selected_lags.nc

Coastlines and country borders use cartopy (Natural Earth, downloaded automatically on the
first run). If cartopy is not installed, the figure is drawn without coastlines and a warning
is printed; install it with:  pip install cartopy

Run from the project root:
    python3 07_scripts/15i_plot_onset_pathway_maps.py 2>&1 | tee 09_logs/step16i_onset_maps.log
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
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

try:
    import geopandas as gpd
    HAVE_GPD = True
except ImportError:  # pragma: no cover
    HAVE_GPD = False

SIG_FILE = C.PROJECT_ROOT / "03_intermediate" / "era5_significance" / "netcdf" / "ERA5_SIG_z500_selected_lags.nc"
WAF_FILE = C.PROJECT_ROOT / "03_intermediate" / "tn_waf" / "netcdf" / "ERA5_TN_WAF_250hPa_selected_lags.nc"
BOUNDARY = C.PROJECT_ROOT / "02_metadata" / "step7d_bangladesh_boundary.gpkg"
OUT_NAME = "fig_m3_onset_pathway_maps"

Z500_LAGS = [-10, -5, 0]
WAF_LAGS = [-5, -3, 0]
STIPPLE_STRIDE = 3
VECTOR_STRIDE = 4
MIN_VECTOR_FRACTION = 0.2  # vectors shorter than 20% of the reference vector are not drawn
MIN_VECTOR_FRACTION = 0.15  # hide robust vectors shorter than 15% of the reference vector
LAG = "lag_day_from_onset"

Z500_VARS = {"field": "composite_mean_anomaly", "mask": "robust_fdr_agreement060"}
WAF_VARS = {"field": "z_anomaly_all_event_composite", "fx": "waf_x_all_event_composite_field",
            "fy": "waf_y_all_event_composite_field", "mask": "waf_all_event_robust_vector"}

MM = 1 / 25.4
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "savefig.dpi": 300, "pdf.fonttype": 42})


def open_checked(path: Path, needed: list[str], lags: list[int]) -> xr.Dataset:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    ds = xr.open_dataset(path)
    missing = [v for v in needed if v not in ds.data_vars]
    if missing:
        raise KeyError(f"{path.name} lacks variables {missing}. Available: {list(ds.data_vars)}")
    have = [int(x) for x in ds[LAG].values]
    absent = [lag for lag in lags if lag not in have]
    if absent:
        raise KeyError(f"{path.name} lacks lags {absent}. Available lags: {have}")
    return ds


def day_label(lag: int) -> str:
    return "0" if lag == 0 else f"{lag:+d}".replace("-", "−")


def day_label(lag: int) -> str:
    return "0" if lag == 0 else f"{lag:+d}".replace("-", "\u2212")


def symmetric_limit(fields: list[np.ndarray], step: float = 10.0) -> float:
    q = np.nanpercentile(np.abs(np.concatenate([f.ravel() for f in fields])), 99)
    return max(step, np.ceil(q / step) * step)


def load_boundary():
    if not (HAVE_GPD and BOUNDARY.exists()):
        return None
    b = gpd.read_file(BOUNDARY)
    return b.to_crs(4326) if b.crs is not None else b


def make_axes(fig, gs_cell):
    if HAVE_CARTOPY:
        return fig.add_subplot(gs_cell, projection=ccrs.PlateCarree())
    return fig.add_subplot(gs_cell)


def decorate(ax, lon, lat, boundary, show_left: bool, show_bottom: bool) -> None:
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    if HAVE_CARTOPY:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5, edgecolor="0.25")
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3, edgecolor="0.5")
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.7", linestyle=":",
                          xlocs=range(30, 130, 20), ylocs=range(10, 80, 10))
        gl.top_labels = gl.right_labels = False
        gl.left_labels = show_left
        gl.bottom_labels = show_bottom
        gl.xlabel_style = gl.ylabel_style = {"size": 6.5}
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
    if boundary is not None:
        kw = {"transform": ccrs.PlateCarree()} if HAVE_CARTOPY else {}
        for geom in boundary.geometry:
            polys = getattr(geom, "geoms", [geom])
            for poly in polys:
                x, y = poly.exterior.xy
                ax.plot(x, y, color="k", lw=0.8, **kw)


def main() -> None:
    C.ensure_dirs()
    if not HAVE_CARTOPY:
        print("WARNING: cartopy not installed - drawing without coastlines. Install with: pip install cartopy")
    C.write_policy("step16i_onset_map_policy.json", {
        "step": "Step 16I", "title": "Journal-quality onset-pathway maps (redraw only)",
        "sources": {"z500": str(SIG_FILE.relative_to(C.PROJECT_ROOT)),
                    "waf": str(WAF_FILE.relative_to(C.PROJECT_ROOT))},
        "z500_lags": Z500_LAGS, "waf_lags": WAF_LAGS,
        "z500_stippling": f"{Z500_VARS['mask']} == 1, every {STIPPLE_STRIDE}rd grid point",
        "waf_vectors": f"only where {WAF_VARS['mask']} == 1 and |WAF| >= {MIN_VECTOR_FRACTION} x reference "
                       f"vector, every {VECTOR_STRIDE}rd grid point",
        "coastlines": "cartopy Natural Earth 50m" if HAVE_CARTOPY else "not drawn (cartopy unavailable)"})

    sig = open_checked(SIG_FILE, list(Z500_VARS.values()), Z500_LAGS)
    waf = open_checked(WAF_FILE, list(WAF_VARS.values()), WAF_LAGS)
    boundary = load_boundary()
    kw = {"transform": ccrs.PlateCarree()} if HAVE_CARTOPY else {}

    fig = plt.figure(figsize=(180 * MM, 100 * MM))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.045], hspace=0.12, wspace=0.08)

    # Row 1: Z500
    lon, lat = sig["lon"].values, sig["lat"].values
    fields = [sig[Z500_VARS["field"]].sel({LAG: lag}).values for lag in Z500_LAGS]
    lim = symmetric_limit(fields)
    mesh = None
    for j, (lag, f) in enumerate(zip(Z500_LAGS, fields)):
        ax = make_axes(fig, gs[0, j])
        mesh = ax.pcolormesh(lon, lat, f, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto", **kw)
        m = sig[Z500_VARS["mask"]].sel({LAG: lag}).values.astype(bool)
        m_thin = np.zeros_like(m)
        m_thin[::STIPPLE_STRIDE, ::STIPPLE_STRIDE] = m[::STIPPLE_STRIDE, ::STIPPLE_STRIDE]
        yy, xx = np.nonzero(m_thin)
        ax.scatter(lon[xx], lat[yy], s=0.6, color="k", alpha=0.55, linewidths=0, **kw)
        decorate(ax, lon, lat, boundary, show_left=(j == 0), show_bottom=False)
        ax.set_title(f"({'abc'[j]}) Z500, day {day_label(lag)}", loc="left")
    cax = fig.add_subplot(gs[0, 3])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("Z500 anomaly (m)")

    # Row 2: Z250 + WAF
    lon, lat = waf["lon"].values, waf["lat"].values
    fields = [waf[WAF_VARS["field"]].sel({LAG: lag}).values for lag in WAF_LAGS]
    lim = symmetric_limit(fields)
    fx_all = [waf[WAF_VARS["fx"]].sel({LAG: lag}).values for lag in WAF_LAGS]
    fy_all = [waf[WAF_VARS["fy"]].sel({LAG: lag}).values for lag in WAF_LAGS]
    masks = [waf[WAF_VARS["mask"]].sel({LAG: lag}).values.astype(bool) for lag in WAF_LAGS]
    mags = np.concatenate([np.hypot(fx, fy)[m].ravel() for fx, fy, m in zip(fx_all, fy_all, masks)])
    ref = float(np.nanpercentile(mags, 90)) if mags.size else 1.0
    ref = float(f"{ref:.1g}") if ref > 0 else 1.0
    LON, LAT = np.meshgrid(lon, lat)
    coverage = {lag: float(m.mean()) for lag, m in zip(WAF_LAGS, masks)}
    for lag, frac in coverage.items():
        print(f"WAF robust-vector mask covers {100 * frac:.1f}% of grid points at day {lag:+d}")
    for j, lag in enumerate(WAF_LAGS):
        ax = make_axes(fig, gs[1, j])
        mesh = ax.pcolormesh(lon, lat, fields[j], cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto", **kw)
        s = (slice(None, None, VECTOR_STRIDE), slice(None, None, VECTOR_STRIDE))
        keep = masks[j] & (np.hypot(fx_all[j], fy_all[j]) >= MIN_VECTOR_FRACTION * ref)
        fx = np.where(keep, fx_all[j], np.nan)[s]
        fy = np.where(keep, fy_all[j], np.nan)[s]
        q = ax.quiver(LON[s], LAT[s], fx, fy, scale=ref * 12, width=0.004, headwidth=3.5,
                      color="k", **kw)
        decorate(ax, lon, lat, boundary, show_left=(j == 0), show_bottom=True)
        ax.set_title(f"({'def'[j]}) Z250 + WAF, day {day_label(lag)}", loc="left")
        if j == len(WAF_LAGS) - 1:
            ax.quiverkey(q, 0.72, 1.07, ref, f"{ref:g} m² s⁻²", labelpos="E", coordinates="axes",
                         fontproperties={"size": 6.5})
    cax = fig.add_subplot(gs[1, 3])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("Z250 anomaly (m)")

    paths = [C.TERM_FIGURES / f"{OUT_NAME}.png", C.TERM_FIGURES / f"{OUT_NAME}.pdf"]
    for p in paths:
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    C.write_checksums("step16i_output_sha256.txt", paths)
    (C.TERM_QC / "step16i_waf_mask_coverage.json").write_text(
        __import__("json").dumps({f"day_{k:+d}": v for k, v in coverage.items()}, indent=2))
    print("Saved:", *[p.relative_to(C.PROJECT_ROOT) for p in paths], sep="\n  ")
    print(f"Reference vector: {ref:g} m2 s-2; Z250 colour limit: +/-{lim:g} m")


if __name__ == "__main__":
    main()
