"""
Step 16R - Study area and station network (manuscript Fig. 1).

(a) Regional domain with every analysis region: South Asia box (15-35N, 70-100E), Bangladesh box
    (20.5-26.8N, 88-93E), NW Indo-Gangetic Plain box (25-32N, 75-88E) and the jet band
    (15-45N, 70-100E). Terrain is shaded if an ERA5 orography file is present (optional).
(b) Bangladesh: the 26 primary BMD stations and their Voronoi area weights (Step 7D).

Terrain: run once with --download-etopo to fetch NOAA ETOPO1 surface elevation (dataset etopo180,
variable altitude, thinned to 0.1 degree, about 1 MB, plain HTTPS, no CDS queue) into
01_raw_data/geospatial/etopo1_south_asia_0p1deg.nc. The same file masks high terrain in Step 16S.

Run from the project root:
    python3 07_scripts/15r_plot_domain_network_map.py 2>&1 | tee 09_logs/step16r_domain_map.log
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "term_common", Path(__file__).resolve().parent / "15_termination_common.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import geopandas as gpd  # noqa: E402

META = C.PROJECT_ROOT / "02_metadata"
BOUNDARY = META / "step7d_bangladesh_boundary.gpkg"
VORONOI = META / "step7d_primary_station_voronoi_weights.gpkg"
STATIONS = META / "step6c_analytical_station_network.csv"
OROG = C.PROJECT_ROOT / "01_raw_data" / "era5" / "invariant" / "ERA5_orography.nc"
ETOPO = C.PROJECT_ROOT / "01_raw_data" / "geospatial" / "etopo1_south_asia_0p1deg.nc"
ETOPO_URL = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc?"
             "altitude%5B(0.0):6:(55.0)%5D%5B(45.0):6:(125.0)%5D")
FIG = "fig_final_01_domain_network"
BOXES = [  # label, lat0, lat1, lon0, lon1, colour, style
    ("SA box", 15.0, 35.0, 70.0, 100.0, "#0072B2", "-"),
    ("NW IGP box", 25.0, 32.0, 75.0, 88.0, "#E69F00", "-"),
    ("BD box", 20.5, 26.8, 88.0, 93.0, "#D55E00", "-"),
    ("Jet band", 15.0, 45.0, 70.0, 100.0, "#CC79A7", "fill"),
]


def download_orography() -> None:
    import cdsapi
    OROG.parent.mkdir(parents=True, exist_ok=True)
    cdsapi.Client().retrieve("reanalysis-era5-single-levels", {
        "product_type": ["reanalysis"], "variable": ["geopotential"], "year": ["2000"], "month": ["01"],
        "day": ["01"], "time": ["00:00"], "area": [50, 50, 0, 120], "data_format": "netcdf",
        "download_format": "unarchived"}, str(OROG))
    print("Saved", OROG)


def download_etopo() -> None:
    import urllib.request
    ETOPO.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading ETOPO1 subset from NOAA ERDDAP ...", flush=True)
    with urllib.request.urlopen(ETOPO_URL, timeout=300) as r:
        ETOPO.write_bytes(r.read())
    print("Saved", ETOPO, f"({ETOPO.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-orography", action="store_true", help="ERA5 geopotential via CDS (slow queue)")
    ap.add_argument("--download-etopo", action="store_true", help="NOAA ETOPO1 via ERDDAP (fast, recommended)")
    args = ap.parse_args()
    if args.download_etopo and not ETOPO.exists():
        download_etopo()
    if args.download_orography and not OROG.exists():
        download_orography()
    C.ensure_dirs()

    MM = 1 / 25.4
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8,
                         "savefig.dpi": 300, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(180 * MM, 82 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.28)
    pc = ccrs.PlateCarree()

    ax = fig.add_subplot(gs[0], projection=pc)
    ax.set_extent([55, 112, 5, 47], crs=pc)
    if ETOPO.exists():
        import xarray as xr
        from matplotlib.colors import LightSource
        with xr.open_dataset(ETOPO) as e:
            alt = e["altitude"].astype("float32")
            lon_e, lat_e = alt["longitude"].values, alt["latitude"].values
            z = alt.values
        land = np.where(z > 0, z, np.nan)
        m = ax.pcolormesh(lon_e, lat_e, land, cmap="YlOrBr", vmin=0, vmax=6000, shading="auto",
                          transform=pc, alpha=0.85, zorder=1)
        shade = LightSource(azdeg=315, altdeg=45).hillshade(np.nan_to_num(np.clip(z, 0, None)), vert_exag=0.02)
        ax.pcolormesh(lon_e, lat_e, np.where(z > 0, shade, np.nan), cmap="Greys_r", vmin=0, vmax=1,
                      shading="auto", transform=pc, alpha=0.25, zorder=2)
        cb = fig.colorbar(m, ax=ax, orientation="horizontal", fraction=0.045, pad=0.08, shrink=0.6)
        cb.set_label("Surface elevation (m, ETOPO1)", fontsize=7)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#e6f0f7", zorder=3)
    elif OROG.exists():
        import xarray as xr
        with xr.open_dataset(OROG) as o:
            zname = "z" if "z" in o else list(o.data_vars)[0]
            z = o[zname].squeeze() / 9.80665
            latn = "latitude" if "latitude" in z.coords else "lat"
            lonn = "longitude" if "longitude" in z.coords else "lon"
            h = z.where(z > 0)
            m = ax.pcolormesh(z[lonn], z[latn], h, cmap="Greys", vmin=0, vmax=6000, shading="auto",
                              transform=pc, alpha=0.75)
            cb = fig.colorbar(m, ax=ax, orientation="horizontal", fraction=0.045, pad=0.08, shrink=0.6)
            cb.set_label("Surface elevation (m)", fontsize=7)
    else:
        print("Note: no terrain file; drawing without terrain (run with --download-etopo to add it).")
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f2efe9")
    if not ETOPO.exists():
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#e6f0f7")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5, zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3, edgecolor="0.4", zorder=4)
    handles = []
    for name, la0, la1, lo0, lo1, col, ls in BOXES:
        if ls == "fill":  # jet band: translucent fill so its edges do not overlap the SA box
            ax.add_patch(mpatches.Rectangle((lo0, la0), lo1 - lo0, la1 - la0, facecolor=col, alpha=0.22,
                                            edgecolor="none", transform=pc, zorder=5))
            handles.append(mpatches.Patch(facecolor=col, alpha=0.4, edgecolor="none", label=name))
            continue
        ax.add_patch(mpatches.Rectangle((lo0, la0), lo1 - lo0, la1 - la0, fill=False, edgecolor=col,
                                        linewidth=1.4, linestyle=ls, transform=pc, zorder=6))
        handles.append(mpatches.Patch(facecolor="none", edgecolor=col, linestyle=ls, linewidth=1.4, label=name))
    ax.legend(handles=handles, loc="lower left", fontsize=6, frameon=True, framealpha=0.9,
              handlelength=1.4, borderpad=0.4, labelspacing=0.3, borderaxespad=0.3).set_zorder(10)
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color="0.6", linestyle=":")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 6.5}
    ax.set_title("(a) Analysis regions", loc="left")

    ax = fig.add_subplot(gs[1], projection=pc)
    ax.set_extent([87.8, 93.0, 20.5, 26.8], crs=pc)
    vor = gpd.read_file(VORONOI).to_crs(4326)
    wcol = next(c for c in vor.columns if "weight" in c.lower())
    vals = vor[wcol] * (100 if vor[wcol].max() <= 1 else 1)
    vor.assign(w=vals).plot(ax=ax, column="w", cmap="YlGnBu", edgecolor="white", linewidth=0.4,
                            transform=pc, legend=True,
                            legend_kwds={"label": "Voronoi area weight (%)", "shrink": 0.6})
    gpd.read_file(BOUNDARY).to_crs(4326).boundary.plot(ax=ax, color="k", linewidth=0.8, transform=pc)
    st = pd.read_csv(STATIONS)
    primary = set(pd.read_csv(C.STATION_WEIGHTS)["station_uid"])
    st = st[st["station_uid"].isin(primary)]
    ax.scatter(st["longitude"], st["latitude"], marker="^", s=14, color="#D55E00", edgecolor="k",
               linewidth=0.4, transform=pc, zorder=5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color="0.6", linestyle=":")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 6.5}
    ax.set_title(f"(b) {len(st)} BMD stations and area weights", loc="left")

    paths = [C.TERM_FIGURES / f"{FIG}.png", C.TERM_FIGURES / f"{FIG}.pdf"]
    for p in paths:
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    C.write_checksums("step16r_output_sha256.txt", paths)
    print("Saved", *[p.relative_to(C.PROJECT_ROOT) for p in paths], sep="\n  ")


if __name__ == "__main__":
    main()
