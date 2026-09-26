from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVENT_LAG_TABLE = (
    PROJECT_ROOT
    / "03_intermediate"
    / "era5_event_composites"
    / "composite_tables"
    / "step9d_event_lag_date_table.csv"
)

PL_STD_DIR = (
    PROJECT_ROOT
    / "03_intermediate"
    / "era5_preprocessed"
    / "pressure_level_standardized"
)

SL_STD_DIR = (
    PROJECT_ROOT
    / "03_intermediate"
    / "era5_preprocessed"
    / "single_level_instant_standardized"
)

CLIM_ROOT = (
    PROJECT_ROOT
    / "03_intermediate"
    / "era5_climatology"
)

OUT_ROOT = (
    PROJECT_ROOT
    / "03_intermediate"
    / "era5_significance"
)

NETCDF_DIR = OUT_ROOT / "netcdf"
TABLE_DIR = PROJECT_ROOT / "08_outputs" / "tables" / "era5"
REPORT_DIR = PROJECT_ROOT / "05_qc_reports" / "era5"

NETCDF_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BOX_RESULTS_CSV = (
    TABLE_DIR
    / "table_13_era5_box_bootstrap_robustness.csv"
)

AREA_FRACTION_CSV = (
    TABLE_DIR
    / "table_14_era5_gridpoint_significant_area_fraction.csv"
)

AGREEMENT_SUMMARY_CSV = (
    TABLE_DIR
    / "table_15_era5_event_agreement_summary.csv"
)

EVENT_LEVEL_CSV = (
    OUT_ROOT
    / "step9g_event_level_box_anomalies.csv"
)

LOG_CSV = (
    REPORT_DIR
    / "step9g_era5_bootstrap_significance_log.csv"
)

SUMMARY_JSON = (
    REPORT_DIR
    / "step9g_era5_bootstrap_significance_summary.json"
)

REPORT_TXT = (
    REPORT_DIR
    / "step9g_era5_bootstrap_significance_report.txt"
)


SELECTED_LAGS = [-10, -5, 0, 5]

COARSEN_FACTOR = 4

N_BOOTSTRAP_MAP = 1000
N_BOOTSTRAP_SCALAR = 5000
BOOTSTRAP_BATCH_SIZE = 25

RANDOM_SEED = 20260711
ALPHA = 0.05

BANGLADESH_BOX = {
    "name": "Bangladesh box",
    "lat_min": 20.5,
    "lat_max": 26.8,
    "lon_min": 88.0,
    "lon_max": 93.0,
}

SOUTH_ASIA_BOX = {
    "name": "South Asia box",
    "lat_min": 15.0,
    "lat_max": 35.0,
    "lon_min": 70.0,
    "lon_max": 100.0,
}

REGIONS = [
    {
        "name": "Full ERA5 domain",
        "lat_min": 10.0,
        "lat_max": 70.0,
        "lon_min": 30.0,
        "lon_max": 120.0,
    },
    BANGLADESH_BOX,
    SOUTH_ASIA_BOX,
]


DIAGNOSTICS = {
    "t2m": {
        "label": "T2m anomaly",
        "units": "degree_Celsius",
        "group": "single_level_instant",
        "variable": "t2m_c",
        "level_hpa": None,
        "input_dir": SL_STD_DIR,
        "input_prefix": "ERA5_SL_INSTANT_STD",
    },
    "mslp": {
        "label": "MSLP anomaly",
        "units": "hPa",
        "group": "single_level_instant",
        "variable": "msl_hpa",
        "level_hpa": None,
        "input_dir": SL_STD_DIR,
        "input_prefix": "ERA5_SL_INSTANT_STD",
    },
    "z500": {
        "label": "500 hPa geopotential-height anomaly",
        "units": "m",
        "group": "pressure_level",
        "variable": "geopotential_height_m",
        "level_hpa": 500,
        "input_dir": PL_STD_DIR,
        "input_prefix": "ERA5_PL_STD",
    },
    "t850": {
        "label": "850 hPa air-temperature anomaly",
        "units": "K",
        "group": "pressure_level",
        "variable": "air_temperature_k",
        "level_hpa": 850,
        "input_dir": PL_STD_DIR,
        "input_prefix": "ERA5_PL_STD",
    },
    "u200": {
        "label": "200 hPa zonal-wind anomaly",
        "units": "m s-1",
        "group": "pressure_level",
        "variable": "u_wind_ms",
        "level_hpa": 200,
        "input_dir": PL_STD_DIR,
        "input_prefix": "ERA5_PL_STD",
    },
}


def safe_variable_name(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace(" ", "_")
        .replace("-", "minus")
        .replace(".", "_")
    )


def climatology_path(config: dict) -> Path:
    safe = safe_variable_name(config["variable"])

    return (
        CLIM_ROOT
        / config["group"]
        / (
            f"ERA5_CLIM_{config['group']}_{safe}"
            "_calendar_day_1991_2020.nc"
        )
    )


def input_file(config: dict, year: int, month: int) -> Path:
    return (
        config["input_dir"]
        / f"{config['input_prefix']}_{year}_{month:02d}.nc"
    )


def coarsen_field(da: xr.DataArray) -> xr.DataArray:
    return (
        da.coarsen(
            lat=COARSEN_FACTOR,
            lon=COARSEN_FACTOR,
            boundary="trim",
        )
        .mean(skipna=True)
        .load()
    )


def region_mask(
    lat: np.ndarray,
    lon: np.ndarray,
    region: dict,
) -> np.ndarray:
    lat_mask = (
        (lat >= region["lat_min"])
        & (lat <= region["lat_max"])
    )

    lon_mask = (
        (lon >= region["lon_min"])
        & (lon <= region["lon_max"])
    )

    return lat_mask[:, None] & lon_mask[None, :]


def cosine_weights(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return (
        np.cos(np.deg2rad(lat))[:, None]
        * np.ones((1, len(lon)), dtype="float64")
    )


def area_mean_events(
    event_data: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    region: dict,
) -> np.ndarray:
    mask = region_mask(lat, lon, region)
    weights = cosine_weights(lat, lon) * mask

    valid = np.isfinite(event_data)

    numerator = np.nansum(
        event_data * weights[None, :, :],
        axis=(1, 2),
    )

    denominator = np.sum(
        valid * weights[None, :, :],
        axis=(1, 2),
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = numerator / denominator

    return mean


def extract_event_anomalies(
    diagnostic_key: str,
    config: dict,
    lag: int,
    event_lag: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    subset = (
        event_lag.loc[
            event_lag["lag_day_from_onset"].eq(lag)
        ]
        .copy()
        .sort_values("event_id")
        .reset_index(drop=True)
    )

    if len(subset) != 85:
        raise RuntimeError(
            f"{diagnostic_key}, lag {lag}: expected 85 events, "
            f"found {len(subset)}"
        )

    clim_path = climatology_path(config)

    if not clim_path.exists():
        raise FileNotFoundError(clim_path)

    event_data = None
    lat = None
    lon = None

    with xr.open_dataset(clim_path) as clim_ds:
        climatology = clim_ds[config["variable"]]

        if config["level_hpa"] is not None:
            climatology = climatology.sel(
                level_hpa=config["level_hpa"]
            )

        for (year, month), month_rows in subset.groupby(
            ["year", "month"],
            sort=True,
        ):
            year = int(year)
            month = int(month)

            path = input_file(config, year, month)

            if not path.exists():
                raise FileNotFoundError(path)

            with xr.open_dataset(path) as ds:
                field_source = ds[config["variable"]]

                if config["level_hpa"] is not None:
                    field_source = field_source.sel(
                        level_hpa=config["level_hpa"]
                    )

                for row_index, row in month_rows.iterrows():
                    date = pd.Timestamp(row["era5_date"])
                    calendar_day = str(row["calendar_day"])

                    field = field_source.sel(
                        time=np.datetime64(date.date())
                    )

                    normal = climatology.sel(
                        calendar_day=calendar_day
                    )

                    anomaly = coarsen_field(field - normal)

                    if event_data is None:
                        lat = anomaly["lat"].values.astype("float32")
                        lon = anomaly["lon"].values.astype("float32")

                        event_data = np.full(
                            (
                                len(subset),
                                len(lat),
                                len(lon),
                            ),
                            np.nan,
                            dtype="float32",
                        )

                    event_data[row_index, :, :] = (
                        anomaly.values.astype("float32")
                    )

            gc.collect()

    if event_data is None or lat is None or lon is None:
        raise RuntimeError(
            f"{diagnostic_key}, lag {lag}: no event data extracted."
        )

    valid_events = np.isfinite(event_data).any(axis=(1, 2))

    if int(valid_events.sum()) != 85:
        raise RuntimeError(
            f"{diagnostic_key}, lag {lag}: only "
            f"{int(valid_events.sum())} events contain valid fields."
        )

    return event_data, subset, lat, lon


def aggregate_by_winter(
    event_data: np.ndarray,
    winters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_winters, winter_inverse = np.unique(
        winters,
        return_inverse=True,
    )

    winter_sum = np.zeros(
        (
            len(unique_winters),
            event_data.shape[1],
            event_data.shape[2],
        ),
        dtype="float32",
    )

    winter_count = np.zeros(
        (
            len(unique_winters),
            event_data.shape[1],
            event_data.shape[2],
        ),
        dtype="float32",
    )

    for winter_index in range(len(unique_winters)):
        subset = event_data[
            winter_inverse == winter_index
        ]

        winter_sum[winter_index] = np.nansum(
            subset,
            axis=0,
        ).astype("float32")

        winter_count[winter_index] = np.sum(
            np.isfinite(subset),
            axis=0,
        ).astype("float32")

    return unique_winters, winter_sum, winter_count


def bootstrap_map(
    event_data: np.ndarray,
    winters: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    composite = np.nanmean(
        event_data,
        axis=0,
    ).astype("float32")

    valid_event_count = np.sum(
        np.isfinite(event_data),
        axis=0,
    ).astype("int16")

    composite_sign = np.sign(composite)

    positive_events = np.sum(
        event_data > 0,
        axis=0,
    )

    negative_events = np.sum(
        event_data < 0,
        axis=0,
    )

    same_sign_count = np.where(
        composite_sign > 0,
        positive_events,
        np.where(
            composite_sign < 0,
            negative_events,
            0,
        ),
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        event_agreement = (
            same_sign_count / valid_event_count
        )

    event_agreement = event_agreement.astype("float32")
    event_agreement[valid_event_count == 0] = np.nan
    event_agreement[composite_sign == 0] = np.nan

    (
        unique_winters,
        winter_sum,
        winter_count,
    ) = aggregate_by_winter(event_data, winters)

    number_of_winters = len(unique_winters)

    rng = np.random.default_rng(seed)

    positive_bootstrap_count = np.zeros(
        composite.shape,
        dtype="int32",
    )

    negative_bootstrap_count = np.zeros(
        composite.shape,
        dtype="int32",
    )

    bootstrap_sum = np.zeros(
        composite.shape,
        dtype="float64",
    )

    bootstrap_sumsq = np.zeros(
        composite.shape,
        dtype="float64",
    )

    completed = 0

    while completed < N_BOOTSTRAP_MAP:
        batch_size = min(
            BOOTSTRAP_BATCH_SIZE,
            N_BOOTSTRAP_MAP - completed,
        )

        sampled = rng.integers(
            0,
            number_of_winters,
            size=(batch_size, number_of_winters),
        )

        weights = np.zeros(
            (batch_size, number_of_winters),
            dtype="float32",
        )

        for batch_index in range(batch_size):
            weights[batch_index] = np.bincount(
                sampled[batch_index],
                minlength=number_of_winters,
            ).astype("float32")

        numerator = np.tensordot(
            weights,
            winter_sum,
            axes=(1, 0),
        )

        denominator = np.tensordot(
            weights,
            winter_count,
            axes=(1, 0),
        )

        with np.errstate(invalid="ignore", divide="ignore"):
            bootstrap_means = numerator / denominator

        positive_bootstrap_count += np.sum(
            bootstrap_means > 0,
            axis=0,
        ).astype("int32")

        negative_bootstrap_count += np.sum(
            bootstrap_means < 0,
            axis=0,
        ).astype("int32")

        bootstrap_sum += np.nansum(
            bootstrap_means,
            axis=0,
        )

        bootstrap_sumsq += np.nansum(
            bootstrap_means ** 2,
            axis=0,
        )

        completed += batch_size

        del numerator, denominator, bootstrap_means
        gc.collect()

    bootstrap_mean = (
        bootstrap_sum / N_BOOTSTRAP_MAP
    )

    bootstrap_variance = (
        bootstrap_sumsq
        - (bootstrap_sum ** 2 / N_BOOTSTRAP_MAP)
    ) / max(N_BOOTSTRAP_MAP - 1, 1)

    bootstrap_variance = np.maximum(
        bootstrap_variance,
        0,
    )

    bootstrap_se = np.sqrt(
        bootstrap_variance
    )

    positive_probability = (
        positive_bootstrap_count / N_BOOTSTRAP_MAP
    )

    negative_probability = (
        negative_bootstrap_count / N_BOOTSTRAP_MAP
    )

    p_value = 2 * np.minimum(
        positive_probability,
        negative_probability,
    )

    p_value = np.clip(
        p_value,
        0,
        1,
    )

    return {
        "composite_mean_anomaly": composite,
        "bootstrap_mean_anomaly": bootstrap_mean.astype("float32"),
        "bootstrap_standard_error": bootstrap_se.astype("float32"),
        "bootstrap_positive_probability": positive_probability.astype("float32"),
        "bootstrap_two_sided_p": p_value.astype("float32"),
        "event_agreement_fraction": event_agreement,
        "valid_event_count": valid_event_count,
    }


def benjamini_hochberg_q(
    p_values: np.ndarray,
) -> np.ndarray:
    flat = p_values.ravel()
    finite = np.isfinite(flat)

    q_flat = np.full(
        flat.shape,
        np.nan,
        dtype="float64",
    )

    p = flat[finite]

    if len(p) == 0:
        return q_flat.reshape(p_values.shape).astype("float32")

    order = np.argsort(p)
    sorted_p = p[order]

    ranks = np.arange(
        1,
        len(sorted_p) + 1,
        dtype="float64",
    )

    sorted_q = (
        sorted_p * len(sorted_p) / ranks
    )

    sorted_q = np.minimum.accumulate(
        sorted_q[::-1]
    )[::-1]

    sorted_q = np.clip(
        sorted_q,
        0,
        1,
    )

    unsorted_q = np.empty_like(sorted_q)
    unsorted_q[order] = sorted_q

    q_flat[finite] = unsorted_q

    return q_flat.reshape(
        p_values.shape
    ).astype("float32")


def scalar_winter_bootstrap(
    values: np.ndarray,
    winters: np.ndarray,
    seed: int,
) -> dict:
    valid = np.isfinite(values)

    values = values[valid]
    winters = winters[valid]

    unique_winters = np.unique(winters)

    winter_sums = np.array(
        [
            np.sum(values[winters == winter])
            for winter in unique_winters
        ],
        dtype="float64",
    )

    winter_counts = np.array(
        [
            np.sum(winters == winter)
            for winter in unique_winters
        ],
        dtype="float64",
    )

    rng = np.random.default_rng(seed)

    bootstrap_values = np.empty(
        N_BOOTSTRAP_SCALAR,
        dtype="float64",
    )

    for bootstrap_index in range(N_BOOTSTRAP_SCALAR):
        sampled = rng.integers(
            0,
            len(unique_winters),
            size=len(unique_winters),
        )

        bootstrap_values[bootstrap_index] = (
            winter_sums[sampled].sum()
            / winter_counts[sampled].sum()
        )

    lower = float(
        np.nanpercentile(
            bootstrap_values,
            2.5,
        )
    )

    upper = float(
        np.nanpercentile(
            bootstrap_values,
            97.5,
        )
    )

    positive_probability = float(
        np.mean(bootstrap_values > 0)
    )

    negative_probability = float(
        np.mean(bootstrap_values < 0)
    )

    p_value = float(
        min(
            1,
            2 * min(
                positive_probability,
                negative_probability,
            ),
        )
    )

    return {
        "bootstrap_mean": float(
            np.mean(bootstrap_values)
        ),
        "bootstrap_ci_lower": lower,
        "bootstrap_ci_upper": upper,
        "bootstrap_p": p_value,
        "bootstrap_significant_95": bool(
            lower > 0 or upper < 0
        ),
    }


def scalar_robustness(
    values: np.ndarray,
    winters: np.ndarray,
) -> dict:
    valid = np.isfinite(values)

    values = values[valid]
    winters = winters[valid]

    event_weighted_mean = float(
        np.mean(values)
    )

    unique_winters = np.unique(winters)

    winter_means = np.array(
        [
            np.mean(values[winters == winter])
            for winter in unique_winters
        ],
        dtype="float64",
    )

    winter_equal_mean = float(
        np.mean(winter_means)
    )

    leave_one_out = []

    for winter in unique_winters:
        keep = winters != winter
        leave_one_out.append(
            float(np.mean(values[keep]))
        )

    leave_one_out = np.array(
        leave_one_out,
        dtype="float64",
    )

    full_sign = np.sign(event_weighted_mean)

    leave_one_out_same_sign_fraction = float(
        np.mean(
            np.sign(leave_one_out) == full_sign
        )
    )

    event_sign_agreement = float(
        np.mean(
            np.sign(values) == full_sign
        )
    )

    return {
        "event_weighted_mean": event_weighted_mean,
        "winter_equal_mean": winter_equal_mean,
        "event_minus_winter_equal": (
            event_weighted_mean
            - winter_equal_mean
        ),
        "event_sign_agreement_fraction": event_sign_agreement,
        "leave_one_winter_out_minimum": float(
            leave_one_out.min()
        ),
        "leave_one_winter_out_maximum": float(
            leave_one_out.max()
        ),
        "leave_one_winter_out_same_sign_fraction": (
            leave_one_out_same_sign_fraction
        ),
        "number_of_events": int(len(values)),
        "number_of_winters": int(len(unique_winters)),
    }


def weighted_mask_fraction(
    condition: np.ndarray,
    valid: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    region: dict,
) -> float:
    mask = region_mask(
        lat,
        lon,
        region,
    )

    weights = cosine_weights(
        lat,
        lon,
    )

    usable = mask & valid

    denominator = float(
        np.sum(weights[usable])
    )

    if denominator == 0:
        return np.nan

    numerator = float(
        np.sum(
            weights[
                usable & condition
            ]
        )
    )

    return numerator / denominator


def write_diagnostic_netcdf(
    diagnostic_key: str,
    config: dict,
    lat: np.ndarray,
    lon: np.ndarray,
    lag_results: list[dict],
) -> Path:
    output = (
        NETCDF_DIR
        / f"ERA5_SIG_{diagnostic_key}_selected_lags.nc"
    )

    dataset = xr.Dataset(
        coords={
            "lag_day_from_onset": np.array(
                SELECTED_LAGS,
                dtype="int32",
            ),
            "lat": lat.astype("float32"),
            "lon": lon.astype("float32"),
        }
    )

    float_variables = [
        "composite_mean_anomaly",
        "bootstrap_mean_anomaly",
        "bootstrap_standard_error",
        "bootstrap_positive_probability",
        "bootstrap_two_sided_p",
        "bootstrap_fdr_q",
        "event_agreement_fraction",
    ]

    for variable in float_variables:
        dataset[variable] = (
            (
                "lag_day_from_onset",
                "lat",
                "lon",
            ),
            np.stack(
                [
                    result[variable]
                    for result in lag_results
                ],
                axis=0,
            ).astype("float32"),
        )

    integer_variables = [
        "valid_event_count",
        "significant_raw_05",
        "significant_fdr_05",
        "agreement_at_least_055",
        "agreement_at_least_060",
        "agreement_at_least_067",
        "robust_fdr_agreement060",
    ]

    for variable in integer_variables:
        dtype = (
            "int16"
            if variable == "valid_event_count"
            else "uint8"
        )

        dataset[variable] = (
            (
                "lag_day_from_onset",
                "lat",
                "lon",
            ),
            np.stack(
                [
                    result[variable]
                    for result in lag_results
                ],
                axis=0,
            ).astype(dtype),
        )

    dataset.attrs.update(
        {
            "title": (
                "ERA5 winter-block bootstrap significance: "
                + config["label"]
            ),
            "diagnostic_key": diagnostic_key,
            "source_variable": config["variable"],
            "pressure_level_hpa": (
                "none"
                if config["level_hpa"] is None
                else str(config["level_hpa"])
            ),
            "units": config["units"],
            "event_count": 85,
            "bootstrap_repetitions": N_BOOTSTRAP_MAP,
            "bootstrap_unit": "winter",
            "fdr_method": "Benjamini-Hochberg",
            "primary_robust_mask": (
                "bootstrap_fdr_q < 0.05 and "
                "event_agreement_fraction >= 0.60"
            ),
            "input_grid": "0.25 degree",
            "bootstrap_grid": "approximately 1 degree",
            "created_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        }
    )

    encoding = {}

    for variable in dataset.data_vars:
        encoding[variable] = {
            "zlib": True,
            "complevel": 4,
        }

    temporary = output.with_suffix(".nc.part")

    if temporary.exists():
        temporary.unlink()

    dataset.to_netcdf(
        temporary,
        engine="netcdf4",
        encoding=encoding,
    )

    temporary.replace(output)
    dataset.close()

    return output


def main() -> None:
    print("STEP 9G: ERA5 BOOTSTRAP SIGNIFICANCE AND ROBUSTNESS")
    print("=" * 70)
    print("Primary bootstrap unit: winter")
    print(f"Map bootstrap repetitions: {N_BOOTSTRAP_MAP}")
    print(f"Box bootstrap repetitions: {N_BOOTSTRAP_SCALAR}")
    print(f"Selected lags: {SELECTED_LAGS}")
    print("Bootstrap grid: approximately 1 degree")
    print("")

    event_lag = pd.read_csv(EVENT_LAG_TABLE)

    event_lag["era5_date"] = pd.to_datetime(
        event_lag["era5_date"]
    )

    box_rows = []
    event_rows = []
    area_fraction_rows = []
    agreement_rows = []
    log_rows = []

    for diagnostic_index, (
        diagnostic_key,
        config,
    ) in enumerate(DIAGNOSTICS.items()):
        print("")
        print("=" * 70)
        print(
            f"DIAGNOSTIC: {diagnostic_key} — "
            f"{config['label']}"
        )
        print("=" * 70)

        diagnostic_lag_results = []
        diagnostic_lat = None
        diagnostic_lon = None

        for lag_index, lag in enumerate(SELECTED_LAGS):
            print("")
            print(
                f"Processing {diagnostic_key}, lag {lag:+d}"
            )

            seed = (
                RANDOM_SEED
                + diagnostic_index * 1000
                + lag_index * 100
            )

            (
                event_data,
                metadata,
                lat,
                lon,
            ) = extract_event_anomalies(
                diagnostic_key=diagnostic_key,
                config=config,
                lag=lag,
                event_lag=event_lag,
            )

            diagnostic_lat = lat
            diagnostic_lon = lon

            winters = metadata[
                "winter_label"
            ].astype(str).to_numpy()

            result = bootstrap_map(
                event_data=event_data,
                winters=winters,
                seed=seed,
            )

            q_value = benjamini_hochberg_q(
                result["bootstrap_two_sided_p"]
            )

            result["bootstrap_fdr_q"] = q_value

            result["significant_raw_05"] = (
                result["bootstrap_two_sided_p"]
                < ALPHA
            ).astype("uint8")

            result["significant_fdr_05"] = (
                q_value < ALPHA
            ).astype("uint8")

            result["agreement_at_least_055"] = (
                result["event_agreement_fraction"]
                >= 0.55
            ).astype("uint8")

            result["agreement_at_least_060"] = (
                result["event_agreement_fraction"]
                >= 0.60
            ).astype("uint8")

            result["agreement_at_least_067"] = (
                result["event_agreement_fraction"]
                >= 0.67
            ).astype("uint8")

            result["robust_fdr_agreement060"] = (
                (q_value < ALPHA)
                & (
                    result["event_agreement_fraction"]
                    >= 0.60
                )
            ).astype("uint8")

            diagnostic_lag_results.append(result)

            valid_grid = np.isfinite(
                result["composite_mean_anomaly"]
            )

            for region in REGIONS:
                region_name = region["name"]

                raw_fraction = weighted_mask_fraction(
                    condition=(
                        result["significant_raw_05"]
                        == 1
                    ),
                    valid=valid_grid,
                    lat=lat,
                    lon=lon,
                    region=region,
                )

                fdr_fraction = weighted_mask_fraction(
                    condition=(
                        result["significant_fdr_05"]
                        == 1
                    ),
                    valid=valid_grid,
                    lat=lat,
                    lon=lon,
                    region=region,
                )

                agreement_fraction = weighted_mask_fraction(
                    condition=(
                        result["agreement_at_least_060"]
                        == 1
                    ),
                    valid=valid_grid,
                    lat=lat,
                    lon=lon,
                    region=region,
                )

                robust_fraction = weighted_mask_fraction(
                    condition=(
                        result[
                            "robust_fdr_agreement060"
                        ]
                        == 1
                    ),
                    valid=valid_grid,
                    lat=lat,
                    lon=lon,
                    region=region,
                )

                robust_positive_fraction = (
                    weighted_mask_fraction(
                        condition=(
                            (
                                result[
                                    "robust_fdr_agreement060"
                                ]
                                == 1
                            )
                            & (
                                result[
                                    "composite_mean_anomaly"
                                ]
                                > 0
                            )
                        ),
                        valid=valid_grid,
                        lat=lat,
                        lon=lon,
                        region=region,
                    )
                )

                robust_negative_fraction = (
                    weighted_mask_fraction(
                        condition=(
                            (
                                result[
                                    "robust_fdr_agreement060"
                                ]
                                == 1
                            )
                            & (
                                result[
                                    "composite_mean_anomaly"
                                ]
                                < 0
                            )
                        ),
                        valid=valid_grid,
                        lat=lat,
                        lon=lon,
                        region=region,
                    )
                )

                area_fraction_rows.append(
                    {
                        "diagnostic": diagnostic_key,
                        "diagnostic_label": config["label"],
                        "lag_day_from_onset": lag,
                        "region": region_name,
                        "raw_p05_area_fraction": raw_fraction,
                        "fdr_q05_area_fraction": fdr_fraction,
                        "agreement_ge060_area_fraction": agreement_fraction,
                        "robust_fdr_agreement060_area_fraction": robust_fraction,
                        "robust_positive_area_fraction": robust_positive_fraction,
                        "robust_negative_area_fraction": robust_negative_fraction,
                    }
                )

                mask = region_mask(
                    lat,
                    lon,
                    region,
                )

                agreement_values = (
                    result["event_agreement_fraction"][
                        mask
                    ]
                )

                agreement_values = agreement_values[
                    np.isfinite(agreement_values)
                ]

                agreement_rows.append(
                    {
                        "diagnostic": diagnostic_key,
                        "diagnostic_label": config["label"],
                        "lag_day_from_onset": lag,
                        "region": region_name,
                        "mean_event_agreement": float(
                            np.mean(agreement_values)
                        ),
                        "median_event_agreement": float(
                            np.median(agreement_values)
                        ),
                        "agreement_25th_percentile": float(
                            np.percentile(
                                agreement_values,
                                25,
                            )
                        ),
                        "agreement_75th_percentile": float(
                            np.percentile(
                                agreement_values,
                                75,
                            )
                        ),
                    }
                )

            for box_index, box in enumerate(
                [
                    BANGLADESH_BOX,
                    SOUTH_ASIA_BOX,
                ]
            ):
                values = area_mean_events(
                    event_data=event_data,
                    lat=lat,
                    lon=lon,
                    region=box,
                )

                bootstrap = scalar_winter_bootstrap(
                    values=values,
                    winters=winters,
                    seed=seed + 10 + box_index,
                )

                robustness = scalar_robustness(
                    values=values,
                    winters=winters,
                )

                robust_primary = bool(
                    bootstrap[
                        "bootstrap_significant_95"
                    ]
                    and robustness[
                        "event_sign_agreement_fraction"
                    ] >= 0.60
                    and robustness[
                        "leave_one_winter_out_same_sign_fraction"
                    ] == 1.0
                    and np.sign(
                        robustness[
                            "event_weighted_mean"
                        ]
                    )
                    == np.sign(
                        robustness[
                            "winter_equal_mean"
                        ]
                    )
                )

                box_rows.append(
                    {
                        "diagnostic": diagnostic_key,
                        "diagnostic_label": config["label"],
                        "units": config["units"],
                        "lag_day_from_onset": lag,
                        "box": box["name"],
                        **robustness,
                        **bootstrap,
                        "primary_robust_result": robust_primary,
                    }
                )

                for event_index, event_value in enumerate(
                    values
                ):
                    event_rows.append(
                        {
                            "event_id": metadata.iloc[
                                event_index
                            ]["event_id"],
                            "winter_label": winters[
                                event_index
                            ],
                            "diagnostic": diagnostic_key,
                            "diagnostic_label": config["label"],
                            "units": config["units"],
                            "lag_day_from_onset": lag,
                            "box": box["name"],
                            "event_anomaly": float(
                                event_value
                            ),
                        }
                    )

            log_rows.append(
                {
                    "diagnostic": diagnostic_key,
                    "lag_day_from_onset": lag,
                    "event_count": 85,
                    "winter_count": int(
                        len(np.unique(winters))
                    ),
                    "lat_count": len(lat),
                    "lon_count": len(lon),
                    "minimum_valid_event_count": int(
                        np.nanmin(
                            result[
                                "valid_event_count"
                            ]
                        )
                    ),
                    "maximum_valid_event_count": int(
                        np.nanmax(
                            result[
                                "valid_event_count"
                            ]
                        )
                    ),
                    "status": "completed",
                }
            )

            pd.DataFrame(log_rows).to_csv(
                LOG_CSV,
                index=False,
            )

            del event_data
            gc.collect()

        output_path = write_diagnostic_netcdf(
            diagnostic_key=diagnostic_key,
            config=config,
            lat=diagnostic_lat,
            lon=diagnostic_lon,
            lag_results=diagnostic_lag_results,
        )

        print(f"Written: {output_path}")

        del diagnostic_lag_results
        gc.collect()

    box_table = pd.DataFrame(box_rows)
    area_table = pd.DataFrame(area_fraction_rows)
    agreement_table = pd.DataFrame(agreement_rows)
    event_table = pd.DataFrame(event_rows)
    log_table = pd.DataFrame(log_rows)

    box_table.to_csv(
        BOX_RESULTS_CSV,
        index=False,
    )

    area_table.to_csv(
        AREA_FRACTION_CSV,
        index=False,
    )

    agreement_table.to_csv(
        AGREEMENT_SUMMARY_CSV,
        index=False,
    )

    event_table.to_csv(
        EVENT_LEVEL_CSV,
        index=False,
    )

    log_table.to_csv(
        LOG_CSV,
        index=False,
    )

    robust_count = int(
        box_table[
            "primary_robust_result"
        ].sum()
    )

    summary = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "event_count": 85,
        "selected_lags": SELECTED_LAGS,
        "diagnostics": list(
            DIAGNOSTICS.keys()
        ),
        "bootstrap_unit": "winter",
        "map_bootstrap_repetitions": N_BOOTSTRAP_MAP,
        "box_bootstrap_repetitions": N_BOOTSTRAP_SCALAR,
        "bootstrap_grid": "approximately 1 degree",
        "fdr_alpha": ALPHA,
        "box_results_count": int(
            len(box_table)
        ),
        "primary_robust_box_results": robust_count,
        "netcdf_files": [
            str(path)
            for path in sorted(
                NETCDF_DIR.glob(
                    "ERA5_SIG_*_selected_lags.nc"
                )
            )
        ],
        "box_results_csv": str(
            BOX_RESULTS_CSV
        ),
        "area_fraction_csv": str(
            AREA_FRACTION_CSV
        ),
        "agreement_summary_csv": str(
            AGREEMENT_SUMMARY_CSV
        ),
        "event_level_csv": str(
            EVENT_LEVEL_CSV
        ),
        "log_csv": str(
            LOG_CSV
        ),
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "STEP 9G: ERA5 BOOTSTRAP SIGNIFICANCE AND ROBUSTNESS",
        "=" * 70,
        "Status: completed",
        "Primary bootstrap unit: winter",
        f"Event count: {summary['event_count']}",
        f"Selected lags: {SELECTED_LAGS}",
        f"Diagnostics: {', '.join(summary['diagnostics'])}",
        f"Map bootstrap repetitions: {N_BOOTSTRAP_MAP}",
        f"Box bootstrap repetitions: {N_BOOTSTRAP_SCALAR}",
        "Bootstrap grid: approximately 1 degree",
        "Multiple-testing control: Benjamini-Hochberg FDR",
        "",
        f"Box-level tests: {len(box_table)}",
        f"Primary robust box results: {robust_count}",
        "",
        f"NetCDF folder: {NETCDF_DIR}",
        f"Box robustness table: {BOX_RESULTS_CSV}",
        f"Significant-area table: {AREA_FRACTION_CSV}",
        f"Agreement table: {AGREEMENT_SUMMARY_CSV}",
        f"Event-level box anomalies: {EVENT_LEVEL_CSV}",
        f"Log CSV: {LOG_CSV}",
        f"Summary JSON: {SUMMARY_JSON}",
    ]

    REPORT_TXT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("")
    print("\n".join(lines))
    print("")
    print(
        "STEP 9G-1 PASSED. ERA5 BOOTSTRAP SIGNIFICANCE "
        "AND ROBUSTNESS OUTPUTS CREATED."
    )


if __name__ == "__main__":
    main()
