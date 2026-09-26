from pathlib import Path
import cdsapi


PROJECT_ROOT = Path(__file__).resolve().parents[1]

target = (
    PROJECT_ROOT
    / "01_raw_data"
    / "era5"
    / "pressure_level"
    / "ERA5_PL_2003_01.nc"
)

dataset = "reanalysis-era5-pressure-levels"

request = {
    "product_type": ["reanalysis"],

    "variable": [
        "geopotential",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vorticity",
        "potential_vorticity",
    ],

    "pressure_level": [
        "1000",
        "850",
        "700",
        "500",
        "300",
        "250",
        "200",
        "100",
        "70",
        "50",
        "30",
        "10",
    ],

    "year": ["2003"],
    "month": ["01"],

    "day": [
        "01", "02", "03", "04", "05", "06", "07",
        "08", "09", "10", "11", "12", "13", "14",
        "15", "16", "17", "18", "19", "20", "21",
        "22", "23", "24", "25", "26", "27", "28",
        "29", "30", "31",
    ],

    "time": ["00:00"],

    # Same regional domain as your existing files:
    # North, West, South, East
    "area": [70, 30, 10, 120],

    # NetCDF output
    "data_format": "netcdf",
    "download_format": "unarchived",
}

print("Downloading to:")
print(target)

client = cdsapi.Client()
client.retrieve(dataset, request, str(target))

print("Download complete:")
print(target)
