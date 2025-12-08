#!/usr/bin/env python

import os

# import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import xarray as xr
from cartopy import crs as ccrs


def test_align_exact(ds1, ds2):
    """
    Test two datasets are exactly aligned
    """
    try:
        ds1_aligned, ds2_aligned = xr.align(ds1, ds2, join="exact")
        return ds1_aligned, ds2_aligned

    except ValueError as e:
        print(e)
        return None


# LPJ
# Daily NPP
lpj_path = "/discover/nobackup/projects/GHGC/LPJ_collaborations/NRT_carbon_budget_const_lu/20250521/S2_RESP_ACCLIM/ncdf_outputs"

lpj_dnpp = f"{lpj_path}/ERA5_S2_RESP_ACCLIM_dnpp.nc"
ds_dnpp = xr.open_dataset(
    lpj_dnpp,
    # chunks="auto"
)


# Daily Rh
lpj_drh = f"{lpj_path}/ERA5_S2_RESP_ACCLIM_drh.nc"
ds_drh = xr.open_dataset(lpj_drh)


# ### Combine to calc NEE (NEE = RH - NPP)
ds_aligned1, ds_aligned2 = test_align_exact(ds_drh, ds_dnpp)


ds_lpj_combined = xr.merge([ds_drh, ds_dnpp])


# MiCASA
micasa_path = "micasa_virtualized/vstore.parquet"
ds_mi = xr.open_dataset(
    f"reference::{micasa_path}",
    engine="zarr",
    consolidated=False,
)

ds_mi_chunk = ds_mi.chunk({"time": 30, "lat": 900, "lon": 1800})


# Align MiCASA and LPJ
# Micasa starts at Jan 2001
ds_lpj_sel = ds_lpj_combined.sel(time=slice("2001", None))
ds_lpj_sel

# LPJ does not include leap days, Downsample to match LPJ
ds_mi_downsample = ds_mi_chunk.coarsen(
    lat=5, lon=5, boundary="trim"
).mean()  # Downsampling (5x5 aggregation since 0.5°/0.1° = 5)

# test_align_exact(ds_lpj_sel, ds_mi_downsample)  # Leap days still not aligned

# Chunking
chunk_config = {"time": 365, "lat": 900, "lon": 1800}

# Drop leap days to match LPJ and drop unneeded vars
leap_day_mask = ~(
    (ds_mi_downsample.time.dt.month == 2) & (ds_mi_downsample.time.dt.day == 29)
)
ds_mi_noleap = ds_mi_downsample.sel(time=leap_day_mask)
# Rechunk after remove leap days
ds_mi_sel = ds_mi_noleap.chunk(chunk_config)[["NEE", "NPP", "Rh"]]

# # Change var names for consistency with micasa
ds_lpj_sel = ds_lpj_sel.rename_dims({"latitude": "lat", "longitude": "lon"})
ds_lpj_sel = ds_lpj_sel.rename_vars({"latitude": "lat", "longitude": "lon"})


# test_align_exact(ds_lpj_sel, ds_mi_sel)  # After renaming they aren't aligned??
# Force Micasa to fit lpj
ds_mi_sel = ds_mi_sel.reindex(lat=ds_lpj_sel.lat, lon=ds_lpj_sel.lon, method="nearest")
ds_lpj_align, ds_mi_align = test_align_exact(ds_lpj_sel, ds_mi_sel)

ds_lpj_align = ds_lpj_align.chunk(chunk_config)

# print(ds_lpj_align["dnpp"].attrs, ds_mi_align["NPP"].attrs, sep="\n\n")
result = (
    ds_lpj_align["dnpp"] / 86400 - ds_mi_align["NPP"]
)  # Conversion? did i do that right


# Pick years slice 2022-2024
result_sel = result.sel(time=slice("2022", "2024"))

# Compute (doesn't improve performance to calculate mean or var first)
results_compute = result_sel.compute()

mask = results_compute.notnull().any(dim=["time"])
results_masked = results_compute.where(mask, drop=True)
results_masked = results_masked.astype("float64")

seasonal_means = results_masked.groupby(result_sel.time.dt.season).mean("time")
seasonal_var = results_masked.groupby(result_sel.time.dt.season).var(
    dim="time", skipna=True
)

# Plots
output_dir = "plots/"
os.makedirs(output_dir, exist_ok=True)

proj = ccrs.PlateCarree()

## Means
for season in seasonal_means.season.values:
    fig, ax = plt.subplots(1, 1, figsize=(16, 8), subplot_kw={"projection": proj})

    # print(i, season)
    plot = seasonal_means.sel(season=season).plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        vmin=-9e-8,
        vmax=9e-8,
        cmap="RdBu",
        add_colorbar=False,
    )
    ax.set_title(f"{season}")
    cb = plt.colorbar(
        plot,
        orientation="horizontal",
        shrink=0.8,
        pad=0.05,
        label="LPJ-EOSIM — MiCASA (kg C m-2 s-1)\n2022-2024 Average",
    )
    fig.suptitle("Mean model difference, NPP", x=0.5, y=0.92, fontsize=15)
    output_filename = f"NPPDiff_means_{season}.png"
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)


## Variance
for season in seasonal_var.season.values:
    fig, ax = plt.subplots(1, 1, figsize=(16, 8), subplot_kw={"projection": proj})

    plot = seasonal_var.sel(season=season).plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="Reds",
        add_colorbar=False,
    )
    ax.set_title(f"{season}")
    cb = plt.colorbar(
        plot,
        orientation="horizontal",
        shrink=0.8,
        pad=0.05,
        label="LPJ-EOSIM — MiCASA (kg C m-2 s-1)\n2022-2024 Average",
    )
    fig.suptitle("Variance of model difference, NPP", x=0.5, y=0.92, fontsize=15)
    output_filename = f"NPPDiff_var_{season}.png"
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path)
