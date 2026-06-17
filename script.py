# imports
import argparse

import numpy as np

import file_io
import plotting
import processing

# pipeline parameters
parser = argparse.ArgumentParser(description="ACT lightcurve viewer pipeline")
parser.add_argument(
    "-t", "--thumbnail-dir", default="./", help="Directory containing thumbnail files"
)
parser.add_argument(
    "-l",
    "--lightcurve-file",
    default="collated_lightcurve.txt",
    help="Path to collated lightcurve text file",
)
parser.add_argument("-c", "--coadd-days", type=int, default=3, help="Number of days to coadd.")
parser.add_argument(
    "-b",
    "--bands",
    nargs="+",
    default=["f090", "f150", "f220"],
    help="ACT frequency bands to process",
)
args = parser.parse_args()

thumbnail_dir = args.thumbnail_dir
lightcurve_file = args.lightcurve_file
coadd_days = args.coadd_days
act_bands = args.bands


# phase 1: thumbnail pipeline
print("\n=== STARTING THUMBNAIL PROCESSING ===")

# 1. load data
all_thumbnails = file_io.read_thumbnails(thumbnail_dir)

# 2. process math (coadding)
coadded_thumbnails = []
for file_data in all_thumbnails:
    coadded = processing.coadd_thumbnails(file_data, time_range=[], time_delta=coadd_days)
    coadded_thumbnails.extend(coadded)

# 3. setup variables for plotting
sources = []
for i in range(len(coadded_thumbnails)):
    sources.append(coadded_thumbnails[i]["source_id"])
sources = np.unique(sources)

mean_observation_times = {}
for source in sources:
    mean_observation_times[source] = {}
    for band in act_bands:
        mean_observation_times[source][band] = []
        for i in range(len(coadded_thumbnails)):
            if (
                coadded_thumbnails[i]["source_id"] != source
                or coadded_thumbnails[i]["freq"] != band
            ):
                continue
            mean_observation_times[source][band].append(
                int(np.mean(coadded_thumbnails[i]["coadded_observation_times"]))
            )

# 4. generate plots
print(f"Generating thumbnail plots for {len(sources)} sources...")
plotting.plot_iqu_maps(coadded_thumbnails, sources)
plotting.plot_time_evolution(coadded_thumbnails, sources, mean_observation_times, coadd_days)


# phase 2: text lightcurve pipeline
print("\n=== STARTING LIGHTCURVE PROCESSING ===")

# 1. load data
print(f"Loading lightcurve data from: {lightcurve_file}")
lc_data = file_io.read_lightcurves(lightcurve_file)

time = lc_data["time"]
source_names = lc_data["source_names"]
bands = lc_data["bands"]
flux = lc_data["flux"]
dflux = lc_data["dflux"]

# 2. process math (noise cuts)
print("Calculating dynamic noise cuts...")
noise_cuts = plotting.calculate_and_plot_noise_cuts(bands, dflux)

# 3. generate plots
print("Generating text lightcurve plots...")
plotting.plot_lightcurves_per_source(time, source_names, bands, flux, dflux)
plotting.plot_lightcurves_per_band_uncut(time, bands, flux, dflux)
plotting.plot_all_sources_cut_lightcurves(time, flux, dflux, bands, noise_cuts)
plotting.plot_flux_histogram_cut(flux, dflux, bands, noise_cuts)
plotting.plot_coadded_aggregate(time, flux, dflux, bands, ndays=coadd_days)
plotting.plot_cut_lightcurves_per_band(time, flux, dflux, bands, noise_cuts)
plotting.plot_snr_all_bands(time, flux, dflux, bands, noise_cuts)

print("\n=== PIPELINE COMPLETE! ===")
print("All plots have been saved to their respective directories.")
