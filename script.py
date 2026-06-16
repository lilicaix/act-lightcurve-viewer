# imports
import os
import numpy as np
import file_io
import processing
import plotting

# pipeline parameters
hdf5_data_dir = '/scratch/gpfs/JDUNKLEY/lc5846/usrp/act_biermann_sources/'
text_data_file = '/scratch/gpfs/JDUNKLEY/lc5846/usrp/act_biermann_sources/collated_lightcurve.txt'

hdf5_coadd_days = 3
text_coadd_days = 30
act_bands = ['f090', 'f150', 'f220']


# phase 1: hdf5 thumbnail pipeline
print("\n=== STARTING HDF5 THUMBNAIL PROCESSING ===")

# 1. load data
all_thumbnails = file_io.read_thumbnails(hdf5_data_dir)

# 2. process math (coadding)
coadded_thumbnails = []
for file_data in all_thumbnails:
    coadded = processing.coadd_thumbnails(file_data, time_range=[], time_delta=hdf5_coadd_days)
    coadded_thumbnails.extend(coadded)

# 3. setup variables for plotting
sources = []
for i in range(len(coadded_thumbnails)):
    sources.append(coadded_thumbnails[i]['source_id'])
sources = np.unique(sources)

mean_observation_times = {}
for source in sources:
    mean_observation_times[source] = {}
    for band in act_bands:
        mean_observation_times[source][band] = []
        for i in range(len(coadded_thumbnails)):
            if coadded_thumbnails[i]['source_id']!=source or coadded_thumbnails[i]['freq']!=band :
                continue
            mean_observation_times[source][band].append(int(np.mean(coadded_thumbnails[i]['coadded_observation_times'])))

# 4. generate plots
print(f"Generating HDF5 image plots for {len(sources)} sources...")
plotting.plot_iqu_maps(coadded_thumbnails, sources)
plotting.plot_time_evolution(coadded_thumbnails, sources, mean_observation_times, hdf5_coadd_days)


# phase 2: text lightcurve pipeline
print("\n=== STARTING TEXT LIGHTCURVE PROCESSING ===")

# 1. load data
print(f"Loading lightcurve data from: {text_data_file}")
lc_data = file_io.read_lightcurves(text_data_file)

time = lc_data['time']
source_names = lc_data['source_names']
bands = lc_data['bands']
flux = lc_data['flux']
dflux = lc_data['dflux']

# 2. process math (noise cuts)
print("Calculating dynamic noise cuts...")
noise_cuts = plotting.calculate_and_plot_noise_cuts(bands, dflux)

# 3. generate plots
print("Generating text lightcurve plots...")
plotting.plot_lightcurves_per_source(time, source_names, bands, flux, dflux)
plotting.plot_lightcurves_per_band_uncut(time, bands, flux, dflux)
plotting.plot_all_sources_cut_lightcurves(time, flux, dflux, bands, noise_cuts)
plotting.plot_flux_histogram_cut(flux, dflux, bands, noise_cuts)
plotting.plot_coadded_aggregate(time, flux, dflux, bands, ndays=text_coadd_days)
plotting.plot_cut_lightcurves_per_band(time, flux, dflux, bands, noise_cuts)
plotting.plot_snr_all_bands(time, flux, dflux, bands, noise_cuts)

print("\n=== PIPELINE COMPLETE! ===")
print("All plots have been saved to their respective directories.")
