# imports
import argparse
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

from act_lightcurve_viewer import file_io, plotting, processing

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
parser.add_argument(
    "-c", "--coadd-days", type=int, default=3, help="Number of days to coadd."
)
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

all_thumbnails = file_io.read_thumbnails(thumbnail_dir)

# load data
master_thumbs = {}
for file_data in all_thumbnails:
    for k, v in file_data.items():
        master_thumbs.setdefault(k, []).extend(v)

master_thumbs = processing.consolidate_data(thumbnails=master_thumbs)

# process math (coadding)
coadded_thumbnails = processing.coadd_thumbnails(
    master_thumbs, time_range=[], time_delta=coadd_days
)

# setup variables for plotting
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
        mean_observation_times[source][band].sort()
'''
# generate plots
print(f"Generating thumbnail plots for {len(sources)} sources...")
plotting.plot_iqu_maps(
    coadded_thumbnails, sources
)
plotting.plot_time_evolution(
    coadded_thumbnails, sources, mean_observation_times, coadd_days
)
plotting.plot_time_evolution_polarization(
    coadded_thumbnails, sources, mean_observation_times, coadd_days
)
'''
# phase 2: text lightcurve pipeline
print("\n=== STARTING LIGHTCURVE PROCESSING ===")

# load data
print(f"Loading lightcurve data from: {lightcurve_file}")
lc_data = file_io.read_lightcurves(lightcurve_file)

time = lc_data["time"]
source_names = lc_data["source_names"]
bands = lc_data["bands"]
flux = lc_data["flux"]
dflux = lc_data["dflux"]

source_names = processing.consolidate_data(sources=source_names)

sort_idx = np.lexsort((time, source_names))
time = time[sort_idx]
source_names = source_names[sort_idx]
flux = flux[sort_idx]
dflux = dflux[sort_idx]
bands = bands[sort_idx]

# process math (noise cuts & flare finding)
print("Calculating raw noise histogram...")
noise_cuts = plotting.calculate_and_plot_noise_cuts(bands, dflux)

print("Applying 5th and 95th percentile noise filter...")
time, source_names, flux, dflux, bands = processing.filter_bad_lightcurves(
    time, source_names, flux, dflux, bands
)

for b in noise_cuts:
    noise_cuts[b] = np.inf

print("Scanning for flares...")
flares = processing.find_flares(time, source_names, flux, dflux, bands, snr_threshold=3.0)
print(f"Found {len(flares)} potential flares across all sources!")

# calculate polarization limits natively
flares_with_pol = processing.calculate_polarization_limits(coadded_thumbnails, flares)

# load into unified dataframe
columns = [
    "star_name", "frequency", "time", "amplitude", "uncertainty", "snr", 
    "intensity", "pol_amplitude", "3_sigma_limit", "max_pol_fraction"
]
final_df = pd.DataFrame(flares_with_pol, columns=columns)

# crossmatch with biermann catalog
biermann_df = pd.read_csv('biermann_events_clean.csv')
biermann_subset = biermann_df[['star_name', 'event_id']].dropna(subset=['star_name']).copy()

def parse_name_to_coord(name):
    name = str(name).strip().replace('J', '')
    sign = '+' if '+' in name else '-'
    ra_part, dec_part = name.split(sign)
    ra_str = f"{ra_part[0:2]}h{ra_part[2:4]}m{ra_part[4:]}s"
    if '.' in dec_part and len(dec_part.split('.')[0]) == 4:
        dec_str = f"{sign}{dec_part[0:2]}d{dec_part[2:]}m"
    else:
        dec_str = f"{sign}{dec_part[0:2]}d{dec_part[2:4]}m{dec_part[4:]}s"
    return SkyCoord(f"{ra_str} {dec_str}", frame='icrs')

# assign event ids based on spatial match
pol_coords = SkyCoord([parse_name_to_coord(n) for n in final_df['star_name']])
biermann_coords = SkyCoord([parse_name_to_coord(n) for n in biermann_subset['star_name']])
idx, d2d, _ = pol_coords.match_to_catalog_sky(biermann_coords)
match_mask = d2d < (60 * u.arcsec)

final_df['event_id'] = pd.Series(dtype='object')
final_df.loc[match_mask, 'event_id'] = biermann_subset.iloc[idx[match_mask]]['event_id'].values
final_df['event_id'] = final_df['event_id'].str.replace(r'[a-zA-Z]', '', regex=True)

# reorder columns so event_id is second
cols = final_df.columns.tolist()
if 'event_id' in cols:
    cols.insert(1, cols.pop(cols.index('event_id')))
final_df = final_df[cols]

# save master spreadsheet
final_df.to_csv("all_detected_flares.csv", index=False)
print("Saved all flares to 'all_detected_flares.csv'!")

# generate plots
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

np.save('coadded_thumbnails.npy', coadded_thumbnails)
