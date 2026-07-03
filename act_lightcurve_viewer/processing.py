# imports
import numpy as np
import csv

from .file_io import get_thumbnail_times


def coadd_thumbnails(
    thumbnails,
    coadd_arrays=["all"],
    time_delta=None,
    time_range=[],
    time_bins=[],
):
    ## coadd_arrays can be list of arrays to coadd, i.e. ['pa4','pa5'] would coadd those two but leave pa6 separate.
    ## use 'all' to coadd all of the available options (works for bands and for arrays).
    ## time_delta in days, if None coadd all available observations, subject to time_range
    ## time_range = [ctime_start,ctime_end] will only coadd over a specific time range. ctime in seconds.
    ## time_range=[] will use the full range of times.
    ## time_bins can be specific array of time bins,ctime in seconds.

    if time_delta:
        time_delta = time_delta * 86400  ## convert days to seconds for binning ctimes

    coadded_thumbs = {}
    output_coadds = []
    for source in thumbnails:
        observation_times = get_thumbnail_times(thumbnails[source])
        if len(observation_times) == 0:
            print("no observations of %s... skipping." % source)
            continue

        min_time = np.amin(observation_times)
        max_time = np.amax(observation_times)
        if time_range:
            min_time = np.amax([min_time, time_range[0]])
            max_time = np.amin([max_time, time_range[1]])

        if not time_delta:
            time_bins = [min_time, max_time + 1.0]
        else:
            time_bins = np.arange(min_time, max_time + time_delta, time_delta)

        if source not in coadded_thumbs:
            coadded_thumbs[source] = {}

        unique_freqs = []
        unique_arrays = []
        for thumb in thumbnails[source]:
            thumb_time = thumb["t"]
            thumb_time_bin = np.digitize(thumb_time, time_bins)
            if thumb_time_bin not in coadded_thumbs[source]:
                coadded_thumbs[source][thumb_time_bin] = {}

            thumb_freq = thumb["freq"]
            if thumb_freq not in unique_freqs:
                unique_freqs.append(thumb_freq)
            if thumb_freq not in coadded_thumbs[source][thumb_time_bin]:
                coadded_thumbs[source][thumb_time_bin][thumb_freq] = {}

            thumb_arr = thumb["arr"]
            if thumb_arr not in unique_arrays:
                unique_arrays.append(thumb_arr)
            if thumb_arr not in coadded_thumbs[source][thumb_time_bin][thumb_freq]:
                coadded_thumbs[source][thumb_time_bin][thumb_freq][thumb_arr] = {}
                coadded_thumbs[source][thumb_time_bin][thumb_freq][thumb_arr]["rho"] = (
                    thumb.pop("rho")
                )
                coadded_thumbs[source][thumb_time_bin][thumb_freq][thumb_arr][
                    "kappa"
                ] = thumb.pop("kappa")
                coadded_thumbs[source][thumb_time_bin][thumb_freq][thumb_arr][
                    "coadded_map_ids"
                ] = [thumb.pop("Id")]
                coadded_thumbs[source][thumb_time_bin][thumb_freq][thumb_arr][
                    "coadded_observation_times"
                ] = [thumb.pop("t")]
            else:
                coadded_thumbs[source][thumb_time_bin][thumb_freq][thumb_arr][
                    "rho"
                ] += thumb.pop("rho")
                coadded_thumbs[source][thumb_time_bin][thumb_freq][thumb_arr][
                    "kappa"
                ] += thumb.pop("kappa")
                coadded_thumbs[source][thumb_time_bin][thumb_freq][thumb_arr][
                    "coadded_map_ids"
                ] += [thumb.pop("Id")]
                coadded_thumbs[source][thumb_time_bin][thumb_freq][thumb_arr][
                    "coadded_observation_times"
                ] += [thumb.pop("t")]

        arrays_to_coadd = []
        if len(coadd_arrays) > 0:
            if coadd_arrays[0] == "all":
                arrays_to_coadd = unique_arrays
            else:
                arrays_to_coadd = coadd_arrays

        for time_bin in coadded_thumbs[source]:
            for freq in coadded_thumbs[source][time_bin]:
                arr_coadd = {}
                for arr in arrays_to_coadd:
                    if arr not in coadded_thumbs[source][time_bin][freq]:
                        continue
                    thumb_keys = list(
                        coadded_thumbs[source][time_bin][freq][arr].keys()
                    )
                    if not arr_coadd:
                        for key in thumb_keys:
                            arr_coadd[key] = coadded_thumbs[source][time_bin][freq][
                                arr
                            ].pop(key)
                        arr_coadd["coadded_arrays"] = [arr]
                    else:
                        for key in thumb_keys:
                            arr_coadd[key] += coadded_thumbs[source][time_bin][freq][
                                arr
                            ].pop(key)
                        arr_coadd["coadded_arrays"] += [arr]
                if arr_coadd:
                    arr_coadd["freq"] = freq
                    arr_coadd["time_min"] = np.amin(
                        arr_coadd["coadded_observation_times"]
                    )
                    arr_coadd["time_max"] = np.amax(
                        arr_coadd["coadded_observation_times"]
                    )
                    arr_coadd["source_id"] = source
                    output_coadds.append(arr_coadd)

                for arr in unique_arrays:
                    if arr in arrays_to_coadd:
                        continue
                    if arr not in coadded_thumbs[source][time_bin][freq]:
                        continue
                    arr_coadd = {}
                    for key in coadded_thumbs[source][time_bin][freq][arr].keys():
                        arr_coadd[key] = coadded_thumbs[source][time_bin][freq][
                            arr
                        ].pop(key)
                    arr_coadd["coadded_arrays"] = [arr]
                    arr_coadd["freq"] = freq
                    arr_coadd["time_min"] = np.amin(
                        arr_coadd["coadded_observation_times"]
                    )
                    arr_coadd["time_max"] = np.amax(
                        arr_coadd["coadded_observation_times"]
                    )
                    arr_coadd["source_id"] = source
                    output_coadds.append(arr_coadd)

    del coadded_thumbs
    return output_coadds


def coadd(time, flux, dflux, ndays=14, minobs=2):
    sort_inds = np.argsort(time)
    mintime = np.nanmin(time)
    maxtime = np.nanmax(time)
    time_bins = np.arange(mintime, maxtime + ndays * 60 * 60 * 24, ndays * 60 * 60 * 24)
    binned_flux = []
    binned_dflux = []
    inds = np.digitize(time, time_bins)
    filled_inds = []
    for i in range(len(time_bins)):
        if np.sum(inds == i) < minobs:
            filled_inds.append(False)
            continue
        # indx = inds
        avg_flux = np.sum(flux[inds == i] * (1.0 / dflux[inds == i]) ** 2) / np.sum(
            (1.0 / dflux[inds == i]) ** 2
        )
        avg_dflux = np.mean(dflux[inds == i]) / np.sqrt(np.sum(inds == i))
        binned_flux.append(avg_flux)
        binned_dflux.append(avg_dflux)
        filled_inds.append(True)

    return time_bins[filled_inds], np.asarray(binned_flux), np.asarray(binned_dflux)

def filter_bad_lightcurves(time, source_names, flux, dflux, bands):
    """
    Filters out observations where the uncertainty (dflux) is
    below the 5th percentile or above the 95th percentile, per band.
    """
    valid_indices = np.ones(len(flux), dtype=bool)

    for b in np.unique(bands):
        band_dflux = dflux[bands == b]
        if len(band_dflux) == 0: 
            continue
            
        lower_limit = np.nanpercentile(band_dflux, 5)
        upper_limit = np.nanpercentile(band_dflux, 95)

        bad_low = (bands == b) & (dflux < lower_limit)
        bad_high = (bands == b) & (dflux > upper_limit)

        valid_indices[bad_low] = False
        valid_indices[bad_high] = False

    clean_time = time[valid_indices]
    clean_sources = source_names[valid_indices]
    clean_flux = flux[valid_indices]
    clean_dflux = dflux[valid_indices]
    clean_bands = bands[valid_indices]

    return clean_time, clean_sources, clean_flux, clean_dflux, clean_bands

def find_flares(time, source_names, flux, dflux, bands, snr_threshold=3.0):
    """
    Scans lightcurve arrays for transient events exceeding a specific SNR threshold.
    Returns a list of lists containing: [star_name, frequency, time, amplitude, uncertainty, snr]
    """
    snr = flux / dflux
    flare_mask = snr >= snr_threshold

    flare_times = time[flare_mask]
    flare_sources = source_names[flare_mask]
    flare_fluxes = flux[flare_mask]
    flare_dfluxes = dflux[flare_mask]
    flare_bands = bands[flare_mask]
    flare_snrs = snr[flare_mask]

    detected_flares = []
    for i in range(len(flare_times)):
        raw_name = str(flare_sources[i])
        
        if " J" in raw_name:
            clean_name = "J" + raw_name.split(" J")[1]
        else:
            clean_name = raw_name.replace("SO-S_", "").replace("SO-SV_", "").replace("SO-", "")

        detected_flares.append([
            clean_name,
            str(flare_bands[i]),
            float(flare_times[i]),
            float(flare_fluxes[i]),
            float(flare_dfluxes[i]),
            float(flare_snrs[i])
        ])

    return detected_flares


def calculate_polarization_limits(coadded_thumbnails, flares, output_filename="polarization_summary_table.csv"):
    print("\n--- STARTING POLARIZATION CROSS-REFERENCE ---")
    results = []
    
    for flare in flares:
        # unpack the flare data directly from the pipeline's list
        star_raw, band_raw, flare_time, intensity, _, _ = flare
        
        # force variables to be strings and delete hidden spaces
        star = str(star_raw).strip()
        band = str(band_raw).strip()
        
        # grab all thumbnails for this specific star and band
        thumbs = [t for t in coadded_thumbnails if star in str(t["source_id"]).strip() and str(t["freq"]).strip() == band]
        
        if not thumbs:
            continue
            
        # find the thumbnail closest to the flare time
        best_thumb = min(thumbs, key=lambda t: abs(np.mean(t["coadded_observation_times"]) - float(flare_time)))
        
        # check the time difference
        time_diff_seconds = abs(np.mean(best_thumb["coadded_observation_times"]) - float(flare_time))
        
        # if it's within our 4-day window, calculate the physics
        if time_diff_seconds < (4 * 86400):
            
            center_y = best_thumb["rho"][1].shape[0] // 2
            center_x = best_thumb["rho"][1].shape[1] // 2
            
            # Extract Q and U specifically at that center pixel!
            Q_flux = best_thumb["rho"][1][center_y, center_x] / best_thumb["kappa"][1][center_y, center_x]
            U_flux = best_thumb["rho"][2][center_y, center_x] / best_thumb["kappa"][2][center_y, center_x]
            
            p_amp = np.sqrt(Q_flux**2 + U_flux**2)
            
            # Extract the noise specifically at the center pixel
            p_noise = 1.0 / np.sqrt(best_thumb["kappa"][1][center_y, center_x])
            upper_limit_3sigma = 3 * p_noise
                        
            results.append([
                star, band, round(float(intensity), 2), round(float(p_amp), 2), 
                round(float(upper_limit_3sigma), 2), round(float(upper_limit_3sigma)/float(intensity), 4)
            ])

    # save the final summary table
    if results:
        with open(output_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Star Name", "Band", "Intensity (mJy)", "Pol Amplitude (mJy)", "3-Sigma Limit (mJy)", "Max Pol Fraction (%)"])
            writer.writerows(results)
        print(f"\nSUCCESS! Saved polarization summary table with {len(results)} rows!")
    else:
        print("\nERROR: Table is empty. No flares were successfully matched.")
