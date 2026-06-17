# imports
import numpy as np

from file_io import get_thumbnail_times


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
                coadded_thumbs[source][thumb_time_bin][thumb_freq][thumb_arr]["rho"] = thumb.pop(
                    "rho"
                )
                coadded_thumbs[source][thumb_time_bin][thumb_freq][thumb_arr]["kappa"] = thumb.pop(
                    "kappa"
                )
                coadded_thumbs[source][thumb_time_bin][thumb_freq][thumb_arr]["coadded_map_ids"] = [
                    thumb.pop("Id")
                ]
                coadded_thumbs[source][thumb_time_bin][thumb_freq][thumb_arr][
                    "coadded_observation_times"
                ] = [thumb.pop("t")]
            else:
                coadded_thumbs[source][thumb_time_bin][thumb_freq][thumb_arr]["rho"] += thumb.pop(
                    "rho"
                )
                coadded_thumbs[source][thumb_time_bin][thumb_freq][thumb_arr]["kappa"] += thumb.pop(
                    "kappa"
                )
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
                    thumb_keys = list(coadded_thumbs[source][time_bin][freq][arr].keys())
                    if not arr_coadd:
                        for key in thumb_keys:
                            arr_coadd[key] = coadded_thumbs[source][time_bin][freq][arr].pop(key)
                        arr_coadd["coadded_arrays"] = [arr]
                    else:
                        for key in thumb_keys:
                            arr_coadd[key] += coadded_thumbs[source][time_bin][freq][arr].pop(key)
                        arr_coadd["coadded_arrays"] += [arr]
                if arr_coadd:
                    arr_coadd["freq"] = freq
                    arr_coadd["time_min"] = np.amin(arr_coadd["coadded_observation_times"])
                    arr_coadd["time_max"] = np.amax(arr_coadd["coadded_observation_times"])
                    arr_coadd["source_id"] = source
                    output_coadds.append(arr_coadd)

                for arr in unique_arrays:
                    if arr in arrays_to_coadd:
                        continue
                    if arr not in coadded_thumbs[source][time_bin][freq]:
                        continue
                    arr_coadd = {}
                    for key in coadded_thumbs[source][time_bin][freq][arr].keys():
                        arr_coadd[key] = coadded_thumbs[source][time_bin][freq][arr].pop(key)
                    arr_coadd["coadded_arrays"] = [arr]
                    arr_coadd["freq"] = freq
                    arr_coadd["time_min"] = np.amin(arr_coadd["coadded_observation_times"])
                    arr_coadd["time_max"] = np.amax(arr_coadd["coadded_observation_times"])
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
