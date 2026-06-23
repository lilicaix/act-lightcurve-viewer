# imports
import os
from datetime import datetime, timezone

import matplotlib.dates as mdates
import numpy as np
from matplotlib import pyplot as plt

from .file_io import decode_filename_to_act
from .processing import coadd


# hdf5 thumbnail plotting functions
def plot_iqu_maps(coadded_thumbnails, sources, bands=["f090", "f150", "f220"]):
    for source in sources:
        for band in bands:
            print(source, band)

            base_filename, top_title, readable_subtitle = decode_filename_to_act(source, band)
            filename = base_filename.replace(".png", "_IQU.png")

            plt.figure(figsize=(12, 4))
            for i in range(len(coadded_thumbnails)):
                if (
                    coadded_thumbnails[i]["source_id"] != source
                    or coadded_thumbnails[i]["freq"] != band
                ):
                    continue

                plt.subplot(131)
                plt.imshow(
                    coadded_thumbnails[i]["rho"][0] / coadded_thumbnails[i]["kappa"][0]
                )
                plt.title(coadded_thumbnails[i]["freq"] + " I")
                plt.subplot(132)
                plt.imshow(
                    coadded_thumbnails[i]["rho"][1] / coadded_thumbnails[i]["kappa"][1]
                )
                plt.title(coadded_thumbnails[i]["freq"] + " Q")
                plt.subplot(133)
                plt.imshow(
                    coadded_thumbnails[i]["rho"][2] / coadded_thumbnails[i]["kappa"][2]
                )
                plt.title(coadded_thumbnails[i]["freq"] + " U")

                unix_time = np.mean(coadded_thumbnails[i]["coadded_observation_times"])
                dt_obj = datetime.fromtimestamp(unix_time, tz=timezone.utc)
                readable_time = dt_obj.strftime("%Y-%m-%d %H:%M:%S")

                plt.tight_layout()
                plt.suptitle(f"{top_title}\n{readable_subtitle}   |   {readable_time}", y=1.05)

            os.makedirs("iqu", exist_ok=True)
            save_path = os.path.join("iqu", filename)
            plt.savefig(save_path, format="png", dpi=150, bbox_inches="tight")
            plt.close()


def plot_time_evolution(
    coadded_thumbnails,
    sources,
    mean_observation_times,
    coadd_days,
    bands=["f090", "f150", "f220"],
):
    flux_ref = 150
    for source in sources:
        for band in bands:
            print(source, band)
            ntimebins = len(mean_observation_times[source][band])
            ref_size = 4
            if len(mean_observation_times[source][band]) > 10:
                ref_size = 2

            # Unpack the new 3-part naming format
            filename, top_title, readable_subtitle = decode_filename_to_act(source, band)

            if len(mean_observation_times[source][band]) < 25:
                ncols = int(np.ceil(np.sqrt(ntimebins)))
                nrows = int(np.ceil(ntimebins / ncols))
                plt.figure(figsize=(ref_size * ncols, ref_size * nrows))

                for ni in range(ntimebins):
                    plt.subplot(nrows, ncols, ni + 1)
                    for i in range(len(coadded_thumbnails)):
                        if (
                            coadded_thumbnails[i]["source_id"] != source
                            or coadded_thumbnails[i]["freq"] != band
                            or int(
                                np.mean(
                                    coadded_thumbnails[i]["coadded_observation_times"]
                                )
                            )
                            != mean_observation_times[source][band][ni]
                        ):
                            continue
                        f = flux_ref if band in ["f090", "f150"] else 2 * flux_ref
                        plt.imshow(
                            coadded_thumbnails[i]["rho"][0]
                            / coadded_thumbnails[i]["kappa"][0],
                            vmin=-f / np.sqrt(coadd_days if coadd_days > 1 else 1),
                            vmax=f / np.sqrt(coadd_days if coadd_days > 1 else 1),
                        )
                    unix_time = mean_observation_times[source][band][ni]
                    dt_obj = datetime.fromtimestamp(unix_time, tz=timezone.utc)
                    readable_time = dt_obj.strftime("%Y-%m-%d\n%H:%M:%S")
                    plt.title(readable_time, fontsize=10)

                plt.tight_layout()
                # Stack the titles
                plt.suptitle(f"{top_title}\n{readable_subtitle}", y=1.05)
                os.makedirs("time evolution", exist_ok=True)
                save_path = os.path.join("time evolution", filename)
                plt.savefig(save_path, format="png", dpi=150, bbox_inches="tight")
                plt.close()

            else:
                sqrtn = int(np.ceil(np.sqrt(ntimebins)))
                plt.figure(figsize=(ref_size * sqrtn, ref_size * sqrtn))
                for ni in range(ntimebins):
                    plt.subplot(sqrtn, sqrtn, ni + 1)
                    for i in range(len(coadded_thumbnails)):
                        if (
                            coadded_thumbnails[i]["source_id"] != source
                            or coadded_thumbnails[i]["freq"] != band
                            or int(
                                np.mean(
                                    coadded_thumbnails[i]["coadded_observation_times"]
                                )
                            )
                            != mean_observation_times[source][band][ni]
                        ):
                            continue
                        plt.imshow(
                            coadded_thumbnails[i]["rho"][0]
                            * coadded_thumbnails[i]["kappa"][0] ** (-0.5),
                            vmin=-3,
                            vmax=3,
                        )
                    unix_time = mean_observation_times[source][band][ni]
                    dt_obj = datetime.fromtimestamp(unix_time, tz=timezone.utc)
                    readable_time = dt_obj.strftime("%Y-%m-%d\n%H:%M:%S")
                    plt.title(readable_time, fontsize=10)

                plt.tight_layout()
                # Stack the titles
                plt.suptitle(f"{top_title}\n{readable_subtitle}", y=1.05)
                os.makedirs("time evolution", exist_ok=True)
                save_path = os.path.join("time evolution", filename)
                plt.savefig(save_path, format="png", dpi=150, bbox_inches="tight")
                plt.close()


# lightcurve plotting functions
def plot_lightcurves_per_source(time, source_names, bands, flux, dflux):
    os.makedirs("lightcurves_per_source", exist_ok=True)

    for source in np.unique(source_names):
        print(source)
        plt.figure(dpi=200)
        source_inds = source_names == source
        print(source_names[source_inds])

        # Unpack the new 3-part naming format
        base_filename, top_title, readable_subtitle = decode_filename_to_act(source, "all_bands")
        filename = base_filename.replace("_all_bands.png", "_lightcurve.png")

        for b in np.unique(bands):
            inds = (source_inds) & (bands == b)
            dates = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in time[inds]]
            plt.errorbar(
                dates, flux[inds], yerr=dflux[inds], marker="o", ls="", label=b
            )

        plt.grid()
        plt.legend()
        # Stack the titles
        plt.title(f"{top_title} Lightcurve\n{readable_subtitle}")
        plt.ylabel("Flux Density (mJy)")
        plt.xlabel("Date (UTC)")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gcf().autofmt_xdate()
        save_path = os.path.join("lightcurves_per_source", filename)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()


def plot_lightcurves_per_band_uncut(time, bands, flux, dflux):
    os.makedirs("lightcurves_per_band", exist_ok=True)

    for b in np.unique(bands):
        plt.figure(dpi=200)

        dates = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in time[bands == b]]

        plt.errorbar(
            dates, flux[bands == b], yerr=dflux[bands == b], marker="o", ls="", label=b
        )
        plt.grid()
        plt.legend()
        safe_band = str(b).strip()
        plt.title(f"All Sources - {safe_band}")
        plt.ylabel("Flux Density (mJy)")
        plt.xlabel("Date (UTC)")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gcf().autofmt_xdate()
        save_path = os.path.join(
            "lightcurves_per_band", f"all_sources_{safe_band}_lightcurve.png"
        )
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()


def calculate_and_plot_noise_cuts(bands, dflux):
    os.makedirs("aggregate_summaries", exist_ok=True)

    noise_cuts = {}
    plt.figure(dpi=200)
    for b in np.unique(bands):
        plt.hist(dflux[bands == b], bins=21, label=b, alpha=0.5)
        noise_cuts[b] = np.nanquantile(dflux[bands == b], 0.9)
    plt.xlabel("Flux Density (mJy)")
    plt.ylabel("Number of Observations")
    plt.legend()
    plt.title("Noise Cuts Distribution")
    plt.savefig(
        os.path.join("aggregate_summaries", "noise_cuts_histogram.png"),
        bbox_inches="tight",
    )
    plt.close()
    return noise_cuts


def plot_all_sources_cut_lightcurves(time, flux, dflux, bands, noise_cuts):
    os.makedirs("aggregate_summaries", exist_ok=True)

    plt.figure(dpi=200)
    for b in np.unique(bands):
        print(b)
        if b == " f030" or b == " f040":
            continue
        inds = (bands == b) & (dflux < noise_cuts[b])
        dates = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in time[inds]]

        plt.errorbar(dates, flux[inds], yerr=dflux[inds], marker="o", ls="", label=b)
    plt.grid()
    plt.legend()
    plt.title("All Sources Cut Lightcurves")
    plt.ylabel("Flux Density (mJy)")
    plt.xlabel("Date (UTC)")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gcf().autofmt_xdate()
    plt.savefig(
        os.path.join("aggregate_summaries", "all_sources_cut_lightcurves.png"),
        bbox_inches="tight",
    )
    plt.close()


def plot_flux_histogram_cut(flux, dflux, bands, noise_cuts):
    os.makedirs("aggregate_summaries", exist_ok=True)

    plt.figure(dpi=200)
    for b in sorted(np.unique(bands)):
        print(b)
        inds = (bands == b) & (dflux < noise_cuts[b])

        # The math here calculates the Inverse-Variance Weighted Average
        plt.hist(
            flux[inds],
            bins=21,
            label="%s : %.2f+-%.2f"
            % (
                b,
                np.average(flux[inds], weights=1.0 / dflux[inds] ** 2),
                np.nanstd(flux[inds]) / np.sqrt(np.sum(inds)),
            ),
            alpha=0.5,
        )

    plt.grid()
    plt.legend()
    plt.title("Flux Histogram (After Cuts)")
    plt.xlabel("Flux Density (mJy)")
    plt.ylabel("Number of Observations")
    plt.savefig(
        os.path.join("aggregate_summaries", "flux_histogram_cut.png"),
        bbox_inches="tight",
    )
    plt.close()


def plot_coadded_aggregate(time, flux, dflux, bands, ndays=30):
    os.makedirs("aggregate_summaries", exist_ok=True)

    plt.figure(dpi=200)
    for b in np.unique(bands):
        t, f, df = coadd(
            time[bands == b], flux[bands == b], dflux[bands == b], ndays=ndays
        )

        dates = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in t]

        plt.errorbar(dates, f, yerr=df, marker="o", ls="", label=b)

    plt.grid()
    plt.legend()
    plt.title(f"{ndays} Day Coadds - All Bands")
    plt.ylabel("Flux Density (mJy)")
    plt.xlabel("Date (UTC)")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gcf().autofmt_xdate()
    plt.savefig(
        os.path.join("aggregate_summaries", f"coadded_{ndays}day_lightcurves.png"),
        bbox_inches="tight",
    )
    plt.close()


def plot_cut_lightcurves_per_band(time, flux, dflux, bands, noise_cuts):
    os.makedirs("lightcurves_per_band", exist_ok=True)

    for b in np.unique(bands):
        print(b)
        if b == " f030" or b == " f040":
            continue
        inds = (bands == b) & (dflux < noise_cuts[b])
        plt.figure(dpi=200)
        dates = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in time[inds]]
        plt.errorbar(dates, flux[inds], yerr=dflux[inds], marker="o", ls="", label=b)
        plt.grid()
        plt.legend()
        safe_band = str(b).strip()
        plt.title(f"All Sources Cut - {safe_band}")
        plt.ylabel("Flux Density (mJy)")
        plt.xlabel("Date (UTC)")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gcf().autofmt_xdate()
        save_path = os.path.join(
            "lightcurves_per_band", f"all_sources_cut_{safe_band}_lightcurve.png"
        )
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()


def plot_snr_all_bands(time, flux, dflux, bands, noise_cuts):
    os.makedirs("aggregate_summaries", exist_ok=True)

    plt.figure(dpi=200)
    for b in np.unique(bands):
        print(b)
        if b == " f030" or b == " f040":
            continue
        inds = (bands == b) & (dflux < noise_cuts[b])
        dates = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in time[inds]]
        plt.errorbar(dates, flux[inds] / dflux[inds], marker="o", ls="", label=b)

    plt.grid()
    plt.legend()
    plt.title("Signal to Noise Ratio (SNR) - All Bands")
    plt.ylabel("Signal-to-Noise Ratio (SNR)")
    plt.xlabel("Date (UTC)")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gcf().autofmt_xdate()
    plt.savefig(
        os.path.join("aggregate_summaries", "snr_all_bands.png"), bbox_inches="tight"
    )
    plt.close()
