import os
import argparse
from datetime import datetime, timezone

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from act_lightcurve_viewer import file_io, processing


def get_core_coord(name_str):
    """extract core coordinates to match biermann catalog naming"""
    if "J" not in str(name_str):
        return str(name_str)
    coords = str(name_str).split("J")[1].strip()
    if len(coords) >= 10:
        return coords[0:4] + coords[6:10]
    return coords


def main():
    # setup command line arguments for the script
    parser = argparse.ArgumentParser(description="Plot lightcurve comparisons")
    parser.add_argument(
        "-l",
        "--lightcurve-file",
        default="collated_lightcurve.txt",
        help="Path to collated lightcurve text file",
    )
    args = parser.parse_args()

    print("Loading lightcurve and flare data...")
    
    # load the raw lightcurve data to plot as background points
    try:
        lc_data = file_io.read_lightcurves(args.lightcurve_file)
    except FileNotFoundError:
        print(f"Error: Could not find '{args.lightcurve_file}'.")
        return
        
    time_arr = lc_data["time"]
    source_names_arr = lc_data["source_names"]
    flux_arr = lc_data["flux"]
    dflux_arr = lc_data["dflux"]
    bands_arr = lc_data["bands"]

    source_names_arr = processing.consolidate_data(sources=source_names_arr)

    time_arr, source_names_arr, flux_arr, dflux_arr, bands_arr = processing.filter_bad_lightcurves(
        time_arr, source_names_arr, flux_arr, dflux_arr, bands_arr
    )

    # load the sorted flares and biermann reference catalog
    try:
        new_flares = pd.read_csv("final_detected_flares.csv")
        biermann_ref = pd.read_csv("biermann_events_clean.csv")
    except FileNotFoundError:
        print("Error: Ensure 'final_detected_flares.csv' and 'biermann_events_clean.csv' exist.")
        return

    # convert biermann peak times from utc to unix seconds for plotting
    if "peak_unix" not in biermann_ref.columns:
        biermann_ref["peak_unix"] = pd.to_datetime(biermann_ref["peak_utc"], utc=True).astype("int64") // 10**9

    # convert duration hours into absolute unix timestamps for shading
    if "rise_hr" in biermann_ref.columns and "fall_hr" in biermann_ref.columns:
        biermann_ref["rise_unix"] = biermann_ref["peak_unix"] - (biermann_ref["rise_hr"] * 3600)
        biermann_ref["fall_unix"] = biermann_ref["peak_unix"] + (biermann_ref["fall_hr"] * 3600)
    
    # extract core coordinates for cross-referencing
    biermann_ref["core_coord"] = biermann_ref["star_name"].apply(get_core_coord)

    # create a folder to save the output images
    os.makedirs("comparison_plots", exist_ok=True)

    # get a list of unique stars that have a new flare
    stars_to_plot = new_flares["star_name"].unique()
    print(f"Generating comparison plots for {len(stars_to_plot)} stars...")

    for star in stars_to_plot:
        # create a single-panel plot with flux
        plt.figure(figsize=(12, 6), dpi=200)

        # find all rows in the raw data that match the current star
        star_mask = np.array([star in str(raw_name) for raw_name in source_names_arr])
        if not star_mask.any():
            print(f"Warning: Data for {star} not found in raw lightcurve. Skipping.")
            plt.close()
            continue

        # extract the data for just this specific star
        s_time = time_arr[star_mask]
        s_flux = flux_arr[star_mask]
        s_dflux = dflux_arr[star_mask]
        s_bands = bands_arr[star_mask]
        
        # filter out zero uncertainty to prevent divide-by-zero math errors
        valid_snr = s_dflux > 0
        s_time = s_time[valid_snr]
        s_flux = s_flux[valid_snr]
        s_dflux = s_dflux[valid_snr]
        s_bands = s_bands[valid_snr]

        # map each frequency band to a specific color
        band_colors = {"f090": "cornflowerblue", "f150": "mediumseagreen", "f220": "salmon"}
        
        # plot the background lightcurve data for each band
        for b in np.unique(s_bands):
            b_str = str(b).strip()
            b_mask = s_bands == b
            dates_b = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in s_time[b_mask]]
            c = band_colors.get(b_str, "lightgray")
            
            plt.errorbar(dates_b, s_flux[b_mask], yerr=s_dflux[b_mask], marker="o", ls="", color=c, alpha=0.3, label=f"Obs ({b_str})")

        # find matching flares for this star in the biermann catalog
        core_star = get_core_coord(star)
        b_matches = biermann_ref[biermann_ref["core_coord"] == core_star]

        # process and plot biermann flares if any exist
        if not b_matches.empty:
            # shade rise/fall periods
            added_shade_label = False
            if "rise_unix" in b_matches.columns and "fall_unix" in b_matches.columns:
                for _, row in b_matches.iterrows():
                    try:
                        r_date = datetime.fromtimestamp(row["rise_unix"], tz=timezone.utc)
                        f_date = datetime.fromtimestamp(row["fall_unix"], tz=timezone.utc)
                        
                        # only add label once to prevent legend clutter
                        label = "Biermann Active Period" if not added_shade_label else None
                        plt.axvspan(r_date, f_date, color="lightblue", alpha=0.3, zorder=1, label=label)
                        added_shade_label = True
                    except (ValueError, TypeError):
                        continue
            b_times = b_matches["peak_unix"].values
            b_dates, b_fluxes = [], []
            
            # match biermann timestamps to our actual observation times
            for t in b_times:
                time_diffs = np.abs(s_time - t)
                if len(time_diffs) > 0 and np.min(time_diffs) < (4 * 86400):
                    # find the absolute closest actual observation
                    closest_time = s_time[np.argmin(time_diffs)]
                    
                    # grab all observations that happened simultaneously (within a 1-hour window)
                    simultaneous_mask = np.abs(s_time - closest_time) < 3600
                    
                    if simultaneous_mask.any():
                        best_idx = np.where(simultaneous_mask)[0][np.argmax(s_flux[simultaneous_mask])]
                        b_dates.append(datetime.fromtimestamp(s_time[best_idx], tz=timezone.utc))
                        b_fluxes.append(s_flux[best_idx])

            # plot the matched biermann flares
            if b_dates:
                plt.scatter(b_dates, b_fluxes, color="indigo", marker="*", s=125, zorder=5, label="Biermann Detected")

        # find matching flares for this star in the new pipeline data
        n_matches = new_flares[new_flares["star_name"] == star]
        n_times = n_matches["time"].values
        n_dates = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in n_times]
        n_fluxes = n_matches["amplitude"].values

        # plot the flares detected by the new pipeline
        if n_dates:
            plt.scatter(n_dates, n_fluxes, color="magenta", marker="*", s=125, zorder=6, label="New")

        # extract biermann event id to display under the title
        biermann_label = ""
        if not b_matches.empty:
            # check for the exact column name used for the event sequence
            if "event_id" in b_matches.columns:
                event_ids = b_matches["event_id"].astype(str).unique()
                biermann_label = f"\nBiermann Event ID: {', '.join(event_ids)}"
            elif "Seq" in b_matches.columns:
                event_ids = b_matches["Seq"].astype(str).unique()
                biermann_label = f"\nBiermann Event ID: {', '.join(event_ids)}"
            else:
                biermann_label = "\nBiermann Event: Known Match"

        # decode the star name into readable titles
        # add a prefix if missing so the file_io regex can cleanly find the ' J' delimiter
        formatted_star = star if " J" in star else f"ACT {star}"
        _, top_title, readable_subtitle = file_io.decode_filename_to_act(formatted_star, "all")
        
        # apply formatting and labels to the flux plot, including the new biermann label
        plt.title(f"{top_title} - Detection Comparison\n{readable_subtitle}{biermann_label}")
        plt.ylabel("Flux Density (mJy)")
        plt.xlabel("Date (UTC)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gcf().autofmt_xdate()

        # save the finished plot as an image file
        safe_name = star.replace(" ", "_")
        plt.savefig(os.path.join("comparison_plots", f"{safe_name}_comparison.png"), bbox_inches="tight")
        plt.close()

    print("\nComplete! Check the 'comparison_plots' folder for your images.")


if __name__ == "__main__":
    main()