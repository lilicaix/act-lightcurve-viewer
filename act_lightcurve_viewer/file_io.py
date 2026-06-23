# imports
import glob
import os
import re

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates.angles import Angle
from numpy import mod as npmod


# string + naming functions
def radec_to_str_name(
    ra: float, dec: float, source_class="pointsource", observatory="SO"
):
    """
    ## stolen from spt3g_software -AF

    Convert RA & dec (in degrees) to IAU-approved string name.

    Arguments
    ---------
    ra : float
        Source right ascension in degrees
    dec : float
        Source declination in degrees
    source_class : str
        The class of source to which this name is assigned.  Supported classes
        are ``pointsource``, ``cluster`` or ``transient``.  Shorthand class
        names for these sources are also allowed (``S``, ``CL``, or ``SV``,
        respectively).  Alternatively, the class can be ``None`` or ``short``,
        indicating a simple source identifier in the form ``"HHMM-DD"``.

    Returns
    -------
    name : str
        A unique identifier for the source with coordinates truncated to the
        appropriate number of significant figures for the source class.
    """

    source_class = str(source_class).lower()
    if source_class in ["sv", "t", "tr", "transient", "v", "var", "variable"]:
        source_class = "SV"
    elif source_class in ["c", "cl", "cluster"]:
        source_class = "CL"
    elif source_class in ["s", "p", "ps", "pointsource", "point_source"]:
        source_class = "S"
    elif source_class in ["none", "short"]:
        source_class = "short"
    else:
        print("Unrecognized source class {}".format(source_class))
        print("Defaulting to [ S ]")
        source_class = "S"

    # ra in (0, 360)
    ra = npmod(ra, 360)

    opts = dict(sep="", pad=True, precision=3)
    rastr = Angle(ra * u.deg).to_string(**opts, unit="hour")
    decstr = Angle(dec * u.deg).to_string(alwayssign=True, **opts)

    if source_class == "SV":
        rastr = rastr[:8]
        decstr = decstr.split(".")[0]
    elif source_class == "CL":
        rastr = rastr[:4]
        decstr = decstr[:5]
    elif source_class == "S":
        rastr = rastr.split(".")[0]
        decr = "{:.3f}".format(dec * 60).split(".")[1]
        decstr = decstr[:5] + ".{}".format(decr[:1])
    elif source_class == "short":
        rastr = rastr[:4]
        decstr = decstr[:3]
        return "{}{}".format(rastr, decstr)

    name = "{}-{} J{}{}".format(observatory, source_class, rastr, decstr)
    return name


def decode_filename_to_act(source_name, band):
    """
    Reverses the IAU string format to standard RA/Dec degrees.
    Returns: (filename, top_title, readable_subtitle)
    """
    source_str = str(source_name)
    if " J" in source_str:
        coord_str = source_str.split(" J")[1]
        match = re.match(r"(\d{6})([+-]\d{4}\.?\d*)", coord_str)
        if match:
            ra_raw, dec_raw = match.groups()
            ra_formatted = f"{ra_raw[0:2]}h{ra_raw[2:4]}m{ra_raw[4:6]}s"
            dec_formatted = f"{dec_raw[0:3]}d{dec_raw[3:]}m"
            
            # 1. Strict IAU filename for your advisor's cross-match
            filename = f"J{coord_str}_{band}.png"
            
            # 2. Bold top title
            top_title = f"ACT J{coord_str}"
            
            # 3. Readable subtitle (Using your original SkyCoord math)
            try:
                c = SkyCoord(f"{ra_formatted} {dec_formatted}")
                readable_subtitle = f"RA: {c.ra.deg:.3f}°, Dec: {c.dec.deg:.3f}°"
            except ValueError:
                readable_subtitle = f"RA: {ra_formatted}, Dec: {dec_formatted}"
                
            return filename, top_title, readable_subtitle
            
    # Fallback for names that don't follow IAU standard
    safe_source = source_str.replace(" ", "_").replace("SO-S_", "").replace("SO-SV_", "")
    filename = f"{safe_source}_{band}.png"
    top_title = f"ACT {safe_source}"
    readable_subtitle = "Coordinates Unknown"
    
    return filename, top_title, readable_subtitle


# hdf5 thumbnail tools
def load_custom_hdf5(fname):
    import h5py
    from pixell import enmap

    context = h5py.File(fname, "r")
    thumbnails = {}
    for key in context:
        thumb_dict = {}
        maptype = enmap.fix_python3(context[key]["maptype"][()])
        Id = enmap.fix_python3(context[key]["Id"][()]).split(f"_{maptype}")[0]
        thumb_maps = enmap.read_hdf(context[key])

        for k in context[key].keys():
            if k == "wcs" or k == "data":
                continue
            thumb_dict[k] = enmap.fix_python3(context[key][k][()])

        if Id not in thumbnails:
            thumbnails[Id] = thumb_dict
        thumbnails[Id][maptype] = thumb_maps

    return thumbnails


def organize_thumbnails_by_source(thumbnails):
    org_thumbs = {}
    for thumb_id in thumbnails:
        source = thumb_id.split("_")[0]
        if source not in org_thumbs:
            org_thumbs[source] = []
        org_thumbs[source].append(thumbnails[thumb_id])

    return org_thumbs


def get_thumbnail_times(thumbnails):
    times = []
    for i in range(len(thumbnails)):
        times.append(float(thumbnails[i]["t"]))
    return times


def read_thumbnails(data_dir):
    # find and sort files in one line
    files = sorted(glob.glob(os.path.join(data_dir, "collated_thumbnail_*.hdf5")))
    print(f"Found {len(files)} files. Loading...")
    all_data = []
    for f in files:
        thumbs = load_custom_hdf5(f)
        all_data.append(organize_thumbnails_by_source(thumbs))
    return all_data


# lightcurve tools
def read_lightcurves(filepath):
    data = np.loadtxt(filepath, delimiter=",", unpack=True, dtype=str)
    lc_data = {
        "time": np.asarray(data[0], dtype=float),
        "source_names": np.asarray(data[1], dtype=str),
        "ra": np.asarray(data[2], dtype=float),
        "dec": np.asarray(data[3], dtype=float),
        "arrs": data[4],
        "bands": data[5],
        "flux": np.asarray(data[7], dtype=float),
        "dflux": np.asarray(data[8], dtype=float),
    }
    return lc_data


def load_biermann_catalog(
    filepath: str = "data/act_depth1_transient_catalog.txt",
) -> pd.DataFrame:
    """
    Load the Biermann et al. ACT Depth1 Transient Catalog
    file into a DataFrame.

    The catalog is a fixed-width formatted text file with
    specific column widths and names.
    """
    colspecs = [
        (0, 14),  # Name
        (15, 18),  # Seq
        (19, 26),  # RAdeg
        (27, 34),  # DEdeg
        (35, 39),  # PosErr
        (40, 46),  # f220-Peak
        (47, 53),  # f150-Peak
        (54, 60),  # f090-Peak
        (61, 66),  # e_f220-Peak
        (67, 72),  # e_f150-Peak
        (73, 77),  # e_f090-Peak
        (78, 82),  # f220-Mean
        (83, 87),  # f150-Mean
        (88, 92),  # f090-Mean
        (93, 96),  # e_f220-Mean
        (97, 100),  # e_f150-Mean
        (101, 104),  # e_f090-Mean
        (105, 123),  # f220-tpeak
        (124, 142),  # f150-tpeak
        (143, 161),  # f090-tpeak
        (162, 166),  # Sp+Index
        (167, 170),  # e_Sp+Index
        (171, 197),  # Simbad-Id
        (198, 207),  # Type
        (208, 219),  # SpType
        (220, 225),  # Sep
        (226, 235),  # pval
        (236, 243),  # Dist
    ]
    names = [
        "Name",
        "Seq",
        "RAdeg",
        "DEdeg",
        "PosErr",
        "f220_Peak",
        "f150_Peak",
        "f090_Peak",
        "e_f220_Peak",
        "e_f150_Peak",
        "e_f090_Peak",
        "f220_Mean",
        "f150_Mean",
        "f090_Mean",
        "e_f220_Mean",
        "e_f150_Mean",
        "e_f090_Mean",
        "f220_tpeak",
        "f150_tpeak",
        "f090_tpeak",
        "Sp_Index",
        "e_Sp_Index",
        "Simbad_Id",
        "Type",
        "SpType",
        "Sep",
        "pval",
        "Dist",
    ]
    df = pd.read_fwf(
        filepath, colspecs=colspecs, names=names, skiprows=52, na_values=""
    )
    df["Name"] = df["Name"].str.strip()
    df["Seq"] = df["Seq"].str.strip()
    df["Simbad_Id"] = df["Simbad_Id"].str.strip()
    df["Type"] = df["Type"].str.strip()
    df["SpType"] = df["SpType"].str.strip()

    return df
