# imports
import glob
import os
import re

import h5py
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates.angles import Angle
from numpy import mod as npmod
from pixell import enmap


# string + naming functions
def radec_to_str_name(ra: float, dec: float, source_class="pointsource", observatory="SO"):
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
    """Reverses the IAU string format to standard RA/Dec degrees."""
    source_str = str(source_name)
    if " J" in source_str:
        coord_str = source_str.split(" J")[1]
        match = re.match(r"(\d{6})([+-]\d{4}\.?\d*)", coord_str)
        if match:
            ra_raw, dec_raw = match.groups()
            ra_formatted = f"{ra_raw[0:2]}h{ra_raw[2:4]}m{ra_raw[4:6]}s"
            dec_formatted = f"{dec_raw[0:3]}d{dec_raw[3:]}m"
            try:
                c = SkyCoord(f"{ra_formatted} {dec_formatted}")
                return f"ACT_RA{c.ra.deg:.3f}_Dec{c.dec.deg:.3f}_{band}.png"
            except ValueError:
                pass
    safe_source = source_str.replace(" ", "_").replace("SO-S_", "ACT_")
    return f"{safe_source}_{band}.png"


# hdf5 thumbnail tools
def load_custom_hdf5(fname):
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
