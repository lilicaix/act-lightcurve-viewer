# act-lightcurve-viewer

Tools for analyzing and plotting ACT depth1 thumbnails and associated lightcurves.

## Installation

**1. Create and activate a virtual environment**

```bash
python -m venv myvenv
source myvenv/bin/activate
```

**2. Install the package**

To install directly from GitHub:

```bash
pip install git+https://github.com/lilicaix/act-lightcurve-viewer.git
```

Or, to install in editable mode from a local clone (recommended for development):

```bash
git clone https://github.com/lilicaix/act-lightcurve-viewer.git
cd act-lightcurve-viewer
pip install -e .
```

## Usage

Run `script.py` with arguments pointing to your data files:

```bash
python script.py -t [/path/to/thumbnails/] -c [coadd days]
```

This will plot the thumbnails and lightcurves with the specified number of coadded days.