# act-lightcurve-viewer

Tools for analyzing and plotting ACT depth1 thumbnails and associated lightcurves.

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/lilicaix/act-lightcurve-viewer.git
cd act-lightcurve-viewer
```

**2. Create and activate a virtual environment**

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
```

**3. Install the package**

For regular use:

```bash
uv pip install -e .
```

For development (includes pytest, ruff, etc.):

```bash
uv pip install -e ".[dev]"
```

## Usage

Run `script.py` with arguments pointing to your data files:

```bash
python script.py -t [/path/to/thumbnails/] -c [coadd days]
```

This will plot the thumbnails and lightcurves with the specified number of coadded days.