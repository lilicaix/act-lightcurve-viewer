import random
from pathlib import Path

import pytest
from astropy import units as u
from astropy.time import Time

from act_lightcurve_viewer.file_io import load_biermann_catalog
from act_lightcurve_viewer.transients import Flare, Transient

DATA_FILE = Path(__file__).parent.parent / "data" / "biermann_transient_candidates.txt"


@pytest.fixture(scope="module")
def catalog():
    return load_biermann_catalog(DATA_FILE)


def test_add_and_get_flares():
    flare = Flare(
        name="J175954+104419",
        seq="1",
        ra=269.975 * u.deg,
        dec=10.739 * u.deg,
        pos_err=13.0 * u.arcsec,
        f150_peak=186.0 * u.mJy,
        e_f150_peak=56.0 * u.mJy,
        f150_tpeak=Time(57970.142063589832, format="mjd"),
    )
    t = Transient(name="J175954+104419", simbad_id="1SWASP J175954.36+104418.9")
    assert t.get_flares() == []
    t.add_flare(flare)
    flares = t.get_flares()
    assert len(flares) == 1
    assert flares[0] is flare


def test_get_flares_returns_copy():
    t = Transient(name="test")
    t.add_flare(
        Flare(
            name="x", seq="1", ra=0.0 * u.deg, dec=0.0 * u.deg, pos_err=0.0 * u.arcsec
        )
    )
    flares = t.get_flares()
    flares.clear()
    assert len(t.get_flares()) == 1


def test_from_catalog_single_flare(catalog):
    # Source 1: J175954+104419 — fully populated, one flare
    rows = catalog[catalog["Name"] == "J175954+104419"]
    t = Transient.from_catalog(rows)

    assert t.name == "J175954+104419"
    assert t.simbad_id == "1SWASP J175954.36+104418.9"
    assert t.source_type == "BYDraV*"
    assert pytest.approx(t.dist, rel=1e-4) == 307.78

    flares = t.get_flares()
    assert len(flares) == 1
    f = flares[0]

    assert f.seq == "1"
    assert pytest.approx(f.ra.to(u.deg).value, rel=1e-5) == 269.975
    assert pytest.approx(f.dec.to(u.deg).value, rel=1e-5) == 10.739
    assert pytest.approx(f.f220_peak.to(u.mJy).value, rel=1e-4) == 274.0
    assert pytest.approx(f.f150_peak.to(u.mJy).value, rel=1e-4) == 186.0
    assert pytest.approx(f.f090_peak.to(u.mJy).value, rel=1e-4) == 145.0
    assert pytest.approx(f.f150_tpeak.mjd, rel=1e-10) == 57970.142063589832
    assert pytest.approx(f.sp_index, rel=1e-4) == 0.7


def test_from_catalog_multi_flare(catalog):
    # Sources 2a and 2b are two flares from the same physical source
    rows = catalog[catalog["Seq"].isin(["2a", "2b"])]
    t = Transient.from_catalog(rows)

    assert len(t.get_flares()) == 2
    seqs = {f.seq for f in t.get_flares()}
    assert seqs == {"2a", "2b"}


def test_from_catalog_random_source(catalog):
    random.seed(42)
    name = random.choice(catalog["Name"].tolist())
    rows = catalog[catalog["Name"] == name]
    t = Transient.from_catalog(rows)

    assert t.name == name
    assert len(t.get_flares()) == len(rows)
    for flare in t.get_flares():
        assert flare.ra.unit == u.deg
        assert flare.dec.unit == u.deg
        if flare.f150_peak is not None:
            assert flare.f150_peak.unit == u.mJy
        if flare.f150_tpeak is not None:
            assert isinstance(flare.f150_tpeak, Time)
