from __future__ import annotations

from astropy import units as u
from astropy.time import Time
from dataclasses import dataclass
import pandas as pd
from typing import Optional


@dataclass
class Flare:
    """A single transient detection event."""

    name: str
    seq: str

    ra: u.Quantity
    dec: u.Quantity
    pos_err: u.Quantity

    f220_peak: Optional[u.Quantity] = None
    f150_peak: Optional[u.Quantity] = None
    f090_peak: Optional[u.Quantity] = None
    e_f220_peak: Optional[u.Quantity] = None
    e_f150_peak: Optional[u.Quantity] = None
    e_f090_peak: Optional[u.Quantity] = None

    f220_mean: Optional[u.Quantity] = None
    f150_mean: Optional[u.Quantity] = None
    f090_mean: Optional[u.Quantity] = None
    e_f220_mean: Optional[u.Quantity] = None
    e_f150_mean: Optional[u.Quantity] = None
    e_f090_mean: Optional[u.Quantity] = None

    f220_tpeak: Optional[Time] = None
    f150_tpeak: Optional[Time] = None
    f090_tpeak: Optional[Time] = None

    sp_index: Optional[float] = None
    e_sp_index: Optional[float] = None


class Transient:
    """A physical transient source, which may have one or more flare events."""

    def __init__(
        self,
        name: str,
        simbad_id: Optional[str] = None,
        source_type: Optional[str] = None,
        sp_type: Optional[str] = None,
        sep: Optional[u.Quantity] = None,
        pval: Optional[float] = None,
        dist: Optional[u.Quantity] = None,
    ):
        self.name = name
        self.simbad_id = simbad_id
        self.source_type = source_type
        self.sp_type = sp_type
        self.sep = sep
        self.pval = pval
        self.dist = dist
        self._flares: list[Flare] = []

    def add_flare(self, flare: Flare) -> None:
        self._flares.append(flare)

    def get_flares(self) -> list[Flare]:
        return list(self._flares)

    def __repr__(self) -> str:
        n = len(self._flares)
        return (
            f"Transient(name={self.name!r}, simbad_id={self.simbad_id!r}, "
            f"flares={n})"
        )

    @classmethod
    def from_catalog(cls, df_rows: pd.DataFrame) -> Transient:
        """Build a Transient from one or more rows of the Biermann catalog DataFrame."""
        import numpy as np

        first = df_rows.iloc[0]

        def _val(v):
            return None if (v is None or (isinstance(v, float) and np.isnan(v))) else v

        def _qty(v, unit):
            val = _val(v)
            return None if val is None else val * unit

        def _time(v):
            val = _val(v)
            return None if val is None else Time(val, format="mjd")

        transient = cls(
            name=first["Name"],
            simbad_id=_val(first["Simbad_Id"]),
            source_type=_val(first["Type"]),
            sp_type=_val(first["SpType"]),
            sep=_val(first["Sep"]),
            pval=_val(first["pval"]),
            dist=_val(first["Dist"]),
        )
        for _, row in df_rows.iterrows():
            transient.add_flare(
                Flare(
                    name=row["Name"],
                    seq=row["Seq"],
                    ra=row["RAdeg"] * u.deg,
                    dec=row["DEdeg"] * u.deg,
                    pos_err=_qty(row["PosErr"], u.arcsec),
                    f220_peak=_qty(row["f220_Peak"], u.mJy),
                    f150_peak=_qty(row["f150_Peak"], u.mJy),
                    f090_peak=_qty(row["f090_Peak"], u.mJy),
                    e_f220_peak=_qty(row["e_f220_Peak"], u.mJy),
                    e_f150_peak=_qty(row["e_f150_Peak"], u.mJy),
                    e_f090_peak=_qty(row["e_f090_Peak"], u.mJy),
                    f220_mean=_qty(row["f220_Mean"], u.mJy),
                    f150_mean=_qty(row["f150_Mean"], u.mJy),
                    f090_mean=_qty(row["f090_Mean"], u.mJy),
                    e_f220_mean=_qty(row["e_f220_Mean"], u.mJy),
                    e_f150_mean=_qty(row["e_f150_Mean"], u.mJy),
                    e_f090_mean=_qty(row["e_f090_Mean"], u.mJy),
                    f220_tpeak=_time(row["f220_tpeak"]),
                    f150_tpeak=_time(row["f150_tpeak"]),
                    f090_tpeak=_time(row["f090_tpeak"]),
                    sp_index=_val(row["Sp_Index"]),
                    e_sp_index=_val(row["e_Sp_Index"]),
                )
            )
        return transient

