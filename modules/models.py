"""
SAMLab - Submarine Acoustic Monitoring Laboratory
Copyright (C) 2026 Universitat Politècnica de València

This file is part of SAMLab.

Licensed under the GNU Affero General Public License v3.0 or later.
See the LICENSE file for details.
"""

from dataclasses import dataclass

@dataclass
class DSP:
    gain: float
    nbits: int

    @property
    def scale_factor(self):
        return 10 ** (self.gain / 20) * 2 ** (self.nbits - 1)

@dataclass
class Bands:
    number: list[int]
    label: list[str]
    sh: list[float]


EVENT_FIELDS_TYPES = [
    ("start", "uint32"),
    ("end", "uint32"),
    ("fmin", "float32"),
    ("fmax", "float32"),
    ("f0", "float32"),
    ("BW", "float32"),
    ("ICI", "float32"),
    ("SPL", "float32"),
    ("score", "int8"),
    ("user", "float32"), # User defined data for several purposes
    ("type", "string"),
    ("tag", "string"),
    ("Tdata", "object"), # variable-length arrays
    ("Fdata", "object"),
]


def default_dsp() -> DSP:
    gain = 0
    nbits = 16
    return DSP(gain=gain, nbits=nbits,
    )


def default_bands() -> Bands:
    return Bands(
        number=[18, 21, 33, 37],
        label=[
            "Mean SPL 63 Hz band[dB re 1µPa^2]",
            "Mean SPL 125 Hz band[dB re 1µPa^2]",
            "Mean SPL 2000_Hz band[dB re 1µPa^2]",
            "Mean SPL 5000_Hz band[dB re 1µPa^2]",
        ],
        sh=[-156.87, -157.33, -164.21, -164.21],
    )