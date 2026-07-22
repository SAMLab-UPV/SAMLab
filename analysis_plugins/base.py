"""
SAMLab - Submarine Acoustic Monitoring Laboratory
Copyright (C) 2026 Universitat Politècnica de València

This file is part of SAMLab.

Licensed under the GNU General Public License v3.0 (GPLv3) or later.
See the LICENSE file for details.
"""

class AnalysisBase:
    """Base class for all analysis plugins."""
    name = "Unnamed Analysis"
    description = "No description provided."
    default_params = {}  # GUI uses this to build input fields

    def __init__(self, **params):
        self.params = {**self.default_params, **params}

    def analyze(self, x, fs, dsp, bands, verbose=0):
        """
        Analyze an audio signal.

        Parameters
        ----------
        x : np.ndarray
            Audio samples, usually in digital counts.
        fs : int | float
            Sampling frequency in Hz.
        dsp : DSP
            DSP calibration information.
        bands : Bands
            Hydrophone sensitivity / band calibration data.
        verbose : int | bool
            If true, print progress information.

        Returns
        -------
        events, indicators
            events: pandas.DataFrame
            indicators: dict
        """
        raise NotImplementedError("Subclasses must implement this method")
