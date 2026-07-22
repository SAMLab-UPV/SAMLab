"""
SAMLab - Submarine Acoustic Monitoring Laboratory
Copyright (C) 2026 Universitat Politècnica de València

This file is part of SAMLab.

Licensed under the GNU General Public License v3.0 (GPLv3) or later.
See the LICENSE file for details.
"""

from analysis_plugins.base import AnalysisBase

import time
import numpy as np
import pandas as pd

from modules.models import EVENT_FIELDS_TYPES


class AMBIENT_noise_level(AnalysisBase):
    #
    # AMBIENT 1/3 OCTAVE SOUND PRESSURE LEVEL
    #
    # It does not return any events (E). Only indicators for the CSV file.
    # It computes (per file):
    #
    #  * 1/3 octave SPL in the bands of 63 Hz, 125 Hz, 2 kHz and 5 kHz
    #
    # Mean of the SPLs for an integration interval of T=1 second.
    # (c) Ramon Miralles 2019
    

    name = "AMBIENT_noise_level"
    description = (
        "<b>AMBIENT 1/3 OCTAVE SOUND PRESSURE LEVEL</b><br>"
        "It computes 1/3 octave SPL in the bands of 63Hz, 125 Hz, 2kHz and 5 kHz "
        "for integration interval of T=1 second.<br>"
        "(c) Ramon Miralles 2019<br>"
    )

    outputs = {
        "events": [],
        "indicators": {
            "Mean SPL 63 Hz band[dB re 1\\muPa^2]": float,
            "Mean SPL 125 Hz band[dB re 1\\muPa^2]": float,
            "Mean SPL 2000_Hz band[dB re 1\\muPa^2]": float,
            "Mean SPL 5000_Hz band[dB re 1\\muPa^2]": float
        }
    }

    version = "1.0"

   

    @staticmethod
    def clipp_test(signal: np.ndarray) -> float:
        # clipp_test : Test the clipping of an audio signal based on the histogram.
        #              This function implements the algorithm described in: 
        #              "Detection of Clipped Fragments in Speech Signals, Sergei
        #              Aleinik, Yuri Matveev, International Journal of Computer and Information Engineering, 2014"
        #
        #   [ score ] = clipp_test( x )
        #   Input:
        #   signal : Signal fragment (or complete signal) to test for clipping. 
        #            Signal should be large enough to compute the
        #            histogram with 50 bins (2000 samples aprox.)
        #
        #   score  : Score value of the clipping.
        #            0: No clipping
        #            1: Severe clipping

        signal = np.asarray(signal, dtype=float).ravel()

        if signal.size == 0:
            return 1.0

        s_med = np.mean(signal)

        # MATLAB hist(signal, 50) returns 50 bin centers and counts
        y, edges = np.histogram(signal, bins=50)
        x = (edges[:-1] + edges[1:]) / 2.0

        if len(x) < 2:
            return 1.0

        res_x = x[1] - x[0]

        y_left = y[x < s_med]
        if y_left.size == 0:
            return 1.0
        if y_left[0] > 10:
            Dist_l = np.sum(y_left < y_left[0])
        else:
            Dist_l = 0

        y_right = y[x > s_med]
        if y_right.size == 0:
            return 1.0
        if y_right[-1] > 10:
            Dist_r = np.sum(y_right < y_right[-1])
        else:
            Dist_r = 0

        denom = x[-1] - x[0]
        if denom <= 0:
            return 1.0

        score = min(1.0, 2.0 * max(Dist_l, Dist_r) * res_x / denom)
        return float(score)

    @staticmethod
    def one_third_octave_SPL(x, fs, band, T):
        #
        # FUNTION    : one_third_octave_SPL (Computes the one third octave SPL for 
        #              the given band and for T a seconds average)
        #
        # SYNTAX     : [ xSPL,Tsa ] = one_third_octave_SPL( x,fs,band, T,'total' )
        #
        # INPUT      :
        #            x   : 1D signal
        #            fs  : Sample frequency
        #            band: IEC/ANSI Octave band (18: 63 Hz, 21: 125 Hz, 33: 2 kHz, 37: 5 kHz)
        #                  Band can be an array if computing several bands is
        #                  needed in one call band=[18, 21, 33, 37]
        #                  Band can also be 'total'. In this case the xSPL is
        #                  obtained for the whole bandwith.            
        #            T   : Time span in seconds to compute the (xSPL)
        #
        # OUTPUT     :
        #            xSPL: one third octave SPL for 
        #                  the given band and for T a seconds average
        #            Tsa : Time scale (start points of the T seconds of the
        #                  averaging interval start)
        #
        # Copyright 2018. iTEAM (GTS) - Universitat Politecnica de Valencia (UPV). 
        # Guillermo Lara and Ramon Miralles. All rights reserved. 
        # Terms of use: Only for private use of the QUIETMED project partners.
        # Please do not distribute.
        #---------------------------------------------------------------------
        
        x = np.asarray(x, dtype=float).ravel()

        Tsamp = int(round(T * fs))
        Tsamppow2 = int(2 * round(Tsamp / 2))  # even number, matching MATLAB

        f = np.arange(0, Tsamppow2 // 2 + 1) * fs / Tsamppow2

        if isinstance(band, str) and band == "total":
            ilow = 1   # MATLAB ilow=2, but Python is 0-based
            ihigh = len(f) - 1
            band_list = ["total"]
        else:
            band = np.asarray(band).ravel()
            G = 10 ** (3 / 10)
            fm = 1000 * G ** ((band - 30) / 3)
            flow = fm * G ** (-1 / 6)
            fhigh = fm * G ** (1 / 6)

            ilow = np.zeros(len(band), dtype=int)
            ihigh = np.zeros(len(band), dtype=int)

            for ii in range(len(band)):
                ilow[ii] = np.where(f >= flow[ii])[0][0]
                ihigh[ii] = np.where(f <= fhigh[ii])[0][-1]

            band_list = band

        # MATLAB: frames = 1:Tsamp:(length(x)-Tsamp);
        # Python 0-based equivalent:
        frames = np.arange(0, len(x) - Tsamp + 1, Tsamp, dtype=int)

        if isinstance(band, str) and band == "total":
            xSPL = np.zeros(len(frames), dtype=float)
        else:
            xSPL = np.zeros((len(band_list), len(frames)), dtype=float)

        for i, start_idx in enumerate(frames):
            xfrag = x[start_idx:start_idx + Tsamp]
            clip_score = AMBIENT_noise_level.clipp_test(xfrag)

            # MATLAB: if clip_score < 0.25
            if clip_score < 0.25:
                Xfrag = np.abs(np.fft.fft(xfrag, n=Tsamppow2))
                Xfrag = Xfrag[:(Tsamppow2 // 2 + 1)]

                if isinstance(band, str) and band == "total":
                    val = 2 * np.sum(Xfrag[ilow:ihigh + 1] ** 2)
                    val += Xfrag[0] ** 2
                    xSPL[i] = val
                else:
                    for l in range(len(band_list)):
                        xSPL[l, i] = 2 * np.sum(Xfrag[ilow[l]:ihigh[l] + 1] ** 2)
            else:
                if isinstance(band, str) and band == "total":
                    xSPL[i] = np.nan
                else:
                    xSPL[:, i] = np.nan

        xSPL = 10 * np.log10(xSPL / (Tsamppow2 ** 2))
        Tsa = frames / fs

        return xSPL, Tsa

    def analyze(self, x, fs, dsp, bands, verbose=0):
        start_time = time.time()

        events = pd.DataFrame({
            name: pd.Series(dtype=dtype)
            for name, dtype in EVENT_FIELDS_TYPES
        })

        indicators = {
            key: np.nan
            for key in self.outputs["indicators"]
        }

        avint = 1  # T = 1 second

        if verbose:
            print("Computing ambient noise level indicators...")

        # MATLAB:
        # scale_factor = 10^((dsp.gain)/20) * 2^(dsp.nbits-1);
        # xv = x' / scale_factor;
        scale_factor = 10 ** (dsp.gain / 20.0) * 2 ** (dsp.nbits - 1)
        xv = np.asarray(x, dtype=float).ravel() / scale_factor

        # MATLAB:
        # [one_third_at_all_bands, Tsa] = one_third_octave_SPL(xv, fs, bands.number, avint);
        one_third_at_all_bands, Tsa = self.one_third_octave_SPL(
            xv, fs, bands.number, avint
        )

        # MATLAB:
        # mean_one_third_at_all_bands = nanmean(one_third_at_all_bands,2) - bands.sh';
        mean_one_third_at_all_bands = (
            np.nanmean(one_third_at_all_bands, axis=1) - np.asarray(bands.sh, dtype=float).ravel()
        )

        for key, value in zip(
            self.outputs["indicators"],
            mean_one_third_at_all_bands
        ):
            indicators[key] = float(value)

        if verbose:
            print(f"Analysis finished in {time.time() - start_time:.2f} s")

        return events, indicators