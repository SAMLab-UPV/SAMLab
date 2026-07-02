"""
SAMLab - Submarine Acoustic Monitoring Laboratory
Copyright (C) 2026 Universitat Politècnica de València

This file is part of SAMLab.

Licensed under the GNU Affero General Public License v3.0 or later.
See the LICENSE file for details.
"""

from analysis_plugins.base import AnalysisBase
import numpy as np
from scipy.fft import fft
import pandas as pd
import time

from modules.models import EVENT_FIELDS_TYPES

class FIN_WHALE_20Hz_detector(AnalysisBase):
    name = "FIN_WHALE_20Hz_detector"
    description = (
        "<b>FIN WHALE 20 Hz PULSE DETECTOR (Legacy SAMLab implementation)</b><br>"
        "Based on Fin Whale Index=Whale band / No Whale band.<br>"
        "Removes pulses shorter than 400 msec., with f0 out of the interval 22 +- 3Hz, and without periodicity.<br>"
        "(c) Ramon Miralles 14-3-2022<br>"
        )
    
    outputs = {
        "events": [
            {
                "type": "FW20",
                "tag": "Fin Whale 20 Hz Pulse",
            }
        ],
        "indicators": {
            "Fin Whale 20 Hz pulses[#]": int,
            "Fin Whale 20 Hz IPI[sec.]": float
        }
    }

    version='1.0' # Object detector version - Don´t forget to increase version number when changes are are done!
    

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

        if verbose:
            print("Checking for Fin Whale 20-Hz pulses using legacy SAMLAB detector...")

        # ----------------- Settings -----------------
        delta_f = 3  #Desired frequency resolution in [Hz] of the spectrogram delta_f=fs/Npoints
        #
        # The specific values for whale call frequency band and noise frequency band are inspired in the work:
        #
        # Seasonality of blue andfin whale calls andthe influence of sea ice in the Western Antarctic Peninsula
        # Ana Sirovic, John A. Hildebrand, Deep-Sea Research II 51 (2004) 2327?2344
        #
        # Fin whale calls are repetitive down sweeps in frequency from 28 to 15 Hz (1 second
        # duration). In addition a narrowband 89 Hz component is present.
        
        whale_fband = 22 # Whale call frequency band (main frequency band finwhale call)
        no_whale_fband = [10, 30] # No Whale (noise) Fin whale pulse around (19-28 Hz)
        acceptable_fwhale_f0 = [whale_fband - 3, whale_fband + 3] #Aceptable central frequencies of Fin whale pulses
        thres = 2 #Threshold to get detections after comparing the "whale_fband/mean(no_whale_fbands)"
        min_duration = 0.4  # Minimum duration in sec. of a Fin Whale pulse

        Npoints = int(np.ceil(fs / delta_f)) #Number of points of the sliding DTFT so that we get the desired frequency resolution

        # Scale factor from digital counts to volts
        scale_factor = 10 ** (dsp.gain / 20) * 2 ** (dsp.nbits - 1)
        sh = np.mean(bands.sh)

        # ---- 1st ACOUSTIC POWER FIN WHALE INDEX
        Ntot = len(x)
        fd = round(whale_fband / fs * Npoints)
        fd1 = round(no_whale_fband[0] / fs * Npoints)
        fd2 = round(no_whale_fband[1] / fs * Npoints)

        Xfwhalek = np.zeros(Ntot - Npoints + 1, dtype=complex)
        Xfno_whale1k = np.zeros(Ntot - Npoints + 1, dtype=complex)
        Xfno_whale2k = np.zeros(Ntot - Npoints + 1, dtype=complex)

        n = np.arange(Npoints)
        Xk = np.sum(x[n] * np.exp(-1j * 2 * np.pi * fd / Npoints * n))
        Xk1 = np.sum(x[n] * np.exp(-1j * 2 * np.pi * fd1 / Npoints * n))
        Xk2 = np.sum(x[n] * np.exp(-1j * 2 * np.pi * fd2 / Npoints * n))

        Xfwhalek[0] = Xk
        Xfno_whale1k[0] = Xk1
        Xfno_whale2k[0] = Xk2

        exp_factor = np.exp(1j * 2 * np.pi * fd / Npoints)
        exp_factor1 = np.exp(1j * 2 * np.pi * fd1 / Npoints)
        exp_factor2 = np.exp(1j * 2 * np.pi * fd2 / Npoints)

        
        for k in range(Ntot - Npoints):
            xdFactor = x[k + Npoints] - x[k]
            Xk = exp_factor * (Xk + xdFactor)
            Xk1 = exp_factor1 * (Xk1 + xdFactor)
            Xk2 = exp_factor2 * (Xk2 + xdFactor)
            Xfwhalek[k + 1] = Xk
            Xfno_whale1k[k + 1] = Xk1
            Xfno_whale2k[k + 1] = Xk2
        end_time = time.time()
    

        # ----------------- Detection -----------------
        freq_median = np.median(np.vstack([np.abs(Xfwhalek), np.abs(Xfno_whale1k), np.abs(Xfno_whale2k)]), axis=0)
        Xfw_det = np.abs(Xfwhalek) / freq_median
        Xfw_det = Xfw_det > thres

        # ----------------- Remove short events -----------------
        transitions = np.diff(Xfw_det.astype(int))
        run_starts = np.where(transitions == 1)[0]
        run_ends = np.where(transitions == -1)[0]

        if len(run_starts) > len(run_ends):
            run_starts = run_starts[:-1]
        if len(run_starts) < len(run_ends):
            run_ends = run_ends[1:]

        run_lengths = run_ends - run_starts
        mask = run_lengths >= int(round(min_duration * fs)) - 1
        run_starts = run_starts[mask]
        run_ends = run_ends[mask]

        # Compute the differences for IEI
        n = len(run_starts)
        diffs = np.empty(n, dtype=float)
        if n ==0:
            diffs = np.array([])
        elif n==1:
            diffs[0] = 0.0   # or np.nan
        else:
            diffs = np.empty_like(run_starts, dtype=float)
            diffs[1:-1] = (run_starts[2:] - run_starts[:-2]) / 2
            diffs[0] = run_starts[1] - run_starts[0]
            diffs[-1] = run_starts[-1] - run_starts[-2]
            diffs=diffs/fs # To seconds

        event_type = self.outputs["events"][0]["type"]
        event_tag = self.outputs["events"][0]["tag"]
        events_list = []

        for start, end, IEI in zip(run_starts, run_ends, diffs):
            x_frag = x[start:end]
            x_frag_v = x_frag / scale_factor
            X_frag = np.abs(fft(x_frag))
            X_frag = X_frag[:len(X_frag)//2]
            freq_vec = np.linspace(0, fs/2, len(X_frag))
    
            fi = np.where((freq_vec > no_whale_fband[0]) & (freq_vec < no_whale_fband[1]))[0]
            centfreq_est = freq_vec[fi[np.argmax(X_frag[fi])]]
    
            if acceptable_fwhale_f0[0] <= centfreq_est <= acceptable_fwhale_f0[1]:
                events_list.append({
                    "start": np.uint32(start),
                    "end": np.uint32(end),
                    "fmin": np.float32(acceptable_fwhale_f0[0]),
                    "fmax": np.float32(acceptable_fwhale_f0[1]),
                    "f0": np.float32(centfreq_est),
                    "BW": np.float32(0),
                    "ICI": np.float32(IEI),
                    "SPL": np.float32(10 * np.log10(np.mean(x_frag_v ** 2)) - sh),
                    "score": np.int8(0),
                    "user": np.float32(np.nan),
                    "type": event_type,
                    "tag": event_tag,
                    "Tdata": np.nan,
                    "Fdata": np.nan,
                })


        if events_list:  # make sure events_list is not empty
            events = pd.DataFrame(events_list)
            indicators["Fin Whale 20 Hz pulses[#]"] = len(events)
            indicators["Fin Whale 20 Hz IPI[sec.]"] = events["ICI"].median()
        else:
            indicators["Fin Whale 20 Hz pulses[#]"] = 0
            indicators["Fin Whale 20 Hz IPI[sec.]"] = np.nan

        if verbose:
            print(f"Analysis finished in {time.time()-start_time:.2f} s")

        return events, indicators

