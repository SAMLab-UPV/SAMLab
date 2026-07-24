"""
SAMLab - Submarine Acoustic Monitoring Laboratory
Copyright (C) 2026 Universitat Politècnica de València

This file is part of SAMLab.

Licensed under the GNU General Public License v3.0 (GPLv3) or later.
See the LICENSE file for details.
"""

from analysis_plugins.base import AnalysisBase

import numpy as np
import pandas as pd
from scipy.signal import spectrogram, find_peaks, windows
from scipy.ndimage import median_filter, gaussian_filter, binary_closing, label
import time

from modules.models import EVENT_FIELDS_TYPES

class DOLPHIN_CONTOUR_WHISTLE_detector(AnalysisBase):
    name = "DOLPHIN_CONTOUR_WHISTLE_detector"
    description = (
        "<b>DOLPHIN CONTOUR WHISTLE DETECTOR</b><br>"
        "Dolphin whistle detector and contour tracking<br>"
        "based on image Processing+Peaks (c) 2022 Ramon Miralles<br>"
        )

    outputs = {
        "events": [
            {
                "type": "DCW",
                "tag": "Dolphin Contour Whistle",
            }
        ],
        "indicators": {
            "Dolphin Contour Whistles [#]": int,
            "Mean duration of the whistles [sec.]": float
        }
    }

    version = "2.1"

    def analyze(self, x, fs, dsp, bands, verbose):
        start_time = time.time()

        events_data = []

        events = pd.DataFrame({
            name: pd.Series(dtype=dtype)
            for name, dtype in EVENT_FIELDS_TYPES
        })

        indicators = {
            key: np.nan
            for key in self.outputs["indicators"]
        }

        event_type = self.outputs["events"][0]["type"]
        event_tag = self.outputs["events"][0]["tag"]

        if verbose:
            print("Checking for Contour Whistle events...")

        # Settings
        secs_analysis = 4
        sh = np.mean(bands.sh) # I should use the sensitivity of the whistle frequency range, but for now I will use the mean of the bands
        tfr_resolution = 50 # TFR resolution in Hz (around 45-46 Hz) for whistles
        zoom_freq_up_limit = 21000
        #zoom_freq_low_limit = 6000
        zoom_freq_low_limit = 2000 # Testing to be able to detect Pilot whales
        whis_frag_shortD = 0.07 # Shortest whistle fragment duration in seconds
        whis_frag_minDf = 500 # Minimum whistle fragment frequency variation
        #min_valid_SPL = 46 # Min SPL so that everything below that is not taken into account
        min_valid_SPL = 26 # Min SPL so that everything below that is not taken into account

        DnFaintW = 6 # Look 5 pixels ahead (or behind) for the next whistle candidate (to trace faint whistles)
        OctaveJump = 1.15 # Number of octaves that a whistle can jump
        Slope_tol = 0.0 # Tolerance angle to join

        # Convert from digital counts to [Volts].
        xv = np.asarray(x, dtype=np.float64) / dsp.scale_factor 

        N_max = len(x)
        N_analysis = int(round(secs_analysis * fs))

        tfr_window = int(round(fs / tfr_resolution))
        tfr_overlap = int(round(tfr_window * 7 / 10))


        # Pre-compute window once
        hann_window = windows.hamming(tfr_window)
        
        idx = np.arange(0, N_max, N_analysis)

        for i0 in idx:
            y_part = xv[i0:i0 + N_analysis]
            if y_part.size == 0:
                continue

            try:
                # OPTIMIZATION: Use nperseg directly without recomputing
                f, n, ps = spectrogram(
                    y_part,
                    fs=fs,
                    window=hann_window,
                    nperseg=tfr_window,
                    noverlap=tfr_overlap,
                    nfft=tfr_window,
                    mode="psd",
                )
            except Exception:
                continue

            # Select only wanted frequency range - vectorized
            freq_mask = (f >= zoom_freq_low_limit) & (f <= zoom_freq_up_limit)
            if not np.any(freq_mask):
                continue

            f = f[freq_mask]
            ps = ps[freq_mask, :]

            # Convert to dB and apply threshold
            ps = 10 * np.log10(ps + 1e-20) - sh
            ps[ps < min_valid_SPL] = min_valid_SPL

            # -------- Median filtering / denoising --------
            # OPTIMIZATION: Use scipy.ndimage with optimized settings
            order = 30
            matrix_mm2 = median_filter(ps, size=(2 * order + 1, 1), mode="nearest")
            ps_median = ps - matrix_mm2

            # -------- Average subtraction / constant tone removal (Gillespie) --------
            # OPTIMIZATION: Vectorized instead of loop
            alfa = 0.02
            ps_denoise = np.zeros_like(ps_median)

            ncols = ps_median.shape[1]
            
            # Forward pass (vectorized)
            Ytnorm = np.zeros(ps_median.shape[0], dtype=np.float64)
            for l in range(ncols):
                Ytnorm = (1 - alfa) * Ytnorm + alfa * ps_median[:, l] # Normalize across time with exponential moving average
                ps_denoise[:, l] = ps_median[:, l] - np.abs(Ytnorm)

            # Backward pass (vectorized)
            # Ytnorminv = np.zeros(ps_median.shape[0], dtype=np.float64)
            # for l in range(ncols - 1, -1, -1):
            #     li = ncols - 1 - l
            #     Ytnorminv = (1 - alfa) * Ytnorminv + alfa * ps_median[:, li]
            #     ps_denoise[:, li] = ps_denoise[:, li] - np.abs(Ytnorminv)
            Ytnorminv = np.zeros(ps_median.shape[0], dtype=np.float64)
            for l in range(ncols - 1, -1, -1):
                Ytnorminv = (1 - alfa) * Ytnorminv + alfa * ps_median[:, l]
                ps_denoise[:, l] = ps_denoise[:, l] - np.abs(Ytnorminv)

            ps_smoothed = ps_denoise
            ps_smoothed[ps_smoothed < 0] = 0

            # -------- Peaks in vertical direction --------
            # OPTIMIZATION: Pre-allocate and use numpy operations
            peaks_in_whistle = np.zeros_like(ps_smoothed)
            peaks_in_whistle_grad2 = np.zeros_like(ps_smoothed)

            for ni in range(ps_smoothed.shape[1]):
                col = ps_smoothed[:, ni]
                #locs, _ = find_peaks(col, prominence=0.8, distance=100)
                df = np.median(np.diff(f))
                peak_distance_bins = max(1, int(round(150 / df)))  # ~150 Hz
                locs, _ = find_peaks(col, prominence=0.8, distance=peak_distance_bins)

                # Compute second derivative efficiently
                if col.size >= 3:
                    grad2 = np.zeros_like(col)
                    grad2[1:-1] = np.diff(col, n=2)
                    peaks_in_whistle_grad2[:, ni] = grad2

                peaks_in_whistle[locs, ni] = ps[locs, ni]

            peaks_in_whistle = (peaks_in_whistle_grad2 < -2) * peaks_in_whistle

            # -------- Peaks in horizontal direction --------
            peaks_in_whistle2 = np.zeros_like(ps_smoothed)
            peaks_in_whistle2_grad2 = np.zeros_like(ps_smoothed)

            for fi in range(ps_smoothed.shape[0]):
                row = ps_smoothed[fi, :]
                locs, _ = find_peaks(row, prominence=1.0)

                # Compute gradient more efficiently
                grad2 = np.gradient(np.gradient(row))
                peaks_in_whistle2_grad2[fi, :] = grad2
                peaks_in_whistle2[fi, locs] = ps[fi, locs]

            peaks_in_whistle2 = (peaks_in_whistle2_grad2 < -3) * peaks_in_whistle2
            ii = peaks_in_whistle2 != 0
            peaks_in_whistle[ii] = peaks_in_whistle2[ii]

            # -------- Variable threshold with frequency --------
            min_thres = np.percentile(ps_smoothed, 90, axis=1, keepdims=True)
            ps_smoothed = np.maximum(ps_smoothed - min_thres, 0)

            # -------- Derivative mask --------
            DY, DX = np.gradient(ps_smoothed)
            mag = np.sqrt(DX**2 + DY**2)  # More efficient than complex
            DY2, DX2 = np.gradient(mag)
            Imask = np.sqrt(DX2**2 + DY2**2) > 2.5

            structure = _disk_structure(5)
            ImaskClose = binary_closing(Imask, structure=structure)

            peaks_in_whistle = peaks_in_whistle * ImaskClose

            # -------- Gradient map for slope compatibility --------
            # OPTIMIZATION: Avoid complex operations, use direct angle
            GradTF = np.arctan2(DY, DX)
            GradTF = np.hstack([GradTF, np.zeros((GradTF.shape[0], 2))])
            GradTF = gaussian_filter(GradTF, sigma=2)
            GradTF = GradTF[:, 2:]
            GradTF[peaks_in_whistle == 0] = 0

            # -------- First pass: local labels --------
            ws1stpass, _ = label(peaks_in_whistle > 0)

            # -------- Second pass: global whistle fragment connection --------
            wsout, labelVal = join_whistle_parts(
                ws1stpass=ws1stpass.copy(),
                GradTF=GradTF,
                peaks_in_whistle=peaks_in_whistle,
                DnFaintW=DnFaintW,
                OctaveJump=OctaveJump,
                Slope_tol=Slope_tol,
            )

            for j in range(1, labelVal):
                rr, cc = np.where(wsout == j)
                if rr.size == 0:
                    continue

                #f_object = f[rr]
                #t_object = n[cc]

                pairs = {}
                for r, c in zip(rr, cc):
                    val = peaks_in_whistle[r, c]
                    if (c not in pairs) or (val > pairs[c][1]):
                        pairs[c] = (r, val)

                cc_u = np.array(sorted(pairs.keys()))
                rr_u = np.array([pairs[c][0] for c in cc_u])

                t_object = n[cc_u]
                f_object = f[rr_u]

                 # ADD THIS HERE
                order_idx = np.argsort(t_object)
                t_object = t_object[order_idx]
                f_object = f_object[order_idx]

                nmin = float(np.min(t_object))
                nmax = float(np.max(t_object))
                fmin = float(np.min(f_object))
                fmax = float(np.max(f_object))

                if (nmax - nmin) > whis_frag_shortD and (fmax - fmin) > whis_frag_minDf:
                    start_sample = int(nmin * fs + i0)
                    end_sample = int(nmax * fs + i0)

                    spl_vals = peaks_in_whistle[rr, cc]
                    spl_mean = float(np.mean(spl_vals)) if spl_vals.size > 0 else np.nan

                    events_data.append({
                        "start": np.uint32(start_sample),
                        "end": np.uint32(end_sample),
                        "fmin": np.float32(fmin),
                        "fmax": np.float32(fmax),
                        "f0": np.float32((fmin + fmax) / 2.0),
                        "BW": np.float32(fmax - fmin),
                        "ICI": np.float32(np.nan),
                        "SPL": np.float32(spl_mean),
                        "score": np.int8(-1),
                        "user": np.nan,
                        "type": event_type,
                        "tag": event_tag,
                        "Tdata": (t_object + (i0 / fs)).astype(np.float32),
                        "Fdata": f_object.astype(np.float32),
                    })

        # -------- Outputs --------
        columns = ["start", "end", "fmin", "fmax", "f0", "BW", "ICI", "SPL", "score", "user", "type", "tag", "Tdata", "Fdata"]

        # if events:
        #     E = pd.DataFrame(events, columns=columns)
        #     durations = (E["end"].astype(float) - E["start"].astype(float)) / float(fs)
        #     I = [int(len(E)), float(np.mean(durations))]
        # else:
        #     E = pd.DataFrame(columns=columns)
        #     I = [0, 0]
        if events_data:
            events_data = remove_duplicate_whistles(
                events_data,
                fs=fs,
                min_time_overlap_ratio=0.6,
                max_mean_freq_diff=300.0,
                max_point_freq_diff=450.0,
                max_gap_sec=0.08,
                merge=True,
            )

            events = pd.DataFrame(events_data, columns=columns)
            durations = (events["end"].astype(float) - events["start"].astype(float)) / float(fs)
            indicators["Dolphin Contour Whistles [#]"] = int(len(events))
            indicators["Mean duration of the whistles [sec.]"] = float(np.mean(durations))    
        else:
            indicators["Dolphin Contour Whistles [#]"] = 0
            indicators["Mean duration of the whistles [sec.]"] = 0    

        if verbose:
            print(f"Analysis finished in {time.time()-start_time:.2f} s")
            
        return events, indicators


def join_whistle_parts(ws1stpass, GradTF, peaks_in_whistle, DnFaintW, OctaveJump, Slope_tol):
    """Optimized whistle joining with numpy operations"""
    M = 5
    w = 0.5

    spec_shape = ws1stpass.shape
    wsout = np.zeros(spec_shape, dtype=np.int32)
    labelVal = 1

    labels, counts = np.unique(ws1stpass[ws1stpass > 0], return_counts=True)
    remaining = {int(lbl): int(cnt) for lbl, cnt in zip(labels, counts)}

    while np.any(ws1stpass):
        ilabelstart = max(remaining, key=remaining.get)
        remaining.pop(ilabelstart, None)

        ind = np.argwhere(ws1stpass == ilabelstart)
        ws1stpass[ws1stpass == ilabelstart] = 0
        wsout[tuple(ind.T)] = labelVal

        # Forward propagation
        finished = False
        while not finished:
            ind = np.argwhere(wsout == labelVal)
            if ind.size < M:
                break
                
            ind = ind[np.argsort(ind[:, 1])]
            row = ind[:, 0]
            col = ind[:, 1]

            end_pts = ind[-M:]
            end_grad = np.mean(GradTF[end_pts[:, 0], end_pts[:, 1]])
            end_strength = np.mean(peaks_in_whistle[end_pts[:, 0], end_pts[:, 1]])

            idrest = np.argwhere(ws1stpass != 0)
            if idrest.size == 0:
                break

            # OPTIMIZATION: Vectorized candidate search
            idcand = idrest[(idrest[:, 1] > col[-1]) & (idrest[:, 1] <= col[-1] + DnFaintW)]
            if idcand.size == 0:
                break

            cand_labels = np.unique(ws1stpass[idcand[:, 0], idcand[:, 1]])

            candidates = []
            for cand in cand_labels:
                pts = np.argwhere(ws1stpass == cand)
                if pts.size < M:
                    pts = pts[np.argsort(pts[:, 1])]
                else:
                    pts = pts[np.argsort(pts[:, 1])]
                    
                rowi = pts[:, 0]
                coli = pts[:, 1]

                if coli[0] < col[-1]:
                    continue

                end_slope = (row[-1] - row[-2]) if len(row) >= 2 else 0
                extRow = row[-1] + end_slope * (coli[0] - col[-1])

                if abs(extRow - rowi[0]) > abs(extRow * OctaveJump - extRow):
                    continue

                beg_pts = pts[:M] if len(pts) >= M else pts
                beg_grad = np.mean(GradTF[beg_pts[:, 0], beg_pts[:, 1]])
                beg_strength = np.mean(peaks_in_whistle[beg_pts[:, 0], beg_pts[:, 1]])

                if my_sign(beg_grad, Slope_tol) != my_sign(end_grad, Slope_tol):
                    continue

                cost = w * abs(end_grad - beg_grad) + (1 - w) * abs(end_strength - beg_strength)
                candidates.append((cost, int(cand)))

            if candidates:
                _, chosen = min(candidates, key=lambda z: z[0])
                pts = np.argwhere(ws1stpass == chosen)
                ws1stpass[ws1stpass == chosen] = 0
                wsout[tuple(pts.T)] = labelVal
                remaining.pop(chosen, None)
            else:
                finished = True

        # Backward propagation (same optimization)
        finished = False
        while not finished:
            ind = np.argwhere(wsout == labelVal)
            if ind.size < M:
                break
                
            ind = ind[np.argsort(ind[:, 1])]
            row = ind[:, 0]
            col = ind[:, 1]

            start_pts = ind[:M]
            start_grad = np.mean(GradTF[start_pts[:, 0], start_pts[:, 1]])
            start_strength = np.mean(peaks_in_whistle[start_pts[:, 0], start_pts[:, 1]])

            idrest = np.argwhere(ws1stpass != 0)
            if idrest.size == 0:
                break

            idcand = idrest[(idrest[:, 1] < col[0]) & (idrest[:, 1] >= col[0] - DnFaintW)]
            if idcand.size == 0:
                break

            cand_labels = np.unique(ws1stpass[idcand[:, 0], idcand[:, 1]])

            candidates = []
            for cand in cand_labels:
                pts = np.argwhere(ws1stpass == cand)
                if pts.size < M:
                    pts = pts[np.argsort(pts[:, 1])]
                else:
                    pts = pts[np.argsort(pts[:, 1])]
                    
                rowi = pts[:, 0]
                coli = pts[:, 1]

                if coli[-1] > col[0]:
                    continue

                start_slope = (row[0] - row[1]) if len(row) >= 2 else 0
                extRow = row[0] + start_slope * (coli[-1] - col[0])

                if abs(extRow - rowi[-1]) > abs(extRow * OctaveJump - extRow):
                    continue

                end_pts = pts[-M:] if len(pts) >= M else pts
                end_grad = np.mean(GradTF[end_pts[:, 0], end_pts[:, 1]])
                end_strength = np.mean(peaks_in_whistle[end_pts[:, 0], end_pts[:, 1]])

                if my_sign(end_grad, Slope_tol) != my_sign(start_grad, Slope_tol):
                    continue

                cost = w * abs(start_grad - end_grad) + (1 - w) * abs(start_strength - end_strength)
                candidates.append((cost, int(cand)))

            if candidates:
                _, chosen = min(candidates, key=lambda z: z[0])
                pts = np.argwhere(ws1stpass == chosen)
                ws1stpass[ws1stpass == chosen] = 0
                wsout[tuple(pts.T)] = labelVal
                remaining.pop(chosen, None)
            else:
                finished = True

        labelVal += 1

    return wsout, labelVal

def contours_should_merge(ev1, ev2,
                          min_time_overlap_ratio=0.6,
                          max_mean_freq_diff=300.0,
                          max_point_freq_diff=450.0,
                          max_gap_sec=0.0):
    """
    Decide whether two contours are duplicates or near-duplicates.
    """
    t1 = np.asarray(ev1["Tdata"], dtype=float)
    f1 = np.asarray(ev1["Fdata"], dtype=float)
    t2 = np.asarray(ev2["Tdata"], dtype=float)
    f2 = np.asarray(ev2["Fdata"], dtype=float)

    if len(t1) < 2 or len(t2) < 2:
        return False

    # sort by time
    idx1 = np.argsort(t1)
    idx2 = np.argsort(t2)
    t1, f1 = t1[idx1], f1[idx1]
    t2, f2 = t2[idx2], f2[idx2]

    start1, end1 = t1[0], t1[-1]
    start2, end2 = t2[0], t2[-1]

    overlap = min(end1, end2) - max(start1, start2)
    gap = max(start1, start2) - min(end1, end2)

    dur1 = end1 - start1
    dur2 = end2 - start2
    min_dur = max(min(dur1, dur2), 1e-9)

    overlap_ratio = max(0.0, overlap) / min_dur

    # either enough overlap, or a tiny gap
    if not (overlap_ratio >= min_time_overlap_ratio or (overlap < 0 and gap <= max_gap_sec)):
        return False

    # compare frequencies where both contours exist
    if overlap > 0:
        t0 = max(start1, start2)
        t1c = min(end1, end2)

        if t1c <= t0:
            return False

        n_samples = max(10, int(np.ceil((t1c - t0) / 0.01)))
        tt = np.linspace(t0, t1c, n_samples)

        f1i = np.interp(tt, t1, f1)
        f2i = np.interp(tt, t2, f2)

        fdiff = np.abs(f1i - f2i)

        if np.mean(fdiff) > max_mean_freq_diff:
            return False
        if np.percentile(fdiff, 90) > max_point_freq_diff:
            return False

        return True

    # if no overlap but very small gap, compare end/start frequencies
    if end1 < start2:
        df = abs(f1[-1] - f2[0])
    else:
        df = abs(f2[-1] - f1[0])

    return df <= max_point_freq_diff


def merge_two_contours(ev1, ev2, fs):
    """
    Merge two contours into one.
    """
    t1 = np.asarray(ev1["Tdata"], dtype=float)
    f1 = np.asarray(ev1["Fdata"], dtype=float)
    t2 = np.asarray(ev2["Tdata"], dtype=float)
    f2 = np.asarray(ev2["Fdata"], dtype=float)

    idx1 = np.argsort(t1)
    idx2 = np.argsort(t2)
    t1, f1 = t1[idx1], f1[idx1]
    t2, f2 = t2[idx2], f2[idx2]

    t_all = np.unique(np.round(np.concatenate([t1, t2]), 6))
    t_all.sort()

    f1i = np.interp(t_all, t1, f1, left=np.nan, right=np.nan)
    f2i = np.interp(t_all, t2, f2, left=np.nan, right=np.nan)

    f_out = np.where(
        np.isnan(f1i), f2i,
        np.where(np.isnan(f2i), f1i, 0.5 * (f1i + f2i))
    )

    valid = ~np.isnan(f_out)
    t_out = t_all[valid]
    f_out = f_out[valid]

    if len(t_out) == 0:
        return ev1

    out = ev1.copy()
    out["start"] = np.uint32(max(0, int(round(t_out[0] * fs))))
    out["end"] = np.uint32(max(0, int(round(t_out[-1] * fs))))
    out["fmin"] = np.float32(np.min(f_out))
    out["fmax"] = np.float32(np.max(f_out))
    out["f0"] = np.float32((out["fmin"] + out["fmax"]) / 2.0)
    out["BW"] = np.float32(out["fmax"] - out["fmin"])
    out["Tdata"] = t_out.astype(np.float32)
    out["Fdata"] = f_out.astype(np.float32)

    spl1 = ev1.get("SPL", np.nan)
    spl2 = ev2.get("SPL", np.nan)
    out["SPL"] = np.float32(np.nanmean([spl1, spl2]))

    return out


def remove_duplicate_whistles(events,
                              fs,
                              min_time_overlap_ratio=0.6,
                              max_mean_freq_diff=300.0,
                              max_point_freq_diff=450.0,
                              max_gap_sec=0.0,
                              merge=True):
    """
    Remove or merge duplicate contours.
    """
    if not events:
        return events

    used = [False] * len(events)
    out = []

    for i in range(len(events)):
        if used[i]:
            continue

        base = events[i].copy()

        for j in range(i + 1, len(events)):
            if used[j]:
                continue

            if contours_should_merge(
                base, events[j],
                min_time_overlap_ratio=min_time_overlap_ratio,
                max_mean_freq_diff=max_mean_freq_diff,
                max_point_freq_diff=max_point_freq_diff,
                max_gap_sec=max_gap_sec
            ):
                if merge:
                    base = merge_two_contours(base, events[j], fs)
                    used[j] = True
                else:
                    len_i = len(base["Tdata"])
                    len_j = len(events[j]["Tdata"])
                    spl_i = base.get("SPL", -np.inf)
                    spl_j = events[j].get("SPL", -np.inf)

                    score_i = len_i + 0.05 * (0 if np.isnan(spl_i) else spl_i)
                    score_j = len_j + 0.05 * (0 if np.isnan(spl_j) else spl_j)

                    if score_i >= score_j:
                        used[j] = True
                    else:
                        used[i] = True
                        base = events[j].copy()

        if not used[i]:
            out.append(base)

    return out

def my_sign(x, thres):
    if abs(x) < thres:
        return 0
    return np.sign(x)


def _disk_structure(radius):
    """Create disk structure for morphological operations"""
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (xx * xx + yy * yy) <= radius * radius