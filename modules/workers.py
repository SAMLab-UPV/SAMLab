"""
SAMLab - Submarine Acoustic Monitoring Laboratory
Copyright (C) 2026 Universitat Politècnica de València

This file is part of SAMLab.

Licensed under the GNU General Public License v3.0 (GPLv3) or later.
See the LICENSE file for details.
"""

import os
import pandas as pd
from PySide6.QtCore import Signal, QThread

# I use a worker to move the long processing outside the main GUI thread
class PluginWorker(QThread):

    finished = Signal(object)  # send results back if needed
    failed = Signal(str) # send error message if needed

    def __init__(self, selected_plugins, x, fs, dsp, bands, verbose):
        super().__init__()
        self.selected_plugins = selected_plugins
        self.x = x
        self.fs = fs
        self.dsp = dsp
        self.bands = bands
        self.verbose = verbose

    def run(self):
        
        try:
            events = []

            #for cls, params in self.selected_plugins.items():
            for cls in self.selected_plugins:
                if self.isInterruptionRequested():
                    print("Cancelled by user.")
                    break
                # cast_params = {}
                # for k, v in params.items():
                #     try:
                #         cast_params[k] = eval(v)
                #     except:
                #         cast_params[k] = v

                # plugin = cls(**cast_params)
                plugin = cls()
                # All prints inside analyze() will now go to your status window
                E, I = plugin.analyze(self.x, self.fs, self.dsp, self.bands, self.verbose)
                events.append(E)
            self.finished.emit(events)
            
        except Exception as e:
            self.failed.emit(str(e))

# I create a worker for the Deployment analysis (similar to the previous, but adds filenname and indicatoris)
class DeploymentFileWorker(QThread):
    finished = Signal(object, object)  # events, file_indicators
    failed = Signal(str)

    def __init__(self, selected_plugins, x, fs, dsp, bands, wavpath, verbose):
        super().__init__()
        self.selected_plugins = selected_plugins
        self.x = x
        self.fs = fs
        self.dsp = dsp
        self.bands = bands
        self.wavpath = wavpath
        self.verbose = verbose

    def run(self):
        try:
            events = pd.DataFrame()
            # file_indicators = []  # Changed to a dictionary to store indicators for each file
            file_indicators = {}

            for cls in self.selected_plugins:
                if self.isInterruptionRequested():
                    break

                plugin = cls()
                E, I = plugin.analyze(
                    self.x,
                    self.fs,
                    self.dsp,
                    self.bands,
                    self.verbose
                )

                if E is not None and len(E) > 0:
                    if isinstance(E, pd.DataFrame):
                        E = E.copy()
                    else:
                        E = pd.DataFrame(E)

                    if "filename" not in E.columns:
                        E.insert(0, "filename", self.wavpath.stem)

                    events = pd.concat([events, E], ignore_index=True)

                if I is not None:
                    #file_indicators.extend(I.values())
                    file_indicators.update(I) # Changed to work with dictionary of indicators

            self.finished.emit(events, file_indicators)

        except Exception as e:
            self.failed.emit(str(e))

# To Fix problemns accesing CSV files from slow USB / HDD disks
# -------- Worker thread that scans CSVs safely --------
class CsvScanner(QThread):
    finished = Signal(list)

    def __init__(self, folder):
        super().__init__()
        self.folder = folder

    def run(self):
        csv_files = []
        try:
            # scandir is MUCH faster than listdir
            with os.scandir(self.folder) as it:
                for entry in it:
                    if entry.is_file() and entry.name.lower().endswith(".csv"):
                        csv_files.append(entry.path)
        except Exception as e:
            print("Scan error:", e)

        self.finished.emit(csv_files)