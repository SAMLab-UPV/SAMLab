"""
SAMLab - Submarine Acoustic Monitoring Laboratory
Copyright (C) 2026 Universitat Politècnica de València

This file is part of SAMLab.

Licensed under the GNU Affero General Public License v3.0 or later.
See the LICENSE file for details.
"""

import csv
import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from scipy.io import loadmat, wavfile,savemat
from datetime import datetime
from re import split as re_split # split with multiple delimiters

from PySide6.QtWidgets import QDialog, QMessageBox, QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QPushButton, QCheckBox, QLineEdit
from PySide6.QtCore import Qt
from PySide6.QtCore import QEventLoop

# Imports to handle analysis plugins
import analysis_plugins
from analysis_plugins import plugin_loader
from modules.ui_helpers import StatusWindow, EmittingStream
from modules.workers import DeploymentFileWorker



# --- Helper: load .mat either via h5py (v7.3 HDF5) or scipy (older MAT) ---
def load_mat_any(matpath: Path) -> dict:
    """
    Load a .mat file. If it's v7.3 (HDF5) use h5py, otherwise use scipy.loadmat.
    Returns a Python dict with variable names as keys.
    NOTE: h5py returns datasets which may need conversion to numpy arrays.
    """
    matpath = Path(matpath)
    if not matpath.exists():
        raise FileNotFoundError(matpath)
    try:
        # Try using h5py to detect HDF5 MAT-file (v7.3)
        with h5py.File(matpath, 'r') as f:
            # If h5py can open and has MATLAB fields, convert datasets to numpy
            data = {}
            def _to_py(name, obj):
                # only top-level variables
                if isinstance(obj, h5py.Dataset):
                    try:
                        arr = obj[()]
                        # convert bytes to str where appropriate
                        if arr.dtype.kind == 'S':
                            # bytes -> str
                            arr = arr.astype('U')
                        data[name] = arr
                    except Exception:
                        data[name] = None
            for name, obj in f.items():
                _to_py(name, obj)
            return data
    except (OSError, IOError):
        # Not HDF5 MAT v7.3 — fallback to scipy
        mat = loadmat(str(matpath), squeeze_me=True, struct_as_record=False)
        # Clean up scipy keys
        # Remove MATLAB meta keys
        for k in ['__header__','__version__','__globals__']:
            mat.pop(k, None)
        return mat

def save_partial_lock(filename, plugins_data):

    with h5py.File(filename, "w") as f:
        plugins_grp = f.create_group("plugins")

        for i, plugin_cls in enumerate(plugins_data):
            grp = plugins_grp.create_group(str(i))
            grp.attrs["name"] = plugin_cls.name
            grp.attrs["version"] = plugin_cls.version

def load_partial_lock(filename):
    plugins = []

    with h5py.File(filename, "r") as h5f:
        for key in h5f["plugins"]:
            grp = h5f["plugins"][key]

            plugins.append({
                "name": grp.attrs["name"],
                "version": grp.attrs["version"]
                #"version": grp.attrs["version"][:]
            })

    return plugins

# --- Helper: show yes/no dialog  ---
def ask_yes_no(self,title: str, message: str) -> bool:
        reply = QMessageBox.question(None, title, message,
                                     QMessageBox.StandardButton.Yes,
                                     QMessageBox.StandardButton.No)
        return reply == QMessageBox.StandardButton.Yes


def dlg_analyze_deplyment_warning(self):
    #
    # Show a simple dialog warning about analyzing a deployment.
    # Returns True if user chooses Yes, False if No, None if closed.
    #
    dialog = QDialog(self)
    dialog.setWindowTitle("Analyze Deployment Warning")
    dialog.setModal(True)

    # Main layout
    layout = QVBoxLayout()
    text1='<b>A valid deployment directory must have:</b><br><br>'
    text2='<b>1.- A "deployment_info.mat" file with acoustic campaign information<br> (can be created from Analyze Menu).</b><br><br>'
    text3='<b>2.- All the audio files (*.WAV) named as:</b><br>'
    text4='<ul><li>XXX_N_1_YYYYMMDD_HHMMSS.WAV</li><br>'
    text5='<li>XXX: 3 Letter location</li><br>'
    text6='<li>N: Geographical position (North, South, East, West)</li><br>'
    text7='<li>YYYYMMDD_HHMMSS: Day & Time when the file started</li></ul><br>'
    text8='<b><font color=red>WARNING:</font> Depending on the number of files this might take a lot of time <br>(typically several days).</b><br><br>'
    text9='<b>Are you sure you want to proceed?</b>'
    # Message
    label = QLabel(f"{text1}{text2}{text3}{text4}{text5}{text6}{text7}{text8}{text9}")
    layout.addWidget(label)

    # Buttons
    buttons_layout = QHBoxLayout()
    yes_btn = QPushButton("Yes")
    no_btn = QPushButton("No")

    buttons_layout.addWidget(yes_btn)
    buttons_layout.addWidget(no_btn)
    layout.addLayout(buttons_layout)

    dialog.setLayout(layout)

    # Result container
    result = {"value": None}

    # Button actions
    def yes():
        result["value"] = True
        dialog.accept()

    def no():
        result["value"] = False
        dialog.reject()

    yes_btn.clicked.connect(yes)
    no_btn.clicked.connect(no)

    dialog.exec()
    return result["value"]

def dlg_analyze_deployment_settings(self, parent=None):
    #
    # Presents a dialog for quering the PathName of the deployment and
    # a character that indicates if restart the analysis or continue.
    # 
    # Returns:
    # 
    #    deployment_folder (str or None)
    #    restart_tasks ('R' or 'C' or None)
    #
    
    def choose_folder():
        folder = QFileDialog.getExistingDirectory(
            None,
            "Select folder",
            "")
        if folder:
            folder_edit.setText(folder)

    def restart_clicked():
        restart_cb.setChecked(True)
        continue_cb.setChecked(False)

    def continue_clicked():
        restart_cb.setChecked(False)
        continue_cb.setChecked(True)
    
    def ok_pressed():
        # return values
        result["folder"] = folder_edit.text()
        result["restart"] = "R" if restart_cb.isChecked() else "C"
        dialog.accept()

    def cancel_pressed():
        dialog.reject()

    dialog = QDialog(parent)
    dialog.setWindowTitle("Analyze Deployment Dialog")
    dialog.setModal(True)
    dialog.resize(400, 200)

    # --- MAIN LAYOUT ---
    layout = QVBoxLayout(dialog)

    # --- Deployment folder selection ---
    folder_layout = QHBoxLayout()
    folder_btn = QPushButton("Deployment Folder")
    folder_edit = QLineEdit("?")
    folder_edit.setReadOnly(True)

    folder_layout.addWidget(folder_btn)
    folder_layout.addWidget(folder_edit)
    layout.addLayout(folder_layout)

    # --- Restart / Continue ---
    label = QLabel("If previous analysis found... Restart or Continue?")
    layout.addWidget(label)

    checkbox_layout = QHBoxLayout()
    restart_cb = QCheckBox("Restart")
    continue_cb = QCheckBox("Continue or Add Tasks")
    continue_cb.setChecked(True)  # default like MATLAB version

    checkbox_layout.addWidget(restart_cb, alignment=Qt.AlignmentFlag.AlignCenter)
    checkbox_layout.addWidget(continue_cb, alignment=Qt.AlignmentFlag.AlignCenter)
    layout.addLayout(checkbox_layout)

    # --- OK / Cancel ---
    btn_layout = QHBoxLayout()
    ok_btn = QPushButton("OK")
    cancel_btn = QPushButton("Cancel")

    btn_layout.addWidget(ok_btn)
    btn_layout.addWidget(cancel_btn)
    layout.addLayout(btn_layout)

    # Output container
    result = {"folder": None, "restart": None}

    # Button actions
    folder_btn.clicked.connect(choose_folder)
    restart_cb.clicked.connect(restart_clicked)
    continue_cb.clicked.connect(continue_clicked)
    ok_btn.clicked.connect(ok_pressed)
    cancel_btn.clicked.connect(cancel_pressed)

    # Show dialog (modal)
    dialog.exec()

    return result["folder"], result["restart"]

def analyze_samaruc_deployment(self,PathName: str, restart_tasks: str, wposition) -> None:
#
#
# FUNTION    : analyze_SAMARUC_deployment
#
# SYNTAX     : analyze_SAMARUC_deployment(PathName,restart_tasks,wposition)
#
#              PathName     : Directory of the deployment (must have a valid
#                             "deployment_info.mat" file with all deployment data)
#                             The function looks and analyze all the *.dat or
#                             *.DAT files in the given PathName
#
#              restart_tasks: Char with 'R' or 'C'.
#                             'R' : Restart analysis of the whole
#                                   deployment (Erasing what it has)
#                             'C' : Continue with the analysis where it was
#                                   left with the possibility of adding
#                                   detection tasks.
#              wposition    : Window position of the calling function
#
#
#---------------------------------------------------------------------

    PathName = Path(PathName)
    verbose = 1
    
    # ---  GET INFO ABOUT ALL POSSIBLE ANALYSIS TASKS (CLASES) ---
    plugins = plugin_loader.discover_plugins(analysis_plugins)
    analysis_tasks= [plugin_cls.name for plugin_cls in plugins]
    analysis_tasks_help= [plugin_cls.description for plugin_cls in plugins]
    analysis_tasks_indtag= [list(plugin_cls.outputs["indicators"].keys()) for plugin_cls in plugins]
    analysis_tasks_version= [plugin_cls.version for plugin_cls in plugins]
    

    # ------  LOAD DEPLOYMENT INFORMATION: DATE, TIME, FS and MONO  -------
    deployment_info_path = PathName / "deployment_info.mat"
    if not deployment_info_path.exists():
        raise FileNotFoundError(f"{deployment_info_path} not found")
    mat = load_mat_any(deployment_info_path)
    # Expecting keys: ddate, fs, mono, dsp, bands (as in MATLAB)
    ddate = mat.get('ddate', None)
    fs = int(np.squeeze(mat.get('fs'))) if 'fs' in mat else None
    mono = int(np.squeeze(mat.get('mono'))) if 'mono' in mat else None
    dsp = mat.get('dsp', {})    # may be structured
    bands = mat.get('bands', {})
    print(f"Deployment started at day/time: {ddate}")
    print(f"Recording setup: fs={fs} mono={mono}")

    # ------ PREPARE FILE LISTS AND FILENAMES ------
    #allfiles = sorted(PathName.glob('*.wav'))  
    allfiles = sorted(
        Path(entry.path)
        for entry in os.scandir(PathName)
        if entry.is_file() and (
            entry.name.endswith(".wav") or
            entry.name.endswith(".WAV")
            )
    )
    csvheader = ['Filename', 'Date']  # This fields are always present in the CSV

    # Create a variable with the name of the CSV and the *.H5 resulting from the analysis
    deployment_indicators_filename = PathName / 'deployment_indicators.csv'
    deployment_indicators_filename_backup = PathName / 'deployment_indicators_backup.csv'
    detected_events_filename = PathName / 'detected_events.h5'   # using h5 for events (MAT v7.3 style)
    detected_events_filename_backup = PathName / 'detected_events_backup.h5'
    partial_analysis_lock_filename = PathName / 'partial_lock.h5'
    summary_log = PathName / 'summary_log_analysis.txt'

    # ---------- If restart requested, ask confirmation and backup existing outputs ----------
    if restart_tasks == 'R':
        ok = ask_yes_no(self,"Restart?", "Are you sure you want to restart the analysis of the deployment?\nThis will backup/delete previous analysis files.")
        if not ok:
            print("User canceled restart")
            return
        # backup files if they exist
        if deployment_indicators_filename.exists():
            deployment_indicators_filename.replace(deployment_indicators_filename_backup)
        if Path(detected_events_filename).exists():
            Path(detected_events_filename).replace(detected_events_filename_backup)
        if partial_analysis_lock_filename.exists():
            partial_analysis_lock_filename.unlink() # Delete lock file

    # ----------- Open log file -------------
    with open(summary_log, 'a', encoding='utf-8') as fidlog:
        user = os.getenv('USER') or os.getenv('USERNAME') or 'unknown'
        fidlog.write(f"{user} started analysis of the deployment on {datetime.now():%Y_%m_%d %H:%M:%S}\n")

    # ---------- Check for previous analysis  ----------

    firstfiletoanalyze = 1
    results_events = pd.DataFrame()  # pandas DataFrame to store event rows
    csvheader_current = list(csvheader)  # mutable copy
    tasksDone = None
    existing_indicators = None
    existing_indicators_backup = None

    # Helper lambda for existence check
    def exists(p: Path) -> bool:
        return p.exists()

    # CASE 1: Lock EXIST, detections and indicator files EXIST, but BACKUP NOT EXIST 
    #         FIRST TIME ANALIZED BUT UNFINISHED ANALYSIS
    if exists(deployment_indicators_filename) and not exists(deployment_indicators_filename_backup) \
       and exists(detected_events_filename) and not exists(detected_events_filename_backup) \
       and exists(partial_analysis_lock_filename):
        # Resume interrupted analysis
        with open(summary_log, 'a', encoding='utf-8') as fidlog:
            fidlog.write('Deployment analysis interrupted. Resuming...\n')
        lock = load_partial_lock(partial_analysis_lock_filename)
        # Get selected analysis names from lock
        detector_names_in_lock=[p["name"] for p in lock]

        # Validate detector_names_in_lock: detector_names_in_lock indexes must not be empty
        if detector_names_in_lock is None:
            raise RuntimeError("Partial lock exists but no selection_values found in lock file.")
        
        # Read header of CSV to get indicators
        with open(deployment_indicators_filename, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        existing_indicators = [s.strip() for s in first_line.split(';') if s.strip()]
        csvheader_current = existing_indicators

        # Build indx list: detectors whose tags are present in existing_indicators
        indx = []
        for l, indtag in enumerate(analysis_tasks_indtag):
            if set(indtag).issubset(set(existing_indicators)):
                indx.append(l)
        
        # Get the inidices from detector_names_in_lock
        selected_indices = [analysis_tasks.index(name)
                            for name in detector_names_in_lock
                            if name in analysis_tasks]
        
        if set(selected_indices) != set(indx):
            print('ERROR: A mismatch between selected tasks to run and CSV fields exist. Finishing without doing anything.')
            return
        
        # mapping names or indices → actual plugin classes.
        self.plugins = plugin_loader.discover_plugins(analysis_plugins)
        selected_plugins = [self.plugins[i] for i in indx]

        # Rebuild the plugin->state mapping for resumed analysis
        selected_plugins_dict = {cls: Qt.Checked for cls in selected_plugins}

        # Read CSV to find last file processed
        B = pd.read_csv(deployment_indicators_filename, delimiter=';', header=None, dtype=str, skiprows=1)
        if B.shape[0] == 0:
            firstfiletoanalyze = 1
        else:
            last_fname = B.iloc[-1, 0].strip()
            idx_find = [i for i, p in enumerate(allfiles) if p.name == last_fname]
            firstfiletoanalyze = idx_find[0] + 2 if idx_find else 1  # +1 matlab->python index; +1 to start next file

        # load detected_events (h5) into results_events if exists
        if Path(detected_events_filename).exists():
            # Expect dataset 'results_events' as a group of arrays or serialized table
            with h5py.File(detected_events_filename, 'r') as hf:
                # Simple assumption: we saved a CSV-like table as attributes/datasets
                if 'csv' in hf:
                    csv_bytes = hf['csv'][()]
                    # convert bytes to string then to DataFrame
                    csv_str = csv_bytes.decode('utf-8')
                    from io import StringIO
                    results_events = pd.read_csv(StringIO(csv_str), delimiter=';')
                # else leave empty

    # CASE 2: backups exist and partial lock exists (interrupted while adding detectors)
    elif exists(deployment_indicators_filename) and exists(deployment_indicators_filename_backup) \
         and exists(detected_events_filename) and exists(detected_events_filename_backup) \
         and exists(partial_analysis_lock_filename):
        with open(summary_log, 'a', encoding='utf-8') as fidlog:
            fidlog.write('Deployment analysis interrupted while adding detectors. Resuming...\n')
        lock = load_partial_lock(partial_analysis_lock_filename)
        # Get selected analysis names from lock
        detector_names_in_lock=[p["name"] for p in lock]

        # Validate detector_names_in_lock: detector_names_in_lock indexes must not be empty
        if detector_names_in_lock is None:
            raise RuntimeError("Partial lock exists but no selection_values found in lock file.")
        
        # Read header of CSV to get indicators
        with open(deployment_indicators_filename, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        existing_indicators = [s.strip() for s in first_line.split(';') if s.strip()]
        csvheader_current = existing_indicators

        # Build indx list: detectors whose indtags are present in existing_indicators
        indx = []
        for l, indtag in enumerate(analysis_tasks_indtag):
            if set(indtag).issubset(set(existing_indicators)):
                indx.append(l)
        
        # Get the inidices from detector_names_in_lock
        selected_indices = [analysis_tasks.index(name)
                            for name in detector_names_in_lock
                            if name in analysis_tasks]
        
        if set(selected_indices) != set(indx):
            print('ERROR: A mismatch between selected tasks to run and CSV fields exist. Finishing without doing anything.')
            return
        
        # mapping names or indices → actual plugin classes.
        self.plugins = plugin_loader.discover_plugins(analysis_plugins)
        selected_plugins = [self.plugins[i] for i in indx]

        # Rebuild the plugin->state mapping for resumed analysis
        selected_plugins_dict = {cls: Qt.Checked for cls in selected_plugins}

        # Get first file to analyze
        B = pd.read_csv(deployment_indicators_filename, delimiter=';', header=None, dtype=str, skiprows=1)
        if B.shape[0] == 0:
            firstfiletoanalyze = 1
        else:
            last_fname = B.iloc[-1, 0].strip()
            idx_find = [i for i, p in enumerate(allfiles) if p.name == last_fname]
            firstfiletoanalyze = idx_find[0] + 2 if idx_find else 1

        # Load backup CSV into B for reading older indicators
        Bbackup = pd.read_csv(deployment_indicators_filename_backup, delimiter=';', header=None, dtype=str, skiprows=1)
        # store results_events from detected_events_filename if exists
        if Path(detected_events_filename).exists():
            with h5py.File(detected_events_filename, 'r') as hf:
                if 'csv' in hf:
                    csv_str = hf['csv'][()].decode('utf-8')
                    from io import StringIO
                    results_events = pd.read_csv(StringIO(csv_str), delimiter=';')

    # CASE 3: Detections and indicator files EXIST, Lock NOT EXIST
    #         Analysis finished previously.
    elif exists(deployment_indicators_filename) and exists(detected_events_filename):
        # Read header of CSV to get indicators
        with open(deployment_indicators_filename, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        existing_indicators = [s.strip() for s in first_line.split(';') if s.strip()]

        # Build indx list: detectors whose indtags are present in existing_indicators
        indx = []
        for l, indtag in enumerate(analysis_tasks_indtag):
            if set(indtag).issubset(set(existing_indicators)):
                indx.append(l)
        
        tasksDone = np.zeros(len(analysis_tasks), dtype=int)
        tasksDone[indx] = 1

        states = {}
        for i, plugin_cls in enumerate(self.plugin_selector_dialog.plugins):
            states[plugin_cls] = (
                Qt.CheckState.PartiallyChecked if tasksDone[i] else Qt.CheckState.Unchecked
            )
        # Set the initial states that were done in the plugin selector dialog
        self.plugin_selector_dialog.set_plugin_states(states)

        if self.plugin_selector_dialog.exec():
            selected_plugins_dict, indexes = self.plugin_selector_dialog.get_selected_plugins()
        else:
            with open(summary_log, 'a', encoding='utf-8') as fidlog:
                fidlog.write('User cancelled plugin selection. NOTHING TO DO!\n')
            return

        # If selection_values equals tasksDone then nothing to add
        if np.array_equal(indexes, indx):
            with open(summary_log, 'a', encoding='utf-8') as fidlog:
                fidlog.write('No new detection tasks added. NOTHING TO DO!\n')
            return
        else:
            # Build CSV header adding selected detectors' indtags
            for l in indexes:
                csvheader_current.extend(analysis_tasks_indtag[l])


            with open(summary_log, 'a', encoding='utf-8') as fidlog:
                fidlog.write('New detection tasks added...\n')
            # backup existing CSV and events
            if deployment_indicators_filename.exists():
                deployment_indicators_filename.replace(deployment_indicators_filename_backup)
            if Path(detected_events_filename).exists():
                Path(detected_events_filename).replace(detected_events_filename_backup)
            firstfiletoanalyze = 1
            # write new csv header
            with open(deployment_indicators_filename, 'w', encoding='utf-8') as f:
                f.write(';'.join(csvheader_current) + '\n')

            # Load previous CSV analysys (just the CSV backup file) to use from there the inidcators that do not change
            if deployment_indicators_filename_backup.exists():
                #Bbackup = pd.read_csv(deployment_indicators_filename_backup, delimiter=';', header=None, dtype=str, skiprows=1)
                Bbackup = pd.read_csv(deployment_indicators_filename_backup, delimiter=';', dtype=str)
            # Initialize an empty results_events table with expected columns
            # columns translated from MATLAB variable_names_types
            columns = ["filename","start","end","fmin","fmax","f0","BW","ICI","SPL","score","user","type","tag","Tdata","Fdata"]
            results_events = pd.DataFrame(columns=columns)

            # log detector versions and decisions
            partial_plugins = []
            checked_plugins = []
            for plugin_cls, state in selected_plugins_dict.items():

                if state == Qt.PartiallyChecked:
                   partial_plugins.append(plugin_cls)

                elif state == Qt.Checked:
                   checked_plugins.append(plugin_cls)

                # plugin display name
                modname = getattr(plugin_cls, "name", plugin_cls.__name__)
                # version from class first, then module
                version = getattr(plugin_cls, "version", None)

                keep_prev = (state == Qt.PartiallyChecked)
                with open(summary_log, 'a', encoding='utf-8') as fidlog:
                    fidlog.write(f"{modname} ver. ({version}) -> "f"{'KEEP PREVIOUS' if keep_prev else 'UPDATED'}\n")

            selected_plugins=list(selected_plugins_dict.keys())

    else:
        # Never analyzed
        with open(summary_log, 'a', encoding='utf-8') as fidlog:
            fidlog.write('-------------------------------------------------------\n')
            fidlog.write('Analysis RESTARTED (*.h5 and *.csv moved to "_backup")\n')
        firstfiletoanalyze = 1
        tasksDone = np.zeros(len(analysis_tasks), dtype=int)
        
        # Bring to front the dialog analysis selector to ask for tasks to run...
        
        if self.plugin_selector_dialog.exec() != QDialog.DialogCode.Accepted:  # User cancelled the dialog
            with open(summary_log, 'a', encoding='utf-8') as fidlog:
                fidlog.write('User cancelled plugin selection. NOTHING TO DO!\n')
            return

        selected_plugins_dict, indx_selected = self.plugin_selector_dialog.get_selected_plugins()

        if not selected_plugins_dict:
            with open(summary_log, 'a', encoding='utf-8') as fidlog:
                fidlog.write('No plugins selected. NOTHING TO DO!\n')
            return
        
        selected_plugins = list(selected_plugins_dict.keys())

        for l in indx_selected:
            csvheader_current.extend(analysis_tasks_indtag[l])
        # write header to new CSV
        with open(deployment_indicators_filename, 'w', encoding='utf-8') as f:
            f.write(';'.join(csvheader_current) + '\n')

        # initialize empty results_events table with expected columns
        columns = ["filename","start","end","fmin","fmax","f0","BW","ICI","SPL","score","user","type","tag","Tdata","Fdata"]
        results_events = pd.DataFrame(columns=columns)

        # --- ADD INFO IN THE LOG OF THE DETECTORS AND VERSION USED IN THE TASK ---
        for plugin_cls, state in selected_plugins_dict.items():
            modname = plugin_cls.name
            version = getattr(plugin_cls, "version", "unknown")

            keep_prev = state == Qt.CheckState.PartiallyChecked

            with open(summary_log, 'a', encoding='utf-8') as fidlog:
                fidlog.write(f"{modname} ver. ({version}) -> "f"{'KEEP PREVIOUS' if keep_prev else 'UPDATED'}\n")

        indx = indx_selected

    # -------------------------------------------------------------------------------------
    # ---------------------- THIS PART IS COMMON - START ANALYZING  -----------------------
    # -------------------------------------------------------------------------------------

    # Save "selected_plugins" that will be used in the lock file indicating the analysis that was being done in case of interruption.
    save_partial_lock(partial_analysis_lock_filename, selected_plugins)

    # open CSV file for appending indicators
    #fids = open(deployment_indicators_filename, 'a', encoding='utf-8')
    fids = open(deployment_indicators_filename, "a", encoding="utf-8", newline="")
    writer = csv.writer(fids, delimiter=";")

    total_files = len(allfiles)
    # convert matlab 1-based firstfiletoanalyze to python 0-based
    start_idx = max(0, int(firstfiletoanalyze) - 1)
    
    # ------ REDIRECT STDOUT ------
    # Store sys.stdout to put back after finished 
    self.original_stdout = sys.stdout

    # Create a QTextEdit to show ther the progress (status window)
    self.status_window = StatusWindow()
    # Set to modal to prevent interaction with the rest of the application while analyzing
    self.status_window.setModal(True)

    # Redirector
    self.redirector = EmittingStream()
    self.redirector.text_written.connect(self.status_window.append_text)    
    # Show the status window
    self.status_window.show()
    self.status_window.text_box.clear()
    # Redirect stdout
    sys.stdout = self.redirector

    deployment_analysis_aborted = False

    # Function to chnage the flag when user clicks cancel in the status window, and also request interruption of the worker thread
    def request_deployment_abort():
        nonlocal deployment_analysis_aborted
        deployment_analysis_aborted = True

    for filesidx in range(start_idx, total_files):
        wavpath = allfiles[filesidx]
        print(f"Processing {wavpath.name} {filesidx+1} of {total_files} :")
        t0 = time.time()

        # parse filename to obtain date/time from naming convention XXX_N_1_YYYYMMDD_HHMMSS.wav
        parts = re_split("[_.]",wavpath.stem)
        real_date_at_i = None
        try:
            # trying to build datetime from parts 4 and 5 (MATLAB code used parts(4) and parts(5))
            # In python index: parts[3] and parts[4]
            dtstr = parts[3] + parts[4]  # 'yyyymmdd' + 'HHMMSS'
            real_date_at_i = datetime.strptime(dtstr, "%Y%m%d%H%M%S")
        except Exception:
            real_date_at_i = None

        # read audio info and samples
       
        fs_file, x = wavfile.read(str(wavpath))
        # If original MATLAB used 'native' (digital counts), x will be integer dtype; keep as is but convert to float when needed
        xdc = x.astype(np.float64)

        # If previous CSV B exists, get row for this file
        B = None
        if 'Bbackup' in locals():
            B = Bbackup

        plugins_to_run = []
        file_indicators = []

        for cls in selected_plugins:
            plugin_state = selected_plugins_dict.get(cls, Qt.Checked)
            if plugin_state == Qt.PartiallyChecked:
                # keep previous indicator values from backup CSV
                indtags = list(cls.outputs["indicators"].keys())
                try:
                    row = B.loc[B.iloc[:, 0].astype(str).str.strip() == wavpath.name].iloc[0]
                    values = [float(str(row[indtag]).replace(',', '.')) for indtag in indtags]  
                    file_indicators.extend(values)
                except Exception:
                    msg = ("ERROR: Backup CSV is inconsistent. Restart deployment analysis.")
                    with open(summary_log, 'a', encoding='utf-8') as fidlog:
                        fidlog.write(msg + "\n")
                    raise RuntimeError(msg)
                continue
            else:
                plugins_to_run.append(cls)

        if plugins_to_run:
            loop = QEventLoop()
            result = {
                "events": pd.DataFrame(),
                "file_indicators": [],
                "error": None,
                "aborted": False,
            }

            worker = DeploymentFileWorker(
                plugins_to_run,
                xdc,
                fs_file,
                dsp,
                bands,
                wavpath,
                verbose
                )

            def on_finished(events, indicators):
                result["events"] = events
                result["file_indicators"] = indicators
                loop.quit()

            def on_failed(msg):
                result["error"] = msg
                loop.quit()

            worker.finished.connect(on_finished)
            worker.failed.connect(on_failed)

            # Optional, but recommended: add this signal to PluginWorkerDeployment
            # worker.failed.connect(on_failed)

            self.status_window.cancel_requested.connect(request_deployment_abort)
            self.status_window.cancel_requested.connect(worker.requestInterruption)

            worker.start()
            loop.exec()

            try:
                self.status_window.cancel_requested.disconnect(worker.requestInterruption)
            except TypeError:
                pass

            if deployment_analysis_aborted:
                print("Deployment analysis aborted by user.")
                break

            if result["error"] is not None:
                raise RuntimeError(result["error"])

            if result["events"] is not None and not result["events"].empty:
                results_events = pd.concat(
                    [results_events, result["events"]],
                    ignore_index=True
                )

            file_indicators.extend(result["file_indicators"])


        elapsed = time.time() - t0
        print(f"Total time processing file: {elapsed:.2f} [s]")

        # ---------- Save indicators row into CSV ----------
        def format_csv_value(val):
            try:
                return f"{float(val):f}".replace(".", ",")
            except Exception:
                return ""

        row = [
            wavpath.name,
            real_date_at_i.strftime("%d-%b-%Y %H:%M:%S") if real_date_at_i else "",
        ]

        row.extend(format_csv_value(v) for v in file_indicators)

        writer.writerow(row)
        fids.flush()

        # row_fields = []
        # row_fields.append(f"{wavpath.name} ;")
        # row_fields.append(f"{real_date_at_i.strftime('%d-%b-%Y %H:%M:%S') if real_date_at_i else ''} ;")
        # # For the rest of columns, MATLAB built using file_indicators list mapping to csvheader entries from index 3 onward
        # for val in file_indicators[:-1]:
        #     try:
        #         s = f"{float(val):f} ;"
        #     except Exception:
        #         s = " ;"
        #     s = s.replace('.', ',')
        #     row_fields.append(s)
        # # last one without trailing ; but replacing decimal separator
        # if file_indicators:
        #     try:
        #         last = f"{float(file_indicators[-1]):f}"
        #     except Exception:
        #         last = ""
        #     last = last.replace('.', ',')
        #     row_fields.append(last)
        # row_fields.append("\n")
        # # write to file
        # fids.write(''.join(row_fields))
        # fids.flush()

        # Save detected events to HDF5 (here simple strategy: save CSV serialization of results_events)
        with h5py.File(detected_events_filename, 'w') as hf:
            # store CSV text as bytes for portability
            csv_bytes = results_events.to_csv(sep=';', index=False).encode('utf-8')
            hf.create_dataset('csv', data=csv_bytes)

    # End file loop
    fids.close()

    # -----PUT STDOUT BACK ------ 
    sys.stdout=self.original_stdout
    print("Processing completed.")
    # Communicate the status window that processing is finshed so that it changes the button action...
    self.status_window.mark_tasks_finished()

    # FUSIONATE OLD AND NEW DETECTED EVENTS IN A SINGLE *.MAT FILE
    # If backups existed and user asked to merge old + new and not a full restart
    if Path(detected_events_filename_backup).exists() and restart_tasks != 'R':
        # load old detected_events (backup)
        old_df = pd.DataFrame()
        try:
            with h5py.File(detected_events_filename_backup, 'r') as hf:
                if 'csv' in hf:
                    csv_str = hf['csv'][()].decode('utf-8')
                    from io import StringIO
                    old_df = pd.read_csv(StringIO(csv_str), delimiter=';')
        except Exception:
            old_df = pd.DataFrame()
        # remove types requested in selection_values==2 (to_remove)
        if not old_df.empty:
            types_to_replace = []

            for plugin_cls, state in selected_plugins_dict.items():
                if state == Qt.Checked:
                    for ev in plugin_cls.outputs["events"]:
                        types_to_replace.append(ev["type"])

            if types_to_replace and "type" in old_df.columns:
                old_df = old_df[~old_df["type"].isin(types_to_replace)]

            merged = pd.concat([old_df, results_events], ignore_index=True)
        else:
            merged = results_events.copy()
        # save merged into detected_events_filename
        with h5py.File(detected_events_filename, "w") as hf:
            hf.create_dataset(
                "csv",
                data=merged.to_csv(sep=";", index=False).encode("utf-8")
            )

    # finish log and cleanup
    with open(summary_log, 'a', encoding='utf-8') as fidlog:
        fidlog.write(f"The analysis ended on {datetime.now():%Y_%m_%d %H:%M:%S}\n")

    # delete partial lock ONLY if analysis was not aborted, otherwise keep it for resuming later
    if not deployment_analysis_aborted:
        if partial_analysis_lock_filename.exists():
            partial_analysis_lock_filename.unlink()
    else:
            print("Partial lock kept so analysis can be resumed.")

    print("Analysis finished.")

# End function
