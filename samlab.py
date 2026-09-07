"""
SAMLab - Submarine Acoustic Monitoring Laboratory
Copyright (C) 2026 Universitat Politècnica de València

This file is part of SAMLab.

Licensed under the GNU General Public License v3.0 (GPLv3) or later.
See the LICENSE file for details.
"""

import os
from shutil import copy2
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QScrollBar, QComboBox, QPushButton, QGroupBox, QRadioButton,
    QCheckBox, QDockWidget, QSizePolicy, QSpacerItem,QMessageBox, QFileDialog, QDialog
)
from PySide6.QtGui import QAction, QIntValidator
from PySide6.QtCore import Qt
from PySide6.QtCore import QSettings # To handle the settings such as AnnotationPathName
from PySide6.QtCore import QTimer  # To execute code every x seconds
from PySide6.QtGui import QIcon, QPixmap
import sys

from pathlib import Path
from os.path import isfile  # To check if a file exist 
from os.path import split as os_path_split # split directory and file parts
from os.path import join as os_path_join # join directory and file parts
from getpass import getuser as getpass_getuser # To get the user in many OSs

import pandas as pd # Import Pandas for CSV reading
from datetime import datetime # date manipulation functions
from re import split as re_split # split with multiple delimiters
from pandas import unique as pd_unique # np unique does not preserv the order and sort!

import numpy as np
import time # To meassure time
from math import floor
# from threading import Timer # To execute code every x seconds
from PIL import Image  # To read images (logos, etc)
from scipy.io import loadmat, wavfile
from scipy import signal
from mpl_toolkits.axes_grid1 import make_axes_locatable
from textwrap import dedent #Unindent for multiline text
import matplotlib as mpl
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle  # To draw rectangle in miniature view of time signal
from matplotlib.lines import Line2D # To draw a line for marker when paying
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from matplotlib.collections import LineCollection
import sounddevice as sd # To play audio files (better than pyaudio and supports Apple Silicion M1/M2)

# Now let´s import Samlab modules needed...

# Local models
from modules.models import DSP, Bands, EVENT_FIELDS_TYPES, default_dsp, default_bands
from modules.select_and_save_dialog import SelectFragmentAndSaveDialog
from modules.ui_helpers import  ask_open_file, is_number, clamp_posx, StatusWindow, EmittingStream, ReliableCsvDialog
from modules.ui_double_range_slider import DoubleVerticalRangeSlider
from modules.plugin_selector_dialog import  PluginSelector
from modules.drawing_modules import draw_tfr, big_data_graph, draw_auxiliary_nav_graph
from modules.deployment_analysis_ui import dlg_analyze_deplyment_warning, dlg_analyze_deployment_settings,analyze_samaruc_deployment
from modules.annotation_ui import dlg_input_ma, ginput_rectangle, ginput_point
# Imports for running detectors, redirect the progress and move the heavy load outside the main GUI to avoid freezing the app.
from modules.workers import PluginWorker
# Import the dialog to create/edit deployment_info file.
from modules.deployment_creator_dialog import DeploymentInfoDialog, _mat_struct_to_dict, _scalar, _float_list
from modules.ui_constants import LABEL_FONT_SIZE
# Import resources to load program icons and images 
from modules.resources import resource

# Imports to handle analysis plugins
import analysis_plugins
from analysis_plugins import plugin_loader

# Import SAMLab version and Release Date from version.py
from version import VERSION, RELEASE_DATE

# ----- FREQ ZOOM Group for Center Panel 
def create_freq_groupbox(self):
    group = QGroupBox("Frequency zoom")
    group.setCheckable(True)
    group.setChecked(False)
    group.setFocusPolicy(Qt.FocusPolicy.NoFocus) # Avoid “Enter unchecks the groupbox”
    group.toggled.connect(self.freq_zoom_Callback)

    top_layout = QHBoxLayout()

    top_layout.addWidget(QLabel("["))
    self.zoom_freq_low_limit_textbox = QLineEdit()
    self.zoom_freq_low_limit_textbox.setFixedWidth(60)
    self.zoom_freq_low_limit_textbox.setText("0")  # default
    self.zoom_freq_low_limit=0
    self.zoom_freq_low_limit_textbox.editingFinished.connect(self.edit_freqzoomdown_Callback)
    top_layout.addWidget(self.zoom_freq_low_limit_textbox)
    top_layout.addWidget(QLabel("-"))
    self.zoom_freq_up_limit_textbox = QLineEdit()
    self.zoom_freq_up_limit_textbox.setFixedWidth(60)
    self.zoom_freq_up_limit_textbox.setText("250")  # default
    self.zoom_freq_up_limit=250
    self.zoom_freq_up_limit_textbox.editingFinished.connect(self.edit_freqzoomup_Callback)
    top_layout.addWidget(self.zoom_freq_up_limit_textbox)
    top_layout.addWidget(QLabel("] Hz"))

    self.hann_w_enabled=QRadioButton("Hann Window")
    self.hann_w_enabled.toggled.connect(self.window_type_Callback)
    top_layout.addWidget(self.hann_w_enabled)

    bottom_layout = QHBoxLayout()
    bottom_layout.addWidget(QLabel("FFT points #:"))
    values_str=list(2**np.arange(5,17))  # Create a list for the combobox of posible FFT# points
    values_str[0]=" ".join([str(values_str[0])," - most wideband"])
    values_str[-1]=" ".join([str(values_str[-1])," - most narrowband"])
    values_str[7]=" ".join([str(values_str[7])," - default"])
    self.FFTpoints = QComboBox()
    self.FFTpoints.addItems([str(v) for v in values_str])
    self.FFTpoints.setCurrentIndex(7)
    self.FFTpoints.currentIndexChanged.connect(self.FFTpointsChanged)
    bottom_layout.addWidget(self.FFTpoints)
    self.frec_res_wzoom_textlabel=QLabel("46.9 Hz")
    bottom_layout.addWidget(self.frec_res_wzoom_textlabel)
    
    # Main layout (stacks rows vertically)
    main_layout = QVBoxLayout()
    main_layout.addLayout(top_layout)
    main_layout.addLayout(bottom_layout)
    group.setLayout(main_layout)
    return group

# ----- DYNAMIC RANGE Group for Center Panel    
def create_DynRangeGroupBox(self):
    group = QGroupBox("Manual Dynamic Range")
    group.setCheckable(True)
    group.setChecked(False)
    group.toggled.connect(self.manual_CL_chkbox_Callback)  # Avoid “Enter unchecks the groupbox”
    
    layout = QHBoxLayout()
    layout.addWidget(QLabel("["))
    self.minDR_textbox = QLineEdit()
    self.minDR_textbox.setFixedWidth(60)
    self.minDR_textbox.setText("0")  # default
    self.minDR_textbox.editingFinished.connect(self.edit_mindr_Callback)
    layout.addWidget(self.minDR_textbox)
    layout.addWidget(QLabel("-"))
    self.maxDR_textbox = QLineEdit()
    self.maxDR_textbox.setFixedWidth(60)
    self.maxDR_textbox.setText("250")  # default
    self.maxDR_textbox.editingFinished.connect(self.edit_maxdr_Callback)
    layout.addWidget(self.maxDR_textbox)
    layout.addWidget(QLabel("] dB"))
            
    group.setLayout(layout)
    return group

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"SAMLab v{VERSION} ({RELEASE_DATE})")
        self.setGeometry(100, 100, 1600, 900)
        
        # Cross-platform start location
        if sys.platform == "darwin":
            self.start_dir  = "/Volumes"
        elif sys.platform == "win32":
            self.start_dir  = ""
        else:
            self.start_dir  =     "/media"

         # Keep only the last 10 opened files
        self.recent_files = []
        self.max_recent_files = 10

        # Set global font sizes for matplotlib to ensure consistency across all plots
        mpl.rcParams["axes.titlesize"] = 10
        mpl.rcParams["axes.labelsize"] = 10

        # Use QSettings (Native Qt Solution) to store "annotation_path" settings persistently across sessions
        self.settings = QSettings()
        # ---- Load saved path on startup ----
        self.annotation_path = self.settings.value(
            "annotation/path",
            "",      # default value if not found
            type=str
        )
        # --- Load recent files on startup ---
        files = self.settings.value("recentFiles", [])
        # Sometimes QSettings may return None or a single string depending on context
        if files is None:
            files = []
        elif isinstance(files, str):
            files = [files]
        # Keep only existing files and max self.max_recent_files items
        self.recent_files = [f for f in files if os.path.exists(f)][:self.max_recent_files]

        # Create a color palette for whistle contours
        self.color_palette_WC = np.array([[0, 0, 1],  # blue
                                        [0, 1, 0],  # green
                                        [1, 1, 1],  # white
                                        [0, 0, 0]   # black
                                        ])
        
        # === Central (Main) widget ===
        self.center_panel = self.create_center_panel()
        self.setCentralWidget(self.center_panel)

        # === Left Dockable Panel ===
        left_dock = QDockWidget("Deployment Navigation Graph", self)
        left_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        left_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable|
            QDockWidget.DockWidgetFeature.DockWidgetFloatable) # Allow floating around but prevent closing
        left_widget = self.create_left_panel()
        left_dock.setWidget(left_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, left_dock)

        # === Right Control Panel (fixed for now) ===
        right_dock = QDockWidget("Event Navigation Control", self)
        right_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        right_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)  # Fixed panel
        right_widget = self.create_right_panel()
        right_dock.setWidget(right_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, right_dock)

        # Create the menu bar
        menu_bar = self.menuBar()

        # A QAction "About" or "Exit" disappear in OSX because the operating system reserves
        #  this  keywords like for its own menu bar. Workaround "About..." instead "About"
        # ---------------- ABOUT Menu ----------------
        samlab_menu = menu_bar.addMenu("SAMLAB")

        about_action = QAction(" &About...", self)
        exit_action = QAction(" &Quit SAMLAB", self)

        about_action.triggered.connect(self.aboutSamLABMenu)
        exit_action.triggered.connect(self.close)

        samlab_menu.addAction(about_action)
        samlab_menu.addSeparator()
        samlab_menu.addAction(exit_action)

        # ---------------- File Menu ----------------
        file_menu = menu_bar.addMenu("File")

        open_action = QAction("Open File...", self)
        open_action.setShortcut("Ctrl+O")
        open_DeploymentAnalysisFile_action = QAction("Open Deployment Analysis File...", self)
        self.open_previous_action_menu = QAction("Open Previous File in Deployment Analysis", self)
        self.open_previous_action_menu.setEnabled(False)  # disable initially
        self.open_next_action_menu = QAction("Open Next File in Deployment Analysis", self)
        self.open_next_action_menu.setEnabled(False)  # disable initially
        save_wav_copy_action=QAction("Save WAV Copy...", self)
        save_fragment_action=QAction("Select Fragment and Save...", self)
        save_wav_copy_action.setEnabled(False)  # disable initially
        save_fragment_action.setEnabled(False)  # disable initially

        open_action.triggered.connect(self.open_file)
        self.open_previous_action_menu.triggered.connect(self.openPreviousFile)
        self.open_next_action_menu.triggered.connect(self.openNextFile)
        open_DeploymentAnalysisFile_action.triggered.connect(self.openDeploymentAnalysisFile)
        save_wav_copy_action.triggered.connect(self.save_wav_copy_Callback)
        save_fragment_action.triggered.connect(self.select_and_save_fragment_Callback)

        file_menu.addAction(open_action)
        self.recent_menu = file_menu.addMenu("Recent Files") # Submenu for recent files, will be populated dynamically
        file_menu.addSeparator()
        file_menu.addAction(open_DeploymentAnalysisFile_action)
        file_menu.addAction(self.open_next_action_menu)
        file_menu.addAction(self.open_previous_action_menu)
        file_menu.addSeparator()
        file_menu.addAction(save_wav_copy_action)
        file_menu.addAction(save_fragment_action)

        self.save_wav_copy_action=save_wav_copy_action
        self.save_fragment_action=save_fragment_action

        # ---------------- Play Menu ----------------
        play_menu = menu_bar.addMenu("Play")

        play_action = QAction("Play Segment", self)
        play_action.setShortcut("Ctrl+P")
        play10_action = QAction("Play Segment x10", self)

        play_action.triggered.connect(self.playAudioMenuCallback)
        play10_action.triggered.connect(self.playx10AudioMenuCallback)

        play_menu.addAction(play_action)
        play_menu.addAction(play10_action)

        play_menu.setEnabled(False)

        self.play_menu=play_menu
       
        # ---------------- View Menu ----------------
        view_menu = menu_bar.addMenu("View")

        self.filter_automatic_det_action = QAction("Filter automatic detections...", self)
        self.filter_automatic_det_action.setEnabled(False)
        self.filter_automatic_det_action.triggered.connect(self.filter_automatic_det_Callback)
        view_menu.addAction(self.filter_automatic_det_action)

        self.filer_manual_anotations_action = QAction("Manual Annotations", self, checkable=True)
        self.filer_manual_anotations_action.setChecked(True)
        self.filer_manual_anotations_action.setEnabled(False)
        self.filer_manual_anotations_action.triggered.connect(self.filter_manual_annotations_Callback)
        view_menu.addAction(self.filer_manual_anotations_action)

        # ---------------- Analyze Menu ----------------
        analyze_menu = menu_bar.addMenu("Analyze")

        analyze_action = QAction("Analyze current File...", self)
        analyze_action.setShortcut("Ctrl+R")
        analyze_deployment_action = QAction("Analyze Deployment...", self)
        create_deployment_info_action = QAction("Create/Edit ""deployment_info"" file...", self)

        analyze_action.triggered.connect(self.lookforevents_Callback)
        analyze_action.setEnabled(False)
        self.analyze_action=analyze_action
        analyze_deployment_action.triggered.connect(self.analyze_deployment_mb_Callback)
        create_deployment_info_action.triggered.connect(self.create_deployment_info_Callback)

        analyze_menu.addAction(analyze_action)
        analyze_menu.addAction(analyze_deployment_action)
        analyze_menu.addSeparator()
        analyze_menu.addAction(create_deployment_info_action)
        

        # ---------------- Manual Annotation Menu ----------------
        manual_annottation_menu = menu_bar.addMenu("Manual Annotation")
        self.add_manual_annottation_action = QAction("Add...", self)
        self.add_manual_annottation_action.setShortcut("Ctrl+A")
        self.add_manual_annottation_action.setEnabled(False)
        manual_annottation_menu.addAction(self.add_manual_annottation_action)
        self.delete_manual_annottation_action = QAction("Delete...", self)
        self.delete_manual_annottation_action.setShortcut("Ctrl+D")
        self.delete_manual_annottation_action.setEnabled(False)
        manual_annottation_menu.addAction(self.delete_manual_annottation_action)
        self.edit_manual_annottation_action = QAction("Edit...", self)
        self.edit_manual_annottation_action.setShortcut("Ctrl+E")
        self.edit_manual_annottation_action.setEnabled(False)
        manual_annottation_menu.addAction(self.edit_manual_annottation_action)
        self.save_manual_annottation_action = QAction("Save Manual Annotations", self)
        self.save_manual_annottation_action.setEnabled(False)
        manual_annottation_menu.addAction(self.save_manual_annottation_action)
        manual_annottation_menu.addSeparator()
        get_user_action = QAction("Get User annotation information...", self)
        manual_annottation_menu.addAction(get_user_action)
        set_annotation_directory_action = QAction("Set Annotation Directory...", self)
        manual_annottation_menu.addAction(set_annotation_directory_action)

        self.add_manual_annottation_action.triggered.connect(self.add_manual_annotation_Callback)
        self.delete_manual_annottation_action.triggered.connect(self.delete_manual_annotation_Callback)
        self.edit_manual_annottation_action.triggered.connect(self.edit_manual_annotation_Callback)
        self.save_manual_annottation_action.triggered.connect(self.save_manual_annotations_Callback)
        get_user_action.triggered.connect(self.get_user_Callback)
        set_annotation_directory_action.triggered.connect(self.set_annotation_directory_Callback)

        # ---------------- Experimental Menu ----------------
        experimental_menu = menu_bar.addMenu("Experimental")
        self.meassure_action = QAction("Meassure...", self)
        self.meassure_action.setEnabled(False)
        self.meassure_action.triggered.connect(self.measure_mb_Callback)
        experimental_menu.addAction(self.meassure_action)

        # Update the recent files menu based on loaded recent files
        self.update_recent_files_menu()

        # Add a status bar
        self.statusBar().showMessage("Ready")

        # === Initialize general variables ===
        self.posx=1  # Position in file in samples
        self.tanalisis=10
        self.fs=44100
        self.ventana=round(self.tanalisis*self.fs)
        self.siz=0 # Recording size in samples

        #  Initialize a pandas object to store detected events in the file
        self.event_fields_types = EVENT_FIELDS_TYPES
        
        self.events = pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in self.event_fields_types})
        self.m_annotated_events: list[dict] = []
        self.eventstobeshown = [] # Empty means show none

        # Create plugin selector dialog to be reused when analyzing files
        self.plugins = plugin_loader.discover_plugins(analysis_plugins)
        self.plugin_selector_dialog = PluginSelector(self.plugins,parent=self)   # or PluginSelector([PluginA, PluginB])

    # ----- LEFT PANEL (Dockable) -----
    def create_left_panel(self):
        frame = QWidget()
        layout = QVBoxLayout()

        # Top controls
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Visualization by:"),0)
        self.fieldselect=QComboBox()
        self.fieldselect.setEnabled(False)
        # Activate the binding of fieldselect
        self.fieldselect.currentIndexChanged.connect(self.fieldselect_Callback)
        control_layout.addWidget(self.fieldselect,3)
        control_layout.addWidget(QPushButton("Combine"),0)
        layout.addLayout(control_layout)

        # Deployment inspector navigator graph 
        self.figDeployInspect = Figure(figsize=(7, 10), dpi=80)
        self.figDeployInspect.patch.set_alpha(0.0)  # transparent for dark mode
        self.deploy_canvas = FigureCanvas(self.figDeployInspect)
        self.deploy_canvas.setStyleSheet("background: transparent;") # transparent for dark mode
        
        # Dual knob range slider
        self.double_slider = DoubleVerticalRangeSlider(
            minimum=0,
            maximum=100,
            low=0,
            high=100,
        )
        # Connect the rangeChanged signal of the double slider to the update_image method
        self.double_slider.rangeChanged.connect(self.update_deployment_nav_image)
        
        # Create a click ID to handle clicking in the deployment inspector and initialize to None until a deployment is loaded
        self.deploy_click_cid = None

        image_nav_layout = QHBoxLayout()
        # Add the canvas and dual slider to the Qt layout
        image_nav_layout.addWidget(self.deploy_canvas)
        image_nav_layout.addWidget(self.double_slider)
        layout.addLayout(image_nav_layout)

        # Create an axis for the image
        self.ax4 = self.figDeployInspect.add_subplot(111)

        # set the initial spectrogram image displayed
        bitmap = Image.open(resource("samaruclogo.png"))
        self.deployment_nav_bitmap = self.ax4.imshow(bitmap,cmap='hot_r',aspect='auto',origin='lower',interpolation='none')
        self.ax4.yaxis.set_inverted(True)     # To reverse YDir
        self.ax4.set_xlabel('+h (CET)')

        # Put in the ylabel the actual date to automatically adjust the tight_layout
        date_str = datetime.now().strftime('%d-%b-%Y')
        self.ax4.yaxis.set_major_formatter(FuncFormatter(lambda *_: date_str))
        self.ax4.tick_params(axis='both', which='major', length=0,labelsize=LABEL_FONT_SIZE)


        divider = make_axes_locatable(self.ax4)
        cax = divider.append_axes("right", size="5%", pad=0.08) # To make colorbar narrower
        self.HC=self.figDeployInspect.colorbar(self.deployment_nav_bitmap, ax=self.ax4,cax=cax) #Color bar
        self.HC.ax.tick_params(labelsize=LABEL_FONT_SIZE) # Adjust colorbar tick label size
        #self.canvas = FigureCanvasTkAgg(self.figDeployInspect, master=self.deploy_ins_frame)
        #self.canvas.get_tk_widget().grid(row=0,  column=0, columnspan=3, padx=0,  pady=0)

        self.figDeployInspect.tight_layout()

        # Rectangle Marking in the click
        self.ll = self.ax4.add_patch(Rectangle((0, 0), 1, 1,
                            linewidth=1, edgecolor='r',
                            facecolor='none',
                            visible=False))
        # Patch remarking the text
        self.lp=self.ax4.add_patch( Rectangle((0, 0),
                    400, 100,
                        #fc='none',
                        edgecolor ='black',
                        linewidth = 1,
                        #linestyle="dotted",
                        facecolor="white",
                        alpha=0.7,
                        zorder=20,
                        visible=False) )
        # The text itself
        self.lt=self.ax4.text(0, 0, 'File info',
                    horizontalalignment='left',
                    verticalalignment='top',
                    zorder=20,
                    visible=False,
                    fontsize=8)
        
        
        self.deploy_canvas.draw_idle()

        frame.setLayout(layout)
        return frame

    # ----- CENTER PANEL -----
    def create_center_panel(self):
        # Create a frame that will hold all widgets for the center panel
        frame = QWidget()
        layout = QVBoxLayout(frame)

        # --- Matplotlib figure and canvas ---
        self.figSpect = Figure(figsize=(8, 7), dpi=80) # Before it was 9,8
        self.figSpect.patch.set_alpha(0.0)  # transparent for dark mode

        self.spect_canvas = FigureCanvas(self.figSpect)
        self.spect_canvas.setStyleSheet("background: transparent;")
        layout.addWidget(self.spect_canvas)

        # Create the main axis
        self.ax0 = self.figSpect.add_subplot(111)
        self.ax0.clear()
        bitmap = (np.zeros((51, 3700))).astype(np.uint8)
        self.TFR_bitmap = self.ax0.imshow(bitmap, cmap='jet', aspect='auto', origin='lower', interpolation='none')
        self.ax0.set_title('Spectrum Level [dB re count^2/Hz]', fontsize=LABEL_FONT_SIZE)
        self.ax0.set_ylabel('Frequency [Hz]',fontsize=LABEL_FONT_SIZE)
        self.ax0.tick_params(axis='both', which='major', labelsize=LABEL_FONT_SIZE)


        # Divider axes for time view and colorbar
        divider = make_axes_locatable(self.ax0)
        self.ax2 = divider.append_axes("bottom", size="15%", pad=0.28)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        cbar =self.figSpect.colorbar(self.TFR_bitmap, ax=self.ax0, cax=cax)
        cbar.ax.tick_params(labelsize=LABEL_FONT_SIZE) # Adjust colorbar tick label size

        # Time view
        mpl.rcParams['path.simplify_threshold'] = 1.0
        self.time_view_lines, = self.ax2.plot([], [], linewidth=0.5)
        self.ax2.yaxis.tick_right()
        self.ax2.set_xlim(0, 10)
        self.ax2.set_ylim(-32768, 32767)
        self.ax2.set_ylabel('Amplitude\n[counts]', fontsize=LABEL_FONT_SIZE)
        self.ax2.set_xlabel('Time (sec.)', fontsize=LABEL_FONT_SIZE)
        self.ax2.tick_params(axis='both', which='major', labelsize=LABEL_FONT_SIZE)

        self.figSpect.subplots_adjust(left=0.02, right=0.88, top=0.98, bottom=0.02)
        self.figSpect.tight_layout(pad=0.1)
        self.spect_canvas.draw_idle()

        # --- Navigation controls ---
        nav_layout = QGridLayout()
        nav_layout.addWidget(QLabel("Time span (sec.):"), 0, 0)
        nav_layout.addWidget(QLabel("Go to sec."), 0, 4)

        self.timeLabel = QLineEdit("10")
        self.timeLabel.setFixedWidth(60)
        self.timeLabel.setEnabled(False)
        self.timeLabel.editingFinished.connect(self.timeSpanChanged)
        nav_layout.addWidget(self.timeLabel, 1, 0)

        self.slider = QScrollBar(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(50)
        #self.slider.setFixedWidth(600)
        self.slider.setFixedWidth(450)
        self.slider.setEnabled(False)
        nav_layout.addWidget(self.slider, 1, 1, 1, 3, Qt.AlignmentFlag.AlignCenter)
        # Activate the binding of the slider after the file is opened
        self.slider.sliderMoved.connect(self.sliderChanged)

        self.gotoTimeLabel = QLineEdit("0")
        self.gotoTimeLabel.setFixedWidth(60)
        self.gotoTimeLabel.setEnabled(False)
        self.gotoTimeLabel.editingFinished.connect(self.gotoTimeLabelChanged)
        nav_layout.addWidget(self.gotoTimeLabel, 1, 4)


        self.previousBut = QPushButton("Previous Time Frame")
        self.previousBut.setEnabled(False)
        self.advanceBut = QPushButton("Auto Advance")
        self.advanceBut.setEnabled(False)
        self.nextBut = QPushButton("Next Time Frame")
        self.nextBut.setEnabled(False)

        self.previousBut.clicked.connect(self.previousTimeFrame)
        self.advanceBut.clicked.connect(self.advanceButTimeFrame)
        self.nextBut.clicked.connect(self.nextTimeFrame)

        nav_layout.addWidget(self.previousBut, 2, 1)
        nav_layout.addWidget(self.advanceBut, 2, 2)
        nav_layout.addWidget(self.nextBut, 2, 3)

        layout.addLayout(nav_layout)

        # --- Frequency zoom controls ---
        self.lf_zoom_enabled = create_freq_groupbox(self)
        self.lf_zoom_enabled.setEnabled(False)
        self.manual_CL_chkbox_enabled = create_DynRangeGroupBox(self)
        self.manual_CL_chkbox_enabled.setEnabled(False)

        freq_layout = QHBoxLayout()
        freq_layout.addWidget(self.lf_zoom_enabled)
        freq_layout.addWidget(self.manual_CL_chkbox_enabled)
        layout.addLayout(freq_layout)

        # Set the layout for the frame
        frame.setLayout(layout)

        # Return the frame to be added to the main layout of your central widget
        return frame


    # ----- RIGHT PANEL -----
    def create_right_panel(self):
        frame = QWidget()
        layout = QVBoxLayout()

        # Event graph section
        group = QGroupBox("Event navigation graph")
        group_layout = QVBoxLayout()

        rb_layout= QHBoxLayout()
        self.select_SPL_graph = QRadioButton("SPL")
        self.select_ICI_graph = QRadioButton("IEI (Inter-Event Interval)") 
        # Inter-Pulse Interval (IPI), Inter-Click Interval (ICI) or Inter-Note Interval (INI)
        self.select_f0_graph = QRadioButton("f0")
        self.select_SPL_graph.setEnabled(False)
        self.select_ICI_graph.setEnabled(False)
        self.select_f0_graph.setEnabled(False)
        self.select_SPL_graph.setChecked(True)

        rb_layout.addWidget(self.select_SPL_graph)
        rb_layout.addWidget(self.select_ICI_graph)
        rb_layout.addWidget(self.select_f0_graph)

        # Add the row of radio buttons to the group layout
        group_layout.addLayout(rb_layout)

        # for opt in ["SPL", "ICI", "f0"]:
        #     event_layout.addWidget(QRadioButton(opt))

        # Create a matplotlib Figure and Axes for the Events plot
        #self.figMiniature = Figure(figsize=(4.1,2), dpi=80)
        self.figMiniature = Figure(figsize=(4.1,4), dpi=80)
        self.figMiniature.patch.set_alpha(0.0)  # transparent for dark mode
        # Create an axis for the image
        self.ax3 = self.figMiniature.add_subplot(211)  # Subplot for the signal miniature
        self.figMiniature.add_subplot(212).axis('off')  # Blank subplot to leave space for the legend

        # Clear previous plots
        self.ax3.clear()
        self.ax3.set_xlabel('Time (sec.)',fontsize=LABEL_FONT_SIZE)
        self.ax3.set_ylabel('SPL (dB re 1 uPa)',fontsize=LABEL_FONT_SIZE)
        self.ax3.tick_params(axis='both', which='major', labelsize=LABEL_FONT_SIZE)

        # Add padding to avoid ylabel out of the canvas
        #self.figMiniature.subplots_adjust(left=0.15, right=0.97, top=0.98, bottom=.02)
        self.figMiniature.tight_layout(pad=1.3)

        self.miniature_canvas = FigureCanvas(self.figMiniature)
        self.miniature_canvas.setStyleSheet("background: transparent;")

        # Connect click event
        self.figMiniature.canvas.mpl_connect("button_press_event", self.miniatureOnClick)

        # Add the canvas to the Qt layout
        group_layout.addWidget(self.miniature_canvas)
        
        group.setLayout(group_layout)
        group.setMinimumSize(330,20) # Minimum size whe resizing panel
        layout.addWidget(group)

        # Analyze controls
        self.AnalyzeBut = QPushButton("Analyze Current File")
        self.AnalyzeBut.setEnabled(False)
        self.AnalyzeBut.clicked.connect(self.lookforevents_Callback)
        layout.addWidget(self.AnalyzeBut)

        layout.addWidget(QLabel("Detected events:"))
        self.eventsortcheckbox=QCheckBox("Sort by type")
        self.eventsortcheckbox.stateChanged.connect(self.eventsortcheckbox_Callback)
        layout.addWidget(self.eventsortcheckbox)
        self.Detected_ev_combo = QComboBox()
        self.Detected_ev_combo.setEnabled(False)
        #self.Detected_ev_combo.currentIndexChanged.connect(self.eventslistselect_Callback)
        # The previous lines trigers the function when I do clear in the program!
        self.Detected_ev_combo.activated.connect(self.eventslistselect_Callback)
        layout.addWidget(self.Detected_ev_combo)
        layout.addWidget(QLabel("Manual annotated events:"))
        self.m_annotation_combo=QComboBox()
        self.m_annotation_combo.setEnabled(False)
        self.m_annotation_combo.activated.connect(self.m_annotation_list_Callback)
        layout.addWidget(self.m_annotation_combo)

        # Spacer and logos
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        samLAB_label = QLabel()
        SAMLab_icon=QPixmap(resource("SAMLab_icon.png")).scaled(203, 118)
        samLAB_label.setPixmap(SAMLab_icon)
        #samLAB_label.setScaledContents(True) 
        samLAB_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(samLAB_label)
        layout.addSpacing(40) # Add blank space
        UPVlogo_label = QLabel("")
        pixmapUPVLogo=QPixmap(resource("upv_logo.png"))
        UPVlogo_label.setPixmap(pixmapUPVLogo)
        #UPVlogo_label.setScaledContents(True) 
        UPVlogo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(UPVlogo_label)
        layout.addSpacing(10) # Add blank space

        frame.setLayout(layout)

        # Put all the connects her after the Panel has been created
        self.select_SPL_graph.toggled.connect(self.select_SPL_ICI_f0_graph_Callback)
        self.select_ICI_graph.toggled.connect(self.select_SPL_ICI_f0_graph_Callback)
        self.select_f0_graph.toggled.connect(self.select_SPL_ICI_f0_graph_Callback)

        return frame

    def aboutSamLABMenu(self):
        # aboutString =" ".join(["Submarine Acoustic Monitoring LABoratory (SAMLAB v" + VERSION + ")",
        #                     "\n\n",
        #                     "Copyright (c) 2026 Universitat Politècnica de València and contributors",
        #                     "\n\n",
        #                     "Author:\n",
        #                     "Ramon Miralles (UPV-iTEAM)","\n\n",
        #                     "Licensed under the GNU General Public License v3.0 (GPLv3)\n\n",
        #                     "Alternative commercial licensing is available"])
        aboutString = """

                    Copyright (c) 2026 Universitat Politècnica de València and contributors<br><br>

                    <b>Author:</b><br>
                    Ramon Miralles (UPV-iTEAM)<br><br>

                    Licensed under the GNU General Public License v3.0 (GPLv3)<br><br>

                    <i>Alternative commercial licensing is available</i>
                    """
        msg = QMessageBox()
        msg.setWindowTitle("About SAMLAB")
        msg.setTextFormat(Qt.RichText)
        msg.setText("Submarine Acoustic Monitoring LABoratory (SAMLAB v{version})".format(version=VERSION))
        msg.setInformativeText(aboutString)
        msg.setIcon(QMessageBox.Icon.Information)  # note: enum moved under QMessageBox.Icon
        msg.setWindowIcon(QIcon(resource("SAMLab_icon.png")))  # changes window title icon
        msg.setIconPixmap(QPixmap(resource("SAMLab_program_icon.png")).scaled(64, 64))  # custom message icon
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def closeEvent(self,event):
        msg = QMessageBox()
        msg.setWindowTitle("Exit SAMLab")
        msg.setText("Are you sure you want to exit SAMLab?")
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setWindowIcon(QIcon(resource("SAMLab_icon.png")))
        msg.setIconPixmap(QPixmap(resource("SAMLab_program_icon.png")).scaled(64, 64))  # custom message icon

        yes_button = msg.addButton(
            "Yes",
            QMessageBox.ButtonRole.YesRole
        )
        no_button = msg.addButton(
            "No",
            QMessageBox.ButtonRole.NoRole
        )

        msg.setDefaultButton(no_button)

        msg.exec()

        if msg.clickedButton() == yes_button:
            event.accept()
        else:
            event.ignore()

    def previousTimeFrame(self,event):
        self.posx=np.max([0,self.posx-self.ventana])
        self.y=self.x[self.posx:self.posx+self.ventana]
        draw_tfr(self,self.y,self.fs,self.posx)
    
        # Update rectangle marker in miniature view
        [miny, maxy]=self.ax3.get_ylim()
        self.rect_miniature_view.set_xy((self.posx/self.fs,miny))
        self.figMiniature.canvas.draw_idle()

        # Update Slider position
        self.slider.setValue(self.posx)

        # Update gotoseg field 
        selectedx=self.posx/self.fs  # slider pos in seconds
        self.gotoTimeLabel.setText('{:.3f}'.format(selectedx))  
    
    def nextTimeFrame(self,event):
        self.posx=np.min([self.x.size-self.ventana, self.posx+self.ventana])
        self.y=self.x[self.posx:self.posx+self.ventana]
        draw_tfr(self,self.y,self.fs,self.posx)
    
        # Update rectangle marker in miniature view
        [miny, maxy]=self.ax3.get_ylim()
        self.rect_miniature_view.set_xy((self.posx/self.fs,miny))
        self.figMiniature.canvas.draw_idle()

        # Update Slider position
        self.slider.setValue(self.posx)

        # Update gotoseg field 
        selectedx=self.posx/self.fs  # slider pos in seconds
        self.gotoTimeLabel.setText('{:.3f}'.format(selectedx))      
 
    def sliderChanged(self,event):

        self.posx=self.slider.value()   # get slider value (samples)
        
        self.y=self.x[self.posx:self.posx+self.ventana]
        draw_tfr(self,self.y,self.fs,self.posx)

        # Update gotoseg field 
        selectedx=self.posx/self.fs  # slider pos in seconds   
        self.gotoTimeLabel.setText('{:.3f}'.format(selectedx))

        # Update rectangle marker in miniature view
        [miny, maxy]=self.ax3.get_ylim()
        self.rect_miniature_view.set_xy((selectedx,miny))
        self.figMiniature.canvas.draw_idle()

    def spect_advance_function(self):
        
        next_posx = self.posx + self.ventana
        clamped_posx = clamp_posx(self,next_posx)

        # Stop when we can no longer advance to a new full window
        if clamped_posx == self.posx:
            self.advance_timer.stop()
            self.is_running = False
            self.advanceBut.setText("Auto Advance")
            return
        
        self.posx = clamped_posx
        self.y=self.x[self.posx:self.posx+self.ventana]
        draw_tfr(self,self.y,self.fs,self.posx)
    
        # Update rectangle marker in miniature view
        [miny, maxy]=self.ax3.get_ylim()
        self.rect_miniature_view.set_xy((self.posx/self.fs,miny))
        self.figMiniature.canvas.draw_idle()

        # Update Slider position
        self.slider.setValue(self.posx)

        # Update gotoseg field 
        selectedx=self.posx/self.fs  # slider pos in seconds   
        self.gotoTimeLabel.setText('{:.3f}'.format(selectedx)) 

    def timeSpanChanged(self):

        #self.sidepanel.lookforevents.focus() # Set focus to a non-focusable element to remove focus from this one
        TimeSpan_text = self.timeLabel.text()
        if is_number(TimeSpan_text):
            self.tanalisis=float(TimeSpan_text)
            # Clamp window size so it never exceeds file size
            self.ventana = min(round(self.tanalisis * self.fs), self.siz)
            # Clamp current position so the visible window still fits
            self.posx = clamp_posx(self, self.posx)

            self.y=self.x[self.posx:self.posx+self.ventana]
            draw_tfr(self,self.y,self.fs,self.posx)
    
            # Update the width of rectangle marker in miniature view
            miny, maxy = self.ax3.get_ylim()
            self.rect_miniature_view.set_width(self.ventana / self.fs)
            self.rect_miniature_view.set_xy((self.posx / self.fs, miny))
            self.figMiniature.canvas.draw_idle()

            # Update Slider from - to values
            self.slider.setMinimum(0)
            self.slider.setMaximum(max(0, self.siz - self.ventana))
            self.slider.setValue(self.posx)

            # Update go-to field with the clamped position
            selectedx = self.posx / self.fs
            self.gotoTimeLabel.setText(f"{selectedx:.3f}")

    def gotoTimeLabelChanged(self):

        goTo_text = self.gotoTimeLabel.text()
        if is_number(goTo_text):
            posx_in_sec=float(goTo_text)
            # Clamp position
            self.posx = clamp_posx(self, round(posx_in_sec * self.fs))
            self.y=self.x[self.posx:self.posx+self.ventana]
            draw_tfr(self,self.y,self.fs,self.posx)

            # Update go-to field with the clamped position
            posx_clamped_in_sec = self.posx / self.fs
            self.gotoTimeLabel.setText(f"{posx_clamped_in_sec:.3f}")

            # Update rectangle marker in miniature view
            [miny, _]=self.ax3.get_ylim()
            self.rect_miniature_view.set_xy((posx_clamped_in_sec,miny))
            self.figMiniature.canvas.draw_idle()

            # Update Slider position
            self.slider.setValue(self.posx)

    def miniatureOnClick(self,event):
        # Ignore clicks outside the axes / data area
        if event.xdata is None:
            return

        #print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #  ('double' if event.dblclick else 'single', event.button,
        #   event.x, event.y, event.xdata, event.ydata))

        posx = round(event.xdata * self.fs)
        self.posx = clamp_posx(self, posx)

        self.y=self.x[self.posx:self.posx+self.ventana]
        draw_tfr(self,self.y,self.fs,self.posx)

        # Update gotoseg field 
        selectedx=self.posx/self.fs  # slider pos in seconds   
        self.gotoTimeLabel.setText('{:.3f}'.format(selectedx))

        # Update Slider position
        self.slider.setValue(self.posx)

        # Update rectangle marker in miniature view
        [miny, maxy]=self.ax3.get_ylim()
        self.rect_miniature_view.set_xy((selectedx,miny))
        self.figMiniature.canvas.draw_idle()

    def advanceButTimeFrame(self,event):

        if not(self.is_running):
            self.is_running=True
            self.advanceBut.setText("STOP")
            self.advance_timer = QTimer(self)
            self.advance_timer.timeout.connect(self.spect_advance_function)
            self.advance_timer.start(500)  # milliseconds
            #self.timer_spect_advance = RepeatedTimer(.5,self.spect_advance_function, "World")
        else:
            self.is_running=False
            self.advanceBut.setText("Auto Advance")
            self.advance_timer.stop()

    def edit_freqzoomdown_Callback(self):

        zoom_freq_low_limit_text=self.zoom_freq_low_limit_textbox.text()
        
        if is_number(zoom_freq_low_limit_text):
            self.zoom_freq_low_limit=float(zoom_freq_low_limit_text)
            draw_tfr(self,self.y,self.fs,self.posx)
    
    def edit_freqzoomup_Callback(self):

        zoom_freq_up_limit_text=self.zoom_freq_up_limit_textbox.text()
        
        if is_number(zoom_freq_up_limit_text):
            self.zoom_freq_up_limit=float(zoom_freq_up_limit_text)
            draw_tfr(self,self.y,self.fs,self.posx)

    def FFTpointsChanged(self):
        fft_npoints=2**(self.FFTpoints.currentIndex()+5)
        frec_res=self.fs/fft_npoints
        self.frec_res_wzoom_textlabel.setText(f"= {frec_res:5.1f} Hz")
        
        draw_tfr(self,self.y,self.fs,self.posx)

    def window_type_Callback(self):
        draw_tfr(self,self.y,self.fs,self.posx)

    def freq_zoom_Callback(self):
        draw_tfr(self,self.y,self.fs,self.posx)

    def manual_CL_chkbox_Callback(self):
        try:
            valmin = int(self.minDR_textbox.text())
            valmax = int(self.maxDR_textbox.text())
        except ValueError:
            return
        if valmin >= valmax:
            QMessageBox.warning(self, "Invalid range", "Minimum must be less than maximum.")
            return
        self.TFR_bitmap.set_clim(valmin,valmax)
        draw_tfr(self,self.y,self.fs,self.posx)
    
    def edit_mindr_Callback(self):
        try:
            valmin = int(self.minDR_textbox.text())
            valmax = int(self.maxDR_textbox.text())
        except ValueError:
            return
        if valmin >= valmax:
            QMessageBox.warning(self, "Invalid range", "Minimum must be less than maximum.")
            return
        self.TFR_bitmap.set_clim(valmin,valmax)
        draw_tfr(self,self.y,self.fs,self.posx)
    
    def edit_maxdr_Callback(self):
        try:
            valmin = int(self.minDR_textbox.text())
            valmax = int(self.maxDR_textbox.text())
        except ValueError:
            return
        if valmin >= valmax:
            QMessageBox.warning(self, "Invalid range", "Minimum must be less than maximum.")
            return
        self.TFR_bitmap.set_clim(valmin,valmax)
        draw_tfr(self,self.y,self.fs,self.posx)

    def select_SPL_ICI_f0_graph_Callback(self):
        draw_auxiliary_nav_graph(self)

    def eventsortcheckbox_Callback(self):
        if not self.events.empty:
            if self.eventsortcheckbox.isChecked():
                # Sort by 'type'
                self.events = self.events.sort_values(by="type").reset_index(drop=True)
            else:
                # Sort by 'start'
                self.events = self.events.sort_values(by="start").reset_index(drop=True)
            
            self.Detected_ev_combo.clear()    
            bssep = ' '
            # Fill combo box
            for _, row in self.events.iterrows():
                text = f"{row['start']/self.fs:8.3f} s.,{bssep}{row['type']},  f0={round(row['f0'])} Hz,  BW={round(row['BW'])} Hz,  SC={row['score']:6.2f}"
                self.Detected_ev_combo.addItem(text)

            self.Detected_ev_combo.setEnabled(True)
            self.Detected_ev_combo.setCurrentIndex(0)

    def playAudioMenuCallback(self):

        # Check if resampling is needed for playback fs>48 kHz
        if self.fs>48000:
            # I resample to 48 kHz for playback otherwise at 192kHz I get clicks and the audio is not smooth
            new_fs = 48000  
            # High-quality, fast resampling (polyphase FIR filtering, much faster than signal.resample)
            ynew = signal.resample_poly(self.y, new_fs, self.fs).astype('int16')
        else:
            new_fs=self.fs
            ynew=self.y.astype('int16')
            
        # Auto-scale for maximum volume
        peak = np.max(abs(ynew))
        if peak > 0:
            audio_scaled = ynew / peak  # scale to [-1, 1]
        else:
            audio_scaled = ynew

        # Convert to int16
        ynew = (audio_scaled * 32767).astype('int16')

        dFrameTime=0.2 # seconds
        chunksize = int(dFrameTime*new_fs) 
        # Open a sd.OutputStream object to write the WAV file to
        stream = sd.OutputStream(samplerate=new_fs, channels=1, blocksize=chunksize, dtype='int16')

        # Create a marker line in the spectrogram view
        spect_play_marker = Line2D([0,0], [0,self.fs/2],color ='red',linestyle="dotted")  
        self.ax0.add_line(spect_play_marker)         

        stream.start()  # Start the stream before writing

        for i in range(0, len(ynew), chunksize):
            data = ynew[i:i + chunksize]
            stream.write(data) # Play the sound by writing the audio data to the stream
            # Update marker of playing chunk in spectrogram view
            spect_play_marker.set_xdata([i/new_fs+self.posx/self.fs,i/new_fs+self.posx/self.fs])
            self.figSpect.canvas.draw_idle()
            self.figSpect.canvas.flush_events()
            
        # Close and terminate the stream
        stream.stop()
        stream.close()

        spect_play_marker.remove()
        self.figSpect.canvas.draw_idle()

    def playx10AudioMenuCallback(self):

        # Check if resampling is needed for playback fs>4.8 kHz
        if self.fs>4800:
            # I resample to 4800 Hz for playback otherwise at 192kHz I get clicks and the audio is not smooth
            new_fs = 4800  
            # High-quality, fast resampling (polyphase FIR filtering, much faster than signal.resample)
            ynew = signal.resample_poly(self.y, new_fs, self.fs).astype('int16')
        else:
            new_fs=self.fs
            ynew=self.y.astype('int16')
            
        # Auto-scale for maximum volume
        peak = np.max(abs(ynew))
        if peak > 0:
            audio_scaled = ynew / peak  # scale to [-1, 1]
        else:
            audio_scaled = ynew

        # Convert to int16
        ynew = (audio_scaled * 32767).astype('int16')
            
        dFrameTime=0.2 # seconds
        chunksize = int(dFrameTime*new_fs) 
        # Open a sd.OutputStream object to write the WAV file to
        stream = sd.OutputStream(samplerate=10*new_fs, channels=1, blocksize=chunksize, dtype='int16')

        # Create a marker line in the spectrogram view
        spect_play_marker = Line2D([0,0], [0,self.fs/2],color ='red',linestyle="dotted")  
        self.ax0.add_line(spect_play_marker)         

        stream.start()  # Start the stream before writing

        for i in range(0, len(ynew), chunksize):
            data = ynew[i:i + chunksize]
            stream.write(data) # Play the sound by writing the audio data to the stream
            # Update marker of playing chunk in spectrogram view
            #spect_play_marker.set_xdata([i/new_fs+self.posx/self.fs,i/new_fs+self.posx/self.fs])
            #self.figSpect.canvas.draw()
            #self.figSpect.canvas.flush_events()
            
        # Close and terminate the stream
        stream.stop()
        stream.close()

        spect_play_marker.remove()
        self.figSpect.canvas.draw_idle()

    def open_file(self,filename=None):
        
        if not filename:
            # Open a file dialog restricted to WAV files
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Select a WAV file",
                self.start_dir,
                "WAV Files (*.wav *.WAV)"
            )
            if not filename:
                return
        
        try:
            start = time.time()
            self.fs, self.x = wavfile.read(filename) 
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open file:\n{e}")
            return
        
        # Keep full filename with path
        self.current_file = filename
        # Remove the path and store only filename (with .wav) for later use in annotations
        self.filename=os.path.basename(filename)
              
        # Load, if exist Manual Annotation file
        parent_directory=os.path.basename(os.path.dirname(filename))
        self.current_annotation_file = f"{parent_directory}_manual_annotated.csv"
        annotation_path_file = os_path_join(self.annotation_path, self.current_annotation_file)

        if os.path.isfile(annotation_path_file):
            m_annotated_events_all = pd.read_csv(annotation_path_file, sep=';')
            self.list_of_all_manual_annotated_events_in_deployment = m_annotated_events_all["m_event_type"].dropna().unique() # % Extract only the Manual annotated categories
            # Find annotated events in current file
            base = os.path.basename(filename)
            m_annotated_events_infile = m_annotated_events_all[m_annotated_events_all.iloc[:, 0] == base]
            
            # convert only the subset
            self.m_annotated_events = m_annotated_events_infile.to_dict(orient="records")

            if self.m_annotated_events:
                # Change "," to "." and convert string to numbers...
                for row in self.m_annotated_events:
                    for key in ("tini", "tfin", "freqmin", "freqmax"):
                        val = row.get(key)
                        if val is None:
                            row[key] = None
                            continue
                        try:
                            row[key] = float(str(val).replace(",", "."))
                        except ValueError:
                            row[key] = None   # equivalent to pandas errors="coerce"

                #  ---- FILL THE COMBO BOX ("m_annotation_list") OF DETECTED EVENTS ------
                bssep = ' '
                self.m_annotation_combo.clear()
                for e in self.m_annotated_events:
                    if e.get("tini") is None:
                        continue
                    #text = f"{row['tini']:8.3f} s.,{bssep}{row.get('m_event_type','')}"
                    label = f"{e['tini']:.3f} s. | {e['m_event_type'] or 'Unlabeled'} |  {e['freqmin']:.1f}-{e['freqmax']:.1f}Hz | {e['user']} on {e['date_time_annotation']}"
                    self.m_annotation_combo.addItem(label)

                self.m_annotation_combo.setEnabled(True)
                self.m_annotation_combo.setCurrentIndex(0)
            else: # Exist but empty for this file
                self.m_annotated_events: list[dict] = [] # Empty the list if no annotation file found
                self.m_annotation_combo.clear()
                self.m_annotation_combo.setEnabled(False)
                self.m_annotation_combo.setCurrentIndex(0)
        else: #  If not exist initialize the table to []
            self.m_annotated_events: list[dict] = [] # Empty the list if no annotation file found
            self.list_of_all_manual_annotated_events_in_deployment = []
            self.m_annotation_combo.clear()
            self.m_annotation_combo.setEnabled(False)
            self.m_annotation_combo.setCurrentIndex(0)

        #  Reset to zero previous detected events and create a struct to store extracted events in the file
        self.events = pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in self.event_fields_types})

        self.siz=len(self.x)
        #self.ventana=round(self.tanalisis*self.fs)
        self.ventana = min(round(self.tanalisis * self.fs), self.siz)
        
        # Update Slider from - to values
        self.slider.setMinimum(0)
        #self.slider.setMaximum(self.siz-self.ventana)
        self.slider.setMaximum(max(0, self.siz - self.ventana))
        self.slider.setValue(0)
        self.slider.setEnabled(True)

        # Update go to sec. Text Edit
        self.gotoTimeLabel.setText("0")
        
        self.posx=0
        end = min(self.posx + self.ventana, self.siz) # Ensure we don't go out of bounds
        self.y = self.x[self.posx:end]
        draw_tfr(self,self.y,self.fs,self.posx)
        
        end = time.time()
        etime_wr=end - start
        SizeRecording=self.x.nbytes/1e6

        self.statusBar().showMessage(" File opened in "+ "{:.2f}".format(etime_wr)+" sec."+"  (fs= "+format(self.fs)+" Hz).  Size in memory "+"{:.4f}".format(SizeRecording)+" MB")

        # Create and store a miniature representation of the complete sound file
        NPminiature = min(300, max(1, len(self.x)))
        Ndiv = max(1, len(self.x) // NPminiature) # // = Floor
        
        # To fix some distortions when creating the miniature representation if mean(x)<>0
        xtemp=self.x
        xnm=xtemp-np.mean(xtemp)
        self.mini_x = np.max(np.reshape(xnm[:(len(xnm) // Ndiv) * Ndiv], (len(xnm) // Ndiv, Ndiv)),axis=1)
        peak = np.max(np.abs(self.mini_x))
        if peak > 0:
            self.mini_x *= 0.8 / peak
        else:
            self.mini_x.fill(0.0)
        
        draw_auxiliary_nav_graph(self)
        
        # Erase the self.Detected_ev_combo list and disable it
        self.Detected_ev_combo.clear()
        self.Detected_ev_combo.setPlaceholderText('---')
        self.Detected_ev_combo.setEnabled(False)

        # Enable Time Span and go to time edit text boxes
        self.timeLabel.setEnabled(True)
        self.gotoTimeLabel.setEnabled(True) # Enable goto time label
        # Enable zoom frequency box
        self.lf_zoom_enabled.setEnabled(True)
        # Enable automatic/ manual dynamic range box
        self.manual_CL_chkbox_enabled.setEnabled(True)

        self.zoom_freq_low_limit_textbox.setValidator(QIntValidator(0, floor(self.fs/2)))  # only allows integers from 0 to fs/2
        self.zoom_freq_up_limit_textbox.setValidator(QIntValidator(0, floor(self.fs/2)))  # only allows integers from 0 to fs/2

        # Activate bindings of the buttons 
        self.previousBut.setEnabled(True)
        self.advanceBut.setEnabled(True)
        self.nextBut.setEnabled(True)
        self.is_running = False

        # --- Enable MENU ITEMS from MENUBAR ------- 
        self.save_wav_copy_action.setEnabled(True)  
        self.save_fragment_action.setEnabled(True)
        self.play_menu.setEnabled(True)
        self.analyze_action.setEnabled(True)
        self.add_manual_annottation_action.setEnabled(True)
        self.delete_manual_annottation_action.setEnabled(True)
        self.edit_manual_annottation_action.setEnabled(True)
        self.save_manual_annottation_action.setEnabled(True)
        self.filer_manual_anotations_action.setEnabled(True)
        self.meassure_action.setEnabled(True)
        
        # Enable the Analyze button
        self.AnalyzeBut.setEnabled(True)

        # Update the title with the filename
        self.setWindowTitle(filename)

        self.add_recent_file(filename)
        self.save_recent_files()

    def save_recent_files(self):
        self.settings.setValue("recentFiles", self.recent_files)

    def add_recent_file(self, file_path):
        # If already in the list, remove it first
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)

        # Add newest file at the top
        self.recent_files.insert(0, file_path)

        # Keep only self.max_recent_files items, forget the oldest ones
        self.recent_files = self.recent_files[:self.max_recent_files]

        self.update_recent_files_menu()

    def update_recent_files_menu(self):
        self.recent_menu.clear()

        if not self.recent_files:
            empty_action = QAction("(No recent files)", self)
            empty_action.setEnabled(False)
            self.recent_menu.addAction(empty_action)
            return

        for file_path in self.recent_files:
            action = QAction(os.path.basename(file_path), self)
            action.setToolTip(file_path)
            action.triggered.connect(
                lambda checked=False, path=file_path: self.open_file(path)
            )
            self.recent_menu.addAction(action)

    def openPreviousFile(self):

        if not hasattr(self, "current_file"):
            QMessageBox.information(
                self,
                "No WAV file opened",
                "Open a WAV file from the deployment first."
            )
            return

        current_filename = Path(self.current_file).name

        actual_file_number = self.FILENAME_TIME[self.FILENAME_TIME == current_filename].index[0]

        if actual_file_number > 0:
            filename_to_open = self.FILENAME_TIME.iloc[actual_file_number - 1]
            self.open_file(os_path_join(self.workdir[0], filename_to_open))
            self.updateDeploymentFileMarker(actual_file_number - 1, self.pixx, self.pixy)
        else:
            QMessageBox.information(
                self,
                "Beginning of deployment",
                "Already the first file in the deployment."
            )

    def openNextFile(self):

        if not hasattr(self, "current_file"):
            QMessageBox.information(
                self,
                "No WAV file opened",
                "Open a WAV file from the deployment first."
            )
            return

        current_filename = Path(self.current_file).name

        actual_file_number = self.FILENAME_TIME[self.FILENAME_TIME == current_filename].index[0]

        if actual_file_number < len(self.FILENAME_TIME) - 1:
            filename_to_open = self.FILENAME_TIME.iloc[actual_file_number + 1]
            self.open_file(os_path_join(self.workdir[0], filename_to_open))
            self.updateDeploymentFileMarker(actual_file_number + 1, self.pixx, self.pixy)
        else:
            QMessageBox.information(
                self,
                "End of deployment",
                "Already the last file in the deployment."
            )
            
   

    def openDeploymentAnalysisFile(self):

        folder = QFileDialog.getExistingDirectory(
        None,
        "Select folder",
        "")
        if folder:
            dlg = ReliableCsvDialog(folder)
            if dlg.exec():
                filename=dlg.selected_file()
                self.workdir=os_path_split(filename) # Spearate and keep workdir of the Deployemnt.
                B = pd.read_csv(filename, delimiter=";", dtype=str)
                numberofcolumns=B.shape[1]  # Get the number of columns
                # Check if the deployment has at least two files to analyze (rows), otherwise it is not a valid deployment analysis
                if B.shape[0] < 2:
                    QMessageBox.warning(
                        self,
                        "Invalid deployment analysis",
                        "The deployment analysis contains fewer than two files.\n\n"
                        "The analysis appears to be incomplete or corrupted."
                    )
                    return
                # Check if the deployment analysis has at least three columns (filename, date, and at least one parameter), otherwise it is not a valid deployment analysis
                if numberofcolumns < 3:
                    QMessageBox.warning(
                        self,
                        "Invalid deployment analysis",
                        "The deployment analysis file has no Indicators."
                    )
                    return

                aux_cell=B.columns.values[2:]  # Get the field names (omiting 1st two filename, date)
                field_cell=[i.split('[', 1)[0] for i in aux_cell]
                unit_cell=[i.split('[', 1)[-1].split(']')[0] for i in aux_cell]
                self.parameter_units=unit_cell
        
                self.HC.set_label(self.parameter_units[0], size=10,rotation=90)
                # self.fieldselect.clear() # delete all items of the Field select combo box
                # self.fieldselect.addItems(field_cell)
                # self.fieldselect.setCurrentIndex(0)
        
                B=B.replace(',', '.',regex=True)
                # Convert indicator columns to numeric values.
                # Invalid / empty cells become NaN instead of crashing later.
                self.A = (B.iloc[:, 2:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)) # Remove name and date column

                self.parameterselected = self.A[:, 0] 

                self.FILENAME_TIME=B.iloc[:,0] # Keep the deployment file names in a variable to use it later in the program

                deployment_date = datetime.strptime(B.iloc[0, 1].strip(), '%d-%b-%Y %H:%M:%S')
 
                # Initialization
                self.TIME_STAMP_min = np.zeros(self.A.shape[0])
                self.TIME_STAMP_day = np.zeros(self.A.shape[0])
                self.graph_yticks = np.array(['dd-mmm-yyyy'] * self.A.shape[0])
                self.FILENAME_REAL_TIME = np.array(['HH:MM:SS'] * self.A.shape[0])

                for i, fname in enumerate(self.FILENAME_TIME):
                    
                    if pd.isna(fname):
                        print(f"Skipping row {i}: missing filename")
                        continue

                    fname = str(fname).strip()
                    # Break filename (i.e. "BCA_P_1_20220526_135751.wav") into parts by "_" and "." characters
                    fname_parts = re_split(r"[_.]", fname)

                    # Make sure we have a valid filename with enough parts to extract the timestamp
                    if len(fname_parts) < 5:
                        print(f"Skipping row {i}: malformed filename '{fname}'")
                        continue
                    
                    try:
                        real_date_at_i = pd.to_datetime(fname_parts[3] + fname_parts[4], format='%Y%m%d%H%M%S')
                    except ValueError:
                        print(f"Skipping row {i}: invalid timestamp in '{fname}'")
                        continue

                    ho = real_date_at_i.hour
                    mn = real_date_at_i.minute

                    self.TIME_STAMP_min[i] = ho * 60 + mn
                    self.TIME_STAMP_day[i] = (real_date_at_i.date() - deployment_date.date()).days
                    self.graph_yticks[i] = real_date_at_i.strftime('%d-%b-%Y')
                    self.FILENAME_REAL_TIME[i] = real_date_at_i.strftime('%H:%M:%S')  # To show when moise click
    
                #self.FILENAME_REAL_TIME = str(self.FILENAME_REAL_TIME)
                self.graph_yticks = pd_unique(self.graph_yticks)
        
                #big_data_graph(self.deployment_inspector,self.A[:,0],self.graph_yticks,self.TIME_STAMP_day,self.TIME_STAMP_min)
                big_data_graph(self,self.parameterselected,self.graph_yticks,self.TIME_STAMP_day,self.TIME_STAMP_min)

                # Adjust units and min-max color scale
                self.HC.set_label(self.parameter_units[0], size=10,rotation=90)
                vmin=np.nanmin(self.parameterselected);vmax=np.nanmax(self.parameterselected)
                self.deployment_nav_bitmap.set_clim(vmin,vmax)
                self.figDeployInspect.canvas.draw_idle()

                # Check to see if it makes sense having the double slidere enable and if so
                # set the min and max of the double knot slider
                if np.isclose(vmin, vmax):
                    self.deployment_nav_bitmap.set_clim(vmin - 0.5, vmax + 0.5)

                    self.double_slider.setEnabled(False)
                    self.double_slider.setToolTip("Disabled because all values are equal; no range can be selected.")
                else:
                    self.double_slider.setEnabled(True)
                    self.double_slider.setToolTip("")

                    self.double_slider.setRange(vmin, vmax)

                # Activate the binding of fieldselect
                self.fieldselect.setEnabled(True)

                # Update the title with the deployment file name
                self.setWindowTitle(filename)

                # Enable the "Open Previous" and "Open Next" menu items
                self.open_previous_action_menu.setEnabled(True)
                self.open_next_action_menu.setEnabled(True)

                # If not connected yet, connect click event to action
                if self.deploy_click_cid is None:
                    self.deploy_click_cid = self.figDeployInspect.canvas.mpl_connect("button_press_event",self.deployinspecOnClick)

                # Load if exist Manual Annotation File
                parent_directory=os.path.basename(os.path.dirname(filename))
                self.current_annotation_file = f"{parent_directory}_manual_annotated.csv"
                AnnotationPathName=self.annotation_path
                annotation_path = os_path_join(AnnotationPathName, self.current_annotation_file)
                
                if os.path.isfile(annotation_path):
                    m_annotated_events = pd.read_csv(annotation_path, sep=';')
                    manual_annotated_categories = m_annotated_events["m_event_type"].dropna().unique() # % Extract only the Manual annotated categories

                    for cat in manual_annotated_categories:
                        df_cat = m_annotated_events[m_annotated_events["m_event_type"] == cat]
                        counts = df_cat.groupby("filename").size()  # Count how many annotations for the specific category in each file

                        #filename_to_pos = {fn: i for i, fn in enumerate(self.FILENAME_TIME)}
                        filename_to_pos = {str(fn).strip(): i for i, fn in enumerate(self.FILENAME_TIME)}

                        Abis = np.zeros(self.FILENAME_TIME.shape[0], dtype=float)

                        for fn, cnt in counts.items():
                            pos=filename_to_pos.get(fn)
                            if pos is not None:
                                Abis[pos] = cnt
                        
                        self.A = np.column_stack((self.A, Abis))
                        field_cell.append(f"{cat} (Manual Annotation)")
                        self.parameter_units.append("[#]")
                
                self.fieldselect.clear() # delete all items of the Field select combo box
                self.fieldselect.addItems(field_cell)
                self.fieldselect.setCurrentIndex(0)

    def save_wav_copy_Callback(self):

        source_file = Path(self.current_file)

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save WAV Copy As",
            str(source_file.name),
            "WAV Files (*.wav)"
        )

        if not filename:
            return

        try:
            copy2(source_file, filename)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Save Error",
                f"Could not copy WAV file:\n\n{e}"
            )
            return

        QMessageBox.information(
            self,
            "WAV Copy Saved",
            f"Saved:\n{filename}"
        )
    
    def select_and_save_fragment_Callback(self):
        result = SelectFragmentAndSaveDialog.getSelection(self)

        if result is None:
            return

        mode = result[0]

        if mode == SelectFragmentAndSaveDialog.START_END:
            fs = self.fs
            msaveini = max(int(round(result[1] * fs)),0)
            msavefin = min(int(round(result[2] * fs)),len(self.x))
            temp_lines = []
            
        elif mode == SelectFragmentAndSaveDialog.CENTER_SAMPLES:

            nsamples = result[1]
            x0, _ = ginput_point(self)

            h1 = self.ax0.axvline(x0)

            h2 = self.ax0.axvline(
                x0 - nsamples/(2*self.fs),
                linestyle="--"
            )

            h3 = self.ax0.axvline(
                x0 + nsamples/(2*self.fs),
                linestyle="--"
            )

            msaveini = round(x0*self.fs - nsamples/2)
            msavefin = round(x0*self.fs + nsamples/2)
            temp_lines = [h1, h2, h3]

        elif mode == SelectFragmentAndSaveDialog.CURRENT_SELECTION:
            x1, _ = ginput_point(self)
            h1 = self.ax0.axvline(x1, linestyle="--")

            x2, _ = ginput_point(self)
            h2 = self.ax0.axvline(x2, linestyle="--")

            msaveini = round(min(x1, x2) * self.fs)
            msavefin = round(max(x1, x2) * self.fs)
            temp_lines = [h1, h2]

        filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Selected Fragment As",
                "",
                "WAV Files (*.wav)"
            )
        
        if filename:
            ys = self.x[msaveini:msavefin]
            wavfile.write(filename,self.fs,ys)
            for h in temp_lines:
                try:
                    h.remove()
                except Exception:
                    pass

            self.figSpect.canvas.draw_idle()
            return

    def update_deployment_nav_image(self, vmin, vmax):
        vmin = float(vmin)
        vmax = float(vmax)

        if vmax <= vmin:
            vmax = vmin + 1e-6

        self.deployment_nav_bitmap.set_clim(vmin, vmax)

        if hasattr(self, "deployment_nav_colorbar"):
            self.deployment_nav_colorbar.update_normal(self.deployment_nav_bitmap)

        self.figDeployInspect.canvas.draw_idle()

    def fieldselect_Callback(self,event):
        valsel=self.fieldselect.currentIndex()
        self.parameterselected=self.A[:,valsel]
        big_data_graph(self,self.parameterselected,self.graph_yticks,self.TIME_STAMP_day,self.TIME_STAMP_min)
        self.HC.set_label(self.parameter_units[valsel], size=10,rotation=90)
        vmin=np.nanmin(self.parameterselected);vmax=np.nanmax(self.parameterselected)

        # Check to see if it makes sense having the double slidere enable and if so
        # set the min and max of the double knot slider
        if np.isclose(vmin, vmax):
            self.deployment_nav_bitmap.set_clim(vmin - 0.5, vmax + 0.5)

            self.double_slider.setEnabled(False)
            self.double_slider.setToolTip("Disabled because all values are equal; no range can be selected.")
        else:
            self.double_slider.setEnabled(True)
            self.double_slider.setToolTip("")

            self.double_slider.setRange(vmin, vmax)
            self.deployment_nav_bitmap.set_clim(vmin,vmax)

        self.figDeployInspect.canvas.draw_idle()

    def updateDeploymentFileMarker(self, filenumber,pixx,pixy):
        xs=self.TIME_STAMP_min[filenumber]
        ys=self.TIME_STAMP_day[filenumber]*pixy
        
        output_txt = """\
                    Date: {0}
                    Time: {1}
                    File: {2}
                    File #: {3} Value:{4}\
        """.format(self.graph_yticks[int(self.TIME_STAMP_day[filenumber])],self.FILENAME_REAL_TIME[filenumber],self.FILENAME_TIME[filenumber],filenumber,self.parameterselected[filenumber])

        # Place a marker around the pixel clicked
        self.ll.set_bounds(xs-1, ys, pixx, pixy-1)
        # Set the text to be displayed so that I can get the extents
        self.lt.set_text(dedent(output_txt))
        # Make it visible to get right the extents
        self.lt.set_visible(True)
        self.lp.set_visible(True)
        # Force a draw so renderer is accurate 
        self.figDeployInspect.canvas.draw_idle()
        renderer = self.figDeployInspect.canvas.get_renderer()
        bb_disp = self.lt.get_window_extent(renderer=renderer)
        # Convert bbox display -> data coordinates CORRECTLY This handles the display y-down vs data y-up conversion for you.
        bb_data = self.figDeployInspect.axes[0].transData.inverted().transform_bbox(bb_disp)
        Twidth=bb_data.width
        Theight=bb_data.height # Theight is negative because the axis are inverted.

        # To avoid patch & text outside the graph
        # Top left corner is (0,0) and y grows downwards, so to avoid patch & text outside the graph:
        if ys - Theight > self.TIME_STAMP_day[-1]*pixy:
            yt = ys + Theight
        else:
            yt = ys + pixy
                
        if xs - Twidth < 0:
            xl = xs+pixx
        else:
            xl = xs - Twidth

        self.lp.set_bounds(xl, yt, Twidth, -Theight)
        self.lt.set_position((xl, yt))
        self.lp.set_visible(True)
        self.lt.set_visible(True)   

        # Put rectangle below text visually
        self.ll.set_zorder(1)   # rectangle at the back
        self.lp.set_zorder(2)   # Patch in the middle
        self.lt.set_zorder(3)   # text on top
        #self.lp.set_zorder(self.lt.get_zorder() - 1)
        self.ll.set_visible(True)
        self.lp.set_visible(True)

    def deployinspecOnClick(self,event):
        
        # Safety check to avoid errors when clicking outside the axes area
        if event.xdata is None or event.ydata is None:
            return
        
        # Safety check to make sure there are at least two timestamps to work with
        if len(self.TIME_STAMP_min) < 2:
            return
        
        graph_sep_line=2 # Graphic separation line
        self.pixx=int(self.TIME_STAMP_min[1]-self.TIME_STAMP_min[0])+2
        self.pixy=15+graph_sep_line # Pixel size y (one file)

        # Obtain the filenumeber in the position where the user "clicks"
        day=floor(event.ydata/self.pixy)
        idx_min=np.argwhere(self.TIME_STAMP_day==day)  # Reduce the list to the selected day.
        file=np.argwhere(event.xdata>self.TIME_STAMP_min[idx_min[:,0]])
        
        filenumber=idx_min[file[-1,0]]  # We get the last one of the list
        
        if filenumber[0]<=self.FILENAME_TIME.shape[0]:
            if event.dblclick:  # Open file?
                if isfile(os_path_join(self.workdir[0],self.FILENAME_TIME[filenumber[0]]).strip()):
                    lines = self.FILENAME_TIME[filenumber[0]]
                    result = ask_open_file(lines)
                    if result is True:
                        self.open_file(os_path_join(self.workdir[0],self.FILENAME_TIME[filenumber[0]]).strip())
                    else:
                        print("No, I don't want to open it!")
                else:
                    lines = ["Sorry. Can't find the file "+self.FILENAME_TIME[filenumber[0]]+" in the directory", "Place the *.CSV and *.WAV files in the same directory."]
                    result =  QMessageBox.warning(self,"Ooops","\n".join(lines))
            else: # Normal click
                self.updateDeploymentFileMarker(filenumber[0],self.pixx,self.pixy)
                self.figDeployInspect.canvas.draw_idle()

    def lookforevents_Callback(self):           
        
        # Disable tri-state mode (so far only used in analyze deployment).
        #self.plugin_selector_dialog.set_tristate_mode(False)

        # Open plugin selection dialog
        if self.plugin_selector_dialog.exec()== QDialog.DialogCode.Accepted:          # This actually opens the dialog window!

            #  Reset to zero previous detected events and creat a struct to store extracted events in the file
            self.events = pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in self.event_fields_types})
            # Erase the self.Detected_ev_combo list
            self.Detected_ev_combo.clear()

            # Disbale the filter by event menu until we have events to filter
            self.filter_automatic_det_action.setEnabled(False)

            # Load default dsp and bands callibration.
            dsp = default_dsp()
            bands = default_bands()

            # Replece them if present with the dsp and bands in the deployment_info.mat file
            try:
                deployment_info = Path(self.current_file).parent / "deployment_info.mat"

                if deployment_info.is_file():
                    mat = loadmat(deployment_info, squeeze_me=True, struct_as_record=False)

                    dsp_dict = _mat_struct_to_dict(mat.get("dsp", {}))
                    bands_dict = _mat_struct_to_dict(mat.get("bands", {}))

                    dsp.gain = float(_scalar(dsp_dict.get("gain"), dsp.gain))
                    dsp.nbits = int(float(_scalar(dsp_dict.get("nbits"), dsp.nbits)))

                    if "number" in bands_dict:
                        bands.number = [int(x) for x in _float_list(bands_dict["number"])]

                    if "sh" in bands_dict:
                        bands.sh = _float_list(bands_dict["sh"])

                    if "label" in bands_dict:
                        labels = np.asarray(bands_dict["label"]).ravel().tolist()
                        bands.label = [str(x) for x in labels]
                        
            # deployment_info.mat exists but is corrupt -> print a warning.
            except Exception as e:
                    print(f"Warning: Could not load deployment_info.mat: {e}")

            verbose=1

            selected_plugins, selected_indexes = self.plugin_selector_dialog.get_selected_plugins()

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

            #-------
            # Create worker
            self.worker = PluginWorker(
                selected_plugins,
                self.x, self.fs, dsp, bands, verbose=True
            )
            # Connect the cancel button of the status window to the worker's interruption method
            self.status_window.cancel_requested.connect(self.worker.requestInterruption)
            # When worker finishes: restore stdout and handle results
            self.worker.finished.connect(self.on_lookforevents_finished)
            self.worker.failed.connect(self.on_lookforevents_failed)

            # Start asynchronous thread
            self.worker.start()
            #-------
            
            # #-----
            # events = []
            
            # for cls in selected_plugins:
            #     plugin = cls()
            #     # All prints inside analyze() will now go to your status window
            #     E, I = plugin.analyze(self.x, self.fs, dsp, bands, verbose)
            #     events.append(E)
            # self.on_lookforevents_finished(events)
            # #----

    def on_lookforevents_finished(self, events):
        # Put stdout back as it was for printing
        sys.stdout=self.original_stdout
        print("Processing completed.")

        # Communicate the status window that processing is finshed so that it changes the button action...
        self.status_window.mark_tasks_finished()
        
        # Merge all events received
        for E in events:
            self.events = pd.concat([self.events, E], ignore_index=True)
        
        # Draw the auxiliary navigation graph with the time series miniature and events....   
        draw_auxiliary_nav_graph(self)

        # Fill combobox with the events
        if not self.events.empty:
            if self.eventsortcheckbox.isChecked():      # or .value(), depending on your UI framework
                # Sort by 'type'
                self.events = self.events.sort_values(by="type").reset_index(drop=True)
            else:
                # Sort by 'start'
                self.events = self.events.sort_values(by="start").reset_index(drop=True)

            bssep = ' '
            # Fill combo box
            for _, row in self.events.iterrows():
                text = f"{row['start']/self.fs:8.3f} s.,{bssep}{row['type']},  f0={round(row['f0'])} Hz,  BW={round(row['BW'])} Hz,  SC={row['score']:6.2f}"
                self.Detected_ev_combo.addItem(text)

            self.Detected_ev_combo.setEnabled(True)
            self.Detected_ev_combo.setCurrentIndex(0)

            # Enable the event plot by SPL/IEI/f0 selection 
            self.select_SPL_graph.setEnabled(True)
            self.select_ICI_graph.setEnabled(True)
            self.select_f0_graph.setEnabled(True)

            # Enable the filter by event type checkbox
            self.filter_automatic_det_action.setEnabled(True)

            # Show all detected types by default after a new analysis
            self.eventstobeshown = self.events["type"].dropna().unique().tolist()

        draw_tfr(self,self.y,self.fs,self.posx)
    
    def on_lookforevents_failed(self, error_message):
        # Always restore stdout
        sys.stdout = self.original_stdout

        print("Processing failed.")

        # Update status window
        if hasattr(self, "status_window"):
            self.status_window.append_text(f"\nERROR: {error_message}\n")
            self.status_window.mark_tasks_finished()

        QMessageBox.critical(
            self,
            "Analysis failed",
            f"The current-file analysis failed:\n\n{error_message}"
        )

    def create_deployment_info_Callback(self):
        dsp = default_dsp()
        bands = default_bands()
        dialog = DeploymentInfoDialog(dsp=dsp, bands=bands, fs=self.fs)
        dialog.exec()

    def eventslistselect_Callback(self):

        selected_E = self.Detected_ev_combo.currentIndex()
        event_centert=round((self.events.start[selected_E]+self.events.end[selected_E])/2)
        #posx=np.max([0, event_centert-round(self.ventana/2)]) # To avoid error due to a negative position
        #posx=np.min([posx, self.siz]) # min to avoid reading past the end of file
        #self.posx=posx
        posx = event_centert - round(self.ventana / 2)
        self.posx = clamp_posx(self, posx)

        self.slider.setValue(self.posx)
        # Update gotoseg field 
        selectedx=self.posx/self.fs  # slider pos in seconds
        self.gotoTimeLabel.setText('{:.3f}'.format(selectedx))  

        self.y=self.x[self.posx:self.posx+self.ventana]
        draw_tfr(self,self.y,self.fs,self.posx)
        # Update rectangle marker in miniature view
        [miny, maxy]=self.ax3.get_ylim()
        self.rect_miniature_view.set_xy((selectedx,miny))
        self.figMiniature.canvas.draw_idle()
    
    def m_annotation_list_Callback(self):
        selected_E = self.m_annotation_combo.currentIndex()
        event_centert=round(self.fs*(self.m_annotated_events[selected_E]["tini"]+self.m_annotated_events[selected_E]["tfin"])/2)
        posx=np.max([0, event_centert-round(self.ventana/2)]) # To avoid error due to a negative position
        posx=np.min([posx, self.siz]) # min to avoid reading past the end of file
        self.posx=posx

        self.slider.setValue(self.posx)
        # Update gotoseg field 
        selectedx=self.posx/self.fs  # slider pos in seconds
        self.gotoTimeLabel.setText('{:.3f}'.format(selectedx))  

        self.y=self.x[self.posx:self.posx+self.ventana]
        draw_tfr(self,self.y,self.fs,self.posx)
        # Update rectangle marker in miniature view
        [miny, maxy]=self.ax3.get_ylim()
        self.rect_miniature_view.set_xy((selectedx,miny))
        self.figMiniature.canvas.draw_idle()

    def analyze_deployment_mb_Callback(self):
        if dlg_analyze_deplyment_warning(self):
            deployment_folder, restart_tasks =dlg_analyze_deployment_settings(self)
            if Path(deployment_folder, "deployment_info.mat").is_file():
                print("Starting deployment analysis...")
                analyze_samaruc_deployment(self,deployment_folder, restart_tasks)
            else:
                QMessageBox.warning(self,
                "Invalid Deployment Folder",
                'The folder is not a valid deployment (missing "deployment_info.mat")'
            )

    def measure_mb_Callback(self):

        xpoint, ypoint, delta_t, delta_f=ginput_rectangle(self)
        
        if xpoint is not None: 
            rect = Rectangle(
                (np.min(xpoint), np.min(ypoint)),
                delta_t,
                delta_f,
                fill=False,
                linestyle="--",
                linewidth=1.5,
            )
            self.ax0.add_patch(rect)
            self.ax0.figure.canvas.draw_idle()

            msg = (
                f"Marker 1: {xpoint[0]:.6g} s, {ypoint[0]:.6g} Hz\n"
                f"Marker 2: {xpoint[1]:.6g} s, {ypoint[1]:.6g} Hz\n"
                f"Δt: {delta_t:.6g} s\n"
                f"Δf: {delta_f:.6g} Hz"
            )

            mbox = QMessageBox(self)
            mbox.setWindowTitle("Measure...")
            mbox.setText(msg)
            #mbox.setIcon(QMessageBox.Icon.Information)
            mbox.setIconPixmap(QPixmap(resource("SAMLab_program_icon.png")))
            mbox.setStandardButtons(QMessageBox.StandardButton.Ok)
            mbox.exec()

            # Remove the rectangle after OK pressed
            rect.remove()
            self.ax0.figure.canvas.draw_idle()

    def set_annotation_directory_Callback(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Annotation Directory")

        layout = QVBoxLayout(dialog)

        label_title = QLabel("Current annotation path:")
        label_path = QLabel(self.annotation_path or "(not set)")
        label_path.setWordWrap(True)

        layout.addWidget(label_title)
        layout.addWidget(label_path)

        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        btn_ok = QPushButton("OK")
        btn_change = QPushButton("Change")

        button_layout.addStretch()
        button_layout.addWidget(btn_change)
        button_layout.addWidget(btn_ok)

        # OK → close without changes
        btn_ok.clicked.connect(dialog.accept)

        # Change → pick folder, update path, save, update label
        def change_directory():
            folder = QFileDialog.getExistingDirectory(
                dialog,
                "Select Annotation Folder",
                self.annotation_path or ""
            )

            if folder:
                self.annotation_path = folder
                self.settings.setValue("annotation/path", folder)
                label_path.setText(folder)

        btn_change.clicked.connect(change_directory)
        dialog.exec()
    
    def get_user_Callback(self):

        username=getpass_getuser()

        msg=f"Current annotation user:\n\n{username}"
        
        mbox = QMessageBox(self)
        mbox.setWindowTitle("SAMLab User")
        mbox.setText(msg)
        #mbox.setIcon(QMessageBox.Icon.Information)
        mbox.setIconPixmap(QPixmap(resource("SAMLab_program_icon.png")))
        mbox.setStandardButtons(QMessageBox.StandardButton.Ok)
        mbox.exec()
        

    def add_manual_annotation_Callback(self):
        xpoint, ypoint, delta_t, delta_f=ginput_rectangle(self)
        
        defaultans = [
        "Dolphin whistle",
        "Dolphin echolocation clicks",
        "Fin whale calls",
        "Fish sounds",
        "Airgun arrays",
        "Sonar or acoustic deterrents",
        "Explosions",
        "Impact pile driver"
        ]

        #ma_list = sorted(set(defaultans) | set(self.manual_annotated_categories))
        ma_list = sorted(set(defaultans) | set(self.list_of_all_manual_annotated_events_in_deployment))
        
        if xpoint is not None: 
            rect = Rectangle(
                (np.min(xpoint), np.min(ypoint)),
                delta_t,
                delta_f,
                fill=False,
                linestyle="--",
                linewidth=1.5,
            )
            self.ax0.add_patch(rect)
            self.ax0.figure.canvas.draw_idle()

            m_annotation = {
                "filename": self.filename,
                "freqmin": np.min(ypoint),
                "freqmax": np.max(ypoint),
                "tini": np.min(xpoint),
                "tfin": np.max(xpoint),
                "user": getpass_getuser(),
                "date_time_annotation": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "m_event_type": ""
                }
            #dlg_input_ma(ma_list, last_val_4_manual_annotation, selection_stats)    
            m_annotation, accepted=dlg_input_ma(ma_list,1, m_annotation)   

            # Remove the rectangle after OK pressed
            rect.remove()
            self.ax0.figure.canvas.draw_idle()

            # Do not create an annotation if the dialog is empty or the user canceled
            if not accepted or not m_annotation.get("m_event_type", "").strip():
                return

            # Append the row
            self.m_annotated_events.append(m_annotation)
            # Sort events by start time
            self.m_annotated_events.sort(key=lambda e: e["tini"])

            self.m_annotation_combo.clear()
            for idx, e in enumerate(self.m_annotated_events):
                label = f"{e['tini']:.3f} s. | {e['m_event_type'] or 'Unlabeled'} |  {e['freqmin']:.1f}-{e['freqmax']:.1f}Hz | {e['user']} on {e['date_time_annotation']}"
                self.m_annotation_combo.addItem(label, userData=idx)  # store index so you can retrieve the event
            
            self.m_annotation_combo.setEnabled(True)

            # Enable the view menu to show manual annotations
            self.filer_manual_anotations_action.setEnabled(True)

            draw_tfr(self,self.y,self.fs,self.posx)
            draw_auxiliary_nav_graph(self)

    def delete_manual_annotation_Callback(self):

        # Color for to highlight the selected event
        EvSelColor=[0,0,1]

        x0, y0 = ginput_point(self)
        i_to_delete = [
            i for i, event in enumerate(self.m_annotated_events)
            if (
                x0 > event["tini"] and
                x0 < event["tfin"] and
                y0 > event["freqmin"] and
                y0 < event["freqmax"]
            )
        ]
        
        if i_to_delete:
            # Highlight the event in the TFR representation
            segments = []
            for i in i_to_delete:
                ev = self.m_annotated_events[i]

                rect = [(ev["tini"], ev["freqmin"]),
                    (ev["tfin"], ev["freqmin"]),
                    (ev["tfin"], ev["freqmax"]),
                    (ev["tini"], ev["freqmax"]),
                    (ev["tini"], ev["freqmin"]),
                ]
                segments.append(rect)

            lc = LineCollection(
                segments,
                colors=EvSelColor,
                linewidths=1.5,
                linestyles="--"
            )

            self.ax0.add_collection(lc)
            self.figSpect.canvas.draw_idle()


            quest_text = []
            for i in i_to_delete:
                ev = self.m_annotated_events[i]
                text = (f"Delete manual annotation {ev['m_event_type']} from: "f"{ev['tini']:12.2f} to {ev['tfin']:12.2f} s, done by "f"{ev['user']} on date: {ev['date_time_annotation']}?")
                quest_text.append(text)
            message = "\n".join(quest_text)
            
            reply = QMessageBox.question(self,"Delete Manual Annotation",message,QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)

            if reply == QMessageBox.StandardButton.Yes:
                # Delte from higher to lower indexes to avoid deleting the wrong one after the first deletion
                for i in sorted(i_to_delete, reverse=True):
                    del self.m_annotated_events[i]
                self.m_annotation_combo.clear()
                for idx, e in enumerate(self.m_annotated_events):
                    label = f"{e['tini']:.3f} s. | {e['m_event_type'] or 'Unlabeled'} |  {e['freqmin']:.1f}-{e['freqmax']:.1f}Hz"
                    self.m_annotation_combo.addItem(label, userData=idx)  # store index so you can retrieve the event
                if not self.m_annotated_events:
                    self.m_annotation_combo.setEnabled(False)
                    self.filer_manual_anotations_action.setEnabled(False)
                draw_tfr(self,self.y,self.fs,self.posx)
                draw_auxiliary_nav_graph(self)

            # Remove the highlight rectangle
            lc.remove()
            self.figSpect.canvas.draw_idle()


    def edit_manual_annotation_Callback(self):

        defaultans = [
        "Dolphin whistle",
        "Dolphin echolocation clicks",
        "Fin whale calls",
        "Fish sounds",
        "Airgun arrays",
        "Sonar or acoustic deterrents",
        "Explosions",
        "Impact pile driver"
        ]

        ma_list = sorted(set(defaultans) | set(self.list_of_all_manual_annotated_events_in_deployment))

        # Color for to highlight the selected event
        EvSelColor=[0,0,1]

        x0, y0 = ginput_point(self)
        i_to_edit = [
            i for i, event in enumerate(self.m_annotated_events)
            if (
                x0 > event["tini"] and
                x0 < event["tfin"] and
                y0 > event["freqmin"] and
                y0 < event["freqmax"]
            )
        ]
        
        if i_to_edit:
            ev = self.m_annotated_events[i_to_edit[0]]

            # Highlight the event in the TFR representation
            line, = self.ax0.plot(
                [ev["tini"], ev["tfin"], ev["tfin"], ev["tini"], ev["tini"]],
                [ev["freqmin"], ev["freqmin"], ev["freqmax"], ev["freqmax"], ev["freqmin"]],
                linestyle="--",linewidth=1.5,color=EvSelColor,
            )
            self.figSpect.canvas.draw_idle()

            m_annotation = {
                "filename": self.filename,
                "freqmin": ev['freqmin'],
                "freqmax": ev['freqmax'],   
                "tini": ev['tini'],
                "tfin": ev['tfin'],
                "user": getpass_getuser(),
                "date_time_annotation": ev['date_time_annotation'],
                "m_event_type": ev["m_event_type"]
                }
            #dlg_input_ma(ma_list, last_val_4_manual_annotation, selection_stats)
            idx = next((i for i, v in enumerate(ma_list) if v == m_annotation["m_event_type"]), 0) # If not exist default 0
            m_annotation, accepted=dlg_input_ma(ma_list,idx, m_annotation)
            
            # Do not create an annotation if the dialog is empty or the user canceled
            if not accepted or not m_annotation.get("m_event_type", "").strip():
                line.remove()
                self.figSpect.canvas.draw_idle()
                return

            # Append the row
            self.m_annotated_events[i_to_edit[0]] = m_annotation
            # Sort events by start time
            self.m_annotated_events.sort(key=lambda e: e["tini"])

            self.m_annotation_combo.clear()
            for idx, e in enumerate(self.m_annotated_events):
                label = f"{e['tini']:.3f} s. | {e['m_event_type'] or 'Unlabeled'} |  {e['freqmin']:.1f}-{e['freqmax']:.1f}Hz | {e['user']} on {e['date_time_annotation']}"
                self.m_annotation_combo.addItem(label, userData=idx)  # store index so you can retrieve the event
            
            self.m_annotation_combo.setEnabled(True)

            # Enable the view menu to show manual annotations
            self.filer_manual_anotations_action.setEnabled(True)

            line.remove()
            draw_tfr(self,self.y,self.fs,self.posx)
            draw_auxiliary_nav_graph(self)

    def save_manual_annotations_Callback(self):

        if self.m_annotated_events:
            file_name = self.filename
            full_path = os_path_join(self.annotation_path, self.current_annotation_file)
            
            # Convert current annotations (list[dict]) to DataFrame
            m_events_current_df = pd.DataFrame(self.m_annotated_events)

            for col in ["tini", "tfin", "freqmin", "freqmax"]:
                if col in m_events_current_df.columns:
                    m_events_current_df[col] = (
                        m_events_current_df[col]
                        .astype(str)
                        .str.replace(".", ",", regex=False)
                    )
            if os.path.isfile(full_path):

                saved_annotated_events = pd.read_csv(full_path, sep=";", dtype=str)

                # Remove previous annotated events from the current opened recording
                saved_annotated_events = saved_annotated_events[
                    saved_annotated_events["filename"] != file_name
                ].copy()

                # Append current file annotations
                m_events_to_write = pd.concat(
                    [saved_annotated_events, m_events_current_df],
                    ignore_index=True
                )
            else:
                m_events_to_write = m_events_current_df

            m_events_to_write.to_csv(full_path, sep=";", index=False)

  # ---------- View Menu actions ----------
    def filter_automatic_det_Callback(self):
        if self.events.empty:
            return

        detected_types = list(self.events["type"].dropna().astype(str).unique())
        #detected_tags = list(self.events["tag"].dropna().astype(str).unique())

        dlg = QDialog(self)
        dlg.setWindowTitle("Filter automatic detections")
        layout = QVBoxLayout(dlg)

        checkboxes = []
        currently_visible = set(detected_types if not self.eventstobeshown else self.eventstobeshown)

        for evtype in detected_types:
            cb = QCheckBox(evtype)
            cb.setChecked(evtype in currently_visible)
            layout.addWidget(cb)
            checkboxes.append((evtype, cb))

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        ok_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)

        if dlg.exec():
            selected = [evtype for evtype, cb in checkboxes if cb.isChecked()]
            self.eventstobeshown = selected
            draw_tfr(self, self.y, self.fs, self.posx)
            draw_auxiliary_nav_graph(self)

    def filter_manual_annotations_Callback(self):
        draw_tfr(self, self.y, self.fs, self.posx)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Set the application-wide icon
    app.setWindowIcon(QIcon(resource("SAMLab_program_icon.png")))
    
    # REQUIRED — defines where settings are stored
    app.setOrganizationName("UPV")
    app.setApplicationName("SAMLab")
    window = MainWindow()
    window.show()

    sys.exit(app.exec())
