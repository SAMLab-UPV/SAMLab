"""
SAMLab - Submarine Acoustic Monitoring Laboratory
Copyright (C) 2026 Universitat Politècnica de València

This file is part of SAMLab.

Licensed under the GNU Affero General Public License v3.0 or later.
See the LICENSE file for details.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from scipy.io import loadmat, savemat
from PySide6.QtCore import Qt, QDate
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from modules.models import DSP,Bands


def _mat_struct_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a scipy MATLAB struct-ish object to a simple dict."""
    if isinstance(obj, np.ndarray) and obj.size == 1:
        obj = obj.item()
    if hasattr(obj, "_fieldnames"):
        return {name: getattr(obj, name) for name in obj._fieldnames}
    if isinstance(obj, np.void) and obj.dtype.names:
        return {name: obj[name] for name in obj.dtype.names}
    if isinstance(obj, dict):
        return obj
    return {}


def _scalar(value: Any, default: Any = None) -> Any:
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    item = arr.flat[0]
    if isinstance(item, np.ndarray):
        return _scalar(item, default)
    try:
        return item.item()
    except AttributeError:
        return item


def _float_list(value: Any) -> list[float]:
    arr = np.asarray(value, dtype=float).squeeze()
    if arr.ndim == 0:
        return [float(arr)]
    return [float(x) for x in arr.tolist()]


def _str_from_mat(value: Any) -> str:
    if isinstance(value, str):
        return value
    arr = np.asarray(value)
    if arr.dtype.kind in {"U", "S"}:
        if arr.ndim == 0:
            return str(arr.item())
        return "".join(str(x) for x in arr.flat)
    item = _scalar(value, "")
    return str(item) if item is not None else ""


class DeploymentInfoDialog(QDialog):
    def __init__(
        self,
        dsp: DSP,
        bands: Bands,
        fs: int | float = 192000,
        parent: Optional[QWidget] = None,
    ) -> None:
    
        super().__init__(parent)
        self.setWindowTitle('CREATE DEPLOYMENT INFO FILE...')
        self.resize(700, 450)

        self.dsp = self._coerce_dsp(dsp)
        self.bands = self._coerce_bands(bands)
        self.fs = fs

        self._build_ui()
        self._populate_defaults()

    @staticmethod
    def _coerce_dsp(dsp: DSP) -> DSP:
        if not isinstance(dsp, DSP):
            raise TypeError(
                "DeploymentInfoDialog requires dsp to be a DSP instance. "
                f"Got {type(dsp).__name__}."
            )
        return dsp

    @staticmethod
    def _coerce_bands(bands: Bands) -> Bands:
        if not isinstance(bands, Bands):
            raise TypeError(
                "DeploymentInfoDialog requires bands to be a Bands instance. "
                f"Got {type(bands).__name__}."
            )
        return bands

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        calibration = QGroupBox("Calibration")
        grid = QGridLayout(calibration)

        grid.addWidget(QLabel("Sensitivity [dB re 1V/uPa]"), 0, 0, 1, 4, alignment=Qt.AlignCenter)
        band_names = ["63 Hz", "125 Hz", "2 kHz", "5 kHz"]
        self.band_labels: list[QLabel] = []
        self.sh_edits: list[QLineEdit] = []
        for col, name in enumerate(band_names):
            lbl = QLabel(name)
            lbl.setAlignment(Qt.AlignCenter)
            self.band_labels.append(lbl)
            grid.addWidget(lbl, 1, col)
            edit = QLineEdit()
            edit.setFixedWidth(70)
            self.sh_edits.append(edit)
            grid.addWidget(edit, 2, col)

        grid.addWidget(QLabel("DSP Gain [dB]:"), 1, 4)
        self.dsp_gain_edit = QLineEdit()
        grid.addWidget(self.dsp_gain_edit, 2, 4)

        grid.addWidget(QLabel("Number of bits:"), 1, 5)
        self.dsp_nbits_edit = QLineEdit()
        grid.addWidget(self.dsp_nbits_edit, 2, 5)

        self.mono_radiobtn = QRadioButton("Mono")
        grid.addWidget(self.mono_radiobtn, 3, 1)
        grid.addWidget(QLabel("Sampling frequency [Hz]:"), 3, 2, 1, 2)
        self.samp_freq_edit = QLineEdit()
        grid.addWidget(self.samp_freq_edit, 3, 4)

        main_layout.addWidget(calibration)

        datetime_layout = QGridLayout()
        datetime_layout.addWidget(QLabel("Deployment date:"), 0, 0)
        datetime_layout.addWidget(QLabel("Deployment started at:"), 0, 1, 1, 3)
        datetime_layout.addWidget(QLabel("Hour:"), 1, 1)
        datetime_layout.addWidget(QLabel("Min:"), 1, 2)
        datetime_layout.addWidget(QLabel("Sec.:"), 1, 3)

        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("dd-MMM-yyyy")
        self.date_edit.setDate(QDate.currentDate())
        datetime_layout.addWidget(self.date_edit, 2, 0)

        self.popup_hour = QComboBox()
        self.popup_minutes = QComboBox()
        self.popup_seconds = QComboBox()
        self.popup_hour.addItems([f"{i:02d}" for i in range(24)])
        self.popup_minutes.addItems([f"{i:02d}" for i in range(60)])
        self.popup_seconds.addItems([f"{i:02d}" for i in range(60)])
        datetime_layout.addWidget(self.popup_hour, 2, 1)
        datetime_layout.addWidget(self.popup_minutes, 2, 2)
        datetime_layout.addWidget(self.popup_seconds, 2, 3)
        main_layout.addLayout(datetime_layout)

        main_layout.addWidget(QLabel("Comments about the deployment..."))
        self.comments_edit = QTextEdit()
        self.comments_edit.setPlainText(
            "Don't forget to include GPS coordinates, hydrophone brand and model "
            "and any other important information."
        )
        main_layout.addWidget(self.comments_edit)

        button_layout = QHBoxLayout()
        self.open_btn = QPushButton('Open "deployment_info.mat"')
        self.save_btn = QPushButton('Save "deployment_info.mat"')
        self.cancel_btn = QPushButton("Cancel")
        button_layout.addWidget(self.open_btn)
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(button_layout)

        self.open_btn.clicked.connect(self.open_deployment_info)
        self.save_btn.clicked.connect(self.save_deployment_info)
        self.cancel_btn.clicked.connect(self.reject)

    def _populate_defaults(self) -> None:
        self.dsp_gain_edit.setText(str(self.dsp.gain))
        self.dsp_nbits_edit.setText(str(self.dsp.nbits))
        self.samp_freq_edit.setText(str(self.fs))

        # MATLAB formula: G=10^(3/10); fm=1000*G.^((bands.number-30)/3)
        g = 10 ** (3 / 10)
        fm = [1000 * g ** ((num - 30) / 3) for num in self.bands.number]
        for i, edit in enumerate(self.sh_edits):
            if i < len(fm):
                self.band_labels[i].setText(f"{round(fm[i])} Hz")
            if i < len(self.bands.sh):
                edit.setText(str(self.bands.sh[i]))

    def _collect_values(self) -> Tuple[DSP, Bands, datetime, float, bool, str]:
        dsp = DSP(
            gain=float(self.dsp_gain_edit.text()),
            nbits=int(float(self.dsp_nbits_edit.text())),   
        )

        sh = [float(edit.text()) for edit in self.sh_edits]
        bands_num_txt = [18, 21, 33, 37]
        bands_label_txt = [
            "Mean SPL 63 Hz band[dB re 1\\muPa^2]",
            "Mean SPL 125 Hz band[dB re 1\\muPa^2]",
            "Mean SPL 2000_Hz band[dB re 1\\muPa^2]",
            "Mean SPL 5000_Hz band[dB re 1\\muPa^2]",
        ]
        bands = Bands(
            sh=sh,
            number=bands_num_txt[: len(sh)],
            label=bands_label_txt[: len(sh)],
        )

        qdate = self.date_edit.date()
        ddate = datetime(
            qdate.year(),
            qdate.month(),
            qdate.day(),
            int(self.popup_hour.currentText()),
            int(self.popup_minutes.currentText()),
            int(self.popup_seconds.currentText()),
        )
        fs = float(self.samp_freq_edit.text())
        mono = self.mono_radiobtn.isChecked()
        comments = self.comments_edit.toPlainText()
        return dsp, bands, ddate, fs, mono, comments

    def save_deployment_info(self) -> None:
        try:
            dsp, bands, ddate, fs, mono, comments = self._collect_values()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid input", f"Please check numeric fields.\n\n{exc}")
            return

        folder = QFileDialog.getExistingDirectory(self, 'Choose folder for saving the "deployment_info.mat" file')
        if not folder:
            QMessageBox.warning(self, "Information", 'ERROR: Not valid path. "deployment_info.mat" was not created!')
            self.reject()
            return

        path = Path(folder) / "deployment_info.mat"
        savemat(
            path,
            {
                "dsp": {"gain": dsp.gain, "nbits": dsp.nbits},
                "bands": {"sh": np.array(bands.sh), "number": np.array(bands.number), "label": np.array(bands.label, dtype=object)},
                # Stored as an ISO string for robust Python/MATLAB interchange.
                # MATLAB can parse this with datetime(ddate, 'InputFormat', 'yyyy-MM-dd HH:mm:ss').
                "ddate": ddate.strftime("%Y-%m-%d %H:%M:%S"),
                "fs": fs,
                "mono": int(mono),
                "comments": comments,
            },
        )
        self.dsp, self.bands, self.fs = dsp, bands, fs
        QMessageBox.information(self, "Information", '"deployment_info.mat" successfully created!')
        self.accept()

    def open_deployment_info(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(self, 'Choose the "deployment_info.mat" file', "", "MAT-files (*.mat)")
        if not file_name:
            return

        dinfo = loadmat(file_name, squeeze_me=True, struct_as_record=False)
        dsp_dict = _mat_struct_to_dict(dinfo.get("dsp", {}))
        bands_dict = _mat_struct_to_dict(dinfo.get("bands", {}))

        self.dsp_nbits_edit.setText(str(_scalar(dsp_dict.get("nbits"), self.dsp.nbits)))
        self.dsp_gain_edit.setText(str(_scalar(dsp_dict.get("gain"), self.dsp.gain)))

        sh = _float_list(bands_dict.get("sh", self.bands.sh))
        for i, value in enumerate(sh[:4]):
            self.sh_edits[i].setText(str(value))
        self.bands.sh = sh

        self.comments_edit.setPlainText(_str_from_mat(dinfo.get("comments", "")))
        self.mono_radiobtn.setChecked(bool(_scalar(dinfo.get("mono", 0), 0)))
        self.samp_freq_edit.setText(str(_scalar(dinfo.get("fs", 192000), 192000)))

        ddate_text = _str_from_mat(dinfo.get("ddate", ""))
        parsed = self._parse_datetime(ddate_text)
        if parsed is not None:
            self.date_edit.setDate(QDate(parsed.year, parsed.month, parsed.day))
            self.popup_hour.setCurrentIndex(parsed.hour)
            self.popup_minutes.setCurrentIndex(parsed.minute)
            self.popup_seconds.setCurrentIndex(parsed.second)

    @staticmethod
    def _parse_datetime(value: str) -> Optional[datetime]:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%d-%b-%Y %H:%M:%S", "%d-%b-%Y"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                pass
        return None