"""
SAMLab - Submarine Acoustic Monitoring Laboratory
Copyright (C) 2026 Universitat Politècnica de València

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License v3.

Commercial licenses are available. Contact: rmiralle@dcom.upv.es
"""

from typing import Any, Dict, List, Optional
from PySide6.QtCore import Qt, QEvent, QPointF
from PySide6.QtGui import QCursor, QMouseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QCompleter,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QApplication
)
from numpy import array
from matplotlib.widgets import RectangleSelector
import time


def dlg_input_ma(ma_list: List[str],last_val_4_manual_annotation: int,selection_stats: Dict[str, Any],parent=None):
    #
    # Shows a dialog for manual annotation of events.
    #
    # Args:
    #    ma_list: List of posible Manual Annotated Events categories for quick selection
    #    last_val_4_manual_annotation: Last value used when annotating 
    #    selection_stats: A dict conatinig
    #                       filename: ""
    #                       freqmin: Minimum frequency
    #                       freqmax: Maximum frequency
    #                       tini: Event start time
    #                       tfin: Event end time
    #                       user: User that annotate the event
    #                       date_time_annotation: Date and time of the annotation
    #                       m_event_type: Manual event type (category) of the annotation
    #
    # Returns:
    #    Updated selection_stats dict (evtype set to [string] on OK, '' on Cancel;
    #    tend set to -1 if "All file duration" checked on OK).
    #

    n = len(ma_list)
    if n == 0:
        initial_index = -1
    else:
        initial_index = max(0, min(int(last_val_4_manual_annotation), n - 1)) # Ensure it's a valid index

    # -------- values from selection_stats --------
    tini = float(selection_stats.get("tini", 0.0))
    tend = float(selection_stats.get("tfin", 0.0))
    dt = tend - tini

    minf = float(selection_stats.get("freqmin", 0.0))
    maxf = float(selection_stats.get("freqmax", 0.0))
    author = str(selection_stats.get("user", ""))
    date = str(selection_stats.get("date_time_annotation", ""))

    def make_line3(all_file: bool) -> str:
        if all_file:
            return f"Delta f.: {maxf - minf:.2f} Hz, Duration: All file"
        return f"Delta f.: {maxf - minf:.2f} Hz, Duration: {dt:.3f} s."

    # -------- dialog --------
    dlg = QDialog(parent)
    dlg.setWindowTitle("EVENT MANUAL ANNOTATION...")
    dlg.setModal(True)
    #dlg.setFixedSize(400, 250)

    # -------- widgets --------
    lbl1 = QLabel(f"User: <b>{author}</b> on {date}")
    lbl1.setAlignment(Qt.AlignmentFlag.AlignCenter)

    lbl2 = QLabel(
        f"Central freq.: {(maxf + minf) / 2:.2f} Hz, between [{minf:.2f} - {maxf:.2f}] Hz."
    )
    lbl2.setAlignment(Qt.AlignmentFlag.AlignLeft)

    lbl3 = QLabel(make_line3(False))
    lbl3.setAlignment(Qt.AlignmentFlag.AlignLeft)

    chk_all = QCheckBox("All file duration")

    lbl_choose = QLabel("Choose an event from the drop-down...")
    combo = QComboBox()
    combo.addItems(ma_list)
    if initial_index >= 0:
        combo.setCurrentIndex(initial_index)

    lbl_filter = QLabel("... or use the textbox below to filter/ add a new category:")
    edit = QLineEdit(combo.currentText() if initial_index >= 0 else "")

    # Optional but nice: completer that matches from the start (like MATLAB strncmpi)
    completer = QCompleter(ma_list)
    completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
    completer.setFilterMode(Qt.MatchFlag.MatchStartsWith)
    edit.setCompleter(completer)

    btn_ok = QPushButton("OK")
    btn_cancel = QPushButton("Cancel")

    # -------- filtering behavior (MATLAB-like) --------
    master = list(ma_list)

    def set_combo_items(items: List[str]) -> None:
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(items)
        combo.setCurrentIndex(0 if items else -1)
        combo.blockSignals(False)

    def apply_filter(prefix: str) -> None:
        prefix = prefix or ""
        if prefix.strip() == "":
            combo.setVisible(True)
            set_combo_items(master)
            return

        filtered = [s for s in master if s.lower().startswith(prefix.lower())]
        if filtered:
            combo.setVisible(True)
            set_combo_items(filtered)
        else:
            combo.setVisible(False)

    def on_combo_changed(_: int) -> None:
        if combo.isVisible() and combo.currentIndex() >= 0:
            edit.blockSignals(True)
            edit.setText(combo.currentText())
            edit.blockSignals(False)

    combo.currentIndexChanged.connect(on_combo_changed)

    def on_text_changed(text: str) -> None:
        apply_filter(text)

    edit.textChanged.connect(on_text_changed)

    def on_chk_toggled(checked: bool) -> None:
        lbl3.setText(make_line3(checked))

    chk_all.toggled.connect(on_chk_toggled)

    # Enter key behavior
    edit.returnPressed.connect(lambda: btn_ok.click())

    # OK / Cancel
    def accept_ok() -> None:
        selection_stats["m_event_type"] = edit.text()  # Pythonic: string
        # If you need MATLAB-like cell array, use:
        # selection_stats["m_event_type"] = [edit.text()]
        if chk_all.isChecked():
            selection_stats["tfin"] = -1
        dlg.accept()

    def cancel() -> None:
        selection_stats["m_event_type"] = ""
        dlg.reject()

    btn_ok.clicked.connect(accept_ok)
    btn_cancel.clicked.connect(cancel)

    # If user closes the window (X), treat like Cancel
    dlg.rejected.connect(lambda: selection_stats.__setitem__("m_event_type", ""))

    # -------- layout --------
    root = QVBoxLayout()
    root.setContentsMargins(20, 20, 20, 20)
    root.setSpacing(8)

    root.addWidget(lbl1)
    root.addWidget(lbl2)

    row3 = QHBoxLayout()
    row3.addWidget(lbl3, 1)
    row3.addWidget(chk_all, 0)
    root.addLayout(row3)

    root.addSpacing(6)
    root.addWidget(lbl_choose)
    root.addWidget(combo)
    root.addWidget(lbl_filter)
    root.addWidget(edit)

    btn_row = QHBoxLayout()
    btn_row.addStretch(1)
    btn_row.addWidget(btn_ok)
    btn_row.addWidget(btn_cancel)
    btn_row.addStretch(1)

    root.addStretch(1)
    root.addLayout(btn_row)

    dlg.setLayout(root)

    # Focus OK like MATLAB
    btn_ok.setFocus(Qt.FocusReason.OtherFocusReason)

    result = dlg.exec()
    return selection_stats, (result == QDialog.DialogCode.Accepted)


def ginput_rectangle(self):
    canvas = self.figSpect.canvas
    canvas.setMouseTracking(True)

    old_cursor = canvas.cursor()
    canvas.unsetCursor()
    canvas.setCursor(Qt.CursorShape.CrossCursor)

    canvas.setFocus(Qt.FocusReason.OtherFocusReason)
    canvas.activateWindow()

    pos = canvas.mapFromGlobal(QCursor.pos())
    evt = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(pos),
        QPointF(canvas.mapToGlobal(pos)),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier
    )
    QApplication.sendEvent(canvas, evt)
    QApplication.processEvents()

    xpoint = None
    ypoint = None
    delta_t = None
    delta_f = None
    done = False

    def on_select(eclick, erelease):
        nonlocal xpoint, ypoint, delta_t, delta_f, done

        xpoint = array([eclick.xdata, erelease.xdata], dtype=float)
        ypoint = array([eclick.ydata, erelease.ydata], dtype=float)
        delta_t = abs(xpoint[1] - xpoint[0])
        delta_f = abs(ypoint[1] - ypoint[0])
        done = True

    rect_selector = RectangleSelector(
        self.ax0,
        on_select,
        useblit=True,
        button=[1],  # left mouse button
        minspanx=0.001, # Minimum span in x direction (time)
        minspany=0.1, # Minimum span in y direction (frequency)
        spancoords='data',
        interactive=False
    )

    try:
        while not done:
            QApplication.processEvents()
            time.sleep(0.01)
    finally:
        rect_selector.set_active(False)
        canvas.setCursor(old_cursor)
        canvas.draw_idle()

    return xpoint, ypoint, delta_t, delta_f


def ginput_point(self):
    canvas = self.figSpect.canvas
    canvas.setMouseTracking(True)

    old_cursor = canvas.cursor()
    canvas.unsetCursor()
    canvas.setCursor(Qt.CursorShape.CrossCursor)

    canvas.setFocus(Qt.FocusReason.OtherFocusReason)
    canvas.activateWindow()

    xpoint = None
    ypoint = None
    done = False

    def on_click(event):
        nonlocal xpoint, ypoint, done

        # Only accept left-clicks inside the target axes
        if event.inaxes != self.ax0:
            return

        if event.button != 1:
            return

        if event.xdata is None or event.ydata is None:
            return

        xpoint = float(event.xdata)
        ypoint = float(event.ydata)

        done = True

    cid = canvas.mpl_connect("button_press_event", on_click)

    try:
        while not done:
            QApplication.processEvents()
            time.sleep(0.01)
    finally:
        canvas.mpl_disconnect(cid)
        canvas.setCursor(old_cursor)
        canvas.draw_idle()

    return xpoint, ypoint