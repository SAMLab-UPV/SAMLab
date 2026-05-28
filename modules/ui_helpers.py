"""
SAMLab - Submarine Acoustic Monitoring Laboratory
Copyright (C) 2026 Universitat Politècnica de València

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License v3.

Commercial licenses are available. Contact: rmiralle@dcom.upv.es
"""

from PySide6.QtWidgets import (
    QDialog, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QListView
)
from PySide6.QtCore import QObject, Signal, QStringListModel, Qt
from modules.workers import CsvScanner



# -----------------------------
# Stream redirector (for stdout in a different window)
# -----------------------------
class EmittingStream(QObject):
    text_written = Signal(str)

    def write(self, text):
        self.text_written.emit(text)

    def flush(self):
        pass

# -----------------------------
# Status Window (for stdout in a this window)
# -----------------------------
class StatusWindow(QDialog):

    cancel_requested = Signal() # To handle cancellation request from the user

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Status Tasks Output")
        self.resize(600, 400)

        self.text_box = QTextEdit()
        self.text_box.setReadOnly(True)

        # --- Add a cancel/exit button ---
        self.action_button = QPushButton("Cancel tasks")
        self.action_button.clicked.connect(self.on_button_clicked)

        layout = QVBoxLayout()
        layout.addWidget(self.text_box)
        layout.addWidget(self.action_button)
        self.setLayout(layout)

        # Internal flag
        self.tasks_running = True

    def append_text(self, text):
        cursor = self.text_box.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.text_box.setTextCursor(cursor)
        self.text_box.ensureCursorVisible()
        
    def on_button_clicked(self):
        # Handle button press depending on state...
        if self.tasks_running:
            # Notify caller (e.g., main window) that user requested cancellation
            # You might emit a custom signal here.
            self.append_text("User requested cancellation. Wait for this analysis plugin to finish...\n")
            self.cancel_requested.emit()
        else:
            # Tasks finished → close the window
            self.accept()

    def mark_tasks_finished(self):
        # Call this when all tasks are done.
        self.tasks_running = False
        self.action_button.setText("Close status window")


def ask_open_file(filename):
    
    #
    # Show a simple dialog asking whether to open a file.
    # Returns True if user chooses Yes, False if No, None if closed.
    #

    dialog = QDialog()
    dialog.setWindowTitle("Open File?")
    dialog.setModal(True)

    # Main layout
    layout = QVBoxLayout()

    # Message
    label = QLabel(f"Do you want to open the file:\n{filename}?")
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

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

class ReliableCsvDialog(QDialog):
    def __init__(self, folder, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reliable CSV Selector")
        self.resize(500, 400)

        self.layout = QVBoxLayout(self)
        self.label = QLabel(f"Scanning folder:\n{folder}")
        self.layout.addWidget(self.label)

        self.view = QListView()
        self.model = QStringListModel()
        self.view.setModel(self.model)
        self.layout.addWidget(self.view)

        self.open_btn = QPushButton("Open")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self.accept)
        self.layout.addWidget(self.open_btn)

        # Start scanning
        self.thread = CsvScanner(folder)
        self.thread.finished.connect(self.on_scan_finished)
        self.thread.start()

    def on_scan_finished(self, files):
        self.model.setStringList(files)
        self.label.setText(f"Found {len(files)} CSV files")
        self.open_btn.setEnabled(bool(files))

    def selected_file(self):
        index = self.view.currentIndex()
        if index.isValid():
            return self.model.data(index, Qt.ItemDataRole.DisplayRole)
        return None
    
def clamp_posx(self, posx: int) -> int:
    return max(0, min(posx, max(0, self.siz - self.ventana)))