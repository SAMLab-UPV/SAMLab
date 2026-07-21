from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QButtonGroup,
)


class SelectFragmentAndSaveDialog(QDialog):

    START_END = 0
    CENTER_SAMPLES = 1
    CURRENT_SELECTION = 2

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Select Fragment and Save...")
        self.setModal(True)
        self.resize(450, 180)

        main_layout = QVBoxLayout(self)

        main_layout.addWidget(QLabel("Choose one ..."))

        # ---------------------------------------------------------
        # Option 1: Start-End time
        # ---------------------------------------------------------

        self.rb_start_end = QRadioButton(
            "Start - End time in seconds:"
        )
        self.rb_start_end.setChecked(True)

        self.start_edit = QLineEdit("0")
        self.start_edit.setFixedWidth(80)

        self.end_edit = QLineEdit("300")
        self.end_edit.setFixedWidth(80)

        start_end_layout = QHBoxLayout()
        start_end_layout.addWidget(self.rb_start_end)
        start_end_layout.addWidget(self.start_edit)
        start_end_layout.addWidget(self.end_edit)
        start_end_layout.addStretch()

        main_layout.addLayout(start_end_layout)

        # ---------------------------------------------------------
        # Option 2: Click center + N samples
        # ---------------------------------------------------------

        self.rb_samples = QRadioButton(
            "Click in center and get a total of"
        )

        self.samples_edit = QLineEdit("5000")
        self.samples_edit.setFixedWidth(80)

        samples_layout = QHBoxLayout()
        samples_layout.addWidget(self.rb_samples)
        samples_layout.addWidget(self.samples_edit)
        samples_layout.addWidget(QLabel("samples"))
        samples_layout.addStretch()

        main_layout.addLayout(samples_layout)

        # ---------------------------------------------------------
        # Option 3: Current selection
        # ---------------------------------------------------------

        self.rb_selection = QRadioButton(
            "Start & End selection"
        )

        main_layout.addWidget(self.rb_selection)

        # ---------------------------------------------------------
        # Radio group
        # ---------------------------------------------------------

        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.rb_start_end)
        self.button_group.addButton(self.rb_samples)
        self.button_group.addButton(self.rb_selection)

        self.button_group.buttonClicked.connect(
            self.update_controls
        )

        # ---------------------------------------------------------
        # Buttons
        # ---------------------------------------------------------

        button_layout = QHBoxLayout()

        self.ok_button = QPushButton("Next...")
        self.cancel_button = QPushButton("Cancel")

        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        main_layout.addLayout(button_layout)

        self.update_controls()

    def update_controls(self):

        start_end_enabled = self.rb_start_end.isChecked()
        samples_enabled = self.rb_samples.isChecked()

        self.start_edit.setEnabled(start_end_enabled)
        self.end_edit.setEnabled(start_end_enabled)

        self.samples_edit.setEnabled(samples_enabled)

    def get_result(self):
        """
        Returns:

        None
            Cancel

        (START_END, start_sec, end_sec)

        (CENTER_SAMPLES, nsamples)

        (CURRENT_SELECTION,)
        """

        if self.rb_start_end.isChecked():

            try:
                start_sec = float(self.start_edit.text())
                end_sec = float(self.end_edit.text())
            except ValueError:
                return None

            return (
                self.START_END,
                start_sec,
                end_sec,
            )

        elif self.rb_samples.isChecked():

            try:
                nsamples = int(self.samples_edit.text())
            except ValueError:
                return None

            return (
                self.CENTER_SAMPLES,
                nsamples,
            )

        else:

            return (
                self.CURRENT_SELECTION,
            )

    @staticmethod
    def getSelection(parent=None):

        dlg = SelectFragmentAndSaveDialog(parent)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            return dlg.get_result()

        return None