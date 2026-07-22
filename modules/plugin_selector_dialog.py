"""
SAMLab - Submarine Acoustic Monitoring Laboratory
Copyright (C) 2026 Universitat Politècnica de València

This file is part of SAMLab.

Licensed under the GNU General Public License v3.0 (GPLv3) or later.
See the LICENSE file for details.
"""

from PySide6.QtWidgets import (
    QDialog, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QWidget,
    QScrollArea, QGroupBox, QCheckBox
)
from PySide6.QtCore import Qt

class PluginSelector(QDialog):
    def __init__(self, plugins=None, initial_states=None, parent=None):
        super().__init__(parent)

        self.plugins = plugins or []
        self.initial_states = initial_states or {}
        self.selected_plugins = {}

        self.setWindowTitle("Choose Analysis Plugins to run...")
        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Available Analysis Plugins:"))

        # ---- Select all (BINARY) ----
        self.select_all_checkbox = QCheckBox("Select all")
        self.select_all_checkbox.toggled.connect(self.select_all_toggled)
        layout.addWidget(self.select_all_checkbox)

        # ---- Scroll area ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        content_layout = QVBoxLayout(content)

        for plugin_cls in self.plugins:
            self.add_plugin_entry(content_layout, plugin_cls)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

        # ---- Buttons ----
        buttons = QHBoxLayout()
        ok_btn = QPushButton("Run Analyses")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        buttons.addStretch()
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)

    # ---------- Logic ----------

    def select_all_toggled(self, checked):
        for info in self.selected_plugins.values():
            info["enabled"].setCheckState(
                Qt.Checked if checked else Qt.Unchecked
            )
    
    def _apply_checkbox_state(self, checkbox, state):
        is_tristate = state == Qt.PartiallyChecked
        checkbox.setTristate(is_tristate)

        if is_tristate:
            checkbox.setCheckState(Qt.PartiallyChecked)
        else:
            if state == Qt.Checked:
                checkbox.setCheckState(Qt.Checked)
            else:
                checkbox.setCheckState(Qt.Unchecked)

    def add_plugin_entry(self, parent_layout, plugin_cls):
        group = QGroupBox(plugin_cls.name)
        layout = QHBoxLayout(group)

        checkbox = QCheckBox(plugin_cls.description.split("<br>", 1)[0].replace("<b>", "")
        .replace("</b>", "")+f" (v{plugin_cls.version})")
        
        state = self.initial_states.get(plugin_cls, Qt.Unchecked)
        self._apply_checkbox_state(checkbox, state)

        layout.addWidget(checkbox)

        info_icon = QLabel("ℹ️")
        info_icon.setFixedWidth(28)
        tooltip_text=f"{plugin_cls.description}\n<b>Returns:</b><ul><li>EVENTS:</li>"
        tooltip_text += "<ul>"
        for ev in plugin_cls.outputs["events"]:
            tooltip_text += (f"<li>{ev['tag']} ({ev['type']})</li>")
        tooltip_text += "</ul>"
        tooltip_text += f"<li>INDICATORS:</li>"
        tooltip_text += "<ul>"
        for ind in plugin_cls.outputs["indicators"]:
            tooltip_text += f"<li>{ind}</li>"
        tooltip_text += "</ul></ul>"
        tooltip_text += f"<b>Version:</b> {plugin_cls.version}"
        info_icon.setToolTip(tooltip_text)
        layout.addWidget(info_icon)

        self.selected_plugins[plugin_cls] = {"enabled": checkbox}
        parent_layout.addWidget(group)

    def reset_states(self, initial_states=None):
        if initial_states is not None:
            self.initial_states = initial_states

        for cls, info in self.selected_plugins.items():
            state = self.initial_states.get(cls, Qt.Unchecked)
            self._apply_checkbox_state(info["enabled"], state)
        
        # Select-all is binary, force a known state
        self.select_all_checkbox.setChecked(False)

    # Method to update the plugins initial states
    def set_plugin_states(self, states: dict):
        self.initial_states = states or {}

        for cls, info in self.selected_plugins.items():
            state = self.initial_states.get(cls, Qt.Unchecked)
            self._apply_checkbox_state(info["enabled"], state)

        self.select_all_checkbox.setChecked(False)

    # def get_selected_plugins(self):
    #     selected_plugins = {}
    #     selected_indexes = []

    #     for i, cls in enumerate(self.plugins):
    #         state = self.selected_plugins[cls]["enabled"].checkState()

    #         if state == Qt.Checked:
    #             selected_plugins[cls] = None
    #             selected_indexes.append(i)

    #     return selected_plugins, selected_indexes

    def get_selected_plugins(self):
        selected_plugins = {}
        selected_indexes = []

        for i, cls in enumerate(self.plugins):
            state = self.selected_plugins[cls]["enabled"].checkState()

            if state in (Qt.Checked, Qt.PartiallyChecked):
                selected_plugins[cls] = state
                selected_indexes.append(i)

        return selected_plugins, selected_indexes


# class PluginSelector(QDialog):
#     def __init__(self, plugins=None, parent=None):
#         super().__init__(parent)

#         self.plugins = plugins or []
#         self.selected_plugins = {}

#         self.setWindowTitle("Choose analysis tasks to run...")
#         self.build_ui()

#     def build_ui(self):
#         layout = QVBoxLayout(self)
#         layout.addWidget(QLabel("Available analyses:"))

#         # Select all checkbox
#         self.select_all_cb = QCheckBox("Select all")
#         self.select_all_cb.toggled.connect(self.select_all_toggled)
#         layout.addWidget(self.select_all_cb)

#         # Scroll area
#         scroll = QScrollArea()
#         scroll.setWidgetResizable(True)

#         content = QWidget()
#         content_layout = QVBoxLayout(content)

#         for plugin_cls in self.plugins:
#             self.add_plugin_entry(content_layout, plugin_cls)

#         content_layout.addStretch()
#         scroll.setWidget(content)
#         layout.addWidget(scroll)

#         # Buttons
#         buttons = QHBoxLayout()
#         ok_btn = QPushButton("Run Analyses")
#         ok_btn.clicked.connect(self.accept)
#         cancel_btn = QPushButton("Cancel")
#         cancel_btn.clicked.connect(self.reject)

#         buttons.addStretch()
#         buttons.addWidget(ok_btn)
#         buttons.addWidget(cancel_btn)
#         layout.addLayout(buttons)

#     def select_all_toggled(self, checked):
#         for info in self.selected_plugins.values():
#             info["enabled"].setChecked(checked)

#     def add_plugin_entry(self, parent_layout, plugin_cls):
#         group = QGroupBox(plugin_cls.name)
#         layout = QHBoxLayout(group)

#         checkbox = QCheckBox(plugin_cls.description.split("\n", 1)[0])
#         layout.addWidget(checkbox)

#         info_icon = QLabel("ℹ️")
#         info_icon.setFixedWidth(28)
#         info_icon.setToolTip(
#             f"{plugin_cls.description}\nVersion: {plugin_cls.version}"
#         )
#         layout.addWidget(info_icon)

#         self.selected_plugins[plugin_cls] = {"enabled": checkbox}
#         parent_layout.addWidget(group)

#     def get_selected_plugins(self):
#         # Returns:
#         #   selected_plugins: dict {plugin_cls: params}
#         #   selected_indexes: list of integer indices in self.plugins
#         #
#         selected_plugins = {}
#         selected_indexes = []

#         for i, cls in enumerate(self.plugins):
#             if self.selected_plugins[cls]["enabled"].isChecked():
#                 selected_plugins[cls] = None
#                 selected_indexes.append(i)

#         return selected_plugins, selected_indexes

        #     if info["enabled"].isChecked(): # In hte future maybe I need passing values to some tasks.
        #         #params = {k: v.text() for k, v in info["params"].items()}
        #         #selected_plugins[cls] = params
        #         selected_indexes.append(i)

        # return selected_plugins, selected_indexes