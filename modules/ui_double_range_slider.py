"""
SAMLab - Submarine Acoustic Monitoring Laboratory
Copyright (C) 2026 Universitat Politècnica de València

This file is part of SAMLab.

Licensed under the GNU Affero General Public License v3.0 or later.
See the LICENSE file for details.
"""

from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import QPainter,QColor,QPen,QPainterPath
from PySide6.QtWidgets import QWidget
from .ui_constants import LABEL_FONT_SIZE
import numpy as np
import math

class DoubleVerticalRangeSlider(QWidget):
    rangeChanged = Signal(float, float)

    def __init__(self, minimum=0, maximum=1, low=0.0, high=1.0, parent=None):
        super().__init__(parent)

        self.minimum = minimum
        self.maximum = maximum

        self.low = float(low)
        self.high = float(high)

        self._dragging = None

        self.track_x = 15
        self.margin = 10

        #self.setMinimumWidth(120)
        self.setFixedWidth(70)
        self.setMinimumHeight(500)

    def setRange(self,minimum,maximum,low=None,high=None):

    #    Update slider limits after loading a new image.
    #
    #    If low/high are omitted:
    #        low  = minimum
    #        high = maximum

        minimum = float(minimum)
        maximum = float(maximum)

        # Prevent zero-width ranges
        if maximum <= minimum:
            eps = max(abs(minimum) * 1e-6, 1e-6)
            maximum = minimum + eps

        self.minimum = minimum
        self.maximum = maximum

        if low is None:
            low = self.minimum

        if high is None:
            high = self.maximum

        self.low = float(low)
        self.high = float(high)

        self.rangeChanged.emit(self.low, self.high)
        self.update()
    
    def value_to_y(self, value):
        usable = self.height() - 2 * self.margin

        t = (value - self.minimum) / (
            self.maximum - self.minimum
        )

        return (
            self.height()
            - self.margin
            - t * usable
        )

    def y_to_value(self, y):
        usable = self.height() - 2 * self.margin

        y = max(
            self.margin,
            min(y, self.height() - self.margin),
        )

        t = (
            self.height()
            - self.margin
            - y
        ) / usable

        return (
            self.minimum
            + t * (self.maximum - self.minimum)
        )

    def draw_handle(self, painter, x, y):
        #w = 12
        #h = 8
        w = 7  # Smaller knobs
        h = 5

        points = [
            QPointF(x - w,     y - h),
            QPointF(x + 2,     y - h),
            QPointF(x + h + 2, y),
            QPointF(x + 2,     y + h),
            QPointF(x - w,     y + h)    
        ]

        painter.setPen(QPen(QColor("#8090A0"), 2))
        painter.setBrush(QColor("white"))
        painter.drawPolygon(points)

    def paintEvent(self, event):
        painter = QPainter(self)

        painter.setRenderHint(
            QPainter.Antialiasing
        )

        y_low = self.value_to_y(self.low)
        y_high = self.value_to_y(self.high)

        # Main track

        painter.setPen(
            QPen(QColor("#c0c0c0"), 3)
        )

        painter.drawLine(
            self.track_x,
            self.margin,
            self.track_x,
            self.height() - self.margin,
        )

        # Get the major ticks step limiting the maximum number of ticks
        tick_step = self.nice_tick_step(max_ticks=8)

        first_tick = math.ceil(self.minimum / tick_step) * tick_step
        last_tick = math.floor(self.maximum / tick_step) * tick_step

        major_values = np.arange(
            first_tick,
            last_tick + tick_step * 0.5,
            tick_step
        )

        painter.setPen(QPen(Qt.black, 1))
        font = painter.font()
        font.setPointSize(LABEL_FONT_SIZE)
        painter.setFont(font)

        for value in major_values:
            y = self.value_to_y(value)

            painter.drawLine(
                self.track_x + 20,
                int(y),
                self.track_x + 30,
                int(y),
            )

            painter.drawText(
                self.track_x + 35,
                int(y + 5),
                f"{value:g}",
            )

        
        # major_values = np.arange(
        #     np.floor(self.minimum),
        #     np.ceil(self.maximum) + tick_step,
        #     tick_step
        # )

        # painter.setPen(
        #     QPen(Qt.black, 1)
        # )

        # font = painter.font()
        # font.setPointSize(LABEL_FONT_SIZE)
        # painter.setFont(font)

        
        # for value in major_values:
        #     y = self.value_to_y(value)

        #     painter.drawLine(
        #         self.track_x + 20,
        #         int(y),
        #         self.track_x + 30,
        #         int(y),
        #     )

        #     painter.drawText(
        #         self.track_x + 35,
        #         int(y + 5),
        #         f"{value:.0f}",
        #     )

        
        # Minor ticks

        # if tick_step <= 10:
        #     minor_step = tick_step / 5

        #     minor_values = np.arange(
        #         first_tick,
        #         last_tick + minor_step * 0.5,
        #         minor_step
        #     )

        #     for value in minor_values:
        #         if abs((value / tick_step) - round(value / tick_step)) < 1e-6:
        #             continue

        #         y = self.value_to_y(value)

        #         painter.drawLine(
        #             self.track_x + 22,
        #             int(y),
        #             self.track_x + 27,
        #             int(y),
        #         )

        minor_values = np.arange(
        np.floor(self.minimum),
        np.ceil(self.maximum) + 1,1)

        for value in minor_values:
            if value % tick_step == 0:
                continue

            y = self.value_to_y(value)

            painter.drawLine(
                self.track_x + 22,
                int(y),
                self.track_x + 27,
                int(y),
            )


        # Handles

        self.draw_handle(
            painter,
            self.track_x,
            y_low,
        )

        self.draw_handle(
            painter,
            self.track_x,
            y_high,
        )

    def mousePressEvent(self, event):

        y = event.position().y()

        y_low = self.value_to_y(self.low)
        y_high = self.value_to_y(self.high)

        if abs(y - y_low) < abs(y - y_high):
            self._dragging = "low"
        else:
            self._dragging = "high"

    def mouseMoveEvent(self, event):
        if not self._dragging:
            return

        #value = round(self.y_to_value(event.position().y()))
        value = self.y_to_value(event.position().y())

        if self._dragging == "low":
            self.low = min(value, self.high)
        else:
            self.high = max(value, self.low)

        self.rangeChanged.emit(self.low, self.high)
        self.update()

    def mouseReleaseEvent(self, event):
        self._dragging = None

    def nice_tick_step(self, max_ticks=8):
        span = self.maximum - self.minimum
        if span <= 0:
            return 1

        raw_step = span / max_ticks
        magnitude = 10 ** math.floor(math.log10(raw_step))
        residual = raw_step / magnitude

        if residual <= 1:
            nice = 1
        elif residual <= 2:
            nice = 2
        elif residual <= 5:
            nice = 5
        else:
            nice = 10

        return nice * magnitude
