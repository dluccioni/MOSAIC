# -----------------------------------------------------------------------------
# Scan Wizard
# -----------------------------------------------------------------------------
"""
Multi-step wizard for configuring nD scans.

Steps:
1. Select motors to scan
2. Define ranges and step sizes
3. Configure output options
4. Advanced parameters
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWizard, QWizardPage,
    QLabel, QListWidget, QListWidgetItem, QDoubleSpinBox, QSpinBox,
    QPushButton, QLineEdit, QCheckBox, QComboBox, QGroupBox,
    QFormLayout, QFileDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QMessageBox, QWidget,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gui.state import SimulationState
from gui.workers.scan_worker import ScanConfig, ScanMode


class MotorSelectionPage(QWizardPage):
    """Page for selecting motors to scan."""

    def __init__(self, state: SimulationState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setTitle("Select Motors")
        self.setSubTitle("Choose which motors to scan")
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Available motors
        layout.addWidget(QLabel("Available Motors:"))
        self.available_list = QListWidget()
        self.available_list.setSelectionMode(QAbstractItemView.MultiSelection)
        layout.addWidget(self.available_list)

        # Selected motors
        layout.addWidget(QLabel("Selected for Scan:"))
        self.selected_list = QListWidget()
        layout.addWidget(self.selected_list)

        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add →")
        add_btn.clicked.connect(self._add_motor)
        remove_btn = QPushButton("← Remove")
        remove_btn.clicked.connect(self._remove_motor)
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(remove_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def initializePage(self):
        """Called when page is shown."""
        self.available_list.clear()
        stage = self.state.stage
        detector = self.state.detector

        # Add stage motors
        if stage is not None:
            if hasattr(stage, 'motors'):
                for name in stage.motors.keys():
                    self.available_list.addItem(name)
            elif hasattr(stage, '_motor_name') and stage._motor_name is not None:
                for name in stage._motor_name:
                    self.available_list.addItem(str(name))

        # Add detector axes
        if detector is not None:
            for axis in ["two_theta", "eta", "distance"]:
                self.available_list.addItem(axis)

    def _add_motor(self):
        for item in self.available_list.selectedItems():
            # Check if already added
            existing = [self.selected_list.item(i).text()
                       for i in range(self.selected_list.count())]
            if item.text() not in existing:
                self.selected_list.addItem(item.text())

    def _remove_motor(self):
        for item in self.selected_list.selectedItems():
            self.selected_list.takeItem(self.selected_list.row(item))

    def get_selected_motors(self) -> List[str]:
        return [self.selected_list.item(i).text()
                for i in range(self.selected_list.count())]

    def validatePage(self) -> bool:
        if self.selected_list.count() == 0:
            QMessageBox.warning(self, "No Motors", "Please select at least one motor to scan.")
            return False
        return True


class RangeDefinitionPage(QWizardPage):
    """Page for defining scan ranges."""

    def __init__(self, state: SimulationState, motor_page: MotorSelectionPage, parent=None):
        super().__init__(parent)
        self.state = state
        self.motor_page = motor_page
        self.setTitle("Define Ranges")
        self.setSubTitle("Set the scan range for each motor")
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Motor", "Start", "End", "Steps"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(self.table)

    def initializePage(self):
        """Called when page is shown."""
        motors = self.motor_page.get_selected_motors()
        self.table.setRowCount(len(motors))

        for i, motor in enumerate(motors):
            self.table.setItem(i, 0, QTableWidgetItem(motor))

            start = QDoubleSpinBox()
            start.setDecimals(4)
            start.setRange(-1e9, 1e9)
            start.setValue(0)
            self.table.setCellWidget(i, 1, start)

            end = QDoubleSpinBox()
            end.setDecimals(4)
            end.setRange(-1e9, 1e9)
            end.setValue(10)
            self.table.setCellWidget(i, 2, end)

            steps = QSpinBox()
            steps.setRange(2, 10000)
            steps.setValue(11)
            self.table.setCellWidget(i, 3, steps)

    def get_ranges(self) -> List[tuple]:
        ranges = []
        for i in range(self.table.rowCount()):
            start = self.table.cellWidget(i, 1).value()
            end = self.table.cellWidget(i, 2).value()
            ranges.append((start, end))
        return ranges

    def get_steps(self) -> List[int]:
        steps = []
        for i in range(self.table.rowCount()):
            steps.append(self.table.cellWidget(i, 3).value())
        return steps


class OutputOptionsPage(QWizardPage):
    """Page for configuring output options."""

    def __init__(self, state: SimulationState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setTitle("Output Options")
        self.setSubTitle("Configure how results are saved")
        self._setup_ui()

    def _setup_ui(self):
        layout = QFormLayout(self)

        # Scan mode
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Live Updates", "live")
        self.mode_combo.addItem("Batch Mode", "batch")
        layout.addRow("Mode:", self.mode_combo)

        # Save intermediate
        self.save_intermediate = QCheckBox("Save intermediate images")
        self.save_intermediate.setChecked(True)
        layout.addRow("", self.save_intermediate)

        # Generate summary
        self.generate_summary = QCheckBox("Generate summary plots")
        self.generate_summary.setChecked(True)
        layout.addRow("", self.generate_summary)

        # Output directory
        dir_widget = QWidget()
        dir_layout = QHBoxLayout(dir_widget)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("Select output directory...")
        dir_layout.addWidget(self.output_dir, 1)
        browse_btn = QPushButton("...")
        browse_btn.clicked.connect(self._browse_dir)
        dir_layout.addWidget(browse_btn)
        layout.addRow("Directory:", dir_widget)

        # Plot prefix
        self.plot_prefix = QLineEdit("scan")
        layout.addRow("Prefix:", self.plot_prefix)

    def _browse_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir.setText(dir_path)


class ScanWizard(QWizard):
    """Wizard for configuring nD scans."""

    scan_configured = Signal(object)  # ScanConfig

    def __init__(self, state: SimulationState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("Scan Wizard")
        self.setMinimumSize(600, 500)

        # Create pages
        self.motor_page = MotorSelectionPage(state)
        self.range_page = RangeDefinitionPage(state, self.motor_page)
        self.output_page = OutputOptionsPage(state)

        self.addPage(self.motor_page)
        self.addPage(self.range_page)
        self.addPage(self.output_page)

        self.finished.connect(self._on_finished)

    def _on_finished(self, result):
        if result == QWizard.Accepted:
            config = ScanConfig(
                motors=self.motor_page.get_selected_motors(),
                ranges=self.range_page.get_ranges(),
                steps=self.range_page.get_steps(),
                mode=ScanMode.LIVE if self.output_page.mode_combo.currentData() == "live" else ScanMode.BATCH,
                save_intermediate=self.output_page.save_intermediate.isChecked(),
                output_directory=self.output_page.output_dir.text(),
                plot_prefix=self.output_page.plot_prefix.text(),
                use_gpu=True
            )
            self.scan_configured.emit(config)

    def get_config(self) -> ScanConfig:
        """Get the configured scan parameters."""
        return ScanConfig(
            motors=self.motor_page.get_selected_motors(),
            ranges=self.range_page.get_ranges(),
            steps=self.range_page.get_steps(),
            mode=ScanMode.LIVE if self.output_page.mode_combo.currentData() == "live" else ScanMode.BATCH,
            save_intermediate=self.output_page.save_intermediate.isChecked(),
            output_directory=self.output_page.output_dir.text(),
            plot_prefix=self.output_page.plot_prefix.text(),
            use_gpu=True
        )
