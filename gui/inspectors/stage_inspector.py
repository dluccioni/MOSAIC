# -----------------------------------------------------------------------------
# Stage Inspector
# -----------------------------------------------------------------------------
"""
Inspector panel for stage/goniometer configuration.

Provides controls for:
- Motor management and values
- Absolute/relative movement
- Rotation matrix display
"""

import sys
from pathlib import Path
import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLabel, QDoubleSpinBox, QPushButton, QComboBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QRadioButton, QButtonGroup,
    QLineEdit, QFileDialog,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gui.state import SimulationState
from gui.panels.inspector import InspectorPanel


class StageInspector(InspectorPanel):
    """Inspector for stage/goniometer configuration."""

    stage_created = Signal(object)
    motor_moved = Signal(str, float)

    def __init__(self, state: SimulationState, parent=None):
        super().__init__(state, parent)
        self.set_title("Stage")
        self._setup_stage_ui()
        self._register_observers()

    def _setup_stage_ui(self):
        """Setup stage-specific UI elements."""
        # Directory Selection Group
        dir_group = self.add_group("Directory")
        dir_layout = dir_group.layout()

        dir_widget = QWidget()
        dir_hlayout = QHBoxLayout(dir_widget)
        dir_hlayout.setContentsMargins(0, 0, 0, 0)
        self.directory_edit = QLineEdit()
        # Set placeholder based on global directory
        global_dir = self.state.global_working_directory
        if global_dir:
            self.directory_edit.setPlaceholderText(f"Using global: {global_dir}")
        else:
            self.directory_edit.setPlaceholderText("Select directory for stage files...")
        self.directory_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse_directory)
        dir_hlayout.addWidget(self.directory_edit)
        dir_hlayout.addWidget(browse_btn)
        dir_layout.addRow(dir_widget)

        # Load existing button
        self.load_existing_btn = QPushButton("Load Existing Stage")
        self.load_existing_btn.clicked.connect(self._load_existing_stage)
        self.load_existing_btn.setEnabled(False)
        dir_layout.addRow(self.load_existing_btn)

        # Motor Table Group
        motor_group = self.add_group("Motors")
        motor_layout = motor_group.layout()

        self.motor_table = QTableWidget(0, 4)
        self.motor_table.setHorizontalHeaderLabels(["Motor", "Type", "Value", "Resolution"])
        self.motor_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.motor_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.motor_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.motor_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.motor_table.setMaximumHeight(200)
        motor_layout.addRow(self.motor_table)

        # Movement Group
        move_group = self.add_group("Movement")
        move_layout = move_group.layout()

        # Motor selector
        self.motor_combo = QComboBox()
        move_layout.addRow("Motor:", self.motor_combo)

        # Movement mode
        mode_widget = QWidget()
        mode_layout = QHBoxLayout(mode_widget)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        self.mode_group = QButtonGroup(self)
        self.absolute_radio = QRadioButton("Absolute")
        self.relative_radio = QRadioButton("Relative")
        self.absolute_radio.setChecked(True)
        self.mode_group.addButton(self.absolute_radio)
        self.mode_group.addButton(self.relative_radio)
        mode_layout.addWidget(self.absolute_radio)
        mode_layout.addWidget(self.relative_radio)
        mode_layout.addStretch()
        move_layout.addRow("Mode:", mode_widget)

        # Value input
        self.value_input = QDoubleSpinBox()
        self.value_input.setDecimals(6)
        self.value_input.setRange(-1e9, 1e9)
        self.value_input.setSuffix(" °")
        move_layout.addRow("Value:", self.value_input)

        # Move button
        move_btn = QPushButton("Move Motor")
        move_btn.clicked.connect(self._on_move)
        move_layout.addRow("", move_btn)

        # Zero button
        zero_btn = QPushButton("Zero All Motors")
        zero_btn.clicked.connect(self._on_zero_all)
        move_layout.addRow("", zero_btn)

        # Display Group
        display_group = self.add_group("Orientation")
        display_layout = display_group.layout()

        self.rotation_display = QLabel("Rotation Matrix:\n[Identity]")
        self.rotation_display.setStyleSheet("font-family: monospace; color: #a0a0a0;")
        display_layout.addRow(self.rotation_display)

        self.translation_display = QLabel("Translation: [0, 0, 0]")
        self.translation_display.setStyleSheet("font-family: monospace; color: #a0a0a0;")
        display_layout.addRow(self.translation_display)

        # Create Stage Button
        create_btn = QPushButton("Create Stage")
        create_btn.clicked.connect(self._on_create_stage)
        create_btn.setStyleSheet("QPushButton { background-color: #2a6a2a; }")
        self.content_layout.insertWidget(self.content_layout.count() - 1, create_btn)

    def _register_observers(self):
        self.state.register_observer("stage_changed", self._on_stage_state_changed)
        self.state.register_observer("global_working_directory_changed", self._on_global_dir_changed)

    def _on_global_dir_changed(self, directory):
        """Handle global working directory change."""
        # Update placeholder text if directory field is empty
        if not self.directory_edit.text():
            self.directory_edit.setPlaceholderText(
                f"Using global: {directory}" if directory
                else "Select directory for stage files..."
            )

    def _on_stage_state_changed(self, stage):
        self._refresh_display()

    def _refresh_display(self):
        stage = self.state.stage
        self.motor_table.setRowCount(0)
        self.motor_combo.clear()

        if stage is None:
            self.rotation_display.setText("Rotation Matrix:\n[Identity]")
            self.translation_display.setText("Translation: [0, 0, 0]")
            return

        # Update directory display if stage has one
        if hasattr(stage, '_directory') and stage._directory:
            self.directory_edit.setText(str(stage._directory))

        # Populate motor table using Stage.py's array-based attributes
        if hasattr(stage, '_motor_name') and stage._motor_name is not None:
            motor_names = stage._motor_name
            motor_types = stage._motor_type if hasattr(stage, '_motor_type') else ['rotation'] * len(motor_names)
            motor_values = stage._motor_value if hasattr(stage, '_motor_value') else [0.0] * len(motor_names)

            for i, name in enumerate(motor_names):
                row = self.motor_table.rowCount()
                self.motor_table.insertRow(row)
                self.motor_table.setItem(row, 0, QTableWidgetItem(str(name)))
                motor_type = motor_types[i] if i < len(motor_types) else 'rotation'
                self.motor_table.setItem(row, 1, QTableWidgetItem(str(motor_type)))
                # Values are stored in radians for rotation motors, display in degrees
                value = motor_values[i] if i < len(motor_values) else 0.0
                if motor_type == 'rotation':
                    value = np.degrees(value)
                self.motor_table.setItem(row, 2, QTableWidgetItem(f"{value:.4f}"))
                self.motor_table.setItem(row, 3, QTableWidgetItem("0.0001"))  # Default resolution
                self.motor_combo.addItem(str(name))

        # Update orientation display using Stage.py's _rotation attribute
        if hasattr(stage, '_rotation') and stage._rotation is not None:
            R = stage._rotation
            R_str = "\n".join([f"[{R[i,0]:7.4f} {R[i,1]:7.4f} {R[i,2]:7.4f}]" for i in range(3)])
            self.rotation_display.setText(f"Rotation Matrix:\n{R_str}")
        else:
            self.rotation_display.setText("Rotation Matrix:\n[Identity]")

        # Update translation display using Stage.py's _translation attribute
        if hasattr(stage, '_translation') and stage._translation is not None:
            T = stage._translation
            self.translation_display.setText(f"Translation: [{T[0]:.4f}, {T[1]:.4f}, {T[2]:.4f}]")
        else:
            self.translation_display.setText("Translation: [0, 0, 0]")

    def _on_create_stage(self):
        directory = self.directory_edit.text()
        if not directory:
            QMessageBox.warning(self, "No Directory",
                              "Please select a working directory first.")
            return

        try:
            from Stage import stage
            new_stage = stage(directory=directory)
            new_stage.create_stage()
            # Save metadata after creation
            if hasattr(new_stage, 'write_stage_metadata'):
                new_stage.write_stage_metadata()
            self.state.stage = new_stage
            self.stage_created.emit(new_stage)
            self._refresh_display()
            QMessageBox.information(self, "Success", "Stage created successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create stage:\n{str(e)}")

    def _on_move(self):
        stage = self.state.stage
        if stage is None:
            QMessageBox.warning(self, "No Stage", "Please create a stage first.")
            return

        motor_name = self.motor_combo.currentText()
        if not motor_name:
            QMessageBox.warning(self, "No Motor", "Please select a motor.")
            return

        value = self.value_input.value()

        try:
            # Use Stage.py's set_single_motor_value methods (value in degrees for rotation)
            if self.absolute_radio.isChecked():
                stage.set_single_motor_value_absolute(motor_name, value, degrees=True)
            else:
                stage.set_single_motor_value_relative(motor_name, value, degrees=True)

            # Save metadata after movement
            if hasattr(stage, 'write_stage_metadata'):
                stage.write_stage_metadata()

            self.state.notify_object_modified("stage")
            self.motor_moved.emit(motor_name, value)
            self._refresh_display()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to move motor:\n{str(e)}")

    def _on_zero_all(self):
        stage = self.state.stage
        if stage is None:
            QMessageBox.warning(self, "No Stage", "Please create a stage first.")
            return

        try:
            # Use Stage.py's zero_stage() method
            stage.zero_stage()

            # Save metadata after zeroing
            if hasattr(stage, 'write_stage_metadata'):
                stage.write_stage_metadata()

            self.state.notify_object_modified("stage")
            self._refresh_display()
            QMessageBox.information(self, "Success", "All motors zeroed.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to zero motors:\n{str(e)}")

    def _on_browse_directory(self):
        """Open directory browser dialog."""
        # Use current text, or global directory, or current working directory
        start_dir = self.directory_edit.text() or self.state.get_default_directory()
        directory = QFileDialog.getExistingDirectory(
            self, "Select Stage Directory",
            start_dir
        )
        if directory:
            self.directory_edit.setText(directory)
            # Check if stage metadata exists
            metadata_path = Path(directory) / "stage_metadata.json"
            if metadata_path.exists():
                # Enable and style button green when metadata found
                self.load_existing_btn.setEnabled(True)
                self.load_existing_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #2a6a2a;
                        color: white;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #3a8a3a;
                    }
                """)
            else:
                # Disable button and reset style when no metadata
                self.load_existing_btn.setEnabled(False)
                self.load_existing_btn.setStyleSheet("")

    def _load_existing_stage(self):
        """Load existing stage from selected directory."""
        directory = self.directory_edit.text()
        if not directory:
            QMessageBox.warning(self, "No Directory",
                              "Please select a working directory first.")
            return

        try:
            from Stage import stage
            existing_stage = stage(directory=directory)
            # Try to load metadata
            if hasattr(existing_stage, 'read_stage_metadata'):
                existing_stage.read_stage_metadata()
            self.state.stage = existing_stage
            self.stage_created.emit(existing_stage)
            self._refresh_display()
            QMessageBox.information(self, "Success", "Stage loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load stage:\n{str(e)}")
