# -----------------------------------------------------------------------------
# Detector Inspector
# -----------------------------------------------------------------------------
"""
Inspector panel for detector configuration.

Provides controls for:
- Detector shape and pixel size
- Geometry type (rectangular/ring)
- Position (distance, 2θ, η)
- Data loading/saving
"""

import sys
from pathlib import Path
from typing import Optional
import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QComboBox,
    QCheckBox,
    QMessageBox,
    QSlider,
    QFileDialog,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gui.state import SimulationState
from gui.panels.inspector import InspectorPanel, PropertyDef


class DetectorInspector(InspectorPanel):
    """
    Inspector for detector configuration.

    Signals:
        detector_created: Emitted when a new detector is created
        detector_moved: Emitted when detector position changes
    """

    detector_created = Signal(object)
    detector_moved = Signal()

    def __init__(self, state: SimulationState, parent=None):
        """
        Initialize the detector inspector.

        Args:
            state: SimulationState instance
            parent: Parent widget
        """
        super().__init__(state, parent)
        self.set_title("Detector")
        self._setup_detector_ui()
        self._register_observers()

    def _setup_detector_ui(self):
        """Setup detector-specific UI elements."""
        # Directory Group (required first)
        dir_group = self.add_group("Directory")
        dir_layout = dir_group.layout()

        # Directory selection
        dir_row = QWidget()
        dir_row_layout = QHBoxLayout(dir_row)
        dir_row_layout.setContentsMargins(0, 0, 0, 0)
        dir_row_layout.setSpacing(4)

        self.directory_edit = QLineEdit()
        # Set placeholder based on global directory
        global_dir = self.state.global_working_directory
        if global_dir:
            self.directory_edit.setPlaceholderText(f"Using global: {global_dir}")
        else:
            self.directory_edit.setPlaceholderText("Select directory for detector files...")
        self.directory_edit.setReadOnly(True)
        dir_row_layout.addWidget(self.directory_edit, 1)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse_directory)
        dir_row_layout.addWidget(browse_btn)

        dir_layout.addRow("Path:", dir_row)

        # Info label
        self.dir_info_label = QLabel("Select a directory to create or load a detector.")
        self.dir_info_label.setStyleSheet("color: #808080; font-style: italic;")
        self.dir_info_label.setWordWrap(True)
        dir_layout.addRow("", self.dir_info_label)

        # Load existing button (only enabled when metadata found)
        self.load_existing_btn = QPushButton("Load Existing Detector")
        self.load_existing_btn.clicked.connect(self._on_load_existing)
        self.load_existing_btn.setEnabled(False)
        dir_layout.addRow("", self.load_existing_btn)

        # Shape Group
        shape_group = self.add_group("Shape & Size")
        shape_layout = shape_group.layout()

        # Pixels Y (horizontal)
        self.pixels_y = QSpinBox()
        self.pixels_y.setRange(1, 10000)
        self.pixels_y.setValue(256)
        self.pixels_y.setSingleStep(64)
        shape_layout.addRow("Pixels Y:", self.pixels_y)

        # Pixels Z (vertical)
        self.pixels_z = QSpinBox()
        self.pixels_z.setRange(1, 10000)
        self.pixels_z.setValue(256)
        self.pixels_z.setSingleStep(64)
        shape_layout.addRow("Pixels Z:", self.pixels_z)

        # Pixel size Y (in Angstroms)
        self.pixel_size_y = QDoubleSpinBox()
        self.pixel_size_y.setDecimals(0)
        self.pixel_size_y.setRange(1, 1e9)
        self.pixel_size_y.setValue(100)  # 0.055 mm = 5.5e5 Å
        self.pixel_size_y.setSuffix(" Å")
        self.pixel_size_y.setSingleStep(10)
        shape_layout.addRow("Pixel Size Y:", self.pixel_size_y)

        # Pixel size Z (in Angstroms)
        self.pixel_size_z = QDoubleSpinBox()
        self.pixel_size_z.setDecimals(0)
        self.pixel_size_z.setRange(1, 1e9)
        self.pixel_size_z.setValue(100)  # 0.055 mm = 5.5e5 Å
        self.pixel_size_z.setSuffix(" Å")
        self.pixel_size_z.setSingleStep(10)
        shape_layout.addRow("Pixel Size Z:", self.pixel_size_z)

        # Total size display
        self.total_size = QLabel("Total: 1.41e8 x 1.41e8 Å")
        self.total_size.setStyleSheet("color: #808080;")
        self.pixels_y.valueChanged.connect(self._update_total_size)
        self.pixels_z.valueChanged.connect(self._update_total_size)
        self.pixel_size_y.valueChanged.connect(self._update_total_size)
        self.pixel_size_z.valueChanged.connect(self._update_total_size)
        shape_layout.addRow("", self.total_size)

        # Geometry Group
        geom_group = self.add_group("Geometry")
        geom_layout = geom_group.layout()

        # Geometry type
        self.geometry = QComboBox()
        self.geometry.addItem("Rectangular", "rectangular")
        self.geometry.addItem("Ring", "ring")
        geom_layout.addRow("Type:", self.geometry)

        # Construction mode
        self.construction_mode = QComboBox()
        self.construction_mode.addItem("Plane (Flat)", "plane")
        self.construction_mode.addItem("Shell (Spherical)", "shell")
        self.construction_mode.setToolTip(
            "Plane: Pixels on flat detector surface\n"
            "Shell: Pixels on spherical shell (equal distance from sample)"
        )
        geom_layout.addRow("Construction:", self.construction_mode)

        # Input mode
        self.input_mode = QComboBox()
        self.input_mode.addItem("Spatial (pixels/Å)", "spatial")
        self.input_mode.addItem("Angular (degrees)", "angular")
        self.input_mode.setToolTip(
            "Spatial: Specify pixel counts and spacing\n"
            "Angular: Specify angular range and resolution"
        )
        geom_layout.addRow("Input Mode:", self.input_mode)

        # Connect to update labels when geometry or mode changes
        self.geometry.currentIndexChanged.connect(self._update_parameter_labels)
        self.construction_mode.currentIndexChanged.connect(self._update_parameter_labels)
        self.input_mode.currentIndexChanged.connect(self._update_parameter_labels)

        # Position Group
        pos_group = self.add_group("Position")
        pos_layout = pos_group.layout()

        # Distance (in Angstroms)
        self.distance = QDoubleSpinBox()
        self.distance.setDecimals(0)
        self.distance.setRange(0, 1e12)
        self.distance.setValue(1000)  # 1000 mm = 1e10 Å
        self.distance.setSuffix(" Å")
        self.distance.setSingleStep(100)
        pos_layout.addRow("Distance:", self.distance)

        # Two theta
        two_theta_row = QWidget()
        two_theta_layout = QHBoxLayout(two_theta_row)
        two_theta_layout.setContentsMargins(0, 0, 0, 0)
        two_theta_layout.setSpacing(4)

        self.two_theta = QDoubleSpinBox()
        self.two_theta.setDecimals(4)
        self.two_theta.setRange(-180, 180)
        self.two_theta.setValue(0)
        self.two_theta.setSuffix(" °")
        self.two_theta.setSingleStep(0.1)
        two_theta_layout.addWidget(self.two_theta, 1)

        # Quick 2theta buttons
        for angle in [-10, 0, 10]:
            btn = QPushButton(f"{angle:+d}°")
            btn.setMaximumWidth(40)
            btn.clicked.connect(lambda checked, a=angle: self.two_theta.setValue(a))
            two_theta_layout.addWidget(btn)

        pos_layout.addRow("2θ:", two_theta_row)

        # Eta
        eta_row = QWidget()
        eta_layout = QHBoxLayout(eta_row)
        eta_layout.setContentsMargins(0, 0, 0, 0)
        eta_layout.setSpacing(4)

        self.eta = QDoubleSpinBox()
        self.eta.setDecimals(4)
        self.eta.setRange(-180, 180)
        self.eta.setValue(0)
        self.eta.setSuffix(" °")
        self.eta.setSingleStep(0.1)
        eta_layout.addWidget(self.eta, 1)

        # Quick eta buttons
        for angle in [-90, 0, 90]:
            btn = QPushButton(f"{angle:+d}°")
            btn.setMaximumWidth(40)
            btn.clicked.connect(lambda checked, a=angle: self.eta.setValue(a))
            eta_layout.addWidget(btn)

        pos_layout.addRow("η:", eta_row)

        # Movement mode selection
        self.movement_mode = QComboBox()
        self.movement_mode.addItem("Absolute", "absolute")
        self.movement_mode.addItem("Relative", "relative")
        self.movement_mode.setToolTip("Absolute: Set exact position\nRelative: Move by specified amounts")
        pos_layout.addRow("Mode:", self.movement_mode)

        # Position buttons
        pos_btn_row = QWidget()
        pos_btn_layout = QHBoxLayout(pos_btn_row)
        pos_btn_layout.setContentsMargins(0, 0, 0, 0)
        pos_btn_layout.setSpacing(4)

        move_btn = QPushButton("Move Detector")
        move_btn.clicked.connect(self._on_move_detector)
        pos_btn_layout.addWidget(move_btn)

        center_btn = QPushButton("Center on Bragg")
        center_btn.clicked.connect(self._on_center_bragg)
        center_btn.setToolTip("Center detector on Bragg peak (requires crystal)")
        pos_btn_layout.addWidget(center_btn)

        # Zero position button (for relative mode)
        zero_btn = QPushButton("Zero")
        zero_btn.clicked.connect(self._on_zero_position)
        zero_btn.setToolTip("Reset position values to zero (for relative adjustments)")
        pos_btn_layout.addWidget(zero_btn)

        pos_layout.addRow("", pos_btn_row)

        # Data Actions Group
        data_group = self.add_group("Data")
        data_layout = data_group.layout()

        # Load pixel data button
        load_btn = QPushButton("Load Pixel Data...")
        load_btn.clicked.connect(self._on_load_data)
        data_layout.addRow("", load_btn)

        # Save pixel data button
        save_btn = QPushButton("Save Pixel Data...")
        save_btn.clicked.connect(self._on_save_data)
        data_layout.addRow("", save_btn)

        # Clear button
        clear_btn = QPushButton("Clear Data")
        clear_btn.clicked.connect(self._on_clear_data)
        data_layout.addRow("", clear_btn)

        # Info display
        self.data_info = QLabel("No data loaded")
        self.data_info.setStyleSheet("color: #808080;")
        data_layout.addRow("Status:", self.data_info)

        # Create Detector Button
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 10, 0, 0)
        btn_layout.setSpacing(8)

        create_btn = QPushButton("Create Detector")
        create_btn.clicked.connect(self._on_create_detector)
        create_btn.setStyleSheet("QPushButton { background-color: #2a6a2a; }")
        btn_layout.addWidget(create_btn)

        # Insert before stretch
        self.content_layout.insertWidget(self.content_layout.count() - 1, btn_row)

        # Initialize displays
        self._update_total_size()

    def _register_observers(self):
        """Register for state change notifications."""
        self.state.register_observer("detector_changed", self._on_detector_state_changed)
        self.state.register_observer("global_working_directory_changed", self._on_global_dir_changed)

    def _on_global_dir_changed(self, directory):
        """Handle global working directory change."""
        # Update placeholder text if directory field is empty
        if not self.directory_edit.text():
            self.directory_edit.setPlaceholderText(
                f"Using global: {directory}" if directory
                else "Select directory for detector files..."
            )

    def _on_detector_state_changed(self, detector):
        """Handle detector state change."""
        self._refresh_display()

    def _on_browse_directory(self):
        """Handle directory browse button click."""
        # Use current text, or global directory, or current working directory
        start_dir = self.directory_edit.text() or self.state.get_default_directory()
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Detector Directory",
            start_dir,
            QFileDialog.ShowDirsOnly
        )

        if directory:
            self.directory_edit.setText(directory)

            # Check if metadata exists and enable/style button accordingly
            metadata_path = Path(directory) / "detector_metadata.json"
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
                self.dir_info_label.setText("Metadata found - click 'Load Existing Detector'")
                self.dir_info_label.setStyleSheet("color: #4ec94e; font-style: italic;")
            else:
                # Disable button and reset style when no metadata
                self.load_existing_btn.setEnabled(False)
                self.load_existing_btn.setStyleSheet("")
                self.dir_info_label.setText("No metadata found - configure parameters and click Create.")
                self.dir_info_label.setStyleSheet("color: #d4a94e; font-style: italic;")

    def _on_load_existing(self):
        """Handle load existing detector button click."""
        directory = self.directory_edit.text()
        if directory:
            self._load_existing_detector(directory)

    def _load_existing_detector(self, directory: str):
        """
        Load an existing detector from directory.

        Args:
            directory: Path to detector directory
        """
        try:
            from Detector import detector

            # Create detector with directory
            new_detector = detector(directory=directory)

            # Load existing metadata
            metadata_path = Path(directory) / "detector_metadata.json"
            if metadata_path.exists():
                new_detector.read_detector_metadata()
                self.dir_info_label.setText("Loaded existing detector from metadata.")
                self.dir_info_label.setStyleSheet("color: #4ec94e; font-style: italic;")

            self.state.detector = new_detector
            self._refresh_display()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load detector:\n{str(e)}")
            self.dir_info_label.setText(f"Error: {str(e)}")
            self.dir_info_label.setStyleSheet("color: #e05050; font-style: italic;")

    def _refresh_display(self):
        """Refresh display from current detector state."""
        detector_obj = self.state.detector
        if detector_obj is None:
            self.data_info.setText("No detector created")
            return

        try:
            # Update directory display
            if hasattr(detector_obj, 'directory'):
                self.directory_edit.setText(detector_obj.directory)

            # Shape (Ny, Nz)
            if hasattr(detector_obj, 'shape') and detector_obj.shape is not None:
                self.pixels_y.blockSignals(True)
                self.pixels_z.blockSignals(True)
                self.pixels_y.setValue(detector_obj.shape[0])
                self.pixels_z.setValue(detector_obj.shape[1])
                self.pixels_y.blockSignals(False)
                self.pixels_z.blockSignals(False)

            # Pixel size
            if hasattr(detector_obj, 'pixel_size') and detector_obj.pixel_size is not None:
                self.pixel_size_y.setValue(detector_obj.pixel_size[0])
                self.pixel_size_z.setValue(detector_obj.pixel_size[1])

            # Distance - block signals to avoid triggering callbacks
            if hasattr(detector_obj, 'distance') and detector_obj.distance is not None:
                self.distance.blockSignals(True)
                self.distance.setValue(float(detector_obj.distance))
                self.distance.blockSignals(False)

            # Two theta - stored in radians, display in degrees
            if hasattr(detector_obj, 'two_theta') and detector_obj.two_theta is not None:
                self.two_theta.blockSignals(True)
                two_theta_deg = float(np.degrees(detector_obj.two_theta))
                self.two_theta.setValue(two_theta_deg)
                self.two_theta.blockSignals(False)

            # Eta - stored in radians, display in degrees
            if hasattr(detector_obj, 'eta') and detector_obj.eta is not None:
                self.eta.blockSignals(True)
                eta_deg = float(np.degrees(detector_obj.eta))
                self.eta.setValue(eta_deg)
                self.eta.blockSignals(False)

            # Geometry type (block signals to prevent _update_parameter_labels
            # from resetting shape/pixel_size values set above)
            if hasattr(detector_obj, '_geometry') and detector_obj._geometry is not None:
                self.geometry.blockSignals(True)
                idx = self.geometry.findData(detector_obj._geometry)
                if idx >= 0:
                    self.geometry.setCurrentIndex(idx)
                self.geometry.blockSignals(False)

            # Construction mode
            if hasattr(detector_obj, '_construction_mode') and detector_obj._construction_mode is not None:
                self.construction_mode.blockSignals(True)
                idx = self.construction_mode.findData(detector_obj._construction_mode)
                if idx >= 0:
                    self.construction_mode.setCurrentIndex(idx)
                self.construction_mode.blockSignals(False)

            # Input mode
            if hasattr(detector_obj, '_input_mode') and detector_obj._input_mode is not None:
                self.input_mode.blockSignals(True)
                idx = self.input_mode.findData(detector_obj._input_mode)
                if idx >= 0:
                    self.input_mode.setCurrentIndex(idx)
                self.input_mode.blockSignals(False)

            # Update data status - check _pixel_values directly to avoid warning print
            if hasattr(detector_obj, '_pixel_values') and detector_obj._pixel_values is not None:
                pv = detector_obj._pixel_values
                if hasattr(pv, 'shape'):
                    self.data_info.setText(f"Data: {pv.shape} pixels")
                else:
                    self.data_info.setText("Data loaded")
            else:
                self.data_info.setText("No data")

        except Exception:
            pass

        self._update_total_size()

    def _update_total_size(self):
        """Update total size display (in Angstroms)."""
        size_y = self.pixels_y.value() * self.pixel_size_y.value()
        size_z = self.pixels_z.value() * self.pixel_size_z.value()
        self.total_size.setText(f"Total: {size_y:.2e} x {size_z:.2e} Å")

    def _update_parameter_labels(self):
        """Update UI labels and units based on selected geometry, construction mode, and input mode."""
        geometry = self.geometry.currentData()
        mode = self.construction_mode.currentData()
        input_mode = self.input_mode.currentData()

        # Angular input mode: shape fields become angular bounds, pixel_size is resolution in degrees
        if input_mode == 'angular':
            if geometry == 'rectangular':
                # For rectangular + angular: shape = (theta_y_min, theta_y_max, theta_z_min, theta_z_max)
                # But we only have 2 spinboxes, so we'll interpret as (range_y, range_z) centered at 0
                # User enters half-range, actual range is -val to +val
                # Pixel size is angular resolution in degrees
                self.pixels_y.setRange(1, 90)
                self.pixels_z.setRange(1, 90)
                self.pixels_y.setValue(10)
                self.pixels_z.setValue(10)
                self.pixels_y.setSuffix(" ° (±θy)")
                self.pixels_z.setSuffix(" ° (±θz)")

                self.pixel_size_y.setSuffix(" ° (dθy)")
                self.pixel_size_z.setSuffix(" ° (dθz)")
                self.pixel_size_y.setDecimals(4)
                self.pixel_size_z.setDecimals(4)
                self.pixel_size_y.setValue(0.1)
                self.pixel_size_z.setValue(0.1)
                self.pixel_size_y.setSingleStep(0.01)
                self.pixel_size_z.setSingleStep(0.01)
                self.pixel_size_y.setRange(0.0001, 90)
                self.pixel_size_z.setRange(0.0001, 90)

            elif geometry == 'ring':
                # For ring + angular: shape = (two_theta_inner, two_theta_outer)
                # Pixel size = (d_two_theta, d_eta) in degrees
                self.pixels_y.setRange(0, 89)
                self.pixels_z.setRange(1, 90)
                self.pixels_y.setValue(5)
                self.pixels_z.setValue(45)
                self.pixels_y.setSuffix(" ° (2θ inner)")
                self.pixels_z.setSuffix(" ° (2θ outer)")

                self.pixel_size_y.setSuffix(" ° (d2θ)")
                self.pixel_size_z.setSuffix(" ° (dη)")
                self.pixel_size_y.setDecimals(4)
                self.pixel_size_z.setDecimals(4)
                self.pixel_size_y.setValue(0.5)
                self.pixel_size_z.setValue(5.0)
                self.pixel_size_y.setSingleStep(0.1)
                self.pixel_size_z.setSingleStep(1.0)
                self.pixel_size_y.setRange(0.0001, 90)
                self.pixel_size_z.setRange(0.0001, 360)

        else:  # spatial input mode
            # Reset pixels to integer counts
            self.pixels_y.setRange(1, 10000)
            self.pixels_z.setRange(1, 10000)
            self.pixels_y.setValue(256)
            self.pixels_z.setValue(256)
            self.pixels_y.setSuffix("")
            self.pixels_z.setSuffix("")

            if geometry == 'rectangular':
                if mode == 'shell':
                    # Shell mode: angular units (degrees)
                    self.pixel_size_y.setSuffix(" °")
                    self.pixel_size_z.setSuffix(" °")
                    self.pixel_size_y.setDecimals(4)
                    self.pixel_size_z.setDecimals(4)
                    if self.pixel_size_y.value() > 10:  # Likely in Angstroms, convert to reasonable angular value
                        self.pixel_size_y.setValue(0.1)
                        self.pixel_size_z.setValue(0.1)
                    self.pixel_size_y.setSingleStep(0.01)
                    self.pixel_size_z.setSingleStep(0.01)
                    self.pixel_size_y.setRange(0.0001, 180)
                    self.pixel_size_z.setRange(0.0001, 180)
                else:  # plane mode
                    # Plane mode: length units (Angstroms)
                    self.pixel_size_y.setSuffix(" Å")
                    self.pixel_size_z.setSuffix(" Å")
                    self.pixel_size_y.setDecimals(0)
                    self.pixel_size_z.setDecimals(0)
                    if self.pixel_size_y.value() < 1:  # Likely in degrees, convert to Angstroms
                        self.pixel_size_y.setValue(100)
                        self.pixel_size_z.setValue(100)
                    self.pixel_size_y.setSingleStep(10)
                    self.pixel_size_z.setSingleStep(10)
                    self.pixel_size_y.setRange(1, 1e9)
                    self.pixel_size_z.setRange(1, 1e9)

            elif geometry == 'ring':
                if mode == 'shell':
                    # Shell mode: angular units (degrees)
                    self.pixel_size_y.setSuffix(" ° (2θ)")
                    self.pixel_size_z.setSuffix(" ° (η)")
                    self.pixel_size_y.setDecimals(4)
                    self.pixel_size_z.setDecimals(4)
                    if self.pixel_size_y.value() > 10:  # Likely in Angstroms
                        self.pixel_size_y.setValue(0.5)
                        self.pixel_size_z.setValue(1.0)
                    self.pixel_size_y.setSingleStep(0.1)
                    self.pixel_size_z.setSingleStep(0.1)
                    self.pixel_size_y.setRange(0.0001, 180)
                    self.pixel_size_z.setRange(0.0001, 360)
                else:  # plane mode
                    # Plane mode: length units (Angstroms)
                    self.pixel_size_y.setSuffix(" Å (radial)")
                    self.pixel_size_z.setSuffix(" Å (arc)")
                    self.pixel_size_y.setDecimals(0)
                    self.pixel_size_z.setDecimals(0)
                    if self.pixel_size_y.value() < 1:  # Likely in degrees
                        self.pixel_size_y.setValue(100)
                        self.pixel_size_z.setValue(100)
                    self.pixel_size_y.setSingleStep(10)
                    self.pixel_size_z.setSingleStep(10)
                    self.pixel_size_y.setRange(1, 1e9)
                    self.pixel_size_z.setRange(1, 1e9)

        # Update total size display
        self._update_total_size()

    def _on_create_detector(self):
        """Handle create detector button."""
        # Check if directory is selected
        directory = self.directory_edit.text()
        if not directory:
            QMessageBox.warning(
                self,
                "Directory Required",
                "Please select a directory for detector files first."
            )
            return

        try:
            from Detector import detector

            # Get current settings
            geometry = self.geometry.currentData()
            construction_mode = self.construction_mode.currentData()
            input_mode = self.input_mode.currentData()

            # Build shape and pixel_size based on input_mode
            if input_mode == 'angular':
                if geometry == 'rectangular':
                    # For rectangular + angular: shape = (theta_y_min, theta_y_max, theta_z_min, theta_z_max)
                    # GUI uses half-range (±val), so range is (-val, +val)
                    half_y = self.pixels_y.value()
                    half_z = self.pixels_z.value()
                    shape = (-half_y, half_y, -half_z, half_z)
                else:  # ring
                    # For ring + angular: shape = (two_theta_inner, two_theta_outer)
                    shape = (self.pixels_y.value(), self.pixels_z.value())
                pixel_size = (self.pixel_size_y.value(), self.pixel_size_z.value())
            else:  # spatial mode
                shape = (self.pixels_y.value(), self.pixels_z.value())
                pixel_size = (self.pixel_size_y.value(), self.pixel_size_z.value())

            # Check if detector already exists in state with this directory
            existing_detector = self.state.detector
            if existing_detector is not None and hasattr(existing_detector, 'directory'):
                if existing_detector.directory == directory:
                    # Update existing detector
                    existing_detector.create_detector(
                        shape=shape,
                        pixel_size=pixel_size,
                        geometry=geometry,
                        construction_mode=construction_mode,
                        input_mode=input_mode
                    )
                    # Position detector (degrees=True is default)
                    existing_detector.position_detector_absolute(
                        self.distance.value(),
                        self.two_theta.value(),
                        self.eta.value(),
                        degrees=True
                    )
                    # Save metadata
                    existing_detector.write_detector_metadata()
                    self.state.notify_object_modified("detector")
                    self.detector_created.emit(existing_detector)
                    self.dir_info_label.setText("Detector configured successfully.")
                    self.dir_info_label.setStyleSheet("color: #4ec94e; font-style: italic;")
                    QMessageBox.information(self, "Success", "Detector configured successfully.")
                    return

            # Create new detector with directory
            new_detector = detector(directory=directory)
            new_detector.create_detector(
                shape=shape,
                pixel_size=pixel_size,
                geometry=geometry,
                construction_mode=construction_mode,
                input_mode=input_mode
            )

            # Position detector (degrees=True is default)
            new_detector.position_detector_absolute(
                self.distance.value(),
                self.two_theta.value(),
                self.eta.value(),
                degrees=True
            )

            # Save metadata
            new_detector.write_detector_metadata()

            self.state.detector = new_detector
            self.detector_created.emit(new_detector)

            self.dir_info_label.setText("Detector created successfully.")
            self.dir_info_label.setStyleSheet("color: #4ec94e; font-style: italic;")

            QMessageBox.information(self, "Success", "Detector created successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create detector:\n{str(e)}")

    def _on_move_detector(self):
        """Handle move detector button (absolute or relative based on mode)."""
        detector_obj = self.state.detector
        if detector_obj is None:
            QMessageBox.warning(self, "No Detector", "Please create a detector first.")
            return

        try:
            mode = self.movement_mode.currentData()
            distance = self.distance.value()
            two_theta = self.two_theta.value()
            eta = self.eta.value()

            if mode == "absolute":
                # Absolute positioning
                if hasattr(detector_obj, 'position_detector_absolute'):
                    detector_obj.position_detector_absolute(
                        distance,
                        two_theta,
                        eta,
                        degrees=True
                    )
                else:
                    QMessageBox.warning(self, "Not Supported",
                                       "Detector doesn't support absolute positioning.")
                    return
            else:
                # Relative positioning
                if hasattr(detector_obj, 'position_detector_relative'):
                    detector_obj.position_detector_relative(
                        distance,
                        two_theta,
                        eta,
                        degrees=True
                    )
                else:
                    QMessageBox.warning(self, "Not Supported",
                                       "Detector doesn't support relative positioning.")
                    return

            # Save metadata after move
            if hasattr(detector_obj, 'write_detector_metadata'):
                detector_obj.write_detector_metadata()
            self.state.notify_object_modified("detector")
            self.detector_moved.emit()

            # Update display with new values after move (especially for relative)
            self._refresh_display()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to move detector:\n{str(e)}")

    def _on_zero_position(self):
        """Reset position values to zero (useful for relative adjustments)."""
        self.distance.setValue(0)
        self.two_theta.setValue(0)
        self.eta.setValue(0)

    def _on_center_bragg(self):
        """Handle center on Bragg peak button."""
        detector_obj = self.state.detector
        crystal = self.state.crystal
        beam_obj = self.state.beam

        if detector_obj is None:
            QMessageBox.warning(self, "No Detector", "Please create a detector first.")
            return

        if crystal is None or beam_obj is None:
            QMessageBox.warning(self, "Missing Objects", "Crystal and beam required for Bragg centering.")
            return

        try:
            # Calculate 2theta for primary reflection
            # Get beam energy - stored as _energy
            beam_energy = None
            if hasattr(beam_obj, '_energy') and beam_obj._energy is not None:
                beam_energy = beam_obj._energy
            elif hasattr(beam_obj, 'energy'):
                beam_energy = beam_obj.energy

            if beam_energy is None:
                QMessageBox.warning(self, "No Energy", "Beam energy not set.")
                return

            if hasattr(crystal, 'get_dhkl'):
                # Try to get primary hkl from crystal
                hkl = [0, 0, 1]  # Default
                if hasattr(crystal, 'primary_hkl') and crystal.primary_hkl is not None:
                    hkl = crystal.primary_hkl

                d = crystal.get_dhkl(hkl)
                wavelength = 12398.419 / beam_energy  # Angstroms
                sin_theta = wavelength / (2 * d)

                if abs(sin_theta) <= 1:
                    import math
                    two_theta = 2 * math.degrees(math.asin(sin_theta))
                    self.two_theta.setValue(two_theta)
                    self._on_move_detector()
                    QMessageBox.information(self, "Success", f"Centered on 2θ = {two_theta:.4f}°")
                else:
                    QMessageBox.warning(self, "Invalid", "Cannot reach this reflection at this energy.")
            else:
                QMessageBox.warning(self, "Not Supported", "Crystal doesn't have get_dhkl method.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to center:\n{str(e)}")

    def _on_load_data(self):
        """Handle load pixel data button."""
        from PySide6.QtWidgets import QFileDialog

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Detector Pixel Data",
            "",
            "NumPy (*.npy *.npz);;HDF5 (*.h5 *.hdf5);;TIFF (*.tiff *.tif);;All Files (*)"
        )
        if filename:
            try:
                detector = self.state.detector
                if detector is None:
                    QMessageBox.warning(self, "No Detector", "Please create a detector first.")
                    return

                # Load based on extension
                ext = Path(filename).suffix.lower()
                if ext in ['.npy', '.npz']:
                    data = np.load(filename)
                    if isinstance(data, np.lib.npyio.NpzFile):
                        # Get first array in npz
                        key = list(data.keys())[0]
                        data = data[key]
                elif ext in ['.tiff', '.tif']:
                    from PIL import Image
                    img = Image.open(filename)
                    data = np.array(img)
                else:
                    QMessageBox.warning(self, "Unknown Format", f"Unknown file format: {ext}")
                    return

                # Apply to detector
                if hasattr(detector, 'input_pixel_values'):
                    detector.input_pixel_values(data)
                    self.state.notify_object_modified("detector")
                    self._refresh_display()
                    QMessageBox.information(self, "Success", f"Loaded {data.shape[0]}x{data.shape[1]} pixels")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load data:\n{str(e)}")

    def _on_save_data(self):
        """Handle save pixel data button."""
        from PySide6.QtWidgets import QFileDialog

        detector = self.state.detector
        # Check _pixel_values directly to avoid warning print
        if detector is None or not hasattr(detector, '_pixel_values') or detector._pixel_values is None:
            QMessageBox.warning(self, "No Data", "No pixel data to save.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Detector Pixel Data",
            "",
            "NumPy (*.npy);;TIFF (*.tiff);;PNG (*.png)"
        )
        if filename:
            try:
                ext = Path(filename).suffix.lower()
                data = detector._pixel_values

                if ext == '.npy':
                    np.save(filename, data)
                elif ext in ['.tiff', '.tif']:
                    from PIL import Image
                    img = Image.fromarray(data)
                    img.save(filename)
                elif ext == '.png':
                    from PIL import Image
                    # Normalize for PNG
                    normalized = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
                    img = Image.fromarray(normalized)
                    img.save(filename)

                QMessageBox.information(self, "Success", f"Saved to {Path(filename).name}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save data:\n{str(e)}")

    def _on_clear_data(self):
        """Handle clear data button."""
        detector = self.state.detector
        if detector is None:
            return

        try:
            # Clear private attributes directly to avoid property issues
            if hasattr(detector, '_pixel_values'):
                detector._pixel_values = None
            if hasattr(detector, '_pixel_amplitude'):
                detector._pixel_amplitude = None
            if hasattr(detector, '_pixel_phase'):
                detector._pixel_phase = None
            if hasattr(detector, '_pixel_intensity'):
                detector._pixel_intensity = None

            self.state.notify_object_modified("detector")
            self._refresh_display()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to clear data:\n{str(e)}")

    def get_config(self) -> dict:
        """Get the current configuration."""
        return {
            "shape": [self.pixels_y.value(), self.pixels_z.value()],
            "pixel_size": [self.pixel_size_y.value(), self.pixel_size_z.value()],
            "geometry": self.geometry.currentData(),
            "construction_mode": self.construction_mode.currentData(),
            "input_mode": self.input_mode.currentData(),
            "distance": self.distance.value(),
            "two_theta": self.two_theta.value(),
            "eta": self.eta.value(),
        }

    def set_config(self, config: dict):
        """Set configuration from dict."""
        if "shape" in config:
            self.pixels_y.setValue(config["shape"][0])
            self.pixels_z.setValue(config["shape"][1])

        if "pixel_size" in config:
            self.pixel_size_y.setValue(config["pixel_size"][0])
            self.pixel_size_z.setValue(config["pixel_size"][1])

        if "geometry" in config:
            idx = self.geometry.findData(config["geometry"])
            if idx >= 0:
                self.geometry.setCurrentIndex(idx)

        if "construction_mode" in config:
            idx = self.construction_mode.findData(config["construction_mode"])
            if idx >= 0:
                self.construction_mode.setCurrentIndex(idx)

        if "input_mode" in config:
            idx = self.input_mode.findData(config["input_mode"])
            if idx >= 0:
                self.input_mode.setCurrentIndex(idx)

        if "distance" in config:
            self.distance.setValue(config["distance"])

        if "two_theta" in config:
            self.two_theta.setValue(config["two_theta"])

        if "eta" in config:
            self.eta.setValue(config["eta"])
