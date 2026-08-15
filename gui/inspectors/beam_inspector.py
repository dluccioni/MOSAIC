# -----------------------------------------------------------------------------
# Beam Inspector
# -----------------------------------------------------------------------------
"""
Inspector panel for X-ray beam configuration.

Provides controls for:
- Beam energy and wavelength
- Beam shape (rectangular/circular)
- Beam size and sampling
- Intensity profile (uniform/gaussian)
- Polarization
- Advanced parameters
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
from gui.panels.inspector import InspectorPanel, PropertyDef, SliderWidget


# Physical constants
HC_EV_A = 12398.419  # h*c in eV*Å


class BeamInspector(InspectorPanel):
    """
    Inspector for X-ray beam configuration.

    Signals:
        beam_created: Emitted when a new beam is created
        beam_updated: Emitted when beam parameters are updated
    """

    beam_created = Signal(object)
    beam_updated = Signal()

    def __init__(self, state: SimulationState, parent=None):
        """
        Initialize the beam inspector.

        Args:
            state: SimulationState instance
            parent: Parent widget
        """
        super().__init__(state, parent)
        self.set_title("Beam")
        self._setup_beam_ui()
        self._register_observers()

    def _setup_beam_ui(self):
        """Setup beam-specific UI elements."""
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
            self.directory_edit.setPlaceholderText("Select directory for beam files...")
        self.directory_edit.setReadOnly(True)
        dir_row_layout.addWidget(self.directory_edit, 1)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse_directory)
        dir_row_layout.addWidget(browse_btn)

        dir_layout.addRow("Path:", dir_row)

        # Info label
        self.dir_info_label = QLabel("Select a directory to create or load a beam.")
        self.dir_info_label.setStyleSheet("color: #808080; font-style: italic;")
        self.dir_info_label.setWordWrap(True)
        dir_layout.addRow("", self.dir_info_label)

        # Load existing button (only enabled when metadata found)
        self.load_existing_btn = QPushButton("Load Existing Beam")
        self.load_existing_btn.clicked.connect(self._on_load_existing)
        self.load_existing_btn.setEnabled(False)
        dir_layout.addRow("", self.load_existing_btn)

        # Energy Group
        energy_group = self.add_group("Energy")
        energy_layout = energy_group.layout()

        # Energy input
        self.energy = QDoubleSpinBox()
        self.energy.setDecimals(2)
        self.energy.setRange(100, 1e6)
        self.energy.setValue(10000)
        self.energy.setSuffix(" eV")
        self.energy.setSingleStep(100)
        self.energy.valueChanged.connect(self._on_energy_changed)
        energy_layout.addRow("Energy:", self.energy)

        # Wavelength display (calculated)
        self.wavelength_label = QLabel("0.7293 Å")
        self.wavelength_label.setStyleSheet("font-weight: bold; color: #4ec94e;")
        energy_layout.addRow("Wavelength:", self.wavelength_label)

        # Quick presets
        preset_row = QWidget()
        preset_layout = QHBoxLayout(preset_row)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(4)

        presets = [("Cu Kα", 8048), ("Mo Kα", 17479), ("Ag Kα", 22163)]
        for name, ev in presets:
            btn = QPushButton(name)
            btn.setMaximumWidth(60)
            btn.clicked.connect(lambda checked, e=ev: self.energy.setValue(e))
            preset_layout.addWidget(btn)

        preset_layout.addStretch()
        energy_layout.addRow("Presets:", preset_row)

        # Shape Group
        shape_group = self.add_group("Shape & Size")
        shape_layout = shape_group.layout()

        # Shape type
        self.shape_type = QComboBox()
        self.shape_type.addItem("Rectangular", "rectangular")
        self.shape_type.addItem("Circular", "circular")
        self.shape_type.currentIndexChanged.connect(self._on_shape_changed)
        shape_layout.addRow("Shape:", self.shape_type)

        # Beam size Y
        self.size_y = QDoubleSpinBox()
        self.size_y.setDecimals(1)
        self.size_y.setRange(1, 1e10)
        self.size_y.setValue(1000)  # 0.5 mm = 5e6 Å
        self.size_y.setSuffix(" Å")
        self.size_y.setSingleStep(100)
        shape_layout.addRow("Size Y:", self.size_y)

        # Beam size Z
        self.size_z = QDoubleSpinBox()
        self.size_z.setDecimals(1)
        self.size_z.setRange(1, 1e10)
        self.size_z.setValue(1000)  # 0.5 mm = 5e6 Å
        self.size_z.setSuffix(" Å")
        self.size_z.setSingleStep(100)
        shape_layout.addRow("Size Z:", self.size_z)

        # Beam Grid Group
        sampling_group = self.add_group("Beam Grid")
        sampling_layout = sampling_group.layout()

        # Samples Y
        self.samples_y = QSpinBox()
        self.samples_y.setRange(1, 10000)
        self.samples_y.setValue(512)
        sampling_layout.addRow("Samples Y:", self.samples_y)

        # Samples Z
        self.samples_z = QSpinBox()
        self.samples_z.setRange(1, 10000)
        self.samples_z.setValue(512)
        sampling_layout.addRow("Samples Z:", self.samples_z)

        # Grid points display
        self.total_rays = QLabel("Grid Points: 1")
        self.total_rays.setStyleSheet("color: #808080;")
        self.samples_y.valueChanged.connect(self._update_total_rays)
        self.samples_z.valueChanged.connect(self._update_total_rays)
        sampling_layout.addRow("", self.total_rays)

        # Profile Group
        profile_group = self.add_group("Intensity Profile")
        profile_layout = profile_group.layout()

        # Profile type
        self.profile_type = QComboBox()
        self.profile_type.addItem("Uniform", "uniform")
        self.profile_type.addItem("Gaussian", "gaussian")
        self.profile_type.currentIndexChanged.connect(self._on_profile_changed)
        profile_layout.addRow("Profile:", self.profile_type)

        # Gaussian waist Y
        self.waist_y = QDoubleSpinBox()
        self.waist_y.setDecimals(1)
        self.waist_y.setRange(1, 1e10)
        self.waist_y.setValue(1000)  # 0.1 mm = 1e6 Å
        self.waist_y.setSuffix(" Å")
        self.waist_y.setSingleStep(100)
        self.waist_y.setEnabled(False)
        profile_layout.addRow("Waist Y:", self.waist_y)

        # Gaussian waist Z
        self.waist_z = QDoubleSpinBox()
        self.waist_z.setDecimals(1)
        self.waist_z.setRange(1, 1e10)
        self.waist_z.setValue(1000)  # 0.1 mm = 1e6 Å
        self.waist_z.setSuffix(" Å")
        self.waist_z.setSingleStep(100)
        self.waist_z.setEnabled(False)
        profile_layout.addRow("Waist Z:", self.waist_z)

        # Polarization Group
        pol_group = self.add_group("Polarization")
        pol_layout = pol_group.layout()

        # Polarization rate with slider
        pol_container = QWidget()
        pol_container_layout = QHBoxLayout(pol_container)
        pol_container_layout.setContentsMargins(0, 0, 0, 0)
        pol_container_layout.setSpacing(8)

        self.pol_slider = QSlider(Qt.Horizontal)
        self.pol_slider.setMinimum(0)
        self.pol_slider.setMaximum(100)
        self.pol_slider.setValue(100)
        self.pol_slider.valueChanged.connect(self._on_pol_slider_changed)
        pol_container_layout.addWidget(self.pol_slider, 1)

        self.pol_spinbox = QDoubleSpinBox()
        self.pol_spinbox.setDecimals(2)
        self.pol_spinbox.setRange(0, 1)
        self.pol_spinbox.setValue(0.5)
        self.pol_spinbox.setSingleStep(0.01)
        self.pol_spinbox.valueChanged.connect(self._on_pol_spinbox_changed)
        pol_container_layout.addWidget(self.pol_spinbox)

        pol_layout.addRow("Rate:", pol_container)

        # Advanced Group
        advanced_group = self.add_group("Advanced")
        advanced_layout = advanced_group.layout()

        # Phase tolerance
        self.phase_tol = QDoubleSpinBox()
        self.phase_tol.setDecimals(4)
        self.phase_tol.setRange(0.001, 10)
        self.phase_tol.setValue(0.001)
        self.phase_tol.setSuffix(" rad")
        advanced_layout.addRow("Phase Tolerance:", self.phase_tol)

        # Series mode
        self.series_mode = QCheckBox("Enable series computation")
        self.series_mode.setToolTip("Use series expansion for large samples")
        advanced_layout.addRow("", self.series_mode)

        # Create/Update Beam Button
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 10, 0, 0)
        btn_layout.setSpacing(8)

        create_btn = QPushButton("Create Beam")
        create_btn.clicked.connect(self._on_create_beam)
        create_btn.setStyleSheet("QPushButton { background-color: #2a6a2a; }")
        btn_layout.addWidget(create_btn)

        update_btn = QPushButton("Update Beam")
        update_btn.clicked.connect(self._on_update_beam)
        btn_layout.addWidget(update_btn)

        # Insert before stretch
        self.content_layout.insertWidget(self.content_layout.count() - 1, btn_row)

        # Initialize displays
        self._on_energy_changed()
        self._update_total_rays()

    def _register_observers(self):
        """Register for state change notifications."""
        self.state.register_observer("beam_changed", self._on_beam_state_changed)
        self.state.register_observer("global_working_directory_changed", self._on_global_dir_changed)

    def _on_global_dir_changed(self, directory):
        """Handle global working directory change."""
        # Update placeholder text if directory field is empty
        if not self.directory_edit.text():
            self.directory_edit.setPlaceholderText(
                f"Using global: {directory}" if directory
                else "Select directory for beam files..."
            )

    def _on_beam_state_changed(self, beam):
        """Handle beam state change."""
        self._refresh_display()

    def _on_browse_directory(self):
        """Handle directory browse button click."""
        # Use current text, or global directory, or current working directory
        start_dir = self.directory_edit.text() or self.state.get_default_directory()
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Beam Directory",
            start_dir,
            QFileDialog.ShowDirsOnly
        )

        if directory:
            self.directory_edit.setText(directory)

            # Check if metadata exists and enable/style button accordingly
            metadata_path = Path(directory) / "beam_metadata.json"
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
                self.dir_info_label.setText("Metadata found - click 'Load Existing Beam'")
                self.dir_info_label.setStyleSheet("color: #4ec94e; font-style: italic;")
            else:
                # Disable button and reset style when no metadata
                self.load_existing_btn.setEnabled(False)
                self.load_existing_btn.setStyleSheet("")
                self.dir_info_label.setText("No metadata found - configure parameters and click Create.")
                self.dir_info_label.setStyleSheet("color: #d4a94e; font-style: italic;")

    def _on_load_existing(self):
        """Handle load existing beam button click."""
        directory = self.directory_edit.text()
        if directory:
            self._load_existing_beam(directory)

    def _load_existing_beam(self, directory: str):
        """
        Load an existing beam from directory.

        Args:
            directory: Path to beam directory
        """
        try:
            from Beam import beam

            # Create beam with directory
            new_beam = beam(directory=directory)

            # Load existing metadata
            metadata_path = Path(directory) / "beam_metadata.json"
            if metadata_path.exists():
                new_beam.read_beam_metadata()
                self.dir_info_label.setText("Loaded existing beam from metadata.")
                self.dir_info_label.setStyleSheet("color: #4ec94e; font-style: italic;")

            self.state.beam = new_beam
            self._refresh_display()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load beam:\n{str(e)}")
            self.dir_info_label.setText(f"Error: {str(e)}")
            self.dir_info_label.setStyleSheet("color: #e05050; font-style: italic;")

    def _refresh_display(self):
        """Refresh display from current beam state."""
        beam_obj = self.state.beam
        if beam_obj is None:
            return

        try:
            # Update directory display
            if hasattr(beam_obj, 'directory'):
                self.directory_edit.setText(beam_obj.directory)

            # Energy (stored as _energy)
            if hasattr(beam_obj, '_energy') and beam_obj._energy is not None:
                self.energy.blockSignals(True)
                self.energy.setValue(beam_obj._energy)
                self.energy.blockSignals(False)
                self._update_wavelength(beam_obj._energy)

            # Beam size (stored as _beam_size in Angstroms)
            if hasattr(beam_obj, '_beam_size') and beam_obj._beam_size is not None:
                self.size_y.setValue(beam_obj._beam_size[0])
                self.size_z.setValue(beam_obj._beam_size[1])

            # Beam samples (stored as _beam_samples)
            if hasattr(beam_obj, '_beam_samples') and beam_obj._beam_samples is not None:
                self.samples_y.setValue(beam_obj._beam_samples[0])
                self.samples_z.setValue(beam_obj._beam_samples[1])

            # Beam shape (stored as _beam_shape)
            if hasattr(beam_obj, '_beam_shape') and beam_obj._beam_shape is not None:
                idx = self.shape_type.findData(beam_obj._beam_shape)
                if idx >= 0:
                    self.shape_type.setCurrentIndex(idx)

            # Beam profile (stored as _beam_profile)
            if hasattr(beam_obj, '_beam_profile') and beam_obj._beam_profile is not None:
                idx = self.profile_type.findData(beam_obj._beam_profile)
                if idx >= 0:
                    self.profile_type.setCurrentIndex(idx)

            # Gaussian waist (stored as _gauss_waist in Angstroms)
            if hasattr(beam_obj, '_gauss_waist') and beam_obj._gauss_waist is not None:
                self.waist_y.setValue(beam_obj._gauss_waist[0])
                self.waist_z.setValue(beam_obj._gauss_waist[1])

            # Polarization rate (stored as _pol_perp_rate)
            if hasattr(beam_obj, '_pol_perp_rate') and beam_obj._pol_perp_rate is not None:
                self.pol_spinbox.setValue(beam_obj._pol_perp_rate)

        except Exception:
            pass

    def _on_energy_changed(self):
        """Handle energy value change."""
        energy = self.energy.value()
        self._update_wavelength(energy)

    def _update_wavelength(self, energy):
        """Update wavelength display from energy."""
        wavelength = HC_EV_A / energy
        self.wavelength_label.setText(f"{wavelength:.4f} Å")

    def _on_shape_changed(self, index):
        """Handle shape type change."""
        shape = self.shape_type.currentData()
        # Could adjust UI based on shape if needed

    def _on_profile_changed(self, index):
        """Handle profile type change."""
        is_gaussian = self.profile_type.currentData() == "gaussian"
        self.waist_y.setEnabled(is_gaussian)
        self.waist_z.setEnabled(is_gaussian)

    def _update_total_rays(self):
        """Update total rays display."""
        total = self.samples_y.value() * self.samples_z.value()
        self.total_rays.setText(f"Grid Points: {total:,}")

    def _on_pol_slider_changed(self, value):
        """Handle polarization slider change."""
        self.pol_spinbox.blockSignals(True)
        self.pol_spinbox.setValue(value / 100.0)
        self.pol_spinbox.blockSignals(False)

    def _on_pol_spinbox_changed(self, value):
        """Handle polarization spinbox change."""
        self.pol_slider.blockSignals(True)
        self.pol_slider.setValue(int(value * 100))
        self.pol_slider.blockSignals(False)

    def _on_create_beam(self):
        """Handle create beam button."""
        # Check if directory is selected
        directory = self.directory_edit.text()
        if not directory:
            QMessageBox.warning(
                self,
                "Directory Required",
                "Please select a directory for beam files first."
            )
            return

        try:
            from Beam import beam

            # Check if beam already exists in state with this directory
            existing_beam = self.state.beam
            if existing_beam is not None and hasattr(existing_beam, 'directory'):
                if existing_beam.directory == directory:
                    # Use existing beam object
                    self._apply_settings_to_beam(existing_beam)
                    self.state.notify_object_modified("beam")
                    self.beam_created.emit(existing_beam)
                    QMessageBox.information(self, "Success", "Beam configured successfully.")
                    return

            # Create new beam with directory
            new_beam = beam(directory=directory)
            self._apply_settings_to_beam(new_beam)

            self.state.beam = new_beam
            self.beam_created.emit(new_beam)

            self.dir_info_label.setText("Beam created successfully.")
            self.dir_info_label.setStyleSheet("color: #4ec94e; font-style: italic;")

            QMessageBox.information(self, "Success", "Beam created successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create beam:\n{str(e)}")

    def _on_update_beam(self):
        """Handle update beam button."""
        existing_beam = self.state.beam
        if existing_beam is None:
            QMessageBox.warning(self, "No Beam", "Please create a beam first.")
            return

        try:
            self._apply_settings_to_beam(existing_beam)
            self.state.notify_object_modified("beam")
            self.beam_updated.emit()

            QMessageBox.information(self, "Success", "Beam updated successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update beam:\n{str(e)}")

    def _apply_settings_to_beam(self, beam_obj):
        """Apply current UI settings to a beam object."""
        # Gather parameters
        energy = self.energy.value()
        beam_shape = self.shape_type.currentData()

        # Beam size already in Angstroms
        beam_size = (self.size_y.value(), self.size_z.value())
        beam_samples = (self.samples_y.value(), self.samples_z.value())
        pol_perp_rate = self.pol_spinbox.value()

        # Profile parameters
        beam_profile = self.profile_type.currentData()
        gaussian_waist = None
        if beam_profile == "gaussian":
            # Waist already in Angstroms
            gaussian_waist = (self.waist_y.value(), self.waist_z.value())

        # Create beam with correct parameter names matching Beam.py API
        if hasattr(beam_obj, 'create_beam'):
            kwargs = {
                'energy': energy,
                'eV': True,
                'beam_shape': beam_shape,
                'beam_size': beam_size,
                'beam_samples': beam_samples,
                'beam_profile': beam_profile,
                'pol_perp_rate': pol_perp_rate,
            }

            if beam_profile == "gaussian" and gaussian_waist:
                kwargs['gaussian_waist'] = gaussian_waist

            beam_obj.create_beam(**kwargs)

            # Apply advanced settings
            if hasattr(beam_obj, 'set_phase_tolerance'):
                beam_obj.set_phase_tolerance(self.phase_tol.value())

            # Save metadata after creation
            if hasattr(beam_obj, 'write_beam_metadata'):
                beam_obj.write_beam_metadata()

    def get_config(self) -> dict:
        """Get the current configuration."""
        return {
            "energy": self.energy.value(),
            "shape": self.shape_type.currentData(),
            "beam_size": [self.size_y.value(), self.size_z.value()],
            "samples": [self.samples_y.value(), self.samples_z.value()],
            "profile": self.profile_type.currentData(),
            "waist": [self.waist_y.value(), self.waist_z.value()],
            "polarization_rate": self.pol_spinbox.value(),
            "phase_tolerance": self.phase_tol.value(),
            "series_mode": self.series_mode.isChecked(),
        }

    def set_config(self, config: dict):
        """Set configuration from dict."""
        if "energy" in config:
            self.energy.setValue(config["energy"])

        if "shape" in config:
            idx = self.shape_type.findData(config["shape"])
            if idx >= 0:
                self.shape_type.setCurrentIndex(idx)

        if "beam_size" in config:
            size = config["beam_size"]
            self.size_y.setValue(size[0])
            self.size_z.setValue(size[1])

        if "samples" in config:
            samples = config["samples"]
            self.samples_y.setValue(samples[0])
            self.samples_z.setValue(samples[1])

        if "profile" in config:
            idx = self.profile_type.findData(config["profile"])
            if idx >= 0:
                self.profile_type.setCurrentIndex(idx)

        if "waist" in config:
            waist = config["waist"]
            self.waist_y.setValue(waist[0])
            self.waist_z.setValue(waist[1])

        if "polarization_rate" in config:
            self.pol_spinbox.setValue(config["polarization_rate"])

        if "phase_tolerance" in config:
            self.phase_tol.setValue(config["phase_tolerance"])

        if "series_mode" in config:
            self.series_mode.setChecked(config["series_mode"])
