# -----------------------------------------------------------------------------
# Crystal Inspector
# -----------------------------------------------------------------------------
"""
Inspector panel for crystal structure configuration.

Provides controls for:
- Loading CIF files
- Setting crystal orientation (primary/secondary hkl)
- Lab frame alignment vectors
- Crystal rotation
- Viewing lattice parameters
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
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QCheckBox,
    QAbstractItemView,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gui.state import SimulationState
from gui.panels.inspector import InspectorPanel, PropertyDef, Vector3Widget
from gui.utils.diffraction_calc import DiffractionCalculator


class CrystalInspector(InspectorPanel):
    """
    Inspector for crystal structure configuration.

    Signals:
        crystal_loaded: Emitted when a CIF file is loaded
        orientation_changed: Emitted when orientation is modified
    """

    crystal_loaded = Signal(object)
    orientation_changed = Signal()
    peak_selection_changed = Signal(list)  # Emits list of selected (h,k,l) tuples
    align_to_peak_requested = Signal(tuple)  # Emits (h,k,l) tuple for alignment

    def __init__(self, state: SimulationState, parent=None):
        """
        Initialize the crystal inspector.

        Args:
            state: SimulationState instance
            parent: Parent widget
        """
        super().__init__(state, parent)
        self.set_title("Crystal")

        # Initialize diffraction calculator
        self._diff_calc = DiffractionCalculator()
        self._selected_peaks = []  # List of selected (h,k,l) tuples
        self._peak_data = []  # Cached list of peak info dicts

        self._setup_crystal_ui()
        self._register_observers()

    def _setup_crystal_ui(self):
        """Setup crystal-specific UI elements."""
        # CIF File Group
        cif_group = self.add_group("CIF File")
        cif_layout = cif_group.layout()

        # File path row
        file_row = QWidget()
        file_layout = QHBoxLayout(file_row)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setSpacing(4)

        self.cif_path_edit = QLineEdit()
        self.cif_path_edit.setPlaceholderText("Select a CIF file...")
        self.cif_path_edit.setReadOnly(True)
        file_layout.addWidget(self.cif_path_edit, 1)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse_cif)
        file_layout.addWidget(browse_btn)

        cif_layout.addRow("File:", file_row)

        # Load button
        load_btn = QPushButton("Load Crystal")
        load_btn.clicked.connect(self._on_load_crystal)
        cif_layout.addRow("", load_btn)

        # Lattice Parameters Group (read-only display)
        lattice_group = self.add_group("Lattice Parameters")
        lattice_layout = lattice_group.layout()

        self.lattice_table = QTableWidget(3, 2)
        self.lattice_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.lattice_table.verticalHeader().setVisible(False)
        self.lattice_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.lattice_table.setMaximumHeight(120)

        # Initialize with placeholders
        params = [("a, b, c (Å)", "--"), ("α, β, γ (°)", "--"), ("Volume", "--")]
        for i, (param, val) in enumerate(params):
            self.lattice_table.setItem(i, 0, QTableWidgetItem(param))
            self.lattice_table.setItem(i, 1, QTableWidgetItem(val))

        self.lattice_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.lattice_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

        lattice_layout.addRow(self.lattice_table)

        # Orientation Group
        orient_group = self.add_group("Orientation")
        orient_layout = orient_group.layout()

        # Primary hkl
        primary_row = QWidget()
        primary_layout = QHBoxLayout(primary_row)
        primary_layout.setContentsMargins(0, 0, 0, 0)
        primary_layout.setSpacing(4)

        self.primary_h = QSpinBox()
        self.primary_h.setRange(-20, 20)
        self.primary_h.setValue(0)
        self.primary_k = QSpinBox()
        self.primary_k.setRange(-20, 20)
        self.primary_k.setValue(0)
        self.primary_l = QSpinBox()
        self.primary_l.setRange(-20, 20)
        self.primary_l.setValue(1)

        for label, sb in [("h:", self.primary_h), ("k:", self.primary_k), ("l:", self.primary_l)]:
            primary_layout.addWidget(QLabel(label))
            primary_layout.addWidget(sb)
            sb.valueChanged.connect(self._on_orientation_changed)

        orient_layout.addRow("Primary (hkl):", primary_row)

        # Secondary hkl
        secondary_row = QWidget()
        secondary_layout = QHBoxLayout(secondary_row)
        secondary_layout.setContentsMargins(0, 0, 0, 0)
        secondary_layout.setSpacing(4)

        self.secondary_h = QSpinBox()
        self.secondary_h.setRange(-20, 20)
        self.secondary_h.setValue(0)
        self.secondary_k = QSpinBox()
        self.secondary_k.setRange(-20, 20)
        self.secondary_k.setValue(1)
        self.secondary_l = QSpinBox()
        self.secondary_l.setRange(-20, 20)
        self.secondary_l.setValue(0)

        for label, sb in [("h:", self.secondary_h), ("k:", self.secondary_k), ("l:", self.secondary_l)]:
            secondary_layout.addWidget(QLabel(label))
            secondary_layout.addWidget(sb)
            sb.valueChanged.connect(self._on_orientation_changed)

        orient_layout.addRow("Secondary (hkl):", secondary_row)

        # Lab Frame Group
        lab_group = self.add_group("Lab Frame Alignment")
        lab_layout = lab_group.layout()

        # Primary alignment vector
        self.primary_vec = Vector3Widget([0, 0, 1], decimals=4)
        self.primary_vec.value_changed.connect(self._on_orientation_changed)
        lab_layout.addRow("Primary Vector:", self.primary_vec)

        # Secondary alignment vector
        self.secondary_vec = Vector3Widget([0, 1, 0], decimals=4)
        self.secondary_vec.value_changed.connect(self._on_orientation_changed)
        lab_layout.addRow("Secondary Vector:", self.secondary_vec)

        # Apply orientation button
        apply_orient_btn = QPushButton("Apply Orientation")
        apply_orient_btn.clicked.connect(self._on_apply_orientation)
        lab_layout.addRow("", apply_orient_btn)

        # Rotation Group
        rotation_group = self.add_group("Additional Rotation")
        rotation_layout = rotation_group.layout()

        # Rotation axis
        self.rotation_axis = Vector3Widget([0, 0, 1], decimals=4)
        rotation_layout.addRow("Axis:", self.rotation_axis)

        # Rotation angle
        self.rotation_angle = QDoubleSpinBox()
        self.rotation_angle.setDecimals(4)
        self.rotation_angle.setRange(-180, 180)
        self.rotation_angle.setValue(0)
        self.rotation_angle.setSuffix(" °")
        rotation_layout.addRow("Angle:", self.rotation_angle)

        # Apply rotation button
        apply_rot_btn = QPushButton("Apply Rotation")
        apply_rot_btn.clicked.connect(self._on_apply_rotation)
        rotation_layout.addRow("", apply_rot_btn)

        # d-spacing Calculator Group
        dspacing_group = self.add_group("d-spacing Calculator")
        dspacing_layout = dspacing_group.layout()

        # hkl input
        calc_hkl_row = QWidget()
        calc_hkl_layout = QHBoxLayout(calc_hkl_row)
        calc_hkl_layout.setContentsMargins(0, 0, 0, 0)
        calc_hkl_layout.setSpacing(4)

        self.calc_h = QSpinBox()
        self.calc_h.setRange(-20, 20)
        self.calc_k = QSpinBox()
        self.calc_k.setRange(-20, 20)
        self.calc_l = QSpinBox()
        self.calc_l.setRange(-20, 20)
        self.calc_l.setValue(1)

        for label, sb in [("h:", self.calc_h), ("k:", self.calc_k), ("l:", self.calc_l)]:
            calc_hkl_layout.addWidget(QLabel(label))
            calc_hkl_layout.addWidget(sb)

        dspacing_layout.addRow("(hkl):", calc_hkl_row)

        # Calculate button
        calc_btn = QPushButton("Calculate")
        calc_btn.clicked.connect(self._on_calculate_dspacing)
        dspacing_layout.addRow("", calc_btn)

        # Result
        self.dspacing_result = QLabel("d = -- Å")
        self.dspacing_result.setStyleSheet("font-weight: bold; color: #4ec94e;")
        dspacing_layout.addRow("Result:", self.dspacing_result)

        # Diffraction Peaks Group
        peaks_group = self.add_group("Diffraction Peaks")
        peaks_layout = peaks_group.layout()

        # Energy/wavelength info
        self.energy_info_label = QLabel("Energy: -- eV  |  λ = -- Å")
        self.energy_info_label.setStyleSheet("color: #888;")
        peaks_layout.addRow(self.energy_info_label)

        # Max indices row
        max_idx_row = QWidget()
        max_idx_layout = QHBoxLayout(max_idx_row)
        max_idx_layout.setContentsMargins(0, 0, 0, 0)
        max_idx_layout.setSpacing(4)

        self.max_h = QSpinBox()
        self.max_h.setRange(1, 20)
        self.max_h.setValue(3)
        self.max_k = QSpinBox()
        self.max_k.setRange(1, 20)
        self.max_k.setValue(3)
        self.max_l = QSpinBox()
        self.max_l.setRange(1, 20)
        self.max_l.setValue(3)

        max_idx_layout.addWidget(QLabel("Max h:"))
        max_idx_layout.addWidget(self.max_h)
        max_idx_layout.addWidget(QLabel("k:"))
        max_idx_layout.addWidget(self.max_k)
        max_idx_layout.addWidget(QLabel("l:"))
        max_idx_layout.addWidget(self.max_l)

        peaks_layout.addRow("Indices:", max_idx_row)

        # Filter row - include only specific h,k,l values (empty = show all)
        filter_row = QWidget()
        filter_layout = QHBoxLayout(filter_row)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setSpacing(4)

        filter_layout.addWidget(QLabel("Show:"))

        self.filter_h = QLineEdit()
        self.filter_h.setPlaceholderText("h")
        self.filter_h.setMaximumWidth(40)
        self.filter_h.setToolTip("Only show peaks with this h value (empty = all)")
        filter_layout.addWidget(self.filter_h)

        self.filter_k = QLineEdit()
        self.filter_k.setPlaceholderText("k")
        self.filter_k.setMaximumWidth(40)
        self.filter_k.setToolTip("Only show peaks with this k value (empty = all)")
        filter_layout.addWidget(self.filter_k)

        self.filter_l = QLineEdit()
        self.filter_l.setPlaceholderText("l")
        self.filter_l.setMaximumWidth(40)
        self.filter_l.setToolTip("Only show peaks with this l value (empty = all)")
        filter_layout.addWidget(self.filter_l)

        self.filter_zero_F = QCheckBox("F=0")
        self.filter_zero_F.setToolTip("Show peaks with zero structure factor")
        self.filter_zero_F.setChecked(False)
        filter_layout.addWidget(self.filter_zero_F)

        filter_layout.addStretch()
        peaks_layout.addRow("Filter:", filter_row)

        # Refresh button
        refresh_btn = QPushButton("Refresh Peaks")
        refresh_btn.clicked.connect(self._on_refresh_peaks)
        peaks_layout.addRow("", refresh_btn)

        # Peaks table - now with 6 columns including |F|
        self.peaks_table = QTableWidget(0, 6)
        self.peaks_table.setHorizontalHeaderLabels(["Show", "(h,k,l)", "d (Å)", "2θ (°)", "|F|", "Status"])
        self.peaks_table.verticalHeader().setVisible(False)
        self.peaks_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.peaks_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.peaks_table.setMinimumHeight(150)
        self.peaks_table.setMaximumHeight(250)

        # Set column widths
        self.peaks_table.setColumnWidth(0, 40)   # Show checkbox
        self.peaks_table.setColumnWidth(1, 70)   # hkl
        self.peaks_table.setColumnWidth(2, 60)   # d-spacing
        self.peaks_table.setColumnWidth(3, 55)   # 2theta
        self.peaks_table.setColumnWidth(4, 55)   # |F| structure factor
        self.peaks_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)  # Status

        peaks_layout.addRow(self.peaks_table)

        # Buttons row
        peaks_btn_row = QWidget()
        peaks_btn_layout = QHBoxLayout(peaks_btn_row)
        peaks_btn_layout.setContentsMargins(0, 0, 0, 0)
        peaks_btn_layout.setSpacing(4)

        show_3d_btn = QPushButton("Show in 3D")
        show_3d_btn.clicked.connect(self._on_show_peaks_3d)
        peaks_btn_layout.addWidget(show_3d_btn)

        align_btn = QPushButton("Align to Peak")
        align_btn.clicked.connect(self._on_align_to_peak)
        peaks_btn_layout.addWidget(align_btn)

        peaks_layout.addRow("", peaks_btn_row)

    def _register_observers(self):
        """Register for state change notifications."""
        self.state.register_observer("crystal_changed", self._on_crystal_state_changed)
        self.state.register_observer("beam_changed", self._on_beam_state_changed)
        self.state.register_observer("stage_changed", self._on_stage_state_changed)

    def _on_crystal_state_changed(self, crystal):
        """Handle crystal state change."""
        self._refresh_display()

    def _refresh_display(self):
        """Refresh display from current crystal state."""
        crystal = self.state.crystal
        if crystal is None:
            self.cif_path_edit.setText("")
            self._clear_lattice_params()
            return

        # Update CIF path if available (Crystal uses 'filepath' attribute)
        if hasattr(crystal, 'filepath') and crystal.filepath:
            self.cif_path_edit.setText(str(crystal.filepath))

        # Update lattice parameters
        self._update_lattice_params(crystal)

    def _clear_lattice_params(self):
        """Clear lattice parameter display."""
        self.lattice_table.setItem(0, 1, QTableWidgetItem("--"))
        self.lattice_table.setItem(1, 1, QTableWidgetItem("--"))
        self.lattice_table.setItem(2, 1, QTableWidgetItem("--"))

    def _update_lattice_params(self, crystal):
        """Update lattice parameter display."""
        try:
            # Get lattice lengths (a, b, c) from conventional cell
            if hasattr(crystal, 'lattice_lengths_conventional') and crystal.lattice_lengths_conventional is not None:
                lengths = crystal.lattice_lengths_conventional
                a, b, c = lengths[0], lengths[1], lengths[2]
                self.lattice_table.setItem(0, 1, QTableWidgetItem(f"{a:.4f}, {b:.4f}, {c:.4f}"))

                # Calculate angles from lattice matrix (conventional cell)
                if hasattr(crystal, 'lattice_matrix_conventional') and crystal.lattice_matrix_conventional is not None:
                    matrix = crystal.lattice_matrix_conventional
                    # Lattice vectors are rows of the matrix
                    va, vb, vc = matrix[0], matrix[1], matrix[2]
                    # Calculate angles between vectors
                    alpha = np.degrees(np.arccos(np.clip(np.dot(vb, vc) / (np.linalg.norm(vb) * np.linalg.norm(vc)), -1, 1)))
                    beta = np.degrees(np.arccos(np.clip(np.dot(va, vc) / (np.linalg.norm(va) * np.linalg.norm(vc)), -1, 1)))
                    gamma = np.degrees(np.arccos(np.clip(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)), -1, 1)))
                    self.lattice_table.setItem(1, 1, QTableWidgetItem(f"{alpha:.2f}, {beta:.2f}, {gamma:.2f}"))
                else:
                    self.lattice_table.setItem(1, 1, QTableWidgetItem("--"))
            elif hasattr(crystal, 'lattice_lengths') and crystal.lattice_lengths is not None:
                # Fallback to primitive cell
                lengths = crystal.lattice_lengths
                a, b, c = lengths[0], lengths[1], lengths[2]
                self.lattice_table.setItem(0, 1, QTableWidgetItem(f"{a:.4f}, {b:.4f}, {c:.4f} (prim)"))
                self.lattice_table.setItem(1, 1, QTableWidgetItem("--"))
            else:
                self.lattice_table.setItem(0, 1, QTableWidgetItem("Not loaded"))
                self.lattice_table.setItem(1, 1, QTableWidgetItem("--"))

            # Show volume as additional info (no space group available in Crystal class)
            if hasattr(crystal, 'lattice_volume_conventional') and crystal.lattice_volume_conventional is not None:
                vol = crystal.lattice_volume_conventional
                self.lattice_table.setItem(2, 1, QTableWidgetItem(f"V = {vol:.2f} Å³"))
            else:
                self.lattice_table.setItem(2, 1, QTableWidgetItem("--"))

        except Exception as e:
            self.lattice_table.setItem(0, 1, QTableWidgetItem(f"Error: {str(e)[:20]}"))
            self.lattice_table.setItem(1, 1, QTableWidgetItem("--"))
            self.lattice_table.setItem(2, 1, QTableWidgetItem("--"))

    def _on_browse_cif(self):
        """Handle browse CIF file button."""
        databases_path = Path(self.state.working_directory) / "databases" / "lattice"
        if not databases_path.exists():
            databases_path = Path(self.state.working_directory)

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select CIF File",
            str(databases_path),
            "CIF Files (*.cif);;All Files (*)"
        )
        if filename:
            self.cif_path_edit.setText(filename)

    def _on_load_crystal(self):
        """Handle load crystal button."""
        cif_path = self.cif_path_edit.text()
        if not cif_path:
            QMessageBox.warning(self, "No File", "Please select a CIF file first.")
            return

        if not Path(cif_path).exists():
            QMessageBox.warning(self, "File Not Found", f"File not found: {cif_path}")
            return

        try:
            from Crystal import crystal
            new_crystal = crystal(cif_path)
            # Must call get_lattice_from_cif() to actually load and process the CIF file
            new_crystal.get_lattice_from_cif()
            self.state.crystal = new_crystal
            self.crystal_loaded.emit(new_crystal)
            self._refresh_display()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load crystal:\n{str(e)}")

    def _on_orientation_changed(self):
        """Handle orientation parameter change."""
        self.orientation_changed.emit()

    def _on_apply_orientation(self):
        """Apply the orientation settings to the crystal."""
        crystal = self.state.crystal
        if crystal is None:
            QMessageBox.warning(self, "No Crystal", "Please load a crystal first.")
            return

        try:
            # Get hkl values as column vectors
            primary_hkl = np.array([self.primary_h.value(), self.primary_k.value(), self.primary_l.value()])
            secondary_hkl = np.array([self.secondary_h.value(), self.secondary_k.value(), self.secondary_l.value()])

            # Build orientation_array: shape (3, 2) - column 0 is primary, column 1 is secondary
            orientation_array = np.column_stack([primary_hkl, secondary_hkl])

            # Get lab frame vectors
            primary_lab = self.primary_vec.get_value()
            secondary_lab = self.secondary_vec.get_value()

            # Build alignment_array: shape (3, 2) - column 0 is primary axis, column 1 is plane
            alignment_array = np.column_stack([primary_lab, secondary_lab])

            # Apply orientation using the correct signature
            if hasattr(crystal, 'align_axes'):
                crystal.align_axes(orientation_array, alignment_array)
                self.state.notify_object_modified("crystal")
                self._refresh_display()
                QMessageBox.information(self, "Success", "Orientation applied successfully.")
            else:
                QMessageBox.warning(self, "Not Supported", "Crystal object doesn't support align_axes.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply orientation:\n{str(e)}")

    def _on_apply_rotation(self):
        """Apply additional rotation to the crystal."""
        crystal = self.state.crystal
        if crystal is None:
            QMessageBox.warning(self, "No Crystal", "Please load a crystal first.")
            return

        try:
            axis = self.rotation_axis.get_value()
            angle_deg = self.rotation_angle.value()
            angle_rad = np.radians(angle_deg)  # Convert degrees to radians

            if hasattr(crystal, 'rotate_crystal') and hasattr(crystal, 'get_rotation'):
                # Crystal.rotate_crystal expects a rotation matrix, not axis/angle
                # Use Crystal.get_rotation(axis, angle) to create the rotation matrix
                rotation_matrix = crystal.get_rotation(axis, angle_rad)
                crystal.rotate_crystal(rotation_matrix)
                self.state.notify_object_modified("crystal")
                self._refresh_display()
                QMessageBox.information(self, "Success", f"Rotated {angle_deg}° about axis.")
            else:
                QMessageBox.warning(self, "Not Supported", "Crystal object doesn't support rotate_crystal.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply rotation:\n{str(e)}")

    def _on_calculate_dspacing(self):
        """Calculate d-spacing for the specified hkl."""
        crystal = self.state.crystal
        if crystal is None:
            QMessageBox.warning(self, "No Crystal", "Please load a crystal first.")
            return

        try:
            h = self.calc_h.value()
            k = self.calc_k.value()
            l = self.calc_l.value()

            if hasattr(crystal, 'get_dhkl'):
                d = crystal.get_dhkl([h, k, l])
                self.dspacing_result.setText(f"d = {d:.6f} Å")
            else:
                self.dspacing_result.setText("d = -- Å (not supported)")
        except Exception as e:
            self.dspacing_result.setText(f"Error: {str(e)}")

    def get_orientation_config(self) -> dict:
        """Get the current orientation configuration."""
        return {
            "primary_hkl": [self.primary_h.value(), self.primary_k.value(), self.primary_l.value()],
            "secondary_hkl": [self.secondary_h.value(), self.secondary_k.value(), self.secondary_l.value()],
            "primary_lab": self.primary_vec.get_value().tolist(),
            "secondary_lab": self.secondary_vec.get_value().tolist(),
        }

    def set_orientation_config(self, config: dict):
        """Set orientation from configuration."""
        if "primary_hkl" in config:
            hkl = config["primary_hkl"]
            self.primary_h.setValue(hkl[0])
            self.primary_k.setValue(hkl[1])
            self.primary_l.setValue(hkl[2])

        if "secondary_hkl" in config:
            hkl = config["secondary_hkl"]
            self.secondary_h.setValue(hkl[0])
            self.secondary_k.setValue(hkl[1])
            self.secondary_l.setValue(hkl[2])

        if "primary_lab" in config:
            self.primary_vec.set_value(config["primary_lab"])

        if "secondary_lab" in config:
            self.secondary_vec.set_value(config["secondary_lab"])

    # -------------------------------------------------------------------------
    # Diffraction Peaks Methods
    # -------------------------------------------------------------------------

    def _on_beam_state_changed(self, beam):
        """Handle beam state change - update energy info and peaks."""
        self._update_energy_info()
        self._update_diff_calc_refs()

    def _on_stage_state_changed(self, stage):
        """Handle stage state change - update peaks if needed."""
        self._update_diff_calc_refs()

    def _update_diff_calc_refs(self):
        """Update diffraction calculator with current objects."""
        self._diff_calc.set_crystal(self.state.crystal)
        self._diff_calc.set_beam(self.state.beam)
        self._diff_calc.set_stage(self.state.stage)

    def _update_energy_info(self):
        """Update the energy/wavelength display."""
        beam = self.state.beam
        if beam is None or beam._energy is None:
            self.energy_info_label.setText("Energy: -- eV  |  λ = -- Å")
            return

        try:
            energy_eV = beam._energy
            wavelength_A = beam._wavelength * 1e10  # Convert m to Å
            self.energy_info_label.setText(f"Energy: {energy_eV:.1f} eV  |  λ = {wavelength_A:.4f} Å")
        except Exception:
            self.energy_info_label.setText("Energy: -- eV  |  λ = -- Å")

    def _parse_filter_value(self, text: str):
        """Parse a single filter value from text input.

        Args:
            text: A single integer value or empty string

        Returns:
            Integer value if valid, None if empty or invalid.
        """
        text = text.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    def _on_refresh_peaks(self):
        """Refresh the peaks table with accessible reflections."""
        crystal = self.state.crystal
        beam = self.state.beam

        if crystal is None:
            QMessageBox.warning(self, "No Crystal", "Please load a crystal first.")
            return

        if beam is None or beam._energy is None:
            QMessageBox.warning(self, "No Beam", "Please configure the beam energy first.")
            return

        try:
            # Update diffraction calculator
            self._update_diff_calc_refs()
            self._update_energy_info()

            # Get max indices
            max_h = self.max_h.value()
            max_k = self.max_k.value()
            max_l = self.max_l.value()

            # Get inclusive filter values (None means show all)
            include_h = self._parse_filter_value(self.filter_h.text())
            include_k = self._parse_filter_value(self.filter_k.text())
            include_l = self._parse_filter_value(self.filter_l.text())
            show_zero_F = self.filter_zero_F.isChecked()

            # Enumerate accessible reflections with filtering
            self._peak_data = []
            for hkl in self._diff_calc.enumerate_accessible_reflections(max_h, max_k, max_l):
                h, k, l = hkl

                # Apply h,k,l inclusive filters (None = no filter, show all)
                if include_h is not None and h != include_h:
                    continue
                if include_k is not None and k != include_k:
                    continue
                if include_l is not None and l != include_l:
                    continue

                info = self._diff_calc.get_reflection_info(hkl)

                # Filter out zero structure factor peaks unless checkbox is checked
                if not show_zero_F:
                    F = info.get('structure_factor', None)
                    if F is not None and F < 0.1:  # Threshold for "zero"
                        continue

                self._peak_data.append(info)

            # Sort by 2-theta angle
            self._peak_data.sort(key=lambda x: x.get('two_theta_deg', 0) or 0)

            # Update table
            self._update_peaks_table()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate peaks:\n{str(e)}")

    def _update_peaks_table(self):
        """Update the peaks table with current peak data."""
        self.peaks_table.setRowCount(len(self._peak_data))

        for row, info in enumerate(self._peak_data):
            hkl = info['hkl']

            # Checkbox for showing in 3D
            checkbox = QCheckBox()
            checkbox.setChecked(hkl in self._selected_peaks)
            checkbox.stateChanged.connect(lambda state, h=hkl: self._on_peak_checkbox_changed(h, state))

            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.setAlignment(Qt.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            self.peaks_table.setCellWidget(row, 0, checkbox_widget)

            # hkl
            hkl_item = QTableWidgetItem(f"({hkl[0]},{hkl[1]},{hkl[2]})")
            hkl_item.setFlags(hkl_item.flags() & ~Qt.ItemIsEditable)
            self.peaks_table.setItem(row, 1, hkl_item)

            # d-spacing
            d = info.get('d_spacing', 0)
            d_item = QTableWidgetItem(f"{d:.3f}")
            d_item.setFlags(d_item.flags() & ~Qt.ItemIsEditable)
            self.peaks_table.setItem(row, 2, d_item)

            # 2-theta
            two_theta = info.get('two_theta_deg', 0)
            tt_item = QTableWidgetItem(f"{two_theta:.1f}" if two_theta else "--")
            tt_item.setFlags(tt_item.flags() & ~Qt.ItemIsEditable)
            self.peaks_table.setItem(row, 3, tt_item)

            # Structure factor |F|
            F = info.get('structure_factor', None)
            if F is not None:
                F_item = QTableWidgetItem(f"{F:.1f}")
            else:
                F_item = QTableWidgetItem("--")
            F_item.setFlags(F_item.flags() & ~Qt.ItemIsEditable)
            self.peaks_table.setItem(row, 4, F_item)

            # Status
            status = "-"
            status_item = QTableWidgetItem(status)
            status_item.setFlags(status_item.flags() & ~Qt.ItemIsEditable)
            self.peaks_table.setItem(row, 5, status_item)

    def _on_peak_checkbox_changed(self, hkl, state):
        """Handle peak checkbox change."""
        # stateChanged emits int: 0=Unchecked, 2=Checked
        if state == Qt.CheckState.Checked.value:
            if hkl not in self._selected_peaks:
                self._selected_peaks.append(hkl)
        else:
            if hkl in self._selected_peaks:
                self._selected_peaks.remove(hkl)

        # Emit signal with selected peaks
        self.peak_selection_changed.emit(list(self._selected_peaks))

    def _on_show_peaks_3d(self):
        """Handle show peaks in 3D button."""
        if not self._selected_peaks:
            QMessageBox.information(self, "No Selection", "Please select peaks to show in 3D.")
            return

        # Emit signal - 3D viewport will handle the visualization
        self.peak_selection_changed.emit(list(self._selected_peaks))

    def _on_align_to_peak(self):
        """Handle align to peak button."""
        # Get selected row
        selected_rows = self.peaks_table.selectedIndexes()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select a peak from the table.")
            return

        row = selected_rows[0].row()
        if row < 0 or row >= len(self._peak_data):
            return

        hkl = self._peak_data[row]['hkl']

        # Emit signal - alignment dialog will handle the alignment
        self.align_to_peak_requested.emit(hkl)

    def get_selected_peaks(self) -> list:
        """Get list of currently selected peak (h,k,l) tuples."""
        return list(self._selected_peaks)

    def get_peak_data(self) -> list:
        """Get cached peak data list."""
        return self._peak_data
