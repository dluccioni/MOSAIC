# -----------------------------------------------------------------------------
# Optics Inspector
# -----------------------------------------------------------------------------
"""
Inspector panel for optical components configuration.

Provides controls for:
- Adding/removing optical components
- Reordering component stack
- Component-specific parameters
"""

import sys
from pathlib import Path
import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QPushButton, QComboBox,
    QListWidget, QListWidgetItem, QMessageBox, QStackedWidget,
    QLineEdit, QFileDialog,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gui.state import SimulationState
from gui.panels.inspector import InspectorPanel, Vector3Widget


class OpticsInspector(InspectorPanel):
    """Inspector for optical components configuration."""

    optics_created = Signal(object)
    component_added = Signal(str)

    COMPONENT_TYPES = [
        ("free_space", "Free Space"),
        ("crl", "Compound Refractive Lens (CRL)"),
        ("bragg", "Bragg Magnifier"),
        ("aperture", "Aperture"),
        ("angular_filter", "Angular Filter"),
    ]

    def __init__(self, state: SimulationState, parent=None):
        super().__init__(state, parent)
        self.set_title("Optics")
        self._setup_optics_ui()
        self._register_observers()

    def _setup_optics_ui(self):
        """Setup optics-specific UI elements."""
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
            self.directory_edit.setPlaceholderText("Select directory for optics files...")
        self.directory_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse_directory)
        dir_hlayout.addWidget(self.directory_edit)
        dir_hlayout.addWidget(browse_btn)
        dir_layout.addRow(dir_widget)

        # Load existing button
        self.load_existing_btn = QPushButton("Load Existing Optics")
        self.load_existing_btn.clicked.connect(self._load_existing_optics)
        self.load_existing_btn.setEnabled(False)
        dir_layout.addRow(self.load_existing_btn)

        # Component Stack Group
        stack_group = self.add_group("Component Stack")
        stack_layout = stack_group.layout()

        # Component list
        self.component_list = QListWidget()
        self.component_list.setMaximumHeight(150)
        self.component_list.itemSelectionChanged.connect(self._on_component_selected)
        stack_layout.addRow(self.component_list)

        # Stack control buttons
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)

        up_btn = QPushButton("↑")
        up_btn.setMaximumWidth(40)
        up_btn.clicked.connect(self._on_move_up)
        btn_layout.addWidget(up_btn)

        down_btn = QPushButton("↓")
        down_btn.setMaximumWidth(40)
        down_btn.clicked.connect(self._on_move_down)
        btn_layout.addWidget(down_btn)

        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._on_remove_component)
        btn_layout.addWidget(remove_btn)

        btn_layout.addStretch()
        stack_layout.addRow(btn_row)

        # Add Component Group
        add_group = self.add_group("Add Component")
        add_layout = add_group.layout()

        # Component type selector
        self.type_combo = QComboBox()
        for comp_id, comp_name in self.COMPONENT_TYPES:
            self.type_combo.addItem(comp_name, comp_id)
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        add_layout.addRow("Type:", self.type_combo)

        # Parameter stack (changes based on type)
        self.param_stack = QStackedWidget()
        self._create_param_widgets()
        add_layout.addRow(self.param_stack)

        # Add button
        add_btn = QPushButton("Add Component")
        add_btn.clicked.connect(self._on_add_component)
        add_layout.addRow("", add_btn)

        # Save/Create Buttons Row
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 0, 0, 0)

        # Create Optics Button
        create_btn = QPushButton("Create Optics Stack")
        create_btn.clicked.connect(self._on_create_optics)
        create_btn.setStyleSheet("QPushButton { background-color: #2a6a2a; }")
        buttons_layout.addWidget(create_btn)

        # Save Stack Button
        self.save_stack_btn = QPushButton("Save Stack")
        self.save_stack_btn.clicked.connect(self._on_save_stack)
        self.save_stack_btn.setToolTip("Save component stack to metadata file")
        self.save_stack_btn.setEnabled(False)  # Enable when optics exists
        buttons_layout.addWidget(self.save_stack_btn)

        self.content_layout.insertWidget(self.content_layout.count() - 1, buttons_widget)

    def _create_param_widgets(self):
        """Create parameter widgets for each component type."""
        # Free Space parameters - add_free_space(length_mm) - UI in Angstroms
        free_space_widget = QWidget()
        fs_layout = QFormLayout(free_space_widget)
        self.fs_length = QDoubleSpinBox()
        self.fs_length.setDecimals(0)
        self.fs_length.setRange(0, 1e12)
        self.fs_length.setValue(1000)  # 100 mm = 1e9 Å
        self.fs_length.setSuffix(" Å")
        self.fs_length.setSingleStep(100)
        fs_layout.addRow("Length:", self.fs_length)
        self.param_stack.addWidget(free_space_widget)

        # CRL parameters - add_CRL_box(number, focal_length_mm, thickness_mm, absorption_sigma)
        crl_widget = QWidget()
        crl_layout = QFormLayout(crl_widget)
        self.crl_number = QSpinBox()
        self.crl_number.setRange(1, 1000)
        self.crl_number.setValue(10)
        crl_layout.addRow("Number of lenses:", self.crl_number)
        self.crl_focal_length = QDoubleSpinBox()
        self.crl_focal_length.setDecimals(0)
        self.crl_focal_length.setRange(0, 1e12)
        self.crl_focal_length.setValue(1000)  # 100 mm = 1e9 Å
        self.crl_focal_length.setSuffix(" Å")
        self.crl_focal_length.setSingleStep(100)
        crl_layout.addRow("Focal Length:", self.crl_focal_length)
        self.crl_thickness = QDoubleSpinBox()
        self.crl_thickness.setDecimals(0)
        self.crl_thickness.setRange(0, 1e12)
        self.crl_thickness.setValue(100)  # 1.0 mm = 1e7 Å
        self.crl_thickness.setSuffix(" Å")
        self.crl_thickness.setSingleStep(10)
        crl_layout.addRow("Thickness:", self.crl_thickness)
        self.crl_mu = QDoubleSpinBox()
        self.crl_mu.setDecimals(2)
        self.crl_mu.setRange(0, 1e9)
        self.crl_mu.setValue(0)  # 0 = parabolic absorption disabled
        self.crl_mu.setSuffix(" 1/m")
        self.crl_mu.setSingleStep(10)
        self.crl_mu.setToolTip(
            "Intensity linear attenuation coefficient of the lens material (1/m).\n"
            "0 disables the parabolic (r^2-dependent) CRL absorption."
        )
        crl_layout.addRow("Attenuation μ:", self.crl_mu)
        self.crl_radius = QDoubleSpinBox()
        self.crl_radius.setDecimals(6)
        self.crl_radius.setRange(0, 1e6)
        self.crl_radius.setValue(0)  # 0 = treated as infinite (disabled)
        self.crl_radius.setSuffix(" m")
        self.crl_radius.setSingleStep(1e-4)
        self.crl_radius.setToolTip(
            "Per-surface parabolic apex radius R in meters.\n"
            "0 is treated as infinite (parabolic absorption disabled)."
        )
        crl_layout.addRow("Apex radius R:", self.crl_radius)
        self.param_stack.addWidget(crl_widget)

        # Bragg Magnifier parameters - add_bragg_magnifier_2b(mag_x, mag_y, reflectivity, phase_shift, ...)
        bragg_widget = QWidget()
        bragg_layout = QFormLayout(bragg_widget)
        self.bragg_mag_x = QDoubleSpinBox()
        self.bragg_mag_x.setDecimals(2)
        self.bragg_mag_x.setRange(0.1, 1000)
        self.bragg_mag_x.setValue(1)
        bragg_layout.addRow("Magnification X:", self.bragg_mag_x)
        self.bragg_mag_y = QDoubleSpinBox()
        self.bragg_mag_y.setDecimals(2)
        self.bragg_mag_y.setRange(0.1, 1000)
        self.bragg_mag_y.setValue(1)
        bragg_layout.addRow("Magnification Y:", self.bragg_mag_y)
        self.bragg_reflectivity = QDoubleSpinBox()
        self.bragg_reflectivity.setDecimals(3)
        self.bragg_reflectivity.setRange(0.0, 1.0)
        self.bragg_reflectivity.setValue(1.0)
        bragg_layout.addRow("Reflectivity:", self.bragg_reflectivity)
        self.param_stack.addWidget(bragg_widget)

        # Aperture parameters - add_aperture(width_mm, shape='square') - UI in Angstroms
        aperture_widget = QWidget()
        ap_layout = QFormLayout(aperture_widget)
        self.ap_width = QDoubleSpinBox()
        self.ap_width.setDecimals(0)
        self.ap_width.setRange(0, 1e9)
        self.ap_width.setValue(100)  # 1 mm = 1e7 Å
        self.ap_width.setSuffix(" Å")
        self.ap_width.setSingleStep(10)
        ap_layout.addRow("Width:", self.ap_width)
        self.ap_shape = QComboBox()
        self.ap_shape.addItem("Square", "square")
        self.ap_shape.addItem("Circular", "circular")
        ap_layout.addRow("Shape:", self.ap_shape)
        self.param_stack.addWidget(aperture_widget)

        # Angular Filter parameters - add_angular_filter(half_angle_mrad, center_x_mrad, center_y_mrad, shape, ...)
        angular_widget = QWidget()
        ang_layout = QFormLayout(angular_widget)
        self.ang_half_angle = QDoubleSpinBox()
        self.ang_half_angle.setDecimals(4)
        self.ang_half_angle.setRange(1e-6, 1e6)
        self.ang_half_angle.setValue(125e-4)
        self.ang_half_angle.setSuffix(" mrad")
        ang_layout.addRow("Half Angle:", self.ang_half_angle)
        self.ang_shape = QComboBox()
        self.ang_shape.addItem("Circular", "circular")
        self.ang_shape.addItem("Elliptical", "elliptical")
        ang_layout.addRow("Shape:", self.ang_shape)
        self.ang_rolloff = QComboBox()
        self.ang_rolloff.addItem("Top-hat", "tophat")
        self.ang_rolloff.addItem("Darwin", "darwin")
        ang_layout.addRow("Rolloff:", self.ang_rolloff)
        self.param_stack.addWidget(angular_widget)

    def _on_type_changed(self, index):
        """Handle component type change."""
        self.param_stack.setCurrentIndex(index)

    def _register_observers(self):
        self.state.register_observer("optics_changed", self._on_optics_state_changed)
        self.state.register_observer("global_working_directory_changed", self._on_global_dir_changed)

    def _on_global_dir_changed(self, directory):
        """Handle global working directory change."""
        # Update placeholder text if directory field is empty
        if not self.directory_edit.text():
            self.directory_edit.setPlaceholderText(
                f"Using global: {directory}" if directory
                else "Select directory for optics files..."
            )

    def _on_optics_state_changed(self, optics):
        self._refresh_display()

    def _refresh_display(self):
        self.component_list.clear()
        optics_obj = self.state.optics
        if optics_obj is None:
            self.save_stack_btn.setEnabled(False)
            return

        # Enable save button when optics exists
        self.save_stack_btn.setEnabled(True)

        # Update directory display if available
        if hasattr(optics_obj, 'directory') and optics_obj.directory:
            self.directory_edit.setText(str(optics_obj.directory))

        # Use components property (which returns _components list)
        if not hasattr(optics_obj, 'components'):
            return

        # Conversion factor: mm to Angstroms
        mm_to_A = 1e7

        for i, component in enumerate(optics_obj.components):
            # Components use 'kind' key, not 'type'
            comp_kind = component.get('kind', 'unknown')
            # Build a descriptive label - convert lengths from mm to Angstroms for display
            if comp_kind == 'free space':
                length = component.get('length', 0) * mm_to_A
                item_text = f"{i+1}. Free Space ({length:.2e} Å)"
            elif comp_kind == 'lens box':
                n = component.get('number', 0)
                f = component.get('focal_length', 0) * mm_to_A
                item_text = f"{i+1}. CRL (N={n}, f={f:.2e} Å)"
            elif comp_kind == 'bragg magnifier 2b':
                mx = component.get('magnification_x', 1)
                my = component.get('magnification_y', 1)
                item_text = f"{i+1}. Bragg Mag ({mx:.1f}x, {my:.1f}x)"
            elif comp_kind == 'aperture':
                w = component.get('width', 0) * mm_to_A
                shape = component.get('type', 'square')
                item_text = f"{i+1}. Aperture ({shape}, {w:.2e} Å)"
            elif comp_kind == 'angular filter':
                ha = component.get('half_angle_x_mrad', 0)
                item_text = f"{i+1}. Angular Filter ({ha:.2f} mrad)"
            else:
                item_text = f"{i+1}. {comp_kind}"
            self.component_list.addItem(item_text)

    def _on_component_selected(self):
        pass  # Could show component details

    def _on_move_up(self):
        optics_obj = self.state.optics
        if optics_obj is None or not hasattr(optics_obj, '_components'):
            return

        row = self.component_list.currentRow()
        if row > 0:
            # Access internal _components list directly for modification
            optics_obj._components[row], optics_obj._components[row-1] = \
                optics_obj._components[row-1], optics_obj._components[row]
            self.state.notify_object_modified("optics")
            self._refresh_display()
            self.component_list.setCurrentRow(row - 1)
            # Auto-save after reorder
            if hasattr(optics_obj, 'write_optics_metadata'):
                optics_obj.write_optics_metadata()

    def _on_move_down(self):
        optics_obj = self.state.optics
        if optics_obj is None or not hasattr(optics_obj, '_components'):
            return

        row = self.component_list.currentRow()
        if 0 <= row < len(optics_obj._components) - 1:
            optics_obj._components[row], optics_obj._components[row+1] = \
                optics_obj._components[row+1], optics_obj._components[row]
            self.state.notify_object_modified("optics")
            self._refresh_display()
            self.component_list.setCurrentRow(row + 1)
            # Auto-save after reorder
            if hasattr(optics_obj, 'write_optics_metadata'):
                optics_obj.write_optics_metadata()

    def _on_remove_component(self):
        optics_obj = self.state.optics
        if optics_obj is None or not hasattr(optics_obj, '_components'):
            return

        row = self.component_list.currentRow()
        if 0 <= row < len(optics_obj._components):
            del optics_obj._components[row]
            self.state.notify_object_modified("optics")
            self._refresh_display()
            # Auto-save after removal
            if hasattr(optics_obj, 'write_optics_metadata'):
                optics_obj.write_optics_metadata()

    def _on_add_component(self):
        optics_obj = self.state.optics
        if optics_obj is None:
            QMessageBox.warning(self, "No Optics", "Please create an optics stack first.")
            return

        comp_type = self.type_combo.currentData()

        # Conversion factor: Angstroms to mm
        A_to_mm = 1e-7

        try:
            if comp_type == "free_space":
                # add_free_space(length_mm) - convert from Å to mm
                optics_obj.add_free_space(self.fs_length.value() * A_to_mm)

            elif comp_type == "crl":
                # add_CRL_box(number, focal_length_mm, thickness_mm, ...) - convert from Å to mm
                # mu_per_m and radius_of_curvature_m are SI (1/m, m); a radius of
                # 0 in the UI means "infinite" (parabolic absorption disabled).
                crl_radius_m = self.crl_radius.value()
                optics_obj.add_CRL_box(
                    number=self.crl_number.value(),
                    focal_length_mm=self.crl_focal_length.value() * A_to_mm,
                    thickness_mm=self.crl_thickness.value() * A_to_mm,
                    mu_per_m=self.crl_mu.value(),
                    radius_of_curvature_m=(crl_radius_m if crl_radius_m > 0 else float('inf'))
                )

            elif comp_type == "bragg":
                # add_bragg_magnifier_2b(magnification_x, magnification_y, reflectivity, ...)
                optics_obj.add_bragg_magnifier_2b(
                    magnification_x=self.bragg_mag_x.value(),
                    magnification_y=self.bragg_mag_y.value(),
                    reflectivity=self.bragg_reflectivity.value()
                )

            elif comp_type == "aperture":
                # add_aperture(width_mm, shape='square') - convert from Å to mm
                optics_obj.add_aperture(
                    width_mm=self.ap_width.value() * A_to_mm,
                    shape=self.ap_shape.currentData()
                )

            elif comp_type == "angular_filter":
                # add_angular_filter(half_angle_mrad, shape='circular', rolloff='tophat', ...)
                optics_obj.add_angular_filter(
                    half_angle_mrad=self.ang_half_angle.value(),
                    shape=self.ang_shape.currentData(),
                    rolloff=self.ang_rolloff.currentData()
                )

            self.state.notify_object_modified("optics")
            self.component_added.emit(comp_type)
            self._refresh_display()

            # Auto-save metadata after adding component
            if hasattr(optics_obj, 'write_optics_metadata'):
                optics_obj.write_optics_metadata()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add component:\n{str(e)}")

    def _on_create_optics(self):
        directory = self.directory_edit.text()
        if not directory:
            QMessageBox.warning(self, "No Directory",
                              "Please select a working directory first.")
            return

        try:
            from Optics import optics
            new_optics = optics(directory=directory)
            # Save metadata after creation
            if hasattr(new_optics, 'write_optics_metadata'):
                new_optics.write_optics_metadata()
            self.state.optics = new_optics
            self.optics_created.emit(new_optics)
            self._refresh_display()
            QMessageBox.information(self, "Success", "Optics stack created successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create optics:\n{str(e)}")

    def _on_browse_directory(self):
        """Open directory browser dialog."""
        # Use current text, or global directory, or current working directory
        start_dir = self.directory_edit.text() or self.state.get_default_directory()
        directory = QFileDialog.getExistingDirectory(
            self, "Select Optics Directory",
            start_dir
        )
        if directory:
            self.directory_edit.setText(directory)
            # Check if optics metadata exists
            metadata_path = Path(directory) / "optics_metadata.json"
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

    def _load_existing_optics(self):
        """Load existing optics from selected directory."""
        directory = self.directory_edit.text()
        if not directory:
            QMessageBox.warning(self, "No Directory",
                              "Please select a working directory first.")
            return

        try:
            from Optics import optics
            existing_optics = optics(directory=directory)
            # Try to load metadata
            if hasattr(existing_optics, 'read_optics_metadata'):
                existing_optics.read_optics_metadata()
            self.state.optics = existing_optics
            self.optics_created.emit(existing_optics)
            self._refresh_display()
            n_components = len(existing_optics.components) if hasattr(existing_optics, 'components') else 0
            QMessageBox.information(self, "Success",
                f"Optics loaded successfully.\n{n_components} component(s) in stack.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load optics:\n{str(e)}")

    def _on_save_stack(self):
        """Save the optics component stack to metadata file."""
        optics_obj = self.state.optics
        if optics_obj is None:
            QMessageBox.warning(self, "No Optics", "No optics stack to save.")
            return

        if not hasattr(optics_obj, 'directory') or not optics_obj.directory:
            QMessageBox.warning(self, "No Directory",
                              "Optics has no directory set. Cannot save.")
            return

        try:
            if hasattr(optics_obj, 'write_optics_metadata'):
                success = optics_obj.write_optics_metadata()
                if success:
                    n_components = len(optics_obj.components) if hasattr(optics_obj, 'components') else 0
                    QMessageBox.information(self, "Success",
                        f"Optics stack saved successfully.\n"
                        f"{n_components} component(s) saved to:\n"
                        f"{optics_obj.directory}/optics_metadata.json")
                else:
                    QMessageBox.warning(self, "Warning",
                                      "Failed to save optics metadata.")
            else:
                QMessageBox.warning(self, "Error",
                                  "Optics object does not support metadata saving.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save optics:\n{str(e)}")
