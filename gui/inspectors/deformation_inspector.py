# -----------------------------------------------------------------------------
# Deformation Inspector
# -----------------------------------------------------------------------------
"""
Inspector panel for deformation field configuration.

Provides controls for:
- Importing deformation fields
- Importing FE nodal fields
- Transform controls (scale, rotate, translate)
- Applying deformation to sample
"""

import sys
from pathlib import Path
import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLabel, QDoubleSpinBox, QPushButton, QComboBox, QFileDialog,
    QMessageBox, QProgressDialog, QCheckBox, QRadioButton, QButtonGroup,
    QLineEdit, QSpinBox,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gui.state import SimulationState
from gui.panels.inspector import InspectorPanel, Vector3Widget


class DeformationInspector(InspectorPanel):
    """Inspector for deformation field configuration."""

    deformation_created = Signal(object)
    field_imported = Signal()
    field_applied = Signal()

    def __init__(self, state: SimulationState, parent=None):
        super().__init__(state, parent)
        self.set_title("Deformation")
        self._field_path = None
        self._fe_connectivity_path = None
        # Store imported field data
        self._positions = None
        self._F = None
        self._setup_deformation_ui()
        self._register_observers()

    def _setup_deformation_ui(self):
        """Setup deformation-specific UI elements."""
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
            self.directory_edit.setPlaceholderText("Select directory for deformation files...")
        self.directory_edit.setReadOnly(True)
        browse_dir_btn = QPushButton("Browse...")
        browse_dir_btn.clicked.connect(self._on_browse_directory)
        dir_hlayout.addWidget(self.directory_edit)
        dir_hlayout.addWidget(browse_dir_btn)
        dir_layout.addRow(dir_widget)

        # Load existing button
        self.load_existing_btn = QPushButton("Load Existing Deformation")
        self.load_existing_btn.clicked.connect(self._load_existing_deformation)
        self.load_existing_btn.setEnabled(False)
        dir_layout.addRow(self.load_existing_btn)

        # Create Deformation Button
        create_btn = QPushButton("Create Deformation Object")
        create_btn.clicked.connect(self._on_create_deformation)
        dir_layout.addRow(create_btn)

        # Import Mode Group
        mode_group = self.add_group("Import Mode")
        mode_layout = mode_group.layout()

        self.mode_group = QButtonGroup(self)
        self.field_mode = QRadioButton("Deformation Field (F tensor)")
        self.fe_mode = QRadioButton("FE Nodal Field (displacement)")
        self.field_mode.setChecked(True)
        self.mode_group.addButton(self.field_mode)
        self.mode_group.addButton(self.fe_mode)
        self.mode_group.buttonClicked.connect(self._on_mode_changed)
        mode_layout.addRow(self.field_mode)
        mode_layout.addRow(self.fe_mode)

        # Deformation Field Import Group
        self.df_group = self.add_group("Deformation Field Import")
        df_layout = self.df_group.layout()

        # Preset selection for deformation field
        self.df_preset = QComboBox()
        self.df_preset.addItem("Auto-detect", None)
        self.df_preset.addItem("COMSOL CSV", "comsol_csv")
        self.df_preset.addItem("Generic (whitespace)", "generic")
        df_layout.addRow("Preset:", self.df_preset)

        # Position scale
        self.df_position_scale = QDoubleSpinBox()
        self.df_position_scale.setDecimals(9)
        self.df_position_scale.setRange(1e-12, 1e12)
        self.df_position_scale.setValue(1.0)
        self.df_position_scale.setToolTip("Scale factor for positions (e.g., 1e-3 for mm to m)")
        df_layout.addRow("Position Scale:", self.df_position_scale)

        # File selection
        df_file_row = QWidget()
        df_file_layout = QHBoxLayout(df_file_row)
        df_file_layout.setContentsMargins(0, 0, 0, 0)
        self.df_file_label = QLabel("No file selected")
        self.df_file_label.setStyleSheet("color: #808080;")
        df_file_layout.addWidget(self.df_file_label, 1)
        df_browse_btn = QPushButton("Browse...")
        df_browse_btn.clicked.connect(self._on_browse_deformation_field)
        df_file_layout.addWidget(df_browse_btn)
        df_layout.addRow("File:", df_file_row)

        # Import button
        df_import_btn = QPushButton("Import Deformation Field")
        df_import_btn.clicked.connect(self._on_import_deformation_field)
        df_layout.addRow("", df_import_btn)

        # FE Nodal Field Import Group
        self.fe_group = self.add_group("FE Nodal Field Import")
        fe_layout = self.fe_group.layout()

        # Preset selection for FE nodal field
        self.fe_preset = QComboBox()
        self.fe_preset.addItem("Auto-detect", None)
        self.fe_preset.addItem("COMSOL Nodes CSV", "comsol_nodes_csv")
        self.fe_preset.addItem("COMSOL Nodes TXT", "comsol_nodes_txt")
        self.fe_preset.addItem("Generic (x,y,z,u1,u2,u3)", "generic_xyzu")
        self.fe_preset.addItem("Generic (x0,y0,z0,x1,y1,z1)", "generic_x0x1")
        fe_layout.addRow("Preset:", self.fe_preset)

        # Position scale for FE
        self.fe_position_scale = QDoubleSpinBox()
        self.fe_position_scale.setDecimals(9)
        self.fe_position_scale.setRange(1e-12, 1e12)
        self.fe_position_scale.setValue(1.0)
        self.fe_position_scale.setToolTip("Scale factor for positions (e.g., 1e-9 for nm to m)")
        fe_layout.addRow("Position Scale:", self.fe_position_scale)

        # Nodes file selection
        fe_file_row = QWidget()
        fe_file_layout = QHBoxLayout(fe_file_row)
        fe_file_layout.setContentsMargins(0, 0, 0, 0)
        self.fe_file_label = QLabel("No file selected")
        self.fe_file_label.setStyleSheet("color: #808080;")
        fe_file_layout.addWidget(self.fe_file_label, 1)
        fe_browse_btn = QPushButton("Browse...")
        fe_browse_btn.clicked.connect(self._on_browse_fe_nodal_field)
        fe_file_layout.addWidget(fe_browse_btn)
        fe_layout.addRow("Nodes File:", fe_file_row)

        # Import nodes button
        fe_import_btn = QPushButton("Import FE Nodal Field")
        fe_import_btn.clicked.connect(self._on_import_fe_nodal_field)
        fe_layout.addRow("", fe_import_btn)

        # Connectivity file (optional for tetrahedra)
        fe_layout.addRow(QLabel("--- Connectivity (optional) ---"))

        self.conn_preset = QComboBox()
        self.conn_preset.addItem("Auto-detect", None)
        self.conn_preset.addItem("COMSOL Mesh TXT", "comsol_mesh_txt")
        self.conn_preset.addItem("Generic (4 columns)", "generic")
        fe_layout.addRow("Mesh Preset:", self.conn_preset)

        conn_file_row = QWidget()
        conn_file_layout = QHBoxLayout(conn_file_row)
        conn_file_layout.setContentsMargins(0, 0, 0, 0)
        self.conn_file_label = QLabel("No file selected")
        self.conn_file_label.setStyleSheet("color: #808080;")
        conn_file_layout.addWidget(self.conn_file_label, 1)
        conn_browse_btn = QPushButton("Browse...")
        conn_browse_btn.clicked.connect(self._on_browse_connectivity)
        conn_file_layout.addWidget(conn_browse_btn)
        fe_layout.addRow("Mesh File:", conn_file_row)

        conn_import_btn = QPushButton("Import Connectivity")
        conn_import_btn.clicked.connect(self._on_import_connectivity)
        fe_layout.addRow("", conn_import_btn)

        # Initially hide FE group
        self.fe_group.setVisible(False)

        # Transform Group
        transform_group = self.add_group("Transform Positions")
        transform_layout = transform_group.layout()

        # Scale
        self.scale = QDoubleSpinBox()
        self.scale.setDecimals(6)
        self.scale.setRange(1e-9, 1e9)
        self.scale.setValue(1.0)
        transform_layout.addRow("Scale:", self.scale)

        # Rotation
        transform_layout.addRow(QLabel("Rotation:"))
        self.rot_axis = Vector3Widget([0, 0, 1], decimals=4)
        transform_layout.addRow("Axis:", self.rot_axis)
        self.rot_angle = QDoubleSpinBox()
        self.rot_angle.setDecimals(4)
        self.rot_angle.setRange(-180, 180)
        self.rot_angle.setValue(0)
        self.rot_angle.setSuffix(" °")
        transform_layout.addRow("Angle:", self.rot_angle)

        # Translation
        self.translation = Vector3Widget([0, 0, 0], decimals=2, suffix="Å")
        transform_layout.addRow("Translation:", self.translation)

        # Apply transform button
        apply_transform_btn = QPushButton("Apply Transform to Field")
        apply_transform_btn.clicked.connect(self._on_apply_transform)
        transform_layout.addRow("", apply_transform_btn)

        # Apply to Sample Group
        apply_group = self.add_group("Apply to Sample")
        apply_layout = apply_group.layout()

        # Method selection
        self.apply_method = QComboBox()
        self.apply_method.addItem("kNN + IDW (for deformation field)", "knn_idw")
        self.apply_method.addItem("MLS Fitting (for FE nodal field)", "mls")
        self.apply_method.currentIndexChanged.connect(self._on_method_changed)
        apply_layout.addRow("Method:", self.apply_method)

        # kNN parameters
        self.knn_k = QSpinBox()
        self.knn_k.setRange(1, 100)
        self.knn_k.setValue(8)
        apply_layout.addRow("k Neighbors:", self.knn_k)

        # IDW power
        self.idw_power = QDoubleSpinBox()
        self.idw_power.setDecimals(2)
        self.idw_power.setRange(0.5, 10.0)
        self.idw_power.setValue(2.0)
        apply_layout.addRow("IDW Power:", self.idw_power)

        # Chunk size
        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(1000, 10000000)
        self.chunk_size.setValue(200000)
        self.chunk_size.setSingleStep(10000)
        apply_layout.addRow("Chunk Size:", self.chunk_size)

        # GPU usage
        self.use_gpu = QCheckBox("Use GPU")
        self.use_gpu.setChecked(True)
        apply_layout.addRow("", self.use_gpu)

        # Apply button
        apply_btn = QPushButton("Apply Deformation to Sample")
        apply_btn.clicked.connect(self._on_apply)
        apply_btn.setStyleSheet("QPushButton { background-color: #2a6a2a; }")
        apply_layout.addRow("", apply_btn)

        # Field Info Group
        info_group = self.add_group("Field Info")
        info_layout = info_group.layout()

        self.info_label = QLabel("No field loaded")
        self.info_label.setStyleSheet("color: #808080;")
        self.info_label.setWordWrap(True)
        info_layout.addRow(self.info_label)

    def _register_observers(self):
        self.state.register_observer("deformation_changed", self._on_deformation_state_changed)
        self.state.register_observer("global_working_directory_changed", self._on_global_dir_changed)

    def _on_global_dir_changed(self, directory):
        """Handle global working directory change."""
        # Update placeholder text if directory field is empty
        if not self.directory_edit.text():
            self.directory_edit.setPlaceholderText(
                f"Using global: {directory}" if directory
                else "Select directory for deformation files..."
            )

    def _on_deformation_state_changed(self, deformation):
        self._refresh_display()

    def _on_mode_changed(self, button):
        """Handle import mode change."""
        is_field_mode = self.field_mode.isChecked()
        self.df_group.setVisible(is_field_mode)
        self.fe_group.setVisible(not is_field_mode)

        # Update apply method selection based on mode
        if is_field_mode:
            self.apply_method.setCurrentIndex(0)  # kNN + IDW
        else:
            self.apply_method.setCurrentIndex(1)  # MLS

    def _on_method_changed(self, index):
        """Handle apply method change."""
        is_knn = self.apply_method.currentData() == "knn_idw"
        self.knn_k.setEnabled(is_knn)
        self.idw_power.setEnabled(is_knn)
        self.chunk_size.setEnabled(is_knn)

    def _refresh_display(self):
        deformation = self.state.deformation
        if deformation is None:
            self.info_label.setText("No deformation object")
            return

        # Update directory display if available
        if hasattr(deformation, 'directory') and deformation.directory:
            self.directory_edit.setText(str(deformation.directory))

        info_parts = []

        # Check for stored deformation field data
        if self._positions is not None and self._F is not None:
            n_points = self._positions.shape[0] if hasattr(self._positions, 'shape') else len(self._positions)
            info_parts.append(f"Deformation Field: {n_points:,} points")

            # Get bounds
            if hasattr(self._positions, 'min') and hasattr(self._positions, 'max'):
                try:
                    pos_min = self._positions.min(axis=0)
                    pos_max = self._positions.max(axis=0)
                    if hasattr(pos_min, 'get'):  # CuPy
                        pos_min = pos_min.get()
                        pos_max = pos_max.get()
                    info_parts.append(f"Bounds: [{pos_min[0]:.1f}, {pos_max[0]:.1f}] × "
                                    f"[{pos_min[1]:.1f}, {pos_max[1]:.1f}] × "
                                    f"[{pos_min[2]:.1f}, {pos_max[2]:.1f}]")
                except Exception:
                    pass

        # Check for FE nodal field
        if hasattr(deformation, '_Xref') and deformation._Xref is not None:
            n_nodes = deformation._Xref.shape[0] if hasattr(deformation._Xref, 'shape') else 0
            info_parts.append(f"FE Nodal Field: {n_nodes:,} nodes")

        # Check for FE connectivity
        if hasattr(deformation, '_elem_nodes') and deformation._elem_nodes is not None:
            n_elem = deformation._elem_nodes.shape[0] if hasattr(deformation._elem_nodes, 'shape') else 0
            info_parts.append(f"FE Elements: {n_elem:,} elements")

        if info_parts:
            self.info_label.setText("\n".join(info_parts))
            self.info_label.setStyleSheet("color: #4ec94e;")
        else:
            self.info_label.setText("Deformation object created, no field loaded")
            self.info_label.setStyleSheet("color: #808080;")

    def _on_browse_directory(self):
        """Open directory browser dialog."""
        # Use current text, or global directory, or current working directory
        start_dir = self.directory_edit.text() or self.state.get_default_directory()
        directory = QFileDialog.getExistingDirectory(
            self, "Select Deformation Working Directory",
            start_dir
        )
        if directory:
            self.directory_edit.setText(directory)
            # Check if deformation metadata exists
            metadata_path = Path(directory) / "deformation_metadata.json"
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

    def _load_existing_deformation(self):
        """Load existing deformation from selected directory."""
        directory = self.directory_edit.text()
        if not directory:
            QMessageBox.warning(self, "No Directory",
                              "Please select a working directory first.")
            return

        try:
            from Deformation import deformation
            existing_def = deformation(directory=directory)
            self.state.deformation = existing_def
            self.deformation_created.emit(existing_def)
            self._refresh_display()
            QMessageBox.information(self, "Success", "Deformation object loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load deformation:\n{str(e)}")

    def _on_create_deformation(self):
        """Create a new deformation object."""
        directory = self.directory_edit.text()
        if not directory:
            QMessageBox.warning(self, "No Directory",
                              "Please select a working directory first.")
            return

        try:
            from Deformation import deformation
            new_def = deformation(directory=directory)
            self.state.deformation = new_def
            self.deformation_created.emit(new_def)
            self._refresh_display()
            QMessageBox.information(self, "Success", "Deformation object created.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create deformation:\n{str(e)}")

    def _on_browse_deformation_field(self):
        """Browse for deformation field file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Deformation Field File", "",
            "Deformation Field (*.txt *.dat *.csv *.npy *.npz);;All Files (*)"
        )
        if filename:
            self._field_path = filename
            self.df_file_label.setText(Path(filename).name)
            self.df_file_label.setStyleSheet("color: #4ec94e;")

    def _on_browse_fe_nodal_field(self):
        """Browse for FE nodal field file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select FE Nodal Field File", "",
            "FE Nodal Field (*.txt *.dat *.csv *.npy *.npz);;All Files (*)"
        )
        if filename:
            self._fe_nodal_path = filename
            self.fe_file_label.setText(Path(filename).name)
            self.fe_file_label.setStyleSheet("color: #4ec94e;")

    def _on_browse_connectivity(self):
        """Browse for FE connectivity file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select FE Connectivity File", "",
            "FE Mesh (*.txt *.dat *.csv *.npy *.npz);;All Files (*)"
        )
        if filename:
            self._fe_connectivity_path = filename
            self.conn_file_label.setText(Path(filename).name)
            self.conn_file_label.setStyleSheet("color: #4ec94e;")

    def _on_import_deformation_field(self):
        """Import deformation field using import_deformation_field method."""
        if self._field_path is None:
            QMessageBox.warning(self, "No File", "Please select a deformation field file first.")
            return

        deformation = self.state.deformation
        if deformation is None:
            QMessageBox.warning(self, "No Object", "Please create a deformation object first.")
            return

        try:
            progress = QProgressDialog("Importing deformation field...", "Cancel", 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            preset = self.df_preset.currentData()
            position_scale = self.df_position_scale.value()

            if hasattr(deformation, 'import_deformation_field'):
                # import_deformation_field returns (positions, F)
                positions, F = deformation.import_deformation_field(
                    filepath=self._field_path,
                    preset=preset,
                    position_scale=position_scale,
                    use_gpu=self.use_gpu.isChecked()
                )

                # Store for later use in apply_deformation_chunked
                self._positions = positions
                self._F = F

                progress.close()
                self.state.notify_object_modified("deformation")
                self.field_imported.emit()
                self._refresh_display()

                n_points = positions.shape[0] if hasattr(positions, 'shape') else len(positions)
                QMessageBox.information(self, "Success",
                    f"Deformation field imported successfully.\n{n_points:,} data points loaded.")
            else:
                progress.close()
                QMessageBox.warning(self, "Not Supported",
                    "Deformation object doesn't support import_deformation_field.")

        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Error", f"Failed to import deformation field:\n{str(e)}")

    def _on_import_fe_nodal_field(self):
        """Import FE nodal field using import_fe_nodal_field method."""
        if not hasattr(self, '_fe_nodal_path') or self._fe_nodal_path is None:
            QMessageBox.warning(self, "No File", "Please select an FE nodal field file first.")
            return

        deformation = self.state.deformation
        if deformation is None:
            QMessageBox.warning(self, "No Object", "Please create a deformation object first.")
            return

        try:
            progress = QProgressDialog("Importing FE nodal field...", "Cancel", 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            preset = self.fe_preset.currentData()
            position_scale = self.fe_position_scale.value()

            if hasattr(deformation, 'import_fe_nodal_field'):
                # import_fe_nodal_field returns (Xref, Xcurr) and stores them internally
                Xref, Xcurr = deformation.import_fe_nodal_field(
                    filepath=self._fe_nodal_path,
                    preset=preset,
                    position_scale=position_scale,
                    use_gpu=self.use_gpu.isChecked()
                )

                progress.close()
                self.state.notify_object_modified("deformation")
                self.field_imported.emit()
                self._refresh_display()

                n_nodes = Xref.shape[0] if hasattr(Xref, 'shape') else len(Xref)
                QMessageBox.information(self, "Success",
                    f"FE nodal field imported successfully.\n{n_nodes:,} nodes loaded.")
            else:
                progress.close()
                QMessageBox.warning(self, "Not Supported",
                    "Deformation object doesn't support import_fe_nodal_field.")

        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Error", f"Failed to import FE nodal field:\n{str(e)}")

    def _on_import_connectivity(self):
        """Import FE connectivity using import_fe_connectivity method."""
        if self._fe_connectivity_path is None:
            QMessageBox.warning(self, "No File", "Please select a connectivity file first.")
            return

        deformation = self.state.deformation
        if deformation is None:
            QMessageBox.warning(self, "No Object", "Please create a deformation object first.")
            return

        try:
            progress = QProgressDialog("Importing FE connectivity...", "Cancel", 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            preset = self.conn_preset.currentData()

            if hasattr(deformation, 'import_fe_connectivity'):
                # import_fe_connectivity returns elem_nodes and stores it internally
                elem_nodes = deformation.import_fe_connectivity(
                    filepath=self._fe_connectivity_path,
                    preset=preset,
                    use_gpu=self.use_gpu.isChecked()
                )

                progress.close()
                self.state.notify_object_modified("deformation")
                self._refresh_display()

                n_elem = elem_nodes.shape[0] if hasattr(elem_nodes, 'shape') else len(elem_nodes)
                QMessageBox.information(self, "Success",
                    f"FE connectivity imported successfully.\n{n_elem:,} elements loaded.")
            else:
                progress.close()
                QMessageBox.warning(self, "Not Supported",
                    "Deformation object doesn't support import_fe_connectivity.")

        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Error", f"Failed to import connectivity:\n{str(e)}")

    def _on_apply_transform(self):
        """Apply transform to the loaded field positions."""
        deformation = self.state.deformation
        if deformation is None:
            QMessageBox.warning(self, "No Deformation", "Please create a deformation object first.")
            return

        # Determine which positions to transform
        positions = None
        is_stored_positions = False
        if self._positions is not None:
            positions = self._positions
            is_stored_positions = True
        elif hasattr(deformation, '_Xref') and deformation._Xref is not None:
            positions = deformation._Xref
        else:
            QMessageBox.warning(self, "No Field", "No field loaded to transform.")
            return

        try:
            scale = float(self.scale.value())
            # Get values as Python lists first, then convert to numpy
            translate_vals = self.translation.get_value()
            translate = np.array([float(translate_vals[0]), float(translate_vals[1]), float(translate_vals[2])], dtype=np.float64)

            angle = float(np.radians(self.rot_angle.value()))
            axis_vals = self.rot_axis.get_value()
            axis = np.array([float(axis_vals[0]), float(axis_vals[1]), float(axis_vals[2])], dtype=np.float64)

            use_gpu = self.use_gpu.isChecked()

            # Ensure positions are numpy array with correct dtype
            if hasattr(positions, 'get'):  # CuPy array
                positions = positions.get()
            positions = np.asarray(positions, dtype=np.float64)

            transforms_applied = []

            # Apply scale if not 1.0
            if abs(scale - 1.0) > 1e-9:
                if hasattr(deformation, 'scale_positions'):
                    positions = deformation.scale_positions(positions, scale=scale, use_gpu=use_gpu)
                else:
                    # Fallback: manual scaling
                    positions = positions * scale
                transforms_applied.append(f"Scale: {scale}")

            # Apply rotation if angle is not 0
            if abs(angle) > 1e-9:
                # Check for zero axis vector
                axis_norm = np.linalg.norm(axis)
                if axis_norm < 1e-9:
                    QMessageBox.warning(self, "Invalid Axis",
                        "Rotation axis cannot be zero vector. Please set a valid axis.")
                    return

                # Normalize axis
                axis = axis / axis_norm
                c, s = np.cos(angle), np.sin(angle)

                # Build rotation matrix using Rodrigues' formula
                # This creates R for column-vector convention (R @ v)
                R_col = np.array([
                    [c + axis[0]**2*(1-c), axis[0]*axis[1]*(1-c) - axis[2]*s, axis[0]*axis[2]*(1-c) + axis[1]*s],
                    [axis[1]*axis[0]*(1-c) + axis[2]*s, c + axis[1]**2*(1-c), axis[1]*axis[2]*(1-c) - axis[0]*s],
                    [axis[2]*axis[0]*(1-c) - axis[1]*s, axis[2]*axis[1]*(1-c) + axis[0]*s, c + axis[2]**2*(1-c)]
                ], dtype=np.float64)

                # Deformation.rotate_positions uses row-vector convention (v @ R)
                # So we need to transpose: v @ R.T = (R @ v.T).T
                R_row = R_col.T

                if hasattr(deformation, 'rotate_positions'):
                    positions = deformation.rotate_positions(positions, R_row, use_gpu=use_gpu)
                else:
                    # Fallback: manual rotation with row-vector convention
                    positions = positions @ R_row
                transforms_applied.append(f"Rotation: {np.degrees(angle):.2f}°")

            # Apply translation if not zero
            if np.any(np.abs(translate) > 1e-9):
                if hasattr(deformation, 'translate_positions'):
                    positions = deformation.translate_positions(positions, translate, use_gpu=use_gpu)
                else:
                    # Fallback: manual translation
                    positions = positions + translate.reshape(1, 3)
                transforms_applied.append(f"Translation: [{translate[0]:.2f}, {translate[1]:.2f}, {translate[2]:.2f}]")

            if not transforms_applied:
                QMessageBox.information(self, "No Transform",
                    "No transforms were applied (scale=1, angle=0, translation=0).")
                return

            # Update stored positions
            if is_stored_positions:
                self._positions = positions
            elif hasattr(deformation, '_Xref'):
                # For FE field, transform both Xref and Xcurr consistently
                deformation._Xref = positions
                if hasattr(deformation, '_Xcurr') and deformation._Xcurr is not None:
                    # Apply same transforms to Xcurr
                    Xcurr = deformation._Xcurr
                    if hasattr(Xcurr, 'get'):
                        Xcurr = Xcurr.get()
                    Xcurr = np.asarray(Xcurr, dtype=np.float64)

                    if abs(scale - 1.0) > 1e-9:
                        Xcurr = Xcurr * scale
                    if abs(angle) > 1e-9:
                        Xcurr = Xcurr @ R_row
                    if np.any(np.abs(translate) > 1e-9):
                        Xcurr = Xcurr + translate.reshape(1, 3)

                    deformation._Xcurr = Xcurr

            self.state.notify_object_modified("deformation")
            self._refresh_display()
            QMessageBox.information(self, "Success",
                f"Transform applied to field positions:\n" + "\n".join(transforms_applied))

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to apply transform:\n{str(e)}")

    def _on_apply(self):
        """Apply deformation to sample."""
        deformation = self.state.deformation
        sample = self.state.sample

        if deformation is None:
            QMessageBox.warning(self, "No Deformation", "Please create a deformation object first.")
            return
        if sample is None:
            QMessageBox.warning(self, "No Sample", "Please create a sample first.")
            return

        method = self.apply_method.currentData()

        try:
            progress = QProgressDialog("Applying deformation...", "Cancel", 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            if method == "knn_idw":
                # Apply deformation field using kNN + IDW interpolation
                if self._positions is None or self._F is None:
                    progress.close()
                    QMessageBox.warning(self, "No Field",
                        "No deformation field loaded. Import a deformation field first.")
                    return

                if hasattr(deformation, 'apply_deformation_chunked'):
                    k = self.knn_k.value()
                    power = self.idw_power.value()
                    chunk_size = self.chunk_size.value()
                    use_gpu = self.use_gpu.isChecked()

                    # With a Sample argument the stored chunks are rewritten in
                    # place and the metadata updated; chunk_size only applies
                    # to array input.
                    deformation.apply_deformation_chunked(
                        field_positions=self._positions,
                        field_F=self._F,
                        sample=sample,
                        chunk_size=chunk_size,
                        k=k,
                        power=power,
                        use_gpu=use_gpu
                    )
                else:
                    progress.close()
                    QMessageBox.warning(self, "Not Supported",
                        "Deformation object doesn't support apply_deformation_chunked.")
                    return

            elif method == "mls":
                # Apply FE nodal field using MLS fitting
                if not hasattr(deformation, '_Xref') or deformation._Xref is None:
                    progress.close()
                    QMessageBox.warning(self, "No FE Field",
                        "No FE nodal field loaded. Import an FE nodal field first.")
                    return

                if hasattr(deformation, 'apply_fe_nodal_field'):
                    use_gpu = self.use_gpu.isChecked()
                    # apply_fe_nodal_field uses internally stored _Xref, _Xcurr
                    deformation.apply_fe_nodal_field(sample, use_gpu=use_gpu)
                else:
                    progress.close()
                    QMessageBox.warning(self, "Not Supported",
                        "Deformation object doesn't support apply_fe_nodal_field.")
                    return

            progress.close()
            self.state.notify_object_modified("sample")
            self.field_applied.emit()
            QMessageBox.information(self, "Success", "Deformation applied to sample successfully.")

        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Error", f"Failed to apply deformation:\n{str(e)}")

    def get_config(self) -> dict:
        """Get the current configuration."""
        return {
            "mode": "field" if self.field_mode.isChecked() else "fe",
            "df_preset": self.df_preset.currentData(),
            "df_position_scale": self.df_position_scale.value(),
            "fe_preset": self.fe_preset.currentData(),
            "fe_position_scale": self.fe_position_scale.value(),
            "conn_preset": self.conn_preset.currentData(),
            "scale": self.scale.value(),
            "rotation_axis": self.rot_axis.get_value(),
            "rotation_angle": self.rot_angle.value(),
            "translation": self.translation.get_value(),
            "apply_method": self.apply_method.currentData(),
            "knn_k": self.knn_k.value(),
            "idw_power": self.idw_power.value(),
            "chunk_size": self.chunk_size.value(),
            "use_gpu": self.use_gpu.isChecked(),
        }

    def set_config(self, config: dict):
        """Set the configuration."""
        if "mode" in config:
            if config["mode"] == "field":
                self.field_mode.setChecked(True)
            else:
                self.fe_mode.setChecked(True)
            self._on_mode_changed(None)

        if "df_preset" in config:
            idx = self.df_preset.findData(config["df_preset"])
            if idx >= 0:
                self.df_preset.setCurrentIndex(idx)

        if "df_position_scale" in config:
            self.df_position_scale.setValue(config["df_position_scale"])

        if "fe_preset" in config:
            idx = self.fe_preset.findData(config["fe_preset"])
            if idx >= 0:
                self.fe_preset.setCurrentIndex(idx)

        if "fe_position_scale" in config:
            self.fe_position_scale.setValue(config["fe_position_scale"])

        if "conn_preset" in config:
            idx = self.conn_preset.findData(config["conn_preset"])
            if idx >= 0:
                self.conn_preset.setCurrentIndex(idx)

        if "scale" in config:
            self.scale.setValue(config["scale"])

        if "rotation_axis" in config:
            self.rot_axis.set_value(config["rotation_axis"])

        if "rotation_angle" in config:
            self.rot_angle.setValue(config["rotation_angle"])

        if "translation" in config:
            self.translation.set_value(config["translation"])

        if "apply_method" in config:
            idx = self.apply_method.findData(config["apply_method"])
            if idx >= 0:
                self.apply_method.setCurrentIndex(idx)

        if "knn_k" in config:
            self.knn_k.setValue(config["knn_k"])

        if "idw_power" in config:
            self.idw_power.setValue(config["idw_power"])

        if "chunk_size" in config:
            self.chunk_size.setValue(config["chunk_size"])

        if "use_gpu" in config:
            self.use_gpu.setChecked(config["use_gpu"])
