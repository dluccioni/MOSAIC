# -----------------------------------------------------------------------------
# Sample Inspector
# -----------------------------------------------------------------------------
"""
Inspector panel for sample configuration.

Provides controls for:
- Sample dimensions
- Sample type (single/polycrystalline)
- Temperature effects
- Sample generation
- Import/Export atomic positions
- Transform operations (rotate, translate)
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
    QComboBox,
    QCheckBox,
    QMessageBox,
    QProgressDialog,
    QDialog,
    QDialogButtonBox,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)

# Standard atomic masses in amu for common elements
ATOMIC_MASSES = {
    'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.81, 'C': 12.011,
    'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305,
    'Al': 26.982, 'Si': 28.086, 'P': 30.974, 'S': 32.065, 'Cl': 35.453, 'Ar': 39.948,
    'K': 39.098, 'Ca': 40.078, 'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996,
    'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38,
    'Ga': 69.723, 'Ge': 72.64, 'As': 74.922, 'Se': 78.96, 'Br': 79.904, 'Kr': 83.798,
    'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224, 'Nb': 92.906, 'Mo': 95.96,
    'Ru': 101.07, 'Rh': 102.906, 'Pd': 106.42, 'Ag': 107.868, 'Cd': 112.411, 'In': 114.818,
    'Sn': 118.710, 'Sb': 121.760, 'Te': 127.60, 'I': 126.904, 'Xe': 131.293, 'Cs': 132.905,
    'Ba': 137.327, 'La': 138.905, 'Ce': 140.116, 'Hf': 178.49, 'Ta': 180.948, 'W': 183.84,
    'Re': 186.207, 'Os': 190.23, 'Ir': 192.217, 'Pt': 195.084, 'Au': 196.967, 'Hg': 200.59,
    'Tl': 204.383, 'Pb': 207.2, 'Bi': 208.980, 'U': 238.029,
}

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gui.state import SimulationState
from gui.panels.inspector import InspectorPanel, PropertyDef, Vector3Widget


class MDImportDialog(QDialog):
    """
    Dialog for importing atomic structures from MD simulation output files.

    Provides configuration for:
    - Input file selection
    - Element list mapping (species ID -> element symbol)
    - Header lines to skip
    - Column indices for species ID and positions
    - Scale factor for unit conversion
    """

    def __init__(self, parent=None, initial_directory=""):
        super().__init__(parent)
        self.setWindowTitle("Import MD Atomic Structure")
        self.setMinimumWidth(500)
        self._initial_directory = initial_directory
        self._setup_ui()

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)

        # File Selection Group
        file_group = QGroupBox("Input File")
        file_layout = QFormLayout(file_group)

        file_row = QWidget()
        file_row_layout = QHBoxLayout(file_row)
        file_row_layout.setContentsMargins(0, 0, 0, 0)
        file_row_layout.setSpacing(4)

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select MD output file...")
        file_row_layout.addWidget(self.file_path_edit, 1)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse)
        file_row_layout.addWidget(browse_btn)

        file_layout.addRow("File:", file_row)
        layout.addWidget(file_group)

        # Element Mapping Group
        element_group = QGroupBox("Element Mapping")
        element_layout = QFormLayout(element_group)

        self.element_list_edit = QLineEdit()
        self.element_list_edit.setPlaceholderText("e.g., Fe, C, H (comma-separated)")
        self.element_list_edit.setToolTip(
            "Comma-separated list of element symbols.\n"
            "Maps 1-based species ID to element: ID=1 -> first element, ID=2 -> second, etc."
        )
        element_layout.addRow("Elements:", self.element_list_edit)

        element_help = QLabel("Species ID 1 → first element, ID 2 → second, etc.")
        element_help.setStyleSheet("color: #808080; font-size: 10px;")
        element_layout.addRow("", element_help)

        layout.addWidget(element_group)

        # File Format Group
        format_group = QGroupBox("File Format")
        format_layout = QFormLayout(format_group)

        self.header_lines_spin = QSpinBox()
        self.header_lines_spin.setRange(0, 1000)
        self.header_lines_spin.setValue(9)
        self.header_lines_spin.setToolTip("Number of header lines to skip at the start of the file")
        format_layout.addRow("Header lines:", self.header_lines_spin)

        self.id_column_spin = QSpinBox()
        self.id_column_spin.setRange(0, 100)
        self.id_column_spin.setValue(1)
        self.id_column_spin.setToolTip("0-based column index containing the species ID")
        format_layout.addRow("ID column (0-based):", self.id_column_spin)

        self.position_columns_edit = QLineEdit()
        self.position_columns_edit.setText("2, 3, 4")
        self.position_columns_edit.setToolTip(
            "0-based column indices for X, Y, Z positions (comma-separated)"
        )
        format_layout.addRow("Position columns:", self.position_columns_edit)

        layout.addWidget(format_group)

        # Unit Conversion Group
        units_group = QGroupBox("Unit Conversion")
        units_layout = QFormLayout(units_group)

        self.scale_combo = QComboBox()
        self.scale_combo.addItem("Angstroms (no conversion)", 1e-10)
        self.scale_combo.addItem("Nanometers", 1e-9)
        self.scale_combo.addItem("Meters", 1.0)
        self.scale_combo.addItem("Bohr radii", 5.29177e-11)
        self.scale_combo.addItem("Custom...", None)
        self.scale_combo.currentIndexChanged.connect(self._on_scale_changed)
        units_layout.addRow("Input units:", self.scale_combo)

        self.custom_scale_edit = QLineEdit()
        self.custom_scale_edit.setText("1e-10")
        self.custom_scale_edit.setEnabled(False)
        self.custom_scale_edit.setToolTip(
            "Scale factor: input_value * (scale / 1e-10) = output in Angstroms\n"
            "e.g., 1e-10 for Angstroms, 1e-9 for nm, 1.0 for meters"
        )
        units_layout.addRow("Custom scale:", self.custom_scale_edit)

        layout.addWidget(units_group)

        # Preview Group
        preview_group = QGroupBox("File Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(100)
        self.preview_text.setStyleSheet("font-family: monospace; font-size: 10px;")
        preview_layout.addWidget(self.preview_text)

        preview_btn = QPushButton("Load Preview")
        preview_btn.clicked.connect(self._load_preview)
        preview_layout.addWidget(preview_btn)

        layout.addWidget(preview_group)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _on_browse(self):
        """Handle file browse button."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select MD Output File",
            self._initial_directory,
            "Text Files (*.txt *.dat *.dump *.xyz *.lmp);;All Files (*)"
        )
        if filename:
            self.file_path_edit.setText(filename)
            self._load_preview()

    def _on_scale_changed(self, index):
        """Handle scale combo change."""
        is_custom = self.scale_combo.currentData() is None
        self.custom_scale_edit.setEnabled(is_custom)

    def _load_preview(self):
        """Load and display file preview."""
        filepath = self.file_path_edit.text()
        if not filepath or not Path(filepath).exists():
            self.preview_text.setText("No file selected or file not found.")
            return

        try:
            with open(filepath, 'r') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= 20:  # Show first 20 lines
                        lines.append("...")
                        break
                    lines.append(f"{i:3d}: {line.rstrip()}")
                self.preview_text.setText("\n".join(lines))
        except Exception as e:
            self.preview_text.setText(f"Error reading file: {str(e)}")

    def get_parameters(self):
        """
        Get the import parameters from the dialog.

        Returns:
            dict: Import parameters or None if validation fails
        """
        # Validate file path
        filepath = self.file_path_edit.text()
        if not filepath:
            QMessageBox.warning(self, "Missing File", "Please select an input file.")
            return None
        if not Path(filepath).exists():
            QMessageBox.warning(self, "File Not Found", f"File not found:\n{filepath}")
            return None

        # Parse element list
        element_text = self.element_list_edit.text().strip()
        if not element_text:
            QMessageBox.warning(self, "Missing Elements", "Please enter the element list.")
            return None
        element_list = [e.strip() for e in element_text.split(",")]
        if not element_list or not all(element_list):
            QMessageBox.warning(self, "Invalid Elements", "Please enter valid element symbols.")
            return None

        # Parse position columns
        try:
            pos_text = self.position_columns_edit.text().strip()
            position_columns = [int(x.strip()) for x in pos_text.split(",")]
            if len(position_columns) != 3:
                raise ValueError("Need exactly 3 position columns")
        except Exception as e:
            QMessageBox.warning(
                self, "Invalid Position Columns",
                f"Please enter 3 comma-separated column indices.\nError: {str(e)}"
            )
            return None

        # Get scale factor
        scale_data = self.scale_combo.currentData()
        if scale_data is None:
            try:
                scale = float(self.custom_scale_edit.text())
            except ValueError:
                QMessageBox.warning(self, "Invalid Scale", "Please enter a valid scale factor.")
                return None
        else:
            scale = scale_data

        return {
            "import_file": filepath,
            "element_list": element_list,
            "header_lines": self.header_lines_spin.value(),
            "ID_column": self.id_column_spin.value(),
            "position_columns": position_columns,
            "scale": scale,
        }


class SampleInspector(InspectorPanel):
    """
    Inspector for sample configuration.

    Signals:
        sample_generated: Emitted when a new sample is generated
        sample_imported: Emitted when sample is imported from file
    """

    sample_generated = Signal(object)
    sample_imported = Signal(object)

    def __init__(self, state: SimulationState, parent=None):
        """
        Initialize the sample inspector.

        Args:
            state: SimulationState instance
            parent: Parent widget
        """
        super().__init__(state, parent)
        self.set_title("Sample")
        self._setup_sample_ui()
        self._register_observers()

    def _setup_sample_ui(self):
        """Setup sample-specific UI elements."""
        # Directory Group (required first)
        dir_group = self.add_group("Directory")
        dir_layout = dir_group.layout()

        # Directory path row
        dir_row = QWidget()
        dir_row_layout = QHBoxLayout(dir_row)
        dir_row_layout.setContentsMargins(0, 0, 0, 0)
        dir_row_layout.setSpacing(4)

        self.dir_path_edit = QLineEdit()
        # Set placeholder based on global directory
        global_dir = self.state.global_working_directory
        if global_dir:
            self.dir_path_edit.setPlaceholderText(f"Using global: {global_dir}")
        else:
            self.dir_path_edit.setPlaceholderText("Select directory for sample files...")
        self.dir_path_edit.setReadOnly(True)
        dir_row_layout.addWidget(self.dir_path_edit, 1)

        browse_dir_btn = QPushButton("Browse...")
        browse_dir_btn.clicked.connect(self._on_browse_directory)
        dir_row_layout.addWidget(browse_dir_btn)

        dir_layout.addRow("Directory:", dir_row)

        # Status label for directory
        self.dir_status = QLabel("No directory selected")
        self.dir_status.setStyleSheet("color: #ff8800;")
        dir_layout.addRow("", self.dir_status)

        # Load existing button (only enabled when metadata found)
        self.load_existing_btn = QPushButton("Load Existing Sample")
        self.load_existing_btn.clicked.connect(self._on_load_existing)
        self.load_existing_btn.setEnabled(False)
        dir_layout.addRow("", self.load_existing_btn)

        # Dimensions Group
        dim_group = self.add_group("Dimensions")
        dim_layout = dim_group.layout()

        # X dimension
        self.dim_x = QDoubleSpinBox()
        self.dim_x.setDecimals(1)
        self.dim_x.setRange(1, 1e9)
        self.dim_x.setValue(500)
        self.dim_x.setSuffix(" Å")
        self.dim_x.setSingleStep(100)
        dim_layout.addRow("X (beam):", self.dim_x)

        # Y dimension
        self.dim_y = QDoubleSpinBox()
        self.dim_y.setDecimals(1)
        self.dim_y.setRange(1, 1e9)
        self.dim_y.setValue(500)
        self.dim_y.setSuffix(" Å")
        self.dim_y.setSingleStep(100)
        dim_layout.addRow("Y (horizontal):", self.dim_y)

        # Z dimension
        self.dim_z = QDoubleSpinBox()
        self.dim_z.setDecimals(1)
        self.dim_z.setRange(1, 1e9)
        self.dim_z.setValue(500)
        self.dim_z.setSuffix(" Å")
        self.dim_z.setSingleStep(100)
        dim_layout.addRow("Z (vertical):", self.dim_z)

        # Sample Type Group
        type_group = self.add_group("Sample Type")
        type_layout = type_group.layout()

        self.sample_type = QComboBox()
        self.sample_type.addItem("Single Crystal", "single")
        self.sample_type.addItem("Polycrystalline", "poly")
        self.sample_type.currentIndexChanged.connect(self._on_sample_type_changed)
        type_layout.addRow("Type:", self.sample_type)

        # Polycrystalline Settings Group (shown/hidden based on sample type)
        self.poly_group = self.add_group("Polycrystalline Settings")
        poly_layout = self.poly_group.layout()

        # Number of grains
        self.n_grains = QSpinBox()
        self.n_grains.setRange(1, 10000)
        self.n_grains.setValue(8)
        poly_layout.addRow("Number of Grains:", self.n_grains)

        # Voronoi seed method
        self.voronoi_method = QComboBox()
        self.voronoi_method.addItem("Uniform (Grid-based)", "uniform")
        self.voronoi_method.addItem("Random", "random")
        poly_layout.addRow("Seed Distribution:", self.voronoi_method)

        # Random seed for reproducibility
        self.poly_seed = QSpinBox()
        self.poly_seed.setRange(0, 2147483647)
        self.poly_seed.setSpecialValueText("Random")
        self.poly_seed.setValue(0)
        poly_layout.addRow("Random Seed:", self.poly_seed)

        # Orientation mode
        self.orientation_mode = QComboBox()
        self.orientation_mode.addItem("Random", "random")
        self.orientation_mode.addItem("Textured", "textured")
        self.orientation_mode.currentIndexChanged.connect(self._on_orientation_mode_changed)
        poly_layout.addRow("Grain Orientations:", self.orientation_mode)

        # Texture axis (X, Y, Z) - shown only for textured mode
        self.texture_axis_x = QDoubleSpinBox()
        self.texture_axis_x.setRange(-1.0, 1.0)
        self.texture_axis_x.setDecimals(3)
        self.texture_axis_x.setValue(0.0)

        self.texture_axis_y = QDoubleSpinBox()
        self.texture_axis_y.setRange(-1.0, 1.0)
        self.texture_axis_y.setDecimals(3)
        self.texture_axis_y.setValue(0.0)

        self.texture_axis_z = QDoubleSpinBox()
        self.texture_axis_z.setRange(-1.0, 1.0)
        self.texture_axis_z.setDecimals(3)
        self.texture_axis_z.setValue(1.0)

        axis_widget = QWidget()
        axis_layout = QHBoxLayout(axis_widget)
        axis_layout.setContentsMargins(0, 0, 0, 0)
        axis_layout.addWidget(QLabel("X:"))
        axis_layout.addWidget(self.texture_axis_x)
        axis_layout.addWidget(QLabel("Y:"))
        axis_layout.addWidget(self.texture_axis_y)
        axis_layout.addWidget(QLabel("Z:"))
        axis_layout.addWidget(self.texture_axis_z)
        self.texture_axis_label = QLabel("Texture Axis:")
        poly_layout.addRow(self.texture_axis_label, axis_widget)
        self.texture_axis_widget = axis_widget

        # Texture spread
        self.texture_spread = QDoubleSpinBox()
        self.texture_spread.setRange(0.1, 90.0)
        self.texture_spread.setDecimals(1)
        self.texture_spread.setValue(5.0)
        self.texture_spread.setSuffix("°")
        self.texture_spread_label = QLabel("Texture Spread:")
        poly_layout.addRow(self.texture_spread_label, self.texture_spread)

        # Initially hide polycrystalline group and texture controls
        self.poly_group.setVisible(False)
        self.texture_axis_label.setVisible(False)
        self.texture_axis_widget.setVisible(False)
        self.texture_spread_label.setVisible(False)
        self.texture_spread.setVisible(False)

        # GPU Settings Group (for sample generation)
        gpu_group = self.add_group("GPU Settings")
        gpu_layout = gpu_group.layout()

        # Number of GPUs to use
        self.n_gpus = QSpinBox()
        self.n_gpus.setRange(0, 16)  # 0 = auto-detect
        self.n_gpus.setSpecialValueText("Auto")
        self.n_gpus.setValue(0)
        self.n_gpus.setToolTip("Number of GPUs to use for sample generation. 0 = use all available.")
        gpu_layout.addRow("GPUs to Use:", self.n_gpus)

        # Temperature Effects Group
        temp_group = self.add_group("Temperature Effects")
        temp_layout = temp_group.layout()

        self.temp_enabled = QCheckBox("Enable thermal vibrations")
        self.temp_enabled.stateChanged.connect(self._on_temp_enabled_changed)
        temp_layout.addRow("", self.temp_enabled)

        # Distribution selection
        self.temp_distribution = QComboBox()
        self.temp_distribution.addItem("Gaussian", "gaussian")
        self.temp_distribution.addItem("Einstein", "einstein")
        self.temp_distribution.addItem("Debye", "debye")
        self.temp_distribution.setEnabled(False)
        self.temp_distribution.currentIndexChanged.connect(self._on_distribution_changed)
        temp_layout.addRow("Model:", self.temp_distribution)

        # Sigma field (for Gaussian mode)
        self.temp_sigma = QDoubleSpinBox()
        self.temp_sigma.setDecimals(3)
        self.temp_sigma.setRange(0.001, 10.0)
        self.temp_sigma.setValue(0.1)
        self.temp_sigma.setSuffix(" Å")
        self.temp_sigma.setEnabled(False)
        self.temp_sigma.valueChanged.connect(self._apply_temperature_settings)
        self.temp_sigma_label = QLabel("Sigma:")
        temp_layout.addRow(self.temp_sigma_label, self.temp_sigma)

        # Temperature field (for Einstein/Debye modes)
        self.temperature = QDoubleSpinBox()
        self.temperature.setDecimals(1)
        self.temperature.setRange(0, 10000)
        self.temperature.setValue(300)
        self.temperature.setSuffix(" K")
        self.temperature.setEnabled(False)
        self.temperature.valueChanged.connect(self._apply_temperature_settings)
        self.temperature_label = QLabel("Temperature:")
        temp_layout.addRow(self.temperature_label, self.temperature)

        # Characteristic temperature field (Einstein θE or Debye θD)
        self.char_temp = QDoubleSpinBox()
        self.char_temp.setDecimals(1)
        self.char_temp.setRange(1, 10000)
        self.char_temp.setValue(315)
        self.char_temp.setSuffix(" K")
        self.char_temp.setEnabled(False)
        self.char_temp.valueChanged.connect(self._apply_temperature_settings)
        self.char_temp_label = QLabel("Debye Temp θD:")
        temp_layout.addRow(self.char_temp_label, self.char_temp)

        # Random seed for displacement generation
        self.temp_seed = QSpinBox()
        self.temp_seed.setRange(0, 2147483647)  # Max int32
        self.temp_seed.setValue(40)
        self.temp_seed.setEnabled(False)
        self.temp_seed.valueChanged.connect(self._apply_temperature_settings)
        self.temp_seed_label = QLabel("Random Seed:")
        temp_layout.addRow(self.temp_seed_label, self.temp_seed)

        # Thermal Expansion sub-section
        expansion_header = QLabel("Thermal Expansion")
        expansion_header.setStyleSheet("font-weight: bold; margin-top: 8px;")
        temp_layout.addRow("", expansion_header)

        # Enable expansion checkbox
        self.expansion_enabled = QCheckBox("Enable thermal expansion")
        self.expansion_enabled.stateChanged.connect(self._on_expansion_enabled_changed)
        self.expansion_enabled.setEnabled(False)  # Enabled when temperature is enabled
        temp_layout.addRow("", self.expansion_enabled)

        # Expansion mode (isotropic vs anisotropic)
        self.expansion_mode = QComboBox()
        self.expansion_mode.addItem("Isotropic", "isotropic")
        self.expansion_mode.addItem("Anisotropic", "anisotropic")
        self.expansion_mode.setEnabled(False)
        self.expansion_mode.currentIndexChanged.connect(self._on_expansion_mode_changed)
        self.expansion_mode_label = QLabel("Mode:")
        temp_layout.addRow(self.expansion_mode_label, self.expansion_mode)

        # Isotropic alpha (single value)
        self.expansion_alpha = QDoubleSpinBox()
        self.expansion_alpha.setDecimals(8)
        self.expansion_alpha.setRange(0, 1e-3)
        self.expansion_alpha.setValue(1.2e-5)  # Typical for metals
        self.expansion_alpha.setSuffix(" /K")
        self.expansion_alpha.setEnabled(False)
        self.expansion_alpha.valueChanged.connect(self._apply_expansion_settings)
        self.expansion_alpha_label = QLabel("α:")
        temp_layout.addRow(self.expansion_alpha_label, self.expansion_alpha)

        # Anisotropic alpha (x, y, z) - using Vector3Widget
        self.expansion_alpha_xyz = Vector3Widget([1.2e-5, 1.2e-5, 1.2e-5], decimals=8, suffix="/K")
        self.expansion_alpha_xyz.setEnabled(False)
        self.expansion_alpha_xyz.value_changed.connect(self._apply_expansion_settings)
        self.expansion_alpha_xyz_label = QLabel("α (x,y,z):")
        temp_layout.addRow(self.expansion_alpha_xyz_label, self.expansion_alpha_xyz)

        # Reference temperature
        self.expansion_T_ref = QDoubleSpinBox()
        self.expansion_T_ref.setDecimals(1)
        self.expansion_T_ref.setRange(0, 10000)
        self.expansion_T_ref.setValue(300)
        self.expansion_T_ref.setSuffix(" K")
        self.expansion_T_ref.setEnabled(False)
        self.expansion_T_ref.valueChanged.connect(self._apply_expansion_settings)
        self.expansion_T_ref_label = QLabel("Reference T:")
        temp_layout.addRow(self.expansion_T_ref_label, self.expansion_T_ref)

        # Initialize field visibility for default (Gaussian)
        self._update_temp_field_visibility()
        self._update_expansion_field_visibility()

        # Generation Group
        gen_group = self.add_group("Generation")
        gen_layout = gen_group.layout()

        # Generate button
        generate_btn = QPushButton("Generate Sample")
        generate_btn.clicked.connect(self._on_generate)
        generate_btn.setStyleSheet("QPushButton { background-color: #2a6a2a; }")
        gen_layout.addRow("", generate_btn)

        # Atom count estimate
        self.atom_estimate = QLabel("Estimated atoms: --")
        self.atom_estimate.setStyleSheet("color: #808080;")
        gen_layout.addRow("", self.atom_estimate)

        # Connect dimension changes to estimate update
        self.dim_x.valueChanged.connect(self._update_estimate)
        self.dim_y.valueChanged.connect(self._update_estimate)
        self.dim_z.valueChanged.connect(self._update_estimate)

        # Import/Export Group
        io_group = self.add_group("Import/Export")
        io_layout = io_group.layout()

        # Import row
        import_row = QWidget()
        import_layout = QHBoxLayout(import_row)
        import_layout.setContentsMargins(0, 0, 0, 0)
        import_layout.setSpacing(4)

        import_btn = QPushButton("Import Positions")
        import_btn.clicked.connect(self._on_import)
        import_layout.addWidget(import_btn)

        export_btn = QPushButton("Export Positions")
        export_btn.clicked.connect(self._on_export)
        import_layout.addWidget(export_btn)

        io_layout.addRow("", import_row)

        # Sample Info Group
        info_group = self.add_group("Sample Info")
        info_layout = info_group.layout()

        self.info_atoms = QLabel("Atoms: --")
        info_layout.addRow("", self.info_atoms)

        self.info_dims = QLabel("Dimensions: --")
        info_layout.addRow("", self.info_dims)

        self.info_memory = QLabel("Memory: --")
        info_layout.addRow("", self.info_memory)

        # Transform Group
        transform_group = self.add_group("Transform")
        transform_layout = transform_group.layout()

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

        rotate_btn = QPushButton("Apply Rotation")
        rotate_btn.clicked.connect(self._on_rotate)
        transform_layout.addRow("", rotate_btn)

        # Translation
        transform_layout.addRow(QLabel("Translation:"))
        self.translation = Vector3Widget([0, 0, 0], decimals=1, suffix="Å")
        transform_layout.addRow("Offset:", self.translation)

        translate_btn = QPushButton("Apply Translation")
        translate_btn.clicked.connect(self._on_translate)
        transform_layout.addRow("", translate_btn)

        # Zero to center
        zero_btn = QPushButton("Zero to Center")
        zero_btn.clicked.connect(self._on_zero)
        transform_layout.addRow("", zero_btn)

    def _register_observers(self):
        """Register for state change notifications."""
        self.state.register_observer("sample_changed", self._on_sample_state_changed)
        self.state.register_observer("crystal_changed", self._on_crystal_changed)
        self.state.register_observer("global_working_directory_changed", self._on_global_dir_changed)

    def _on_global_dir_changed(self, directory):
        """Handle global working directory change."""
        # Update placeholder text if directory field is empty
        if not self.dir_path_edit.text():
            self.dir_path_edit.setPlaceholderText(
                f"Using global: {directory}" if directory
                else "Select directory for sample files..."
            )

    def _on_sample_state_changed(self, sample):
        """Handle sample state change."""
        self._refresh_display()
        # Apply current temperature settings to the new sample
        self._apply_temperature_settings()

    def _on_crystal_changed(self, crystal):
        """Handle crystal state change - update estimate."""
        self._update_estimate()

    def _on_browse_directory(self):
        """Handle browse directory button."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory for Sample Files",
            self.state.global_working_directory,
            QFileDialog.ShowDirsOnly
        )
        if directory:
            self.dir_path_edit.setText(directory)
            self.dir_status.setText("Directory selected")
            self.dir_status.setStyleSheet("color: #00cc00;")

            # Check if existing sample metadata exists in this directory
            import os
            metadata_path = os.path.join(directory, "sample_metadata.json")
            if os.path.exists(metadata_path):
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
                self.dir_status.setText("Metadata found - click 'Load Existing Sample'")
            else:
                # Disable button and reset style when no metadata
                self.load_existing_btn.setEnabled(False)
                self.load_existing_btn.setStyleSheet("")

    def _on_load_existing(self):
        """Handle load existing sample button click."""
        directory = self.dir_path_edit.text()
        if directory:
            self._load_existing_sample(directory)

    def _load_existing_sample(self, directory):
        """Load an existing sample from directory."""
        try:
            from Sample import sample
            existing_sample = sample(directory=directory)
            existing_sample.read_sample_metadata()
            self.state.sample = existing_sample
            self._refresh_display()

            # Update dimension inputs from loaded sample
            if existing_sample.dimensions is not None:
                dims = existing_sample.dimensions
                self.dim_x.setValue(float(dims[0]))
                self.dim_y.setValue(float(dims[1]))
                self.dim_z.setValue(float(dims[2]))

            self.dir_status.setText("Existing sample loaded")
            self.dir_status.setStyleSheet("color: #00cc00;")
        except Exception as e:
            QMessageBox.warning(self, "Load Failed", f"Failed to load existing sample:\n{str(e)}")

    def _refresh_display(self):
        """Refresh display from current sample state."""
        sample = self.state.sample
        if sample is None:
            self.info_atoms.setText("Atoms: --")
            self.info_dims.setText("Dimensions: --")
            self.info_memory.setText("Memory: --")
            return

        try:
            # Update dimensions from Sample's dimensions property
            if hasattr(sample, 'dimensions') and sample.dimensions is not None:
                dims = sample.dimensions
                self.info_dims.setText(f"Dimensions: {dims[0]:.0f} x {dims[1]:.0f} x {dims[2]:.0f} Å")

            # Update directory display if available
            if hasattr(sample, 'directory') and sample.directory:
                self.dir_path_edit.setText(sample.directory)
                self.dir_status.setText("Sample loaded")
                self.dir_status.setStyleSheet("color: #00cc00;")

            # Sample doesn't have atom_count directly - check chunk_total and estimate
            if hasattr(sample, '_chunk_total') and sample._chunk_total is not None:
                chunk_count = sample._chunk_total
                self.info_atoms.setText(f"Chunks: {chunk_count}")

                # Estimate atoms from chunk files if available
                import os
                if hasattr(sample, 'directory'):
                    total_atoms = 0
                    for i in range(1, chunk_count + 1):
                        chunk_file = os.path.join(sample.directory, f"atomic_positions_{i}.npy")
                        if os.path.exists(chunk_file):
                            try:
                                positions = np.load(chunk_file, mmap_mode='r')
                                total_atoms += len(positions)
                            except Exception:
                                pass
                    if total_atoms > 0:
                        self.info_atoms.setText(f"Atoms: {total_atoms:,}")
                        # Estimate memory
                        memory_bytes = total_atoms * 3 * 4  # float32
                        if memory_bytes > 1e9:
                            self.info_memory.setText(f"Memory: ~{memory_bytes/1e9:.2f} GB")
                        elif memory_bytes > 1e6:
                            self.info_memory.setText(f"Memory: ~{memory_bytes/1e6:.2f} MB")
                        else:
                            self.info_memory.setText(f"Memory: ~{memory_bytes/1e3:.2f} KB")
            else:
                self.info_atoms.setText("Atoms: Not generated")
                self.info_memory.setText("Memory: --")

        except Exception as e:
            self.info_atoms.setText(f"Atoms: Error")
            self.info_memory.setText("Memory: --")

    def _update_estimate(self):
        """Update the estimated atom count."""
        crystal = self.state.crystal
        if crystal is None:
            self.atom_estimate.setText("Estimated atoms: -- (load crystal first)")
            return

        try:
            # Get sample volume in Å³
            volume = self.dim_x.value() * self.dim_y.value() * self.dim_z.value()

            # Estimate atoms per unit cell volume using Crystal's actual properties
            if hasattr(crystal, 'lattice_volume') and crystal.lattice_volume is not None:
                unit_cell_volume = crystal.lattice_volume

                # Count atoms in primitive cell
                atoms_per_cell = 1  # Default
                if hasattr(crystal, 'lattice_atom_fractional') and crystal.lattice_atom_fractional is not None:
                    atoms_per_cell = len(crystal.lattice_atom_fractional)

                estimated = int(volume / unit_cell_volume * atoms_per_cell)
                self.atom_estimate.setText(f"Estimated atoms: ~{estimated:,}")
            elif hasattr(crystal, 'lattice_lengths') and crystal.lattice_lengths is not None:
                # Fallback: estimate from lattice lengths (assuming orthorhombic)
                lengths = crystal.lattice_lengths
                unit_cell_volume = lengths[0] * lengths[1] * lengths[2]
                atoms_per_cell = 1
                if hasattr(crystal, 'lattice_atom_fractional') and crystal.lattice_atom_fractional is not None:
                    atoms_per_cell = len(crystal.lattice_atom_fractional)
                estimated = int(volume / unit_cell_volume * atoms_per_cell)
                self.atom_estimate.setText(f"Estimated atoms: ~{estimated:,}")
            else:
                self.atom_estimate.setText("Estimated atoms: -- (no lattice info)")
        except Exception as e:
            self.atom_estimate.setText(f"Estimated atoms: -- ({str(e)[:20]})")

    def _on_sample_type_changed(self, index):
        """Show/hide polycrystalline controls based on sample type."""
        sample_type = self.sample_type.currentData()
        self.poly_group.setVisible(sample_type == "poly")

    def _on_orientation_mode_changed(self, index):
        """Show/hide texture controls based on orientation mode."""
        is_textured = self.orientation_mode.currentData() == "textured"
        self.texture_axis_label.setVisible(is_textured)
        self.texture_axis_widget.setVisible(is_textured)
        self.texture_spread_label.setVisible(is_textured)
        self.texture_spread.setVisible(is_textured)

    def _on_temp_enabled_changed(self, state):
        """Handle temperature checkbox change."""
        enabled = self.temp_enabled.isChecked()
        self.temp_distribution.setEnabled(enabled)
        self._update_temp_field_visibility()

        # Enable expansion controls when temperature is enabled
        self.expansion_enabled.setEnabled(enabled)
        if not enabled:
            self.expansion_enabled.setChecked(False)
        self._update_expansion_field_visibility()

        # Apply to existing sample so changes take effect on next simulation
        self._apply_temperature_settings()

    def _on_distribution_changed(self, index):
        """Handle distribution/model selection change."""
        self._update_temp_field_visibility()
        self._apply_temperature_settings()

    def _update_temp_field_visibility(self):
        """Update field visibility and labels based on selected model."""
        enabled = self.temp_enabled.isChecked()
        dist = self.temp_distribution.currentData()

        # Seed is always visible when temperature is enabled
        self.temp_seed.setVisible(True)
        self.temp_seed_label.setVisible(True)
        self.temp_seed.setEnabled(enabled)
        self.temp_seed_label.setEnabled(enabled)

        # Gaussian mode: show sigma, hide temperature fields
        if dist == "gaussian":
            self.temp_sigma.setVisible(True)
            self.temp_sigma_label.setVisible(True)
            self.temp_sigma.setEnabled(enabled)
            self.temp_sigma_label.setEnabled(enabled)

            self.temperature.setVisible(False)
            self.temperature_label.setVisible(False)
            self.char_temp.setVisible(False)
            self.char_temp_label.setVisible(False)

        # Einstein mode: show temperature and θE
        elif dist == "einstein":
            self.temp_sigma.setVisible(False)
            self.temp_sigma_label.setVisible(False)

            self.temperature.setVisible(True)
            self.temperature_label.setVisible(True)
            self.temperature.setEnabled(enabled)
            self.temperature_label.setEnabled(enabled)

            self.char_temp.setVisible(True)
            self.char_temp_label.setVisible(True)
            self.char_temp_label.setText("Einstein Temp θE:")
            self.char_temp.setEnabled(enabled)
            self.char_temp_label.setEnabled(enabled)

        # Debye mode: show temperature and θD
        elif dist == "debye":
            self.temp_sigma.setVisible(False)
            self.temp_sigma_label.setVisible(False)

            self.temperature.setVisible(True)
            self.temperature_label.setVisible(True)
            self.temperature.setEnabled(enabled)
            self.temperature_label.setEnabled(enabled)

            self.char_temp.setVisible(True)
            self.char_temp_label.setVisible(True)
            self.char_temp_label.setText("Debye Temp θD:")
            self.char_temp.setEnabled(enabled)
            self.char_temp_label.setEnabled(enabled)

    def _get_species_masses(self) -> dict:
        """Get species masses using defaults from the periodic table."""
        masses = {}
        crystal = self.state.crystal
        if crystal is None or not hasattr(crystal, 'species') or crystal.species is None:
            return masses

        # Get unique species and assign default masses
        species_list = list(set(crystal.species))
        for species in species_list:
            masses[str(species)] = ATOMIC_MASSES.get(str(species), 28.0)

        return masses

    def _apply_temperature_settings(self):
        """Apply current temperature settings to the existing sample.

        This allows temperature effects to be changed without regenerating the sample.
        The settings are applied on-the-fly when chunks are loaded during simulation.
        """
        sample = self.state.sample
        if sample is None:
            return

        if self.temp_enabled.isChecked():
            sample.enable_temp = True
            dist = self.temp_distribution.currentData()
            seed = self.temp_seed.value()

            if dist == "gaussian":
                # Gaussian: use sigma directly (in Angstroms)
                sigma = self.temp_sigma.value()
                sample.temp_params = ['gaussian', sigma, 1, seed]

            elif dist == "einstein":
                # Einstein model: use temperature and θE directly
                temp_K = self.temperature.value()
                theta_E = self.char_temp.value()
                sample.temp_params = ['einstein', temp_K, 1, seed]
                sample._temp_theta_E_K = theta_E
                # Set per-species masses to defaults
                sample._temp_species_mass_amu = self._get_species_masses()

            elif dist == "debye":
                # True Debye model with Debye integral
                temp_K = self.temperature.value()
                theta_D = self.char_temp.value()
                sample.temp_params = ['debye', temp_K, 1, seed]
                sample._temp_theta_D_K = theta_D
                # Set mass from crystal species (uses first species mass as default)
                species_masses = self._get_species_masses()
                if species_masses:
                    sample._temp_mass_amu = list(species_masses.values())[0]
        else:
            sample.enable_temp = False

        # Also apply expansion settings
        self._apply_expansion_settings()

    def _on_expansion_enabled_changed(self, state):
        """Handle expansion checkbox change."""
        enabled = self.expansion_enabled.isChecked()
        self.expansion_mode.setEnabled(enabled)
        self._update_expansion_field_visibility()
        self._apply_expansion_settings()

    def _on_expansion_mode_changed(self, index):
        """Handle expansion mode change."""
        self._update_expansion_field_visibility()
        self._apply_expansion_settings()

    def _update_expansion_field_visibility(self):
        """Show/hide fields based on expansion mode."""
        enabled = self.expansion_enabled.isChecked() and self.temp_enabled.isChecked()
        mode = self.expansion_mode.currentData()

        # Mode selector enabled when expansion is enabled
        self.expansion_mode.setEnabled(enabled)
        self.expansion_mode_label.setEnabled(enabled)

        # Reference T always visible when expansion enabled
        self.expansion_T_ref.setEnabled(enabled)
        self.expansion_T_ref_label.setEnabled(enabled)

        if mode == "isotropic":
            self.expansion_alpha.setVisible(True)
            self.expansion_alpha_label.setVisible(True)
            self.expansion_alpha.setEnabled(enabled)
            self.expansion_alpha_label.setEnabled(enabled)

            self.expansion_alpha_xyz.setVisible(False)
            self.expansion_alpha_xyz_label.setVisible(False)
        else:  # anisotropic
            self.expansion_alpha.setVisible(False)
            self.expansion_alpha_label.setVisible(False)

            self.expansion_alpha_xyz.setVisible(True)
            self.expansion_alpha_xyz_label.setVisible(True)
            self.expansion_alpha_xyz.setEnabled(enabled)
            self.expansion_alpha_xyz_label.setEnabled(enabled)

    def _apply_expansion_settings(self):
        """Apply expansion settings to sample."""
        sample = self.state.sample
        if sample is None:
            return

        if self.expansion_enabled.isChecked() and self.temp_enabled.isChecked():
            sample._thermal_expansion_enabled = True
            sample._thermal_expansion_T_ref = self.expansion_T_ref.value()

            mode = self.expansion_mode.currentData()
            if mode == "isotropic":
                sample._thermal_expansion_alpha = self.expansion_alpha.value()
                sample._thermal_expansion_alpha_xyz = None
            else:
                sample._thermal_expansion_alpha = None
                sample._thermal_expansion_alpha_xyz = np.array(
                    self.expansion_alpha_xyz.get_value(), dtype=np.float64
                )
        else:
            sample._thermal_expansion_enabled = False

    def _on_generate(self):
        """Handle generate sample button."""
        # Check directory is selected
        directory = self.dir_path_edit.text()
        if not directory:
            QMessageBox.warning(self, "No Directory", "Please select a working directory first.")
            return

        crystal = self.state.crystal
        if crystal is None:
            QMessageBox.warning(self, "No Crystal", "Please load a crystal first.")
            return

        try:
            from Sample import sample

            dimensions = (self.dim_x.value(), self.dim_y.value(), self.dim_z.value())
            sample_type = self.sample_type.currentData()

            # Show progress dialog
            progress = QProgressDialog("Generating sample...", "Cancel", 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()

            try:
                # Create sample with directory (correct API)
                new_sample = sample(directory=directory)

                # Setup sample dimensions and type
                new_sample.create_sample(
                    dimensions=dimensions,
                    sample_type=sample_type
                )

                # Configure temperature effects if enabled
                if self.temp_enabled.isChecked():
                    new_sample.enable_temp = True
                    dist = self.temp_distribution.currentData()
                    seed = self.temp_seed.value()

                    if dist == "gaussian":
                        # Gaussian: use sigma directly (in Angstroms)
                        sigma = self.temp_sigma.value()
                        new_sample.temp_params = ['gaussian', sigma, 1, seed]

                    elif dist == "einstein":
                        # Einstein model: use temperature and θE directly
                        temp_K = self.temperature.value()
                        theta_E = self.char_temp.value()
                        new_sample.temp_params = ['einstein', temp_K, 1, seed]
                        new_sample._temp_theta_E_K = theta_E
                        new_sample._temp_species_mass_amu = self._get_species_masses()

                    elif dist == "debye":
                        # True Debye model with Debye integral
                        temp_K = self.temperature.value()
                        theta_D = self.char_temp.value()
                        new_sample.temp_params = ['debye', temp_K, 1, seed]
                        new_sample._temp_theta_D_K = theta_D
                        # Set mass from crystal species (uses first species mass as default)
                        species_masses = self._get_species_masses()
                        if species_masses:
                            new_sample._temp_mass_amu = list(species_masses.values())[0]

                # Generate atoms - pass crystal as 'material' parameter
                # Get GPU count (0 means auto-detect all available)
                gpu_count = self.n_gpus.value()
                n_gpus_param = None if gpu_count == 0 else gpu_count

                if sample_type == "single":
                    new_sample.generate_sample_single(material=crystal, n_gpus=n_gpus_param)
                else:
                    # Polycrystalline generation (if supported)
                    if hasattr(new_sample, 'generate_sample_poly'):
                        # Get parameters from GUI
                        n_grains = self.n_grains.value()
                        voronoi_method = self.voronoi_method.currentData()
                        orientation_mode = self.orientation_mode.currentData()

                        # Random seed (0 means None for random)
                        seed_val = self.poly_seed.value()
                        randomness_seed = seed_val if seed_val > 0 else None

                        # Texture parameters (only used if textured mode)
                        texture_axis = (
                            self.texture_axis_x.value(),
                            self.texture_axis_y.value(),
                            self.texture_axis_z.value()
                        )
                        texture_spread_deg = self.texture_spread.value()

                        new_sample.generate_sample_poly(
                            material=crystal,
                            n_grains=n_grains,
                            voronoi_method=voronoi_method,
                            randomness_seed=randomness_seed,
                            orientation_mode=orientation_mode,
                            texture_axis=texture_axis,
                            texture_spread_deg=texture_spread_deg,
                            n_gpus=n_gpus_param
                        )
                    else:
                        QMessageBox.warning(self, "Not Supported", "Polycrystalline generation not available.")
                        return

                # Save metadata
                new_sample.write_sample_metadata()

                self.state.sample = new_sample
                self.sample_generated.emit(new_sample)
                self._refresh_display()
                self.dir_status.setText("Sample generated successfully")
                self.dir_status.setStyleSheet("color: #00cc00;")

            finally:
                progress.close()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate sample:\n{str(e)}")

    def _on_import(self):
        """Handle import positions button - opens MD import dialog."""
        # Check directory is selected
        directory = self.dir_path_edit.text()
        if not directory:
            QMessageBox.warning(
                self, "No Directory",
                "Please select a working directory first.\n"
                "Imported atomic data will be written to this directory."
            )
            return

        # Open MD import dialog
        dialog = MDImportDialog(
            self,
            initial_directory=self.state.working_directory or ""
        )

        if dialog.exec() != QDialog.Accepted:
            return

        params = dialog.get_parameters()
        if params is None:
            return

        try:
            from Sample import sample

            # Show progress dialog
            progress = QProgressDialog("Importing atomic structure...", "Cancel", 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()

            try:
                # Create or use existing sample
                new_sample = sample(directory=directory)

                # Call import_atomic_data with parameters from dialog
                new_sample.import_atomic_data(
                    import_file=params["import_file"],
                    element_list=params["element_list"],
                    header_lines=params["header_lines"],
                    ID_column=params["ID_column"],
                    position_columns=params["position_columns"],
                    scale=params["scale"],
                )

                # Save metadata after import
                new_sample.write_sample_metadata()

                # Update state and UI
                self.state.sample = new_sample
                self.sample_imported.emit(new_sample)
                self._refresh_display()

                # Update dimension inputs from imported sample
                if new_sample.dimensions is not None:
                    dims = new_sample.dimensions
                    self.dim_x.setValue(float(dims[0]))
                    self.dim_y.setValue(float(dims[1]))
                    self.dim_z.setValue(float(dims[2]))

                self.dir_status.setText("MD structure imported successfully")
                self.dir_status.setStyleSheet("color: #00cc00;")

                # Show summary
                import_file_name = Path(params["import_file"]).name
                QMessageBox.information(
                    self, "Import Complete",
                    f"Successfully imported atomic structure from:\n{import_file_name}\n\n"
                    f"Chunks created: {new_sample._chunk_total}\n"
                    f"Dimensions: {dims[0]:.1f} × {dims[1]:.1f} × {dims[2]:.1f} Å"
                )

            finally:
                progress.close()

        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import atomic structure:\n{str(e)}")

    def _on_export(self):
        """Handle export positions button."""
        sample = self.state.sample
        if sample is None:
            QMessageBox.warning(self, "No Sample", "Please generate a sample first.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Atomic Positions",
            "",
            "NumPy (*.npy);;HDF5 (*.h5);;XYZ (*.xyz)"
        )
        if filename:
            try:
                # TODO: Implement export based on file type
                QMessageBox.information(self, "Export", f"Export to {Path(filename).name} not yet implemented")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export:\n{str(e)}")

    def _on_rotate(self):
        """Handle rotate sample button."""
        sample = self.state.sample
        if sample is None:
            QMessageBox.warning(self, "No Sample", "Please generate a sample first.")
            return

        try:
            axis = self.rot_axis.get_value()
            angle = self.rot_angle.value()

            # Use correct method name: rotate_sample_relative(axis, dangle, degrees=True)
            if hasattr(sample, 'rotate_sample_relative'):
                sample.rotate_sample_relative(axis, angle, degrees=True)
                sample.write_sample_metadata()  # Save updated state
                self.state.notify_object_modified("sample")
                QMessageBox.information(self, "Success", f"Sample rotated {angle}° about axis.")
            else:
                QMessageBox.warning(self, "Not Supported", "Sample object doesn't support rotation.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to rotate:\n{str(e)}")

    def _on_translate(self):
        """Handle translate sample button."""
        sample = self.state.sample
        if sample is None:
            QMessageBox.warning(self, "No Sample", "Please generate a sample first.")
            return

        try:
            offset = self.translation.get_value()

            # Use correct method name: translate_sample_relative(offset_vector)
            if hasattr(sample, 'translate_sample_relative'):
                sample.translate_sample_relative(offset)
                sample.write_sample_metadata()  # Save updated state
                self.state.notify_object_modified("sample")
                QMessageBox.information(self, "Success", "Sample translated.")
            else:
                QMessageBox.warning(self, "Not Supported", "Sample object doesn't support translation.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to translate:\n{str(e)}")

    def _on_zero(self):
        """Handle zero to center button."""
        sample = self.state.sample
        if sample is None:
            QMessageBox.warning(self, "No Sample", "Please generate a sample first.")
            return

        try:
            # Use correct method name: zero_sample() - centers position and removes rotation
            if hasattr(sample, 'zero_sample'):
                sample.zero_sample()
                sample.write_sample_metadata()  # Save updated state
                self.state.notify_object_modified("sample")
                QMessageBox.information(self, "Success", "Sample centered and rotation reset.")
            elif hasattr(sample, 'zero_sample_position'):
                # Fallback to just centering position
                sample.zero_sample_position()
                sample.write_sample_metadata()
                self.state.notify_object_modified("sample")
                QMessageBox.information(self, "Success", "Sample centered to origin.")
            else:
                QMessageBox.warning(self, "Not Supported", "Sample object doesn't support zeroing.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to zero:\n{str(e)}")

    def get_config(self) -> dict:
        """Get the current configuration."""
        return {
            "dimensions": [self.dim_x.value(), self.dim_y.value(), self.dim_z.value()],
            "type": self.sample_type.currentData(),
            "temperature_enabled": self.temp_enabled.isChecked(),
            "distribution": self.temp_distribution.currentData(),
            "sigma": self.temp_sigma.value(),
            "temperature": self.temperature.value(),
            "char_temperature": self.char_temp.value(),
            "seed": self.temp_seed.value(),
            # Thermal expansion settings
            "expansion_enabled": self.expansion_enabled.isChecked(),
            "expansion_mode": self.expansion_mode.currentData(),
            "expansion_alpha": self.expansion_alpha.value(),
            "expansion_alpha_xyz": list(self.expansion_alpha_xyz.get_value()),
            "expansion_T_ref": self.expansion_T_ref.value(),
        }

    def set_config(self, config: dict):
        """Set configuration from dict."""
        if "dimensions" in config:
            dims = config["dimensions"]
            self.dim_x.setValue(dims[0])
            self.dim_y.setValue(dims[1])
            self.dim_z.setValue(dims[2])

        if "type" in config:
            idx = self.sample_type.findData(config["type"])
            if idx >= 0:
                self.sample_type.setCurrentIndex(idx)

        if "temperature_enabled" in config:
            self.temp_enabled.setChecked(config["temperature_enabled"])

        if "distribution" in config:
            idx = self.temp_distribution.findData(config["distribution"])
            if idx >= 0:
                self.temp_distribution.setCurrentIndex(idx)

        if "sigma" in config:
            self.temp_sigma.setValue(config["sigma"])

        if "temperature" in config:
            self.temperature.setValue(config["temperature"])

        if "char_temperature" in config:
            self.char_temp.setValue(config["char_temperature"])

        if "seed" in config:
            self.temp_seed.setValue(config["seed"])

        # Thermal expansion settings
        if "expansion_enabled" in config:
            self.expansion_enabled.setChecked(config["expansion_enabled"])

        if "expansion_mode" in config:
            idx = self.expansion_mode.findData(config["expansion_mode"])
            if idx >= 0:
                self.expansion_mode.setCurrentIndex(idx)

        if "expansion_alpha" in config:
            self.expansion_alpha.setValue(config["expansion_alpha"])

        if "expansion_alpha_xyz" in config:
            self.expansion_alpha_xyz.set_value(np.array(config["expansion_alpha_xyz"]))

        if "expansion_T_ref" in config:
            self.expansion_T_ref.setValue(config["expansion_T_ref"])

        # Legacy support for old config format
        if "debye_temperature" in config and "char_temperature" not in config:
            self.char_temp.setValue(config["debye_temperature"])
