# -----------------------------------------------------------------------------
# Defects Inspector
# -----------------------------------------------------------------------------
"""
Inspector panel for defect configuration.

Provides tabbed interface for:
- Stacking faults
- Cracks
- Point defects
- Dislocations (OpenDiS import)
"""

import sys
from pathlib import Path
import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QPushButton, QComboBox,
    QTabWidget, QListWidget, QFileDialog, QMessageBox, QCheckBox,
    QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gui.state import SimulationState
from gui.panels.inspector import InspectorPanel, Vector3Widget


class DefectsInspector(InspectorPanel):
    """Inspector for defect configuration with tabbed interface."""

    defects_created = Signal(object)
    defect_added = Signal(str)

    # Common elements for dropdown selection (ordered by atomic number)
    ELEMENTS = [
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu",
    ]

    def __init__(self, state: SimulationState, parent=None):
        super().__init__(state, parent)
        self.set_title("Defects")
        # Track locally configured point defects for display
        self._configured_point_defects = []
        self._setup_defects_ui()
        self._register_observers()

    def _setup_defects_ui(self):
        """Setup defects-specific UI elements."""
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
            self.directory_edit.setPlaceholderText("Select directory for defects files...")
        self.directory_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse_directory)
        dir_hlayout.addWidget(self.directory_edit)
        dir_hlayout.addWidget(browse_btn)
        dir_layout.addRow(dir_widget)

        # Load existing button
        self.load_existing_btn = QPushButton("Load Existing Defects")
        self.load_existing_btn.clicked.connect(self._load_existing_defects)
        self.load_existing_btn.setEnabled(False)
        dir_layout.addRow(self.load_existing_btn)

        # Tab widget for different defect types
        self.tabs = QTabWidget()

        # Stacking Faults Tab
        sf_tab = self._create_stacking_fault_tab()
        self.tabs.addTab(sf_tab, "Stacking Faults")

        # Cracks Tab
        crack_tab = self._create_crack_tab()
        self.tabs.addTab(crack_tab, "Cracks")

        # Point Defects Tab
        point_tab = self._create_point_defect_tab()
        self.tabs.addTab(point_tab, "Point Defects")

        # Dislocations Tab
        disl_tab = self._create_dislocation_tab()
        self.tabs.addTab(disl_tab, "Dislocations")

        self.content_layout.insertWidget(self.content_layout.count() - 1, self.tabs)

        # Create Defects Button
        create_btn = QPushButton("Create Defects Object")
        create_btn.clicked.connect(self._on_create_defects)
        create_btn.setStyleSheet("QPushButton { background-color: #2a6a2a; }")
        self.content_layout.insertWidget(self.content_layout.count() - 1, create_btn)

        # Apply Defects to Sample Button
        self.apply_btn = QPushButton("Apply Defects to Sample")
        self.apply_btn.clicked.connect(self._on_apply_defects_to_sample)
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #6a4a2a;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #8a6a4a;
            }
            QPushButton:disabled {
                background-color: #404040;
            }
        """)
        self.apply_btn.setToolTip("Apply all configured defects to the sample")
        self.content_layout.insertWidget(self.content_layout.count() - 1, self.apply_btn)

    def _create_stacking_fault_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Parameters
        form = QFormLayout()

        self.sf_count = QSpinBox()
        self.sf_count.setRange(0, 1000)
        self.sf_count.setValue(1)
        form.addRow("Number:", self.sf_count)

        self.sf_offset = Vector3Widget([0, 0, 0], decimals=4, suffix="Å")
        form.addRow("Offset:", self.sf_offset)

        # Normal (h,k,l) - integer Miller indices
        normal_row = QWidget()
        normal_layout = QHBoxLayout(normal_row)
        normal_layout.setContentsMargins(0, 0, 0, 0)
        normal_layout.setSpacing(4)

        normal_layout.addWidget(QLabel("h:"))
        self.sf_normal_h = QSpinBox()
        self.sf_normal_h.setRange(-20, 20)
        self.sf_normal_h.setValue(0)
        normal_layout.addWidget(self.sf_normal_h)

        normal_layout.addWidget(QLabel("k:"))
        self.sf_normal_k = QSpinBox()
        self.sf_normal_k.setRange(-20, 20)
        self.sf_normal_k.setValue(0)
        normal_layout.addWidget(self.sf_normal_k)

        normal_layout.addWidget(QLabel("l:"))
        self.sf_normal_l = QSpinBox()
        self.sf_normal_l.setRange(-20, 20)
        self.sf_normal_l.setValue(1)
        normal_layout.addWidget(self.sf_normal_l)

        normal_layout.addStretch()
        form.addRow("Normal (hkl):", normal_row)

        self.sf_spacing = QDoubleSpinBox()
        self.sf_spacing.setDecimals(2)
        self.sf_spacing.setRange(1, 10000)
        self.sf_spacing.setValue(100)
        self.sf_spacing.setSuffix(" Å")
        form.addRow("Spacing:", self.sf_spacing)

        # Burgers vector (h,k,l) - integer Miller indices with automatic prefactor
        burgers_row = QWidget()
        burgers_layout = QHBoxLayout(burgers_row)
        burgers_layout.setContentsMargins(0, 0, 0, 0)
        burgers_layout.setSpacing(4)

        burgers_layout.addWidget(QLabel("h:"))
        self.sf_burgers_h = QSpinBox()
        self.sf_burgers_h.setRange(-20, 20)
        self.sf_burgers_h.setValue(1)
        self.sf_burgers_h.valueChanged.connect(self._update_burgers_prefactor)
        burgers_layout.addWidget(self.sf_burgers_h)

        burgers_layout.addWidget(QLabel("k:"))
        self.sf_burgers_k = QSpinBox()
        self.sf_burgers_k.setRange(-20, 20)
        self.sf_burgers_k.setValue(1)
        self.sf_burgers_k.valueChanged.connect(self._update_burgers_prefactor)
        burgers_layout.addWidget(self.sf_burgers_k)

        burgers_layout.addWidget(QLabel("l:"))
        self.sf_burgers_l = QSpinBox()
        self.sf_burgers_l.setRange(-20, 20)
        self.sf_burgers_l.setValue(0)
        self.sf_burgers_l.valueChanged.connect(self._update_burgers_prefactor)
        burgers_layout.addWidget(self.sf_burgers_l)

        burgers_layout.addStretch()
        form.addRow("Burgers (hkl):", burgers_row)

        # Prefactor display
        self.sf_burgers_prefactor = QLabel("Prefactor: 1/2 · a[1,1,0]")
        self.sf_burgers_prefactor.setStyleSheet("color: #808080; font-style: italic;")
        form.addRow("", self.sf_burgers_prefactor)

        # Fault orientation (list of +1/-1 for each fault)
        self.sf_orientation = QComboBox()
        self.sf_orientation.addItem("Alternating (+1/-1)", "alternating")
        self.sf_orientation.addItem("All Same (+1)", "same")
        form.addRow("Orientation:", self.sf_orientation)

        # Fault gap
        self.sf_gap = QDoubleSpinBox()
        self.sf_gap.setDecimals(2)
        self.sf_gap.setRange(0, 100)
        self.sf_gap.setValue(0)
        self.sf_gap.setSuffix(" Å")
        form.addRow("Gap:", self.sf_gap)

        layout.addLayout(form)

        # Add button
        add_btn = QPushButton("Add Stacking Faults")
        add_btn.clicked.connect(self._on_add_stacking_fault)
        layout.addWidget(add_btn)

        # List of existing faults
        self.sf_list = QListWidget()
        self.sf_list.setMaximumHeight(100)
        layout.addWidget(self.sf_list)

        layout.addStretch()
        return widget

    def _update_burgers_prefactor(self):
        """Update the Burgers vector prefactor display."""
        h = self.sf_burgers_h.value()
        k = self.sf_burgers_k.value()
        l = self.sf_burgers_l.value()

        h2k2l2 = h*h + k*k + l*l
        if h2k2l2 == 0:
            self.sf_burgers_prefactor.setText("Prefactor: (invalid - cannot be [0,0,0])")
            self.sf_burgers_prefactor.setStyleSheet("color: #c08080; font-style: italic;")
        else:
            self.sf_burgers_prefactor.setText(f"Prefactor: 1/{h2k2l2} · a[{h},{k},{l}]")
            self.sf_burgers_prefactor.setStyleSheet("color: #808080; font-style: italic;")

    def _create_crack_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Info label
        info_label = QLabel("Define crack region by specifying vertices.\nAtoms inside the convex hull will be removed.")
        info_label.setStyleSheet("color: #808080; font-style: italic;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Vertex input row
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("X:"))
        self.crack_x = QDoubleSpinBox()
        self.crack_x.setRange(-1e9, 1e9)
        self.crack_x.setDecimals(2)
        self.crack_x.setSuffix(" Å")
        input_layout.addWidget(self.crack_x)

        input_layout.addWidget(QLabel("Y:"))
        self.crack_y = QDoubleSpinBox()
        self.crack_y.setRange(-1e9, 1e9)
        self.crack_y.setDecimals(2)
        self.crack_y.setSuffix(" Å")
        input_layout.addWidget(self.crack_y)

        input_layout.addWidget(QLabel("Z:"))
        self.crack_z = QDoubleSpinBox()
        self.crack_z.setRange(-1e9, 1e9)
        self.crack_z.setDecimals(2)
        self.crack_z.setSuffix(" Å")
        input_layout.addWidget(self.crack_z)

        add_vertex_btn = QPushButton("Add")
        add_vertex_btn.clicked.connect(self._on_add_crack_vertex)
        input_layout.addWidget(add_vertex_btn)

        layout.addLayout(input_layout)

        # Vertices table
        self.crack_table = QTableWidget()
        self.crack_table.setColumnCount(3)
        self.crack_table.setHorizontalHeaderLabels(["X (Å)", "Y (Å)", "Z (Å)"])
        self.crack_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.crack_table.setMaximumHeight(150)
        self.crack_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.crack_table)

        # Vertex status
        self.crack_vertices = QLabel("0 vertices (need at least 4 for 3D hull)")
        self.crack_vertices.setStyleSheet("color: #808080;")
        layout.addWidget(self.crack_vertices)

        # Control buttons
        btn_layout = QHBoxLayout()

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._on_remove_crack_vertex)
        btn_layout.addWidget(remove_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._on_clear_crack)
        btn_layout.addWidget(clear_btn)

        layout.addLayout(btn_layout)

        # Apply crack button
        apply_crack_btn = QPushButton("Add Crack to Defects")
        apply_crack_btn.clicked.connect(self._on_add_crack)
        apply_crack_btn.setStyleSheet("QPushButton { background-color: #2a6a2a; }")
        layout.addWidget(apply_crack_btn)

        # Existing cracks list
        self.crack_list = QListWidget()
        self.crack_list.setMaximumHeight(80)
        layout.addWidget(self.crack_list)

        layout.addStretch()
        return widget

    def _create_point_defect_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        form = QFormLayout()

        self.pd_type = QComboBox()
        self.pd_type.addItem("Vacancy", "vacancy")
        self.pd_type.addItem("Substitution", "substitution")
        self.pd_type.addItem("Interstitial", "interstitial")
        self.pd_type.currentIndexChanged.connect(self._on_point_defect_type_changed)
        form.addRow("Type:", self.pd_type)

        # Fraction/Count input (label changes based on type)
        self.pd_amount_label = QLabel("Fraction:")
        self.pd_fraction = QDoubleSpinBox()
        self.pd_fraction.setDecimals(6)
        self.pd_fraction.setRange(0, 1)
        self.pd_fraction.setValue(0.001)
        form.addRow(self.pd_amount_label, self.pd_fraction)

        # Species (for vacancy filter, substitution "from", or interstitial species)
        self.pd_species_label = QLabel("Species Filter:")
        self.pd_species = QComboBox()
        self.pd_species.setEditable(True)
        self.pd_species.addItem("All")
        # Add all elements to dropdown
        for element in self.ELEMENTS:
            self.pd_species.addItem(element)
        form.addRow(self.pd_species_label, self.pd_species)

        # "To Species" for substitutions (hidden by default)
        self.pd_to_species_label = QLabel("Replace With:")
        self.pd_to_species = QComboBox()
        self.pd_to_species.setEditable(True)
        # Add all elements to dropdown (no "All" option for substitution target)
        for element in self.ELEMENTS:
            self.pd_to_species.addItem(element)
        # Default to carbon (index 5 in ELEMENTS, but index 0 in this combo since no "All")
        self.pd_to_species.setCurrentText("C")
        self.pd_to_species_label.setVisible(False)
        self.pd_to_species.setVisible(False)
        form.addRow(self.pd_to_species_label, self.pd_to_species)

        layout.addLayout(form)

        # Buttons row
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)

        add_btn = QPushButton("Add Point Defects")
        add_btn.clicked.connect(self._on_add_point_defects)
        btn_layout.addWidget(add_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._on_clear_point_defects)
        btn_layout.addWidget(clear_btn)

        layout.addWidget(btn_row)

        # Summary label
        self.pd_summary_label = QLabel("Configured Point Defects:")
        self.pd_summary_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
        layout.addWidget(self.pd_summary_label)

        self.pd_list = QListWidget()
        self.pd_list.setMaximumHeight(120)
        layout.addWidget(self.pd_list)

        layout.addStretch()
        return widget

    def _on_point_defect_type_changed(self, index):
        """Update UI labels and visibility based on point defect type."""
        defect_type = self.pd_type.currentData()

        if defect_type == "vacancy":
            self.pd_amount_label.setText("Fraction:")
            self.pd_fraction.setRange(0, 1)
            self.pd_fraction.setDecimals(6)
            self.pd_fraction.setValue(0.001)
            self.pd_species_label.setText("Species Filter:")
            self.pd_to_species_label.setVisible(False)
            self.pd_to_species.setVisible(False)

        elif defect_type == "substitution":
            self.pd_amount_label.setText("Fraction:")
            self.pd_fraction.setRange(0, 1)
            self.pd_fraction.setDecimals(6)
            self.pd_fraction.setValue(0.001)
            self.pd_species_label.setText("From Species:")
            self.pd_to_species_label.setVisible(True)
            self.pd_to_species.setVisible(True)

        elif defect_type == "interstitial":
            self.pd_amount_label.setText("Count:")
            self.pd_fraction.setRange(0, 100000)
            self.pd_fraction.setDecimals(0)
            self.pd_fraction.setValue(10)
            self.pd_species_label.setText("Species:")
            self.pd_to_species_label.setVisible(False)
            self.pd_to_species.setVisible(False)

    def _create_dislocation_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # OpenDiS import
        form = QFormLayout()

        file_row = QWidget()
        file_layout = QHBoxLayout(file_row)
        file_layout.setContentsMargins(0, 0, 0, 0)
        self.disl_path = QLabel("No file selected")
        self.disl_path.setStyleSheet("color: #808080;")
        file_layout.addWidget(self.disl_path, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse_dislocation)
        file_layout.addWidget(browse_btn)
        form.addRow("OpenDiS File:", file_row)

        layout.addLayout(form)

        import_btn = QPushButton("Import Dislocation Network")
        import_btn.clicked.connect(self._on_import_dislocations)
        layout.addWidget(import_btn)

        # Visualization option
        self.disl_visualize = QCheckBox("Visualize network after import")
        self.disl_visualize.setChecked(True)
        layout.addWidget(self.disl_visualize)

        # Info display
        self.disl_info = QLabel("No dislocation network loaded")
        self.disl_info.setStyleSheet("color: #808080;")
        layout.addWidget(self.disl_info)

        layout.addStretch()
        return widget

    def _register_observers(self):
        self.state.register_observer("defects_changed", self._on_defects_state_changed)
        self.state.register_observer("global_working_directory_changed", self._on_global_dir_changed)
        self.state.register_observer("crystal_changed", self._on_crystal_changed)

    def _on_crystal_changed(self, crystal):
        """Handle crystal change - update species dropdowns with crystal species."""
        self._update_species_dropdowns()

    def _update_species_dropdowns(self):
        """Update species dropdown with species from loaded crystal."""
        crystal = self.state.crystal
        if crystal is None:
            return

        # Get unique species from crystal
        crystal_species = set()
        if hasattr(crystal, '_species_conventional') and crystal._species_conventional is not None:
            for sp in crystal._species_conventional:
                crystal_species.add(str(sp))
        elif hasattr(crystal, '_species') and crystal._species is not None:
            for sp in crystal._species:
                crystal_species.add(str(sp))

        if not crystal_species:
            return

        # Remember current selections
        current_species = self.pd_species.currentText()
        current_to_species = self.pd_to_species.currentText()

        # Update pd_species dropdown - add crystal species at top after "All"
        self.pd_species.clear()
        self.pd_species.addItem("All")
        # Add crystal species first (with indicator)
        for sp in sorted(crystal_species):
            self.pd_species.addItem(f"{sp} (crystal)")
        # Add separator and all elements
        self.pd_species.insertSeparator(self.pd_species.count())
        for element in self.ELEMENTS:
            self.pd_species.addItem(element)

        # Restore selection if possible
        idx = self.pd_species.findText(current_species)
        if idx >= 0:
            self.pd_species.setCurrentIndex(idx)

        # Update pd_to_species dropdown
        self.pd_to_species.clear()
        # Add crystal species first
        for sp in sorted(crystal_species):
            self.pd_to_species.addItem(f"{sp} (crystal)")
        # Add separator and all elements
        self.pd_to_species.insertSeparator(self.pd_to_species.count())
        for element in self.ELEMENTS:
            self.pd_to_species.addItem(element)

        # Restore selection if possible
        idx = self.pd_to_species.findText(current_to_species)
        if idx >= 0:
            self.pd_to_species.setCurrentIndex(idx)

    def _on_global_dir_changed(self, directory):
        """Handle global working directory change."""
        # Update placeholder text if directory field is empty
        if not self.directory_edit.text():
            self.directory_edit.setPlaceholderText(
                f"Using global: {directory}" if directory
                else "Select directory for defects files..."
            )

    def _on_defects_state_changed(self, defects):
        self._refresh_display()

    def _refresh_display(self):
        defects_obj = self.state.defects
        self.sf_list.clear()
        self.crack_list.clear()
        self.pd_list.clear()

        if defects_obj is None:
            return

        # Update directory display if available
        if hasattr(defects_obj, 'directory') and defects_obj.directory:
            self.directory_edit.setText(str(defects_obj.directory))

        # Update stacking faults list (stored as _stacking_faults)
        if hasattr(defects_obj, '_stacking_faults') and defects_obj._stacking_faults is not None:
            sf = defects_obj._stacking_faults
            if hasattr(sf, 'fault_number') and sf.fault_number > 0:
                for i in range(sf.fault_number):
                    self.sf_list.addItem(f"SF {i+1}: spacing={sf.interfault_spacing:.1f}Å")

        # Update cracks list and table (stored as _cracks)
        if hasattr(defects_obj, '_cracks') and defects_obj._cracks is not None:
            cr = defects_obj._cracks
            if hasattr(cr, 'crack_points') and len(cr.crack_points) > 0:
                self.crack_list.addItem(f"Crack: {len(cr.crack_points)} vertices")
                # Populate the vertex table with existing crack points
                self.crack_table.setRowCount(0)
                for vertex in cr.crack_points:
                    row = self.crack_table.rowCount()
                    self.crack_table.insertRow(row)
                    self.crack_table.setItem(row, 0, QTableWidgetItem(f"{vertex[0]:.2f}"))
                    self.crack_table.setItem(row, 1, QTableWidgetItem(f"{vertex[1]:.2f}"))
                    self.crack_table.setItem(row, 2, QTableWidgetItem(f"{vertex[2]:.2f}"))
                self._update_crack_vertex_count()

        # Update point defects list (stored as _point_defects)
        if hasattr(defects_obj, '_point_defects') and defects_obj._point_defects is not None:
            pd = defects_obj._point_defects

            # Show CONFIGURED parameters (before applying)
            # Vacancies
            if hasattr(pd, 'vacancy_fraction') and pd.vacancy_fraction is not None:
                species_filter = ""
                if hasattr(pd, 'vacancy_species_filter') and pd.vacancy_species_filter:
                    species_filter = f" ({', '.join(pd.vacancy_species_filter)})"
                self.pd_list.addItem(f"Vacancy: {pd.vacancy_fraction*100:.3f}%{species_filter}")
            elif hasattr(pd, 'vacancy_count') and pd.vacancy_count is not None:
                species_filter = ""
                if hasattr(pd, 'vacancy_species_filter') and pd.vacancy_species_filter:
                    species_filter = f" ({', '.join(pd.vacancy_species_filter)})"
                self.pd_list.addItem(f"Vacancy: {pd.vacancy_count} atoms{species_filter}")

            # Substitutions
            if hasattr(pd, 'substitution_fraction') and pd.substitution_fraction is not None:
                from_sp = getattr(pd, 'substitution_from', '?')
                to_sp = getattr(pd, 'substitution_to', '?')
                self.pd_list.addItem(f"Substitution: {pd.substitution_fraction*100:.3f}% {from_sp}→{to_sp}")
            elif hasattr(pd, 'substitution_count') and pd.substitution_count is not None:
                from_sp = getattr(pd, 'substitution_from', '?')
                to_sp = getattr(pd, 'substitution_to', '?')
                self.pd_list.addItem(f"Substitution: {pd.substitution_count} atoms {from_sp}→{to_sp}")

            # Interstitials
            if hasattr(pd, 'interstitial_count') and pd.interstitial_count is not None:
                species = getattr(pd, 'interstitial_species', '?')
                self.pd_list.addItem(f"Interstitial: {pd.interstitial_count} {species} atoms")
            elif hasattr(pd, 'interstitial_positions') and pd.interstitial_positions is not None:
                species = getattr(pd, 'interstitial_species', '?')
                count = len(pd.interstitial_positions)
                self.pd_list.addItem(f"Interstitial: {count} {species} atoms (positioned)")

            # Also show APPLIED counts if defects have been applied
            applied_items = []
            if hasattr(pd, '_applied_vacancies') and pd._applied_vacancies:
                count = sum(len(v) for v in pd._applied_vacancies)
                applied_items.append(f"{count} vacancies")
            # applied lists hold one (M, 3) array per chunk; count the rows
            if hasattr(pd, '_applied_substitutions') and pd._applied_substitutions:
                count = sum(len(v) for v in pd._applied_substitutions)
                applied_items.append(f"{count} substitutions")
            if hasattr(pd, '_applied_interstitials') and pd._applied_interstitials:
                count = sum(len(v) for v in pd._applied_interstitials)
                applied_items.append(f"{count} interstitials")

            if applied_items:
                self.pd_list.addItem(f"[Applied: {', '.join(applied_items)}]")

    def _on_create_defects(self):
        directory = self.directory_edit.text()
        if not directory:
            QMessageBox.warning(self, "No Directory",
                              "Please select a working directory first.")
            return

        try:
            from Defects import defects
            new_defects = defects(directory=directory)
            # Save metadata after creation
            if hasattr(new_defects, 'write_defect_metadata'):
                new_defects.write_defect_metadata()
            self.state.defects = new_defects
            self.defects_created.emit(new_defects)
            # Clear local tracking when creating new object
            self._configured_point_defects.clear()
            self._refresh_display()
            QMessageBox.information(self, "Success", "Defects object created.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create defects:\n{str(e)}")

    def _on_add_stacking_fault(self):
        defects_obj = self.state.defects
        if defects_obj is None:
            QMessageBox.warning(self, "No Defects", "Please create a defects object first.")
            return

        try:
            # Build fault_orientation list based on selection
            fault_number = self.sf_count.value()
            if self.sf_orientation.currentData() == "alternating":
                # Alternating +1, -1, +1, -1, ...
                fault_orientation = [1 if i % 2 == 0 else -1 for i in range(fault_number)]
            else:
                # All same
                fault_orientation = [1] * fault_number

            # Get normal (h,k,l) as direction vector
            normal_h = self.sf_normal_h.value()
            normal_k = self.sf_normal_k.value()
            normal_l = self.sf_normal_l.value()
            fault_normal = np.array([normal_h, normal_k, normal_l], dtype=np.float32)

            # Validate normal is not zero
            if np.allclose(fault_normal, 0):
                QMessageBox.warning(self, "Invalid Normal",
                    "Normal vector cannot be (0,0,0). Please set valid h,k,l indices.")
                return

            # Get Burgers vector (h,k,l) with prefactor 1/(h²+k²+l²)
            burgers_h = self.sf_burgers_h.value()
            burgers_k = self.sf_burgers_k.value()
            burgers_l = self.sf_burgers_l.value()
            h2k2l2 = burgers_h**2 + burgers_k**2 + burgers_l**2

            if h2k2l2 == 0:
                QMessageBox.warning(self, "Invalid Burgers Vector",
                    "Burgers vector cannot be (0,0,0). Please set valid h,k,l indices.")
                return

            # Apply prefactor: b = (1/(h²+k²+l²)) * [h,k,l]
            prefactor = 1.0 / h2k2l2
            burgers_vector = prefactor * np.array([burgers_h, burgers_k, burgers_l], dtype=np.float32)

            # Use add_stacking_faults method with correct parameter names
            if hasattr(defects_obj, 'add_stacking_faults'):
                defects_obj.add_stacking_faults(
                    fault_number=fault_number,
                    fault_offset=np.array(self.sf_offset.get_value(), dtype=np.float32),
                    fault_normal=fault_normal,
                    interfault_spacing=self.sf_spacing.value(),
                    burgers_vector=burgers_vector,
                    fault_orientation=fault_orientation,
                    fault_gap=self.sf_gap.value()
                )
                # Save metadata after adding
                if hasattr(defects_obj, 'write_defect_metadata'):
                    defects_obj.write_defect_metadata()
                self.state.notify_object_modified("defects")
                self.defect_added.emit("stacking_fault")
                self._refresh_display()
                QMessageBox.information(self, "Success",
                    f"Added {fault_number} stacking fault(s).\n"
                    f"Normal: ({normal_h},{normal_k},{normal_l})\n"
                    f"Burgers: 1/{h2k2l2} × [{burgers_h},{burgers_k},{burgers_l}]")
            else:
                QMessageBox.warning(self, "Not Supported", "Defects object doesn't support add_stacking_faults.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add stacking faults:\n{str(e)}")

    def _on_add_crack_vertex(self):
        """Add a vertex to the crack vertices table."""
        x = self.crack_x.value()
        y = self.crack_y.value()
        z = self.crack_z.value()

        row = self.crack_table.rowCount()
        self.crack_table.insertRow(row)
        self.crack_table.setItem(row, 0, QTableWidgetItem(f"{x:.2f}"))
        self.crack_table.setItem(row, 1, QTableWidgetItem(f"{y:.2f}"))
        self.crack_table.setItem(row, 2, QTableWidgetItem(f"{z:.2f}"))

        self._update_crack_vertex_count()

    def _on_remove_crack_vertex(self):
        """Remove selected vertices from the crack table."""
        selected_rows = set(item.row() for item in self.crack_table.selectedItems())
        for row in sorted(selected_rows, reverse=True):
            self.crack_table.removeRow(row)
        self._update_crack_vertex_count()

    def _on_clear_crack(self):
        """Clear all vertices from the crack table."""
        self.crack_table.setRowCount(0)
        self._update_crack_vertex_count()

    def _update_crack_vertex_count(self):
        """Update the vertex count label."""
        count = self.crack_table.rowCount()
        if count < 4:
            self.crack_vertices.setText(f"{count} vertices (need at least 4 for 3D hull)")
            self.crack_vertices.setStyleSheet("color: #c08080;")
        else:
            self.crack_vertices.setText(f"{count} vertices defined")
            self.crack_vertices.setStyleSheet("color: #4ec94e;")

    def _get_crack_vertices_from_table(self) -> np.ndarray:
        """Get crack vertices as Nx3 numpy array from table."""
        rows = self.crack_table.rowCount()
        if rows == 0:
            return np.array([], dtype=np.float32).reshape(0, 3)

        vertices = []
        for row in range(rows):
            x = float(self.crack_table.item(row, 0).text())
            y = float(self.crack_table.item(row, 1).text())
            z = float(self.crack_table.item(row, 2).text())
            vertices.append([x, y, z])

        return np.array(vertices, dtype=np.float32)

    def _on_add_crack(self):
        """Add the defined crack to the defects object."""
        defects_obj = self.state.defects
        if defects_obj is None:
            QMessageBox.warning(self, "No Defects", "Please create a defects object first.")
            return

        vertices = self._get_crack_vertices_from_table()
        if len(vertices) < 4:
            QMessageBox.warning(self, "Not Enough Vertices",
                "A 3D crack requires at least 4 vertices to form a convex hull.")
            return

        try:
            if hasattr(defects_obj, 'add_cracks'):
                defects_obj.add_cracks(vertices)
                # Save metadata after adding
                if hasattr(defects_obj, 'write_defect_metadata'):
                    defects_obj.write_defect_metadata()
                self.state.notify_object_modified("defects")
                self.defect_added.emit("crack")
                self._refresh_display()
                QMessageBox.information(self, "Success", f"Added crack with {len(vertices)} vertices.")
            else:
                QMessageBox.warning(self, "Not Supported", "Defects object doesn't support add_cracks.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add crack:\n{str(e)}")

    def _on_apply_defects_to_sample(self):
        """Apply all configured defects to the sample."""
        defects_obj = self.state.defects
        sample_obj = self.state.sample
        crystal_obj = self.state.crystal

        if defects_obj is None:
            QMessageBox.warning(self, "No Defects", "Please create a defects object first.")
            return

        if sample_obj is None:
            QMessageBox.warning(self, "No Sample", "Please create a sample first.")
            return

        # Check if sample has been generated (has chunks)
        chunk_total = getattr(sample_obj, 'chunk_total', None)
        if chunk_total is None or chunk_total == 0:
            QMessageBox.warning(self, "Sample Not Generated",
                "The sample must be generated before applying defects.\n"
                "Please generate the sample first using the Sample Inspector.")
            return

        # Check if sample has offset defined (required for stacking faults)
        sample_offset = getattr(sample_obj, 'offset', None)
        if sample_offset is None:
            QMessageBox.warning(self, "Sample Not Initialized",
                "The sample offset is not defined.\n"
                "Please ensure the sample is properly created.")
            return

        applied_types = []

        try:
            # Apply stacking faults if configured
            if defects_obj._stacking_faults is not None:
                if hasattr(defects_obj._stacking_faults, 'apply_to_sample'):
                    # Stacking faults require crystal to transform normal/Burgers vectors
                    if crystal_obj is None:
                        QMessageBox.warning(self, "No Crystal",
                            "Stacking faults require a crystal to be loaded.\n"
                            "Please load a crystal first.")
                        return

                    # Check that crystal has required lattice properties
                    lattice_matrix = getattr(crystal_obj, 'lattice_matrix_conventional', None)
                    lattice_lengths = getattr(crystal_obj, 'lattice_lengths_conventional', None)
                    if lattice_matrix is None or lattice_lengths is None:
                        QMessageBox.warning(self, "Crystal Not Loaded Properly",
                            "The crystal lattice properties are not available.\n"
                            "Please reload the crystal from a CIF file.")
                        return

                    # Must generate global positions before applying
                    # This transforms the fault normal and Burgers vector using the crystal lattice
                    if hasattr(defects_obj._stacking_faults, 'generate_global_positions'):
                        defects_obj._stacking_faults.generate_global_positions(
                            sample_obj, crystal_obj, plotting=False, use_gpu=True)

                    defects_obj._stacking_faults.apply_to_sample(sample_obj)
                    applied_types.append("Stacking Faults")

            # Apply cracks if configured
            if defects_obj._cracks is not None:
                if hasattr(defects_obj._cracks, 'apply_to_sample'):
                    defects_obj._cracks.apply_to_sample(sample_obj)
                    applied_types.append("Cracks")

            # Apply point defects if configured
            if defects_obj._point_defects is not None:
                if hasattr(defects_obj._point_defects, 'apply_to_sample'):
                    defects_obj._point_defects.apply_to_sample(sample_obj)
                    applied_types.append("Point Defects")

            # Check for dislocations (they work differently - via nodal fields)
            has_dislocations = hasattr(defects_obj, '_opendis_nodes_xyz') and defects_obj._opendis_nodes_xyz is not None
            dislocation_note = ""
            if has_dislocations:
                dislocation_note = (
                    "\n\nNote: Dislocations use displacement fields, not direct atom modification.\n"
                    "Use 'generate_nodal_field()' to create the displacement field,\n"
                    "then apply via beam scattering calculations."
                )

            if applied_types:
                self.state.notify_object_modified("sample")
                QMessageBox.information(
                    self, "Success",
                    f"Applied defects to sample:\n• " + "\n• ".join(applied_types) + dislocation_note
                )
            else:
                if has_dislocations:
                    QMessageBox.information(
                        self, "Dislocations Configured",
                        "Dislocation network is loaded but dislocations are applied differently.\n\n"
                        "Dislocations generate displacement fields rather than modifying atoms directly.\n"
                        "Use defects.generate_nodal_field() to create the displacement field,\n"
                        "which is then used during beam scattering calculations."
                    )
                else:
                    QMessageBox.information(
                        self, "No Defects",
                        "No defects have been configured yet.\n\n"
                        "Use the tabs above to configure stacking faults, cracks,\n"
                        "point defects, or dislocations before applying."
                    )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply defects to sample:\n{str(e)}")

    def _clean_species_text(self, text: str) -> str:
        """Remove '(crystal)' suffix from species text if present."""
        if text.endswith(" (crystal)"):
            return text[:-10].strip()
        return text.strip()

    def _on_add_point_defects(self):
        defects_obj = self.state.defects
        if defects_obj is None:
            QMessageBox.warning(self, "No Defects", "Please create a defects object first.")
            return

        try:
            defect_type = self.pd_type.currentData()
            amount = self.pd_fraction.value()
            raw_species = self.pd_species.currentText()
            species = self._clean_species_text(raw_species) if raw_species != "All" else None

            # Build kwargs based on defect type
            kwargs = {}
            if defect_type == "vacancy":
                kwargs['vacancy_fraction'] = amount
                if species:
                    # vacancy_species_filter expects a list of species
                    kwargs['vacancy_species_filter'] = [species]

            elif defect_type == "substitution":
                kwargs['substitution_fraction'] = amount
                # Get "from" species
                from_species = species
                if not from_species:
                    QMessageBox.warning(self, "Species Required",
                        "Substitution requires specifying the species to replace.\n"
                        "Please select a 'From Species' other than 'All'.")
                    return
                kwargs['substitution_from'] = from_species

                # Get "to" species
                raw_to_species = self.pd_to_species.currentText()
                to_species = self._clean_species_text(raw_to_species)
                if not to_species:
                    QMessageBox.warning(self, "Species Required",
                        "Substitution requires specifying the replacement species.\n"
                        "Please enter a 'Replace With' species.")
                    return
                kwargs['substitution_to'] = to_species

            elif defect_type == "interstitial":
                # Interstitials use count directly (UI already shows "Count" label)
                count = max(1, int(amount))
                kwargs['interstitial_count'] = count
                if species:
                    kwargs['interstitial_species'] = species
                else:
                    QMessageBox.warning(self, "Species Required",
                        "Interstitial defects require specifying the species.\n"
                        "Please select or enter a species.")
                    return

            # Use add_point_defects method
            if hasattr(defects_obj, 'add_point_defects'):
                defects_obj.add_point_defects(**kwargs)
                # Save metadata after adding
                if hasattr(defects_obj, 'write_defect_metadata'):
                    defects_obj.write_defect_metadata()
                self.state.notify_object_modified("defects")
                self.defect_added.emit("point_defect")
                self._refresh_display()
                QMessageBox.information(self, "Success", f"Point defects ({defect_type}) configured.")
            else:
                QMessageBox.warning(self, "Not Supported", "Defects object doesn't support add_point_defects.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add point defects:\n{str(e)}")

    def _on_clear_point_defects(self):
        """Clear all configured point defects."""
        defects_obj = self.state.defects
        if defects_obj is None:
            QMessageBox.warning(self, "No Defects", "No defects object to clear.")
            return

        # Clear the point defects from the defects object
        if hasattr(defects_obj, '_point_defects'):
            defects_obj._point_defects = None

        # Clear local tracking
        self._configured_point_defects.clear()

        # Refresh display
        self._refresh_display()
        QMessageBox.information(self, "Cleared", "Point defects configuration cleared.")

    def _on_browse_dislocation(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select OpenDiS File", "",
            "OpenDiS Files (*.ca *.dat *.txt);;All Files (*)"
        )
        if filename:
            self.disl_path.setText(Path(filename).name)
            self._opendis_file = filename

    def _on_import_dislocations(self):
        defects_obj = self.state.defects
        if defects_obj is None:
            QMessageBox.warning(self, "No Defects", "Please create a defects object first.")
            return

        if not hasattr(self, '_opendis_file'):
            QMessageBox.warning(self, "No File", "Please select an OpenDiS file first.")
            return

        # Crystal is required for dislocation import (to compute Burgers vector magnitude)
        crystal_obj = self.state.crystal
        if crystal_obj is None:
            QMessageBox.warning(self, "No Crystal",
                "Dislocation import requires a crystal to be loaded.\n"
                "The crystal is needed to compute Burgers vector magnitudes.\n"
                "Please load a crystal first.")
            return

        try:
            if hasattr(defects_obj, 'import_dislocation_network'):
                defects_obj.import_dislocation_network(self._opendis_file, crystal_obj)
                # Save metadata after import
                if hasattr(defects_obj, 'write_defect_metadata'):
                    defects_obj.write_defect_metadata()
                self.state.notify_object_modified("defects")
                self.disl_info.setText("Dislocation network imported")
                self.disl_info.setStyleSheet("color: #4ec94e;")

                if self.disl_visualize.isChecked() and hasattr(defects_obj, 'visualize_dislocation_network'):
                    defects_obj.visualize_dislocation_network()

                QMessageBox.information(self, "Success", "Dislocation network imported.")
            else:
                QMessageBox.warning(self, "Not Supported", "Defects object doesn't support import_dislocation_network.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to import dislocations:\n{str(e)}")

    def _on_browse_directory(self):
        """Open directory browser dialog."""
        # Use current text, or global directory, or current working directory
        start_dir = self.directory_edit.text() or self.state.get_default_directory()
        directory = QFileDialog.getExistingDirectory(
            self, "Select Defects Directory",
            start_dir
        )
        if directory:
            self.directory_edit.setText(directory)
            # Check if defects metadata exists
            metadata_path = Path(directory) / "defects_metadata.json"
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

    def _load_existing_defects(self):
        """Load existing defects from selected directory."""
        directory = self.directory_edit.text()
        if not directory:
            QMessageBox.warning(self, "No Directory",
                              "Please select a working directory first.")
            return

        try:
            from Defects import defects
            existing_defects = defects(directory=directory)
            # Try to load metadata
            if hasattr(existing_defects, 'read_defect_metadata'):
                existing_defects.read_defect_metadata()
            self.state.defects = existing_defects
            self.defects_created.emit(existing_defects)
            # Clear local tracking when loading
            self._configured_point_defects.clear()
            self._refresh_display()
            QMessageBox.information(self, "Success", "Defects loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load defects:\n{str(e)}")
