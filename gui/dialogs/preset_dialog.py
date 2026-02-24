# -----------------------------------------------------------------------------
# Preset Dialog
# -----------------------------------------------------------------------------
"""
Dialog for browsing, loading, and saving simulation presets.

Features:
- Browse built-in and user presets by category
- Preview preset parameters before loading
- Save current configuration as new preset
- Import/export presets as JSON files
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QListWidget, QListWidgetItem, QPushButton,
    QLineEdit, QTextEdit, QComboBox, QGroupBox,
    QFormLayout, QFileDialog, QMessageBox, QSplitter,
    QTreeWidget, QTreeWidgetItem, QTabWidget,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gui.state import SimulationState


class PresetDialog(QDialog):
    """Dialog for managing simulation presets."""

    preset_loaded = Signal(str)  # preset name
    preset_saved = Signal(str)   # preset name

    def __init__(self, state: SimulationState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("Preset Manager")
        self.setMinimumSize(700, 500)

        # Preset directories
        self.builtin_dir = Path(__file__).parent.parent / "presets"
        self.user_dir = Path.home() / ".xray_simulator" / "presets"
        self.user_dir.mkdir(parents=True, exist_ok=True)

        self._setup_ui()
        self._load_preset_list()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Tabs for Load vs Save
        tabs = QTabWidget()

        # Load tab
        load_widget = QWidget()
        self._setup_load_tab(load_widget)
        tabs.addTab(load_widget, "Load Preset")

        # Save tab
        save_widget = QWidget()
        self._setup_save_tab(save_widget)
        tabs.addTab(save_widget, "Save Preset")

        layout.addWidget(tabs)

    def _setup_load_tab(self, widget):
        layout = QHBoxLayout(widget)

        # Left: Preset tree
        left_panel = QVBoxLayout()

        left_panel.addWidget(QLabel("Available Presets:"))
        self.preset_tree = QTreeWidget()
        self.preset_tree.setHeaderHidden(True)
        self.preset_tree.itemClicked.connect(self._on_preset_selected)
        self.preset_tree.itemDoubleClicked.connect(self._on_preset_double_clicked)
        left_panel.addWidget(self.preset_tree)

        # Import button
        import_btn = QPushButton("Import from File...")
        import_btn.clicked.connect(self._import_preset)
        left_panel.addWidget(import_btn)

        layout.addLayout(left_panel, 1)

        # Right: Preview
        right_panel = QVBoxLayout()

        right_panel.addWidget(QLabel("Preview:"))

        # Preset info
        info_group = QGroupBox("Information")
        info_layout = QFormLayout(info_group)
        self.preview_name = QLabel("-")
        self.preview_category = QLabel("-")
        self.preview_date = QLabel("-")
        self.preview_description = QLabel("-")
        self.preview_description.setWordWrap(True)
        info_layout.addRow("Name:", self.preview_name)
        info_layout.addRow("Category:", self.preview_category)
        info_layout.addRow("Date:", self.preview_date)
        info_layout.addRow("Description:", self.preview_description)
        right_panel.addWidget(info_group)

        # Parameters preview
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)
        self.params_text = QTextEdit()
        self.params_text.setReadOnly(True)
        self.params_text.setStyleSheet("font-family: monospace;")
        params_layout.addWidget(self.params_text)
        right_panel.addWidget(params_group, 1)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.load_btn = QPushButton("Load Preset")
        self.load_btn.setEnabled(False)
        self.load_btn.clicked.connect(self._load_selected_preset)
        btn_layout.addWidget(self.load_btn)

        self.export_btn = QPushButton("Export...")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_preset)
        btn_layout.addWidget(self.export_btn)

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self._delete_preset)
        btn_layout.addWidget(self.delete_btn)

        right_panel.addLayout(btn_layout)

        layout.addLayout(right_panel, 1)

    def _setup_save_tab(self, widget):
        layout = QVBoxLayout(widget)

        # Form
        form_group = QGroupBox("Save Current Configuration")
        form_layout = QFormLayout(form_group)

        self.save_name = QLineEdit()
        self.save_name.setPlaceholderText("my_preset")
        form_layout.addRow("Name:", self.save_name)

        self.save_category = QComboBox()
        self.save_category.addItems(["User", "DFXM", "Laue", "Powder", "Coherent", "Custom"])
        self.save_category.setEditable(True)
        form_layout.addRow("Category:", self.save_category)

        self.save_description = QTextEdit()
        self.save_description.setMaximumHeight(80)
        self.save_description.setPlaceholderText("Optional description of this preset...")
        form_layout.addRow("Description:", self.save_description)

        layout.addWidget(form_group)

        # Component selection
        components_group = QGroupBox("Include Components")
        components_layout = QVBoxLayout(components_group)

        self.include_crystal = self._create_checkbox("Crystal", True)
        self.include_sample = self._create_checkbox("Sample", True)
        self.include_beam = self._create_checkbox("Beam", True)
        self.include_detector = self._create_checkbox("Detector", True)
        self.include_stage = self._create_checkbox("Stage", True)
        self.include_optics = self._create_checkbox("Optics", True)
        self.include_defects = self._create_checkbox("Defects", False)
        self.include_deformation = self._create_checkbox("Deformation", False)

        for cb in [self.include_crystal, self.include_sample, self.include_beam,
                   self.include_detector, self.include_stage, self.include_optics,
                   self.include_defects, self.include_deformation]:
            components_layout.addWidget(cb)

        layout.addWidget(components_group)

        # Preview
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.save_preview = QTextEdit()
        self.save_preview.setReadOnly(True)
        self.save_preview.setStyleSheet("font-family: monospace;")
        preview_layout.addWidget(self.save_preview)
        layout.addWidget(preview_group, 1)

        # Update preview when options change
        self.save_name.textChanged.connect(self._update_save_preview)
        for cb in [self.include_crystal, self.include_sample, self.include_beam,
                   self.include_detector, self.include_stage, self.include_optics,
                   self.include_defects, self.include_deformation]:
            cb.stateChanged.connect(self._update_save_preview)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        save_btn = QPushButton("Save Preset")
        save_btn.clicked.connect(self._save_preset)
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)

        # Initial preview
        self._update_save_preview()

    def _create_checkbox(self, text, checked=True):
        from PySide6.QtWidgets import QCheckBox
        cb = QCheckBox(text)
        cb.setChecked(checked)
        return cb

    def _load_preset_list(self):
        """Load available presets into tree."""
        self.preset_tree.clear()

        # Built-in presets
        if self.builtin_dir.exists():
            builtin_item = QTreeWidgetItem(["Built-in"])
            builtin_item.setFlags(builtin_item.flags() & ~Qt.ItemIsSelectable)
            self.preset_tree.addTopLevelItem(builtin_item)

            for preset_file in sorted(self.builtin_dir.glob("*.json")):
                child = QTreeWidgetItem([preset_file.stem])
                child.setData(0, Qt.UserRole, str(preset_file))
                child.setData(0, Qt.UserRole + 1, "builtin")
                builtin_item.addChild(child)

            builtin_item.setExpanded(True)

        # User presets by category
        user_presets = {}
        for preset_file in self.user_dir.glob("*.json"):
            try:
                with open(preset_file, 'r') as f:
                    data = json.load(f)
                category = data.get("category", "User")
                if category not in user_presets:
                    user_presets[category] = []
                user_presets[category].append(preset_file)
            except Exception:
                continue

        for category in sorted(user_presets.keys()):
            cat_item = QTreeWidgetItem([category])
            cat_item.setFlags(cat_item.flags() & ~Qt.ItemIsSelectable)
            self.preset_tree.addTopLevelItem(cat_item)

            for preset_file in sorted(user_presets[category], key=lambda p: p.stem):
                child = QTreeWidgetItem([preset_file.stem])
                child.setData(0, Qt.UserRole, str(preset_file))
                child.setData(0, Qt.UserRole + 1, "user")
                cat_item.addChild(child)

            cat_item.setExpanded(True)

    def _on_preset_selected(self, item: QTreeWidgetItem, column: int):
        """Handle preset selection."""
        preset_path = item.data(0, Qt.UserRole)
        if not preset_path:
            self.load_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            return

        try:
            with open(preset_path, 'r') as f:
                data = json.load(f)

            self.preview_name.setText(data.get("name", item.text(0)))
            self.preview_category.setText(data.get("category", "Unknown"))
            self.preview_date.setText(data.get("date", "Unknown"))
            self.preview_description.setText(data.get("description", "No description"))

            # Format parameters
            params = data.get("parameters", {})
            params_str = json.dumps(params, indent=2)
            self.params_text.setText(params_str)

            self.load_btn.setEnabled(True)
            self.export_btn.setEnabled(True)

            # Only allow deleting user presets
            preset_type = item.data(0, Qt.UserRole + 1)
            self.delete_btn.setEnabled(preset_type == "user")

        except Exception as e:
            self.params_text.setText(f"Error loading preset: {e}")
            self.load_btn.setEnabled(False)

    def _on_preset_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle preset double-click to load."""
        if item.data(0, Qt.UserRole):
            self._load_selected_preset()

    def _load_selected_preset(self):
        """Load the selected preset."""
        item = self.preset_tree.currentItem()
        if not item:
            return

        preset_path = item.data(0, Qt.UserRole)
        if not preset_path:
            return

        try:
            with open(preset_path, 'r') as f:
                data = json.load(f)

            params = data.get("parameters", {})
            self._apply_preset(params)

            QMessageBox.information(self, "Success", f"Preset '{data.get('name', 'Unknown')}' loaded successfully.")
            self.preset_loaded.emit(data.get("name", ""))
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load preset: {e}")

    def _apply_preset(self, params: Dict[str, Any]):
        """Apply preset parameters to simulation state."""
        # Crystal
        if "crystal" in params and self.state.crystal:
            crystal_params = params["crystal"]
            if "cif_file" in crystal_params:
                self.state.crystal.load_cif(crystal_params["cif_file"])
            if "orientation" in crystal_params:
                # Apply orientation parameters
                pass

        # Sample
        if "sample" in params and self.state.sample:
            sample_params = params["sample"]
            if "dimensions" in sample_params:
                dims = sample_params["dimensions"]
                self.state.sample.Lx = dims.get("Lx", self.state.sample.Lx)
                self.state.sample.Ly = dims.get("Ly", self.state.sample.Ly)
                self.state.sample.Lz = dims.get("Lz", self.state.sample.Lz)

        # Beam
        if "beam" in params and self.state.beam:
            beam_params = params["beam"]
            if "energy" in beam_params:
                self.state.beam.energy = beam_params["energy"]
            if "shape" in beam_params:
                self.state.beam.shape = beam_params["shape"]

        # Detector
        if "detector" in params and self.state.detector:
            det_params = params["detector"]
            if "Ny" in det_params:
                self.state.detector.Ny = det_params["Ny"]
            if "Nz" in det_params:
                self.state.detector.Nz = det_params["Nz"]

        # Stage
        if "stage" in params and self.state.stage:
            stage_params = params["stage"]
            if "motors" in stage_params:
                for name, value in stage_params["motors"].items():
                    if hasattr(self.state.stage, 'set_motor'):
                        self.state.stage.set_motor(name, value)

        # Notify observers
        self.state.notify_observers("preset_loaded")

    def _import_preset(self):
        """Import preset from external file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import Preset",
            "",
            "JSON Files (*.json)"
        )
        if not filename:
            return

        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            # Validate structure
            if "parameters" not in data:
                raise ValueError("Invalid preset file: missing 'parameters' key")

            # Save to user presets
            name = data.get("name", Path(filename).stem)
            dest_path = self.user_dir / f"{name}.json"

            if dest_path.exists():
                reply = QMessageBox.question(
                    self, "Overwrite?",
                    f"Preset '{name}' already exists. Overwrite?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return

            with open(dest_path, 'w') as f:
                json.dump(data, f, indent=2)

            self._load_preset_list()
            QMessageBox.information(self, "Success", f"Preset '{name}' imported successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to import preset: {e}")

    def _export_preset(self):
        """Export selected preset to file."""
        item = self.preset_tree.currentItem()
        if not item:
            return

        preset_path = item.data(0, Qt.UserRole)
        if not preset_path:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Preset",
            f"{item.text(0)}.json",
            "JSON Files (*.json)"
        )
        if not filename:
            return

        try:
            import shutil
            shutil.copy(preset_path, filename)
            QMessageBox.information(self, "Success", f"Preset exported to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export preset: {e}")

    def _delete_preset(self):
        """Delete selected user preset."""
        item = self.preset_tree.currentItem()
        if not item:
            return

        preset_path = item.data(0, Qt.UserRole)
        preset_type = item.data(0, Qt.UserRole + 1)

        if preset_type != "user":
            QMessageBox.warning(self, "Cannot Delete", "Built-in presets cannot be deleted.")
            return

        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete preset '{item.text(0)}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        try:
            Path(preset_path).unlink()
            self._load_preset_list()
            QMessageBox.information(self, "Success", "Preset deleted.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to delete preset: {e}")

    def _update_save_preview(self):
        """Update the save preview text."""
        preset_data = self._build_preset_data()
        self.save_preview.setText(json.dumps(preset_data, indent=2))

    def _build_preset_data(self) -> Dict[str, Any]:
        """Build preset data from current state."""
        params = {}

        if self.include_crystal.isChecked() and self.state.crystal:
            crystal = self.state.crystal
            params["crystal"] = {
                "lattice_parameters": {
                    "a": getattr(crystal, 'a', None),
                    "b": getattr(crystal, 'b', None),
                    "c": getattr(crystal, 'c', None),
                    "alpha": getattr(crystal, 'alpha', None),
                    "beta": getattr(crystal, 'beta', None),
                    "gamma": getattr(crystal, 'gamma', None),
                }
            }

        if self.include_sample.isChecked() and self.state.sample:
            sample = self.state.sample
            params["sample"] = {
                "dimensions": {
                    "Lx": getattr(sample, 'Lx', None),
                    "Ly": getattr(sample, 'Ly', None),
                    "Lz": getattr(sample, 'Lz', None),
                }
            }

        if self.include_beam.isChecked() and self.state.beam:
            beam = self.state.beam
            params["beam"] = {
                "energy": getattr(beam, 'energy', None),
                "shape": getattr(beam, 'shape', None),
                "Ny": getattr(beam, 'Ny', None),
                "Nz": getattr(beam, 'Nz', None),
            }

        if self.include_detector.isChecked() and self.state.detector:
            detector = self.state.detector
            params["detector"] = {
                "Ny": getattr(detector, 'Ny', None),
                "Nz": getattr(detector, 'Nz', None),
                "pixel_size": getattr(detector, 'pixel_size', None),
                "distance": getattr(detector, 'distance', None),
            }

        if self.include_stage.isChecked() and self.state.stage:
            stage = self.state.stage
            if hasattr(stage, 'motors'):
                params["stage"] = {
                    "motors": {name: motor.value for name, motor in stage.motors.items()}
                }

        if self.include_optics.isChecked() and self.state.optics:
            params["optics"] = {
                "components": []  # Would serialize optics stack
            }

        return {
            "name": self.save_name.text() or "unnamed",
            "category": self.save_category.currentText(),
            "description": self.save_description.toPlainText(),
            "date": datetime.now().isoformat(),
            "parameters": params
        }

    def _save_preset(self):
        """Save current configuration as preset."""
        name = self.save_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a preset name.")
            return

        # Sanitize name for filename
        safe_name = "".join(c for c in name if c.isalnum() or c in "._- ")
        if not safe_name:
            QMessageBox.warning(self, "Invalid Name", "Preset name contains no valid characters.")
            return

        dest_path = self.user_dir / f"{safe_name}.json"

        if dest_path.exists():
            reply = QMessageBox.question(
                self, "Overwrite?",
                f"Preset '{name}' already exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        try:
            preset_data = self._build_preset_data()
            with open(dest_path, 'w') as f:
                json.dump(preset_data, f, indent=2)

            self._load_preset_list()
            QMessageBox.information(self, "Success", f"Preset '{name}' saved successfully.")
            self.preset_saved.emit(name)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save preset: {e}")
