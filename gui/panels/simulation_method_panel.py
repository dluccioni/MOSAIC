# -----------------------------------------------------------------------------
# Simulation Method Panel
# -----------------------------------------------------------------------------
"""
Panel for selecting X-ray interaction simulation methods.

Provides controls for:
- Kinematic scattering options
- Transmission options
- Wavefield propagation options
"""

import sys
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QPushButton, QComboBox,
    QCheckBox, QLineEdit, QTabWidget, QScrollArea, QFrame,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gui.state import SimulationState


class SimulationMethodPanel(QWidget):
    """Panel for configuring X-ray simulation methods."""

    method_changed = Signal(str)  # Emitted when simulation method changes
    settings_changed = Signal()   # Emitted when any setting changes

    def __init__(self, state: SimulationState, parent=None):
        super().__init__(parent)
        self.state = state
        self._setup_ui()

    def _setup_ui(self):
        """Setup the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Title
        title = QLabel("Simulation Method")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title)

        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)

        # --- Primary Method Selection ---
        method_group = QGroupBox("Interaction Type")
        method_layout = QFormLayout(method_group)

        self.scattering_enabled = QCheckBox("Kinematic Scattering")
        self.scattering_enabled.setChecked(True)
        self.scattering_enabled.setToolTip("Include kinematic X-ray scattering contribution")
        self.scattering_enabled.stateChanged.connect(self._on_method_changed)
        method_layout.addRow(self.scattering_enabled)

        self.transmission_enabled = QCheckBox("Transmission")
        self.transmission_enabled.setChecked(False)
        self.transmission_enabled.setToolTip("Include X-ray transmission through the sample")
        self.transmission_enabled.stateChanged.connect(self._on_method_changed)
        method_layout.addRow(self.transmission_enabled)

        self.propagation_enabled = QCheckBox("Wavefield Propagation")
        self.propagation_enabled.setChecked(False)
        self.propagation_enabled.setToolTip("Propagate wavefield through optics stack")
        self.propagation_enabled.stateChanged.connect(self._on_method_changed)
        method_layout.addRow(self.propagation_enabled)

        content_layout.addWidget(method_group)

        # --- Scattering Options ---
        self.scattering_group = QGroupBox("Scattering Options")
        sc_layout = QFormLayout(self.scattering_group)

        # Remove forward (f0)
        self.sc_remove_forward = QCheckBox("Remove Forward (f0)")
        self.sc_remove_forward.setChecked(False)
        self.sc_remove_forward.setToolTip("Remove forward scattering term to avoid double-counting with transmission")
        sc_layout.addRow(self.sc_remove_forward)

        # Use depth einsum
        self.sc_use_depth_ein = QCheckBox("Use Depth Einsum")
        self.sc_use_depth_ein.setChecked(False)
        self.sc_use_depth_ein.setToolTip("Use einsum formulation for depth-dependent calculations")
        sc_layout.addRow(self.sc_use_depth_ein)

        # Einsum cache directory
        self.sc_ein_cache_dir = QLineEdit()
        self.sc_ein_cache_dir.setPlaceholderText("Optional cache directory")
        self.sc_ein_cache_dir.setEnabled(False)
        self.sc_use_depth_ein.stateChanged.connect(
            lambda s: self.sc_ein_cache_dir.setEnabled(self.sc_use_depth_ein.isChecked()))
        sc_layout.addRow("Cache Dir:", self.sc_ein_cache_dir)

        # Recompute cache
        self.sc_recompute_cache = QCheckBox("Recompute Cache")
        self.sc_recompute_cache.setChecked(False)
        self.sc_recompute_cache.setEnabled(False)
        self.sc_use_depth_ein.stateChanged.connect(
            lambda s: self.sc_recompute_cache.setEnabled(self.sc_use_depth_ein.isChecked()))
        sc_layout.addRow(self.sc_recompute_cache)

        # Apply polarization
        self.sc_apply_polarization = QCheckBox("Apply Polarization")
        self.sc_apply_polarization.setChecked(False)
        self.sc_apply_polarization.setToolTip("Include polarization effects in scattering")
        sc_layout.addRow(self.sc_apply_polarization)

        # Spherical decay
        self.sc_spherical_decay = QCheckBox("Spherical Decay")
        self.sc_spherical_decay.setChecked(False)
        self.sc_spherical_decay.setToolTip("Apply 1/r spherical decay to scattered amplitude")
        sc_layout.addRow(self.sc_spherical_decay)

        # Analyser mode
        self.sc_analyser_mode = QComboBox()
        self.sc_analyser_mode.addItem("Off", "off")
        self.sc_analyser_mode.addItem("2D Angular Filter", "2d")
        self.sc_analyser_mode.addItem("1D Rocking Curve", "1d")
        self.sc_analyser_mode.currentIndexChanged.connect(self._on_analyser_mode_changed)
        sc_layout.addRow("Analyser:", self.sc_analyser_mode)

        # Analyser acceptance angle
        self.sc_analyser_acceptance = QDoubleSpinBox()
        self.sc_analyser_acceptance.setDecimals(6)
        self.sc_analyser_acceptance.setRange(0, 0.1)
        self.sc_analyser_acceptance.setValue(0.0001)
        self.sc_analyser_acceptance.setSuffix(" rad")
        self.sc_analyser_acceptance.setEnabled(False)
        sc_layout.addRow("Acceptance:", self.sc_analyser_acceptance)

        # Darwin half-width
        self.sc_darwin_halfwidth = QDoubleSpinBox()
        self.sc_darwin_halfwidth.setDecimals(6)
        self.sc_darwin_halfwidth.setRange(0, 0.01)
        self.sc_darwin_halfwidth.setValue(0.00001)
        self.sc_darwin_halfwidth.setSuffix(" rad")
        self.sc_darwin_halfwidth.setEnabled(False)
        sc_layout.addRow("Darwin Width:", self.sc_darwin_halfwidth)

        content_layout.addWidget(self.scattering_group)

        # --- Transmission Options ---
        self.transmission_group = QGroupBox("Transmission Options")
        tr_layout = QFormLayout(self.transmission_group)

        # Number of slices (auto or manual)
        self.tr_auto_slices = QCheckBox("Auto Slices")
        self.tr_auto_slices.setChecked(True)
        self.tr_auto_slices.setToolTip("Automatically determine number of slices based on target phase step")
        self.tr_auto_slices.stateChanged.connect(self._on_auto_slices_changed)
        tr_layout.addRow(self.tr_auto_slices)

        # Manual slice count
        self.tr_n_slices = QSpinBox()
        self.tr_n_slices.setRange(1, 2048)
        self.tr_n_slices.setValue(10)
        self.tr_n_slices.setToolTip("Number of slices for multislice transmission")
        self.tr_n_slices.setEnabled(False)
        tr_layout.addRow("N Slices:", self.tr_n_slices)

        # Target phase step (for auto slices)
        self.tr_target_phase = QDoubleSpinBox()
        self.tr_target_phase.setDecimals(3)
        self.tr_target_phase.setRange(0.001, 1.0)
        self.tr_target_phase.setValue(0.1)
        self.tr_target_phase.setSuffix(" rad")
        self.tr_target_phase.setToolTip("Target max phase step per slice (radians)")
        tr_layout.addRow("Phase Step:", self.tr_target_phase)

        # Pad factor
        self.tr_pad_factor = QDoubleSpinBox()
        self.tr_pad_factor.setDecimals(1)
        self.tr_pad_factor.setRange(1.0, 4.0)
        self.tr_pad_factor.setValue(2.0)
        self.tr_pad_factor.setToolTip("Padding factor for FFT propagation")
        tr_layout.addRow("Pad Factor:", self.tr_pad_factor)

        # Kernel radius (in pixels)
        self.tr_kernel_radius = QSpinBox()
        self.tr_kernel_radius.setRange(0, 100)
        self.tr_kernel_radius.setValue(0)
        self.tr_kernel_radius.setSuffix(" px")
        self.tr_kernel_radius.setToolTip("Gaussian blur radius (pixels) for smoothing. 0 = disabled.")
        tr_layout.addRow("Kernel Radius:", self.tr_kernel_radius)

        # Padding mode
        self.tr_padding_mode = QComboBox()
        self.tr_padding_mode.addItem("Edge", "edge")
        self.tr_padding_mode.addItem("Constant", "constant")
        tr_layout.addRow("Padding:", self.tr_padding_mode)

        # Pad constant
        self.tr_pad_constant = QDoubleSpinBox()
        self.tr_pad_constant.setDecimals(4)
        self.tr_pad_constant.setRange(-1e10, 1e10)
        self.tr_pad_constant.setValue(0.0)
        self.tr_pad_constant.setEnabled(False)
        self.tr_padding_mode.currentIndexChanged.connect(
            lambda idx: self.tr_pad_constant.setEnabled(idx == 1))
        tr_layout.addRow("Pad Value:", self.tr_pad_constant)

        self.transmission_group.setVisible(False)
        content_layout.addWidget(self.transmission_group)

        # --- Wavefield Propagation Options ---
        self.propagation_group = QGroupBox("Propagation Options")
        prop_layout = QFormLayout(self.propagation_group)

        # Max step
        self.prop_step_max = QDoubleSpinBox()
        self.prop_step_max.setDecimals(4)
        self.prop_step_max.setRange(0.001, 1.0)
        self.prop_step_max.setValue(0.02)
        self.prop_step_max.setToolTip("Maximum propagation step size")
        prop_layout.addRow("Step Max:", self.prop_step_max)

        # Pad factor
        self.prop_pad_factor = QDoubleSpinBox()
        self.prop_pad_factor.setDecimals(2)
        self.prop_pad_factor.setRange(0.5, 4.0)
        self.prop_pad_factor.setValue(1.0)
        self.prop_pad_factor.setToolTip("Padding factor for FFT")
        prop_layout.addRow("Pad Factor:", self.prop_pad_factor)

        # Padding mode
        self.prop_padding_mode = QComboBox()
        self.prop_padding_mode.addItem("Edge", "edge")
        self.prop_padding_mode.addItem("Constant", "constant")
        prop_layout.addRow("Padding:", self.prop_padding_mode)

        # Pad constant (enabled when padding mode is "constant")
        self.prop_pad_constant = QDoubleSpinBox()
        self.prop_pad_constant.setDecimals(4)
        self.prop_pad_constant.setRange(-1e10, 1e10)
        self.prop_pad_constant.setValue(0.0)
        self.prop_pad_constant.setEnabled(False)
        self.prop_padding_mode.currentIndexChanged.connect(
            lambda idx: self.prop_pad_constant.setEnabled(idx == 1))
        prop_layout.addRow("Pad Value:", self.prop_pad_constant)

        # Save field
        self.prop_save_field = QCheckBox("Save Field")
        self.prop_save_field.setChecked(True)
        self.prop_save_field.setToolTip("Save propagated field to detector")
        prop_layout.addRow(self.prop_save_field)

        self.propagation_group.setVisible(False)
        content_layout.addWidget(self.propagation_group)

        # --- GPU Options ---
        gpu_group = QGroupBox("Computation")
        gpu_layout = QFormLayout(gpu_group)

        self.use_gpu = QCheckBox("Use GPU")
        self.use_gpu.setChecked(True)
        self.use_gpu.setToolTip("Use GPU acceleration (requires CuPy)")
        gpu_layout.addRow(self.use_gpu)

        content_layout.addWidget(gpu_group)

        # Stretch to fill
        content_layout.addStretch()

        scroll.setWidget(content)
        layout.addWidget(scroll)

    def _on_method_changed(self):
        """Handle method checkbox changes."""
        self.scattering_group.setVisible(self.scattering_enabled.isChecked())
        self.transmission_group.setVisible(self.transmission_enabled.isChecked())
        self.propagation_group.setVisible(self.propagation_enabled.isChecked())

        # Auto-set remove_forward when both scattering and transmission are enabled
        if self.scattering_enabled.isChecked() and self.transmission_enabled.isChecked():
            self.sc_remove_forward.setChecked(True)
            self.sc_remove_forward.setToolTip("Enabled automatically to avoid double-counting with transmission")

        self.method_changed.emit(self._get_method_string())
        self.settings_changed.emit()

    def _on_analyser_mode_changed(self, index):
        """Handle analyser mode change."""
        enabled = index > 0  # Not "off"
        self.sc_analyser_acceptance.setEnabled(enabled)
        self.sc_darwin_halfwidth.setEnabled(enabled)

    def _on_auto_slices_changed(self, state):
        """Handle auto slices checkbox change."""
        auto = self.tr_auto_slices.isChecked()
        self.tr_n_slices.setEnabled(not auto)
        self.tr_target_phase.setEnabled(auto)

    def _get_method_string(self) -> str:
        """Get a string describing the current method selection."""
        methods = []
        if self.scattering_enabled.isChecked():
            methods.append("Scattering")
        if self.transmission_enabled.isChecked():
            methods.append("Transmission")
        if self.propagation_enabled.isChecked():
            methods.append("Propagation")
        return " + ".join(methods) if methods else "None"

    def get_scattering_kwargs(self) -> dict:
        """Get kwargs for atomic_scattering_kinematic method."""
        kwargs = {
            "remove_forward": self.sc_remove_forward.isChecked(),
            "use_depth_ein": self.sc_use_depth_ein.isChecked(),
            "apply_polarization": self.sc_apply_polarization.isChecked(),
            "spherical_decay": self.sc_spherical_decay.isChecked(),
            "use_gpu": self.use_gpu.isChecked(),
        }

        if self.sc_use_depth_ein.isChecked():
            cache_dir = self.sc_ein_cache_dir.text().strip()
            if cache_dir:
                kwargs["ein_cache_dir"] = cache_dir
            kwargs["recompute_cache"] = self.sc_recompute_cache.isChecked()

        analyser_mode = self.sc_analyser_mode.currentData()
        if analyser_mode != "off":
            kwargs["analyser_mode"] = analyser_mode
            kwargs["analyser_acceptance_angle_rad"] = self.sc_analyser_acceptance.value()
            kwargs["analyser_darwin_halfwidth_rad"] = self.sc_darwin_halfwidth.value()

        return kwargs

    def get_transmission_kwargs(self) -> dict:
        """Get kwargs for atomic_transmission method."""
        kwargs = {
            "kernel_radius": self.tr_kernel_radius.value(),
            "padding_mode": self.tr_padding_mode.currentData(),
            "pad_constant": self.tr_pad_constant.value(),
            "pad_factor": self.tr_pad_factor.value(),
            "use_gpu": self.use_gpu.isChecked(),
        }

        # Add slice control parameters
        if self.tr_auto_slices.isChecked():
            kwargs["n_slices"] = None  # Auto mode
            kwargs["target_phase_step"] = self.tr_target_phase.value()
        else:
            kwargs["n_slices"] = self.tr_n_slices.value()

        return kwargs

    def get_propagation_kwargs(self) -> dict:
        """Get kwargs for wavefield_propagation method."""
        return {
            "step_max": self.prop_step_max.value(),
            "pad_factor": self.prop_pad_factor.value(),
            "padding_mode": self.prop_padding_mode.currentData(),
            "pad_constant": self.prop_pad_constant.value(),
            "save_field": self.prop_save_field.isChecked(),
            "use_gpu": self.use_gpu.isChecked(),
        }

    def get_adi_kwargs(self) -> dict:
        """Get kwargs for atomic_direct_interaction method."""
        return {
            "scattering": self.scattering_enabled.isChecked(),
            "sc_kwargs": self.get_scattering_kwargs() if self.scattering_enabled.isChecked() else None,
            "transmission": self.transmission_enabled.isChecked(),
            "tr_kwargs": self.get_transmission_kwargs() if self.transmission_enabled.isChecked() else None,
            "use_gpu": self.use_gpu.isChecked(),
        }

    def is_propagation_enabled(self) -> bool:
        """Check if wavefield propagation is enabled."""
        return self.propagation_enabled.isChecked()

    def get_config(self) -> dict:
        """Get the current configuration."""
        return {
            "scattering_enabled": self.scattering_enabled.isChecked(),
            "transmission_enabled": self.transmission_enabled.isChecked(),
            "propagation_enabled": self.propagation_enabled.isChecked(),
            "use_gpu": self.use_gpu.isChecked(),
            "scattering": {
                "remove_forward": self.sc_remove_forward.isChecked(),
                "use_depth_ein": self.sc_use_depth_ein.isChecked(),
                "ein_cache_dir": self.sc_ein_cache_dir.text(),
                "recompute_cache": self.sc_recompute_cache.isChecked(),
                "apply_polarization": self.sc_apply_polarization.isChecked(),
                "spherical_decay": self.sc_spherical_decay.isChecked(),
                "analyser_mode": self.sc_analyser_mode.currentData(),
                "analyser_acceptance": self.sc_analyser_acceptance.value(),
                "darwin_halfwidth": self.sc_darwin_halfwidth.value(),
            },
            "transmission": {
                "auto_slices": self.tr_auto_slices.isChecked(),
                "n_slices": self.tr_n_slices.value(),
                "target_phase_step": self.tr_target_phase.value(),
                "pad_factor": self.tr_pad_factor.value(),
                "kernel_radius": self.tr_kernel_radius.value(),
                "padding_mode": self.tr_padding_mode.currentData(),
                "pad_constant": self.tr_pad_constant.value(),
            },
            "propagation": {
                "step_max": self.prop_step_max.value(),
                "pad_factor": self.prop_pad_factor.value(),
                "padding_mode": self.prop_padding_mode.currentData(),
                "pad_constant": self.prop_pad_constant.value(),
                "save_field": self.prop_save_field.isChecked(),
            },
        }

    def set_config(self, config: dict):
        """Set the configuration."""
        if "scattering_enabled" in config:
            self.scattering_enabled.setChecked(config["scattering_enabled"])
        if "transmission_enabled" in config:
            self.transmission_enabled.setChecked(config["transmission_enabled"])
        if "propagation_enabled" in config:
            self.propagation_enabled.setChecked(config["propagation_enabled"])
        if "use_gpu" in config:
            self.use_gpu.setChecked(config["use_gpu"])

        if "scattering" in config:
            sc = config["scattering"]
            if "remove_forward" in sc:
                self.sc_remove_forward.setChecked(sc["remove_forward"])
            if "use_depth_ein" in sc:
                self.sc_use_depth_ein.setChecked(sc["use_depth_ein"])
            if "ein_cache_dir" in sc:
                self.sc_ein_cache_dir.setText(sc["ein_cache_dir"])
            if "recompute_cache" in sc:
                self.sc_recompute_cache.setChecked(sc["recompute_cache"])
            if "apply_polarization" in sc:
                self.sc_apply_polarization.setChecked(sc["apply_polarization"])
            if "spherical_decay" in sc:
                self.sc_spherical_decay.setChecked(sc["spherical_decay"])
            if "analyser_mode" in sc:
                idx = self.sc_analyser_mode.findData(sc["analyser_mode"])
                if idx >= 0:
                    self.sc_analyser_mode.setCurrentIndex(idx)
            if "analyser_acceptance" in sc:
                self.sc_analyser_acceptance.setValue(sc["analyser_acceptance"])
            if "darwin_halfwidth" in sc:
                self.sc_darwin_halfwidth.setValue(sc["darwin_halfwidth"])

        if "transmission" in config:
            tr = config["transmission"]
            if "auto_slices" in tr:
                self.tr_auto_slices.setChecked(tr["auto_slices"])
            if "n_slices" in tr:
                self.tr_n_slices.setValue(tr["n_slices"])
            if "target_phase_step" in tr:
                self.tr_target_phase.setValue(tr["target_phase_step"])
            if "pad_factor" in tr:
                self.tr_pad_factor.setValue(tr["pad_factor"])
            if "kernel_radius" in tr:
                self.tr_kernel_radius.setValue(tr["kernel_radius"])
            if "padding_mode" in tr:
                idx = self.tr_padding_mode.findData(tr["padding_mode"])
                if idx >= 0:
                    self.tr_padding_mode.setCurrentIndex(idx)
            if "pad_constant" in tr:
                self.tr_pad_constant.setValue(tr["pad_constant"])

        if "propagation" in config:
            prop = config["propagation"]
            if "step_max" in prop:
                self.prop_step_max.setValue(prop["step_max"])
            if "pad_factor" in prop:
                self.prop_pad_factor.setValue(prop["pad_factor"])
            if "padding_mode" in prop:
                idx = self.prop_padding_mode.findData(prop["padding_mode"])
                if idx >= 0:
                    self.prop_padding_mode.setCurrentIndex(idx)
            if "pad_constant" in prop:
                self.prop_pad_constant.setValue(prop["pad_constant"])
            if "save_field" in prop:
                self.prop_save_field.setChecked(prop["save_field"])

        self._on_method_changed()
