# -----------------------------------------------------------------------------
# Alignment Dialog
# -----------------------------------------------------------------------------
"""
Dialog for aligning stage and detector to a specific diffraction peak.

Features:
- Display target reflection information (d-spacing, Bragg angle)
- Alignment mode selection (stage only, stage+detector, detector only)
- Scattering plane orientation (horizontal, vertical, custom eta)
- Preview calculated motor/detector values
- Apply alignment
"""

import sys
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QPushButton, QGroupBox, QFormLayout,
    QRadioButton, QButtonGroup, QDoubleSpinBox,
    QMessageBox, QFrame,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gui.state import SimulationState
from gui.utils.diffraction_calc import DiffractionCalculator


class AlignmentDialog(QDialog):
    """Dialog for aligning to a specific diffraction peak."""

    alignment_applied = Signal(dict)  # Emits alignment parameters

    def __init__(
        self,
        state: SimulationState,
        hkl: Tuple[int, int, int],
        parent=None
    ):
        """Initialize the alignment dialog.

        Args:
            state: SimulationState instance
            hkl: Target Miller indices (h, k, l)
            parent: Parent widget
        """
        super().__init__(parent)
        self.state = state
        self.hkl = hkl

        # Initialize diffraction calculator
        self._diff_calc = DiffractionCalculator(
            crystal=state.crystal,
            beam=state.beam,
            stage=state.stage
        )

        self.setWindowTitle("Align to Diffraction Peak")
        self.setMinimumWidth(450)
        self.setModal(True)

        self._setup_ui()
        self._update_preview()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Target reflection info
        info_group = QGroupBox("Target Reflection")
        info_layout = QFormLayout(info_group)

        self.hkl_label = QLabel(f"({self.hkl[0]}, {self.hkl[1]}, {self.hkl[2]})")
        self.hkl_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        info_layout.addRow("Miller Indices:", self.hkl_label)

        self.d_spacing_label = QLabel("--")
        info_layout.addRow("d-spacing:", self.d_spacing_label)

        self.bragg_angle_label = QLabel("--")
        info_layout.addRow("Bragg angle:", self.bragg_angle_label)

        self.two_theta_label = QLabel("--")
        info_layout.addRow("2θ:", self.two_theta_label)

        layout.addWidget(info_group)

        # Alignment mode
        mode_group = QGroupBox("Alignment Mode")
        mode_layout = QVBoxLayout(mode_group)

        self.mode_group = QButtonGroup(self)

        self.mode_stage_detector = QRadioButton("Rotate stage + position detector")
        self.mode_stage_detector.setChecked(True)
        self.mode_group.addButton(self.mode_stage_detector, 0)
        mode_layout.addWidget(self.mode_stage_detector)

        self.mode_stage_only = QRadioButton("Rotate stage only")
        self.mode_group.addButton(self.mode_stage_only, 1)
        mode_layout.addWidget(self.mode_stage_only)

        self.mode_detector_only = QRadioButton("Position detector only (current orientation)")
        self.mode_group.addButton(self.mode_detector_only, 2)
        mode_layout.addWidget(self.mode_detector_only)

        self.mode_group.buttonClicked.connect(self._on_mode_changed)

        layout.addWidget(mode_group)

        # Scattering plane orientation
        plane_group = QGroupBox("Scattering Plane")
        plane_layout = QVBoxLayout(plane_group)

        self.plane_group = QButtonGroup(self)

        # Note: Detector.py convention - η=0 is XZ plane (vertical), η=90° is XY plane (horizontal)
        self.plane_vertical = QRadioButton("Vertical (η = 0°)")
        self.plane_vertical.setChecked(True)  # Default to vertical scattering
        self.plane_group.addButton(self.plane_vertical, 0)
        plane_layout.addWidget(self.plane_vertical)

        self.plane_horizontal = QRadioButton("Horizontal (η = 90°)")
        self.plane_group.addButton(self.plane_horizontal, 1)
        plane_layout.addWidget(self.plane_horizontal)

        custom_row = QWidget()
        custom_layout = QHBoxLayout(custom_row)
        custom_layout.setContentsMargins(0, 0, 0, 0)

        self.plane_custom = QRadioButton("Custom:")
        self.plane_group.addButton(self.plane_custom, 2)
        custom_layout.addWidget(self.plane_custom)

        self.custom_eta = QDoubleSpinBox()
        self.custom_eta.setRange(-180, 180)
        self.custom_eta.setValue(0)
        self.custom_eta.setSuffix("°")
        self.custom_eta.setEnabled(False)
        self.custom_eta.valueChanged.connect(self._update_preview)
        custom_layout.addWidget(self.custom_eta)

        custom_layout.addWidget(QLabel("η"))
        custom_layout.addStretch()
        plane_layout.addWidget(custom_row)

        self.plane_group.buttonClicked.connect(self._on_plane_changed)

        layout.addWidget(plane_group)

        # Preview
        preview_group = QGroupBox("Preview")
        preview_layout = QFormLayout(preview_group)

        self.preview_stage = QLabel("--")
        self.preview_stage.setStyleSheet("font-family: monospace;")
        preview_layout.addRow("Stage rotation:", self.preview_stage)

        self.preview_detector = QLabel("--")
        self.preview_detector.setStyleSheet("font-family: monospace;")
        preview_layout.addRow("Detector position:", self.preview_detector)

        layout.addWidget(preview_group)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(self._update_preview)
        btn_layout.addWidget(preview_btn)

        apply_btn = QPushButton("Apply")
        apply_btn.setDefault(True)
        apply_btn.clicked.connect(self._on_apply)
        btn_layout.addWidget(apply_btn)

        layout.addLayout(btn_layout)

        # Populate reflection info
        self._populate_reflection_info()

    def _populate_reflection_info(self):
        """Populate the reflection information fields."""
        try:
            d = self._diff_calc.get_d_spacing(self.hkl)
            self.d_spacing_label.setText(f"{d:.4f} Å")

            theta = self._diff_calc.get_bragg_angle(self.hkl)
            if theta is not None:
                self.bragg_angle_label.setText(f"{np.degrees(theta):.3f}°")
                self.two_theta_label.setText(f"{np.degrees(2 * theta):.3f}°")
            else:
                self.bragg_angle_label.setText("Inaccessible")
                self.two_theta_label.setText("--")

        except Exception as e:
            self.d_spacing_label.setText(f"Error: {e}")
            self.bragg_angle_label.setText("--")
            self.two_theta_label.setText("--")

    def _on_mode_changed(self, button):
        """Handle alignment mode change."""
        self._update_preview()

    def _on_plane_changed(self, button):
        """Handle scattering plane change."""
        # Enable/disable custom eta input
        self.custom_eta.setEnabled(self.plane_custom.isChecked())
        self._update_preview()

    def _get_target_eta(self) -> float:
        """Get the target eta angle in radians.

        Detector.py convention:
        - η = 0 → detector in XZ plane (vertical scattering)
        - η = 90° → detector in XY plane (horizontal scattering)
        """
        if self.plane_horizontal.isChecked():
            return np.pi / 2  # η = 90° for horizontal (XY plane)
        elif self.plane_vertical.isChecked():
            return 0.0  # η = 0° for vertical (XZ plane)
        else:
            return np.radians(self.custom_eta.value())

    def _refresh_diff_calc(self):
        """Refresh diffraction calculator with current state."""
        self._diff_calc.set_crystal(self.state.crystal)
        self._diff_calc.set_beam(self.state.beam)
        self._diff_calc.set_stage(self.state.stage)

    def _update_preview(self):
        """Update the preview with calculated values."""
        try:
            # Refresh calculator with current state
            self._refresh_diff_calc()

            theta = self._diff_calc.get_bragg_angle(self.hkl)
            if theta is None:
                self.preview_stage.setText("Reflection inaccessible")
                self.preview_detector.setText("--")
                return

            two_theta = 2 * theta
            target_eta = self._get_target_eta()

            mode_id = self.mode_group.checkedId()

            # Calculate alignment
            alignment = self._diff_calc.calculate_alignment_motor_values(
                self.hkl, target_eta
            )

            if alignment is None:
                self.preview_stage.setText("Cannot align")
                self.preview_detector.setText("--")
                return

            # Format preview based on mode
            if mode_id == 0:  # Stage + detector
                if 'motor_values' in alignment:
                    mv = alignment['motor_values']
                    self.preview_stage.setText(
                        f"φ={mv.get('phi', 0):.2f}° χ={mv.get('chi', 0):.2f}° ω={mv.get('omega', 0):.2f}°"
                    )
                else:
                    self.preview_stage.setText("Rotation matrix computed")

                self.preview_detector.setText(
                    f"2θ={alignment['two_theta_deg']:.2f}° η={alignment['eta_deg']:.2f}°"
                )

            elif mode_id == 1:  # Stage only
                if 'motor_values' in alignment:
                    mv = alignment['motor_values']
                    self.preview_stage.setText(
                        f"φ={mv.get('phi', 0):.2f}° χ={mv.get('chi', 0):.2f}° ω={mv.get('omega', 0):.2f}°"
                    )
                else:
                    self.preview_stage.setText("Rotation matrix computed")
                self.preview_detector.setText("(not changed)")

            else:  # Detector only
                # Get current Q direction and calculate where detector should go
                det_angles = self._diff_calc.get_detector_angles(self.hkl)
                if det_angles:
                    self.preview_stage.setText("(not changed)")
                    self.preview_detector.setText(
                        f"2θ={np.degrees(det_angles[0]):.2f}° η={np.degrees(det_angles[1]):.2f}°"
                    )
                else:
                    self.preview_stage.setText("(not changed)")
                    self.preview_detector.setText("Cannot determine")

        except Exception as e:
            self.preview_stage.setText(f"Error: {str(e)[:30]}")
            self.preview_detector.setText("--")

    def _on_apply(self):
        """Apply the alignment."""
        try:
            # Refresh calculator with current state
            self._refresh_diff_calc()

            theta = self._diff_calc.get_bragg_angle(self.hkl)
            if theta is None:
                QMessageBox.warning(
                    self, "Cannot Align",
                    "This reflection is not accessible at current beam energy."
                )
                return

            two_theta = 2 * theta
            target_eta = self._get_target_eta()
            mode_id = self.mode_group.checkedId()

            # Calculate alignment
            alignment = self._diff_calc.calculate_alignment_motor_values(
                self.hkl, target_eta
            )

            if alignment is None:
                QMessageBox.warning(
                    self, "Cannot Align",
                    "Unable to calculate alignment for this reflection."
                )
                return

            # Apply based on mode
            applied_changes = []

            if mode_id in [0, 1]:  # Stage rotation needed
                # Apply stage rotation via motor values or rotation matrix
                stage = self.state.stage
                if stage is not None and 'motor_values' in alignment:
                    mv = alignment['motor_values']
                    # Use set_single_motor_value_absolute for each motor
                    for motor_name, angle in mv.items():
                        try:
                            if hasattr(stage, 'set_single_motor_value_absolute'):
                                # Check if motor exists in stage
                                if hasattr(stage, '_motor_name') and motor_name in stage._motor_name:
                                    stage.set_single_motor_value_absolute(motor_name, angle, degrees=True)
                                    applied_changes.append(f"{motor_name}={angle:.2f}°")
                        except Exception as e:
                            print(f"Failed to set motor {motor_name}: {e}")

                    if applied_changes:
                        self.state.notify_object_modified("stage")

            if mode_id in [0, 2]:  # Detector positioning needed
                detector = self.state.detector
                if detector is not None:
                    if mode_id == 2:
                        # Detector only - use current Q direction
                        det_angles = self._diff_calc.get_detector_angles(self.hkl)
                        if det_angles:
                            two_theta_rad, eta_rad = det_angles
                        else:
                            two_theta_rad = two_theta
                            eta_rad = target_eta
                    else:
                        two_theta_rad = alignment['two_theta']
                        eta_rad = alignment['eta']

                    # Get current detector distance
                    distance = getattr(detector, '_distance', None)
                    if distance is None:
                        distance = 500 * 1e7  # Default 500mm in Angstroms

                    if hasattr(detector, 'position_detector_absolute'):
                        # Pass degrees=False since we have radians
                        detector.position_detector_absolute(
                            distance, two_theta_rad, eta_rad, degrees=False
                        )
                        applied_changes.append(
                            f"2θ={np.degrees(two_theta_rad):.2f}°"
                        )
                        applied_changes.append(
                            f"η={np.degrees(eta_rad):.2f}°"
                        )
                        self.state.notify_object_modified("detector")

            # Store result message for parent to display after dialog closes
            if applied_changes:
                self._result_message = (
                    "Alignment Applied",
                    f"Applied:\n" + "\n".join(applied_changes)
                )
            else:
                self._result_message = (
                    "Alignment",
                    "Alignment calculated but no changes were applied.\n"
                    "Motor/detector may not be configured."
                )

            # Emit signal and close
            self.alignment_applied.emit({
                'hkl': self.hkl,
                'mode': mode_id,
                'two_theta': two_theta,
                'eta': target_eta,
                'changes': applied_changes,
            })

            self.accept()

        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to apply alignment:\n{str(e)}"
            )
