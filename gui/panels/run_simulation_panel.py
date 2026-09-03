# -----------------------------------------------------------------------------
# Run Simulation Panel
# -----------------------------------------------------------------------------
"""
Panel for running simulations - single shots and scans.

Provides controls for:
- Single simulation execution
- Multi-dimensional scan orchestration
- Progress monitoring
"""

import sys
from pathlib import Path
import numpy as np

from PySide6.QtCore import Qt, Signal, QThread, QObject
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QPushButton, QComboBox,
    QCheckBox, QLineEdit, QTabWidget, QScrollArea, QFrame,
    QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox, QSizePolicy,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gui.state import SimulationState


class SimulationWorker(QObject):
    """Worker thread for running simulations."""

    progress = Signal(int, str)  # progress %, message
    finished = Signal(object)    # result dict
    error = Signal(str)          # error message
    scan_image = Signal(object)  # {"title": str, "png": bytes} rendered off the GUI thread

    def __init__(self, state, adi_kwargs, propagation_enabled, prop_kwargs, mode="single", scan_config=None):
        """
        Initialize the simulation worker.

        IMPORTANT: All kwargs must be gathered from Qt widgets in the main thread
        before creating this worker. Qt widgets are NOT thread-safe.

        Args:
            state: SimulationState with simulation objects
            adi_kwargs: Dict of kwargs for atomic_direct_interaction (from method_panel.get_adi_kwargs())
            propagation_enabled: Bool indicating if propagation should run (from method_panel.is_propagation_enabled())
            prop_kwargs: Dict of kwargs for wavefield_propagation (from method_panel.get_propagation_kwargs())
            mode: "single" or "scan"
            scan_config: Scan configuration dict (for scan mode)
        """
        super().__init__()
        self.state = state
        self.adi_kwargs = adi_kwargs
        self.propagation_enabled = propagation_enabled
        self.prop_kwargs = prop_kwargs
        self.mode = mode
        self.scan_config = scan_config
        self._cancelled = False

    def cancel(self):
        """Request cancellation."""
        self._cancelled = True

    def run(self):
        """Execute the simulation."""
        try:
            if self.mode == "single":
                self._run_single()
            elif self.mode == "scan":
                self._run_scan()
            elif self.mode == "propagation_only":
                self._run_propagation_only()
        except Exception as e:
            self.error.emit(str(e))

    def _run_single(self):
        """Run a single simulation."""
        self.progress.emit(0, "Preparing simulation...")
        print("[GUI|INFO] Preparing single simulation...")

        beam = self.state.beam
        sample = self.state.sample
        detector = self.state.detector
        stage = self.state.stage
        optics = self.state.optics

        if beam is None:
            self.error.emit("No beam object configured")
            return
        if sample is None:
            self.error.emit("No sample object configured")
            return
        if detector is None:
            self.error.emit("No detector object configured")
            return
        if stage is None:
            self.error.emit("No stage object configured")
            return

        self.progress.emit(10, "Running atomic direct interaction...")
        print("[GUI|INFO] Running atomic direct interaction...")

        # Run atomic direct interaction (kwargs gathered in main thread)
        beam.atomic_direct_interaction(
            sample=sample,
            detector=detector,
            stage=stage,
            **self.adi_kwargs
        )

        self.progress.emit(70, "Atomic interaction complete")
        print("[GUI|INFO] Atomic interaction complete")

        # Run wavefield propagation if enabled (flag checked in main thread)
        if self.propagation_enabled and optics is not None:
            self.progress.emit(75, "Running wavefield propagation...")
            print("[GUI|INFO] Running wavefield propagation...")
            beam.wavefield_propagation(
                detector=detector,
                optics=optics,
                **self.prop_kwargs
            )
            self.progress.emit(95, "Propagation complete")
            print("[GUI|INFO] Propagation complete")

        self.progress.emit(100, "Simulation complete")
        print("[GUI|INFO] Simulation complete")
        self.finished.emit({"success": True, "mode": "single"})

    def _run_propagation_only(self):
        """Run wavefield propagation on existing detector pixel values."""
        self.progress.emit(0, "Preparing propagation...")
        print("[GUI|INFO] Preparing propagation-only simulation...")

        beam = self.state.beam
        detector = self.state.detector
        optics = self.state.optics

        if beam is None:
            self.error.emit("No beam object configured")
            return
        if detector is None:
            self.error.emit("No detector object configured")
            return
        if optics is None:
            self.error.emit("No optics object configured")
            return

        # Check that detector has pixel values to propagate
        if not hasattr(detector, 'pixel_values') or detector.pixel_values is None:
            self.error.emit("Detector has no pixel values. Run a simulation first.")
            return

        # Check that optics has components
        if not hasattr(optics, 'components') or not optics.components:
            self.error.emit("Optics stack is empty. Add at least one component.")
            return

        self.progress.emit(20, "Running wavefield propagation...")
        print("[GUI|INFO] Running wavefield propagation on existing detector data...")
        print(f"[GUI|INFO] Input pixel values shape: {detector.pixel_values.shape}")
        print(f"[GUI|INFO] Input pixel values dtype: {detector.pixel_values.dtype}")

        # Run wavefield propagation on current detector pixel values
        beam.wavefield_propagation(
            detector=detector,
            optics=optics,
            **self.prop_kwargs
        )

        self.progress.emit(100, "Propagation complete")
        print("[GUI|INFO] Propagation complete")
        if hasattr(detector, 'pixel_values') and detector.pixel_values is not None:
            print(f"[GUI|INFO] Output pixel values shape: {detector.pixel_values.shape}")
        self.finished.emit({"success": True, "mode": "propagation_only"})

    def _run_scan(self):
        """Run a scan simulation."""
        if self.scan_config is None:
            self.error.emit("No scan configuration provided")
            return

        self.progress.emit(0, "Preparing scan...")
        print("[GUI|INFO] Preparing scan simulation...")

        beam = self.state.beam
        sample = self.state.sample
        detector = self.state.detector
        stage = self.state.stage
        optics = self.state.optics

        if beam is None:
            self.error.emit("No beam object configured")
            return
        if sample is None:
            self.error.emit("No sample object configured")
            return
        if detector is None:
            self.error.emit("No detector object configured")
            return
        if stage is None:
            self.error.emit("No stage object configured")
            return

        # Get experiment object
        from Experiment import experiment
        exp = experiment()

        self.progress.emit(10, "Starting scan...")
        print(f"[GUI|INFO] Starting scan with {len(self.scan_config.get('motors', []))} axis/axes...")

        # Prepare scan parameters
        ranges = self.scan_config.get("ranges", [])
        stepsizes = self.scan_config.get("stepsizes", [])
        motors = self.scan_config.get("motors", [])
        degrees = self.scan_config.get("degrees", True)
        scan_mode = self.scan_config.get("scan_mode", "absolute")
        couplings = self.scan_config.get("couplings", None)
        per_step_outputs = self.scan_config.get("per_step_outputs", ("Intensity",))
        show_plots = self.scan_config.get("show_plots", True)
        save_dir = self.scan_config.get("save_dir", None)

        # Log coupling info if enabled
        if couplings:
            print(f"[GUI|INFO] Motor couplings enabled: {couplings}")

        # Use kwargs gathered in main thread (thread-safe)
        prop_kwargs = self.prop_kwargs if self.propagation_enabled else None

        def on_step(info):
            # Runs in this worker thread: report progress and hand rendered
            # images to the main thread; never open a window here
            pct = 10 + int(85 * info["step"] / max(1, info["total"]))
            self.progress.emit(pct, f"Step {info['step']}/{info['total']}: {info['position']}")
            if show_plots:
                for name, fig in info["figures"].items():
                    png = self._figure_to_png(fig)
                    if png is not None:
                        self.scan_image.emit({"title": f"Step {info['step']}/{info['total']} {name}", "png": png})

        # Plots are shown on the main thread from the emitted images, so the
        # scan itself never calls plt.show()
        result = exp.scan_nD(
            sample=sample,
            beam=beam,
            detector=detector,
            stage=stage,
            ranges=ranges,
            stepsizes=stepsizes,
            motors=motors,
            degrees=degrees,
            scan_mode=scan_mode,
            optics=optics if self.propagation_enabled else None,
            couplings=couplings,
            per_step_outputs=per_step_outputs,
            show_plots=False,
            save_dir=save_dir,
            adi_kwargs=self.adi_kwargs,
            prop_kwargs=prop_kwargs,
            step_callback=on_step,
        )

        summary_png = self._render_scan_summary(result) if show_plots else None

        self.progress.emit(100, "Scan complete")
        print("[GUI|INFO] Scan complete")
        self.finished.emit({"success": True, "mode": "scan", "result": result, "summary_png": summary_png})

    @staticmethod
    def _figure_to_png(fig, dpi=100):
        """Rasterize a matplotlib figure to PNG bytes (Agg path, no window)."""
        import io
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi)
            return buf.getvalue()
        except Exception as e:
            print(f"[GUI|WARN] Could not rasterize scan figure: {e}")
            return None

    @staticmethod
    def _render_scan_summary(result):
        """Render the summed-intensity summary of a scan_nD result to PNG bytes.

        Uses a standalone Agg figure so it is safe to call from the worker thread.
        """
        import io
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            axes = result["axes"]
            motors = result["motor_names"]
            sumI = np.asarray(result["sum_intensity"])
            fig = Figure(figsize=(6, 4.5))
            FigureCanvasAgg(fig)
            if len(axes) == 1:
                ax = fig.add_subplot(111)
                ax.plot(axes[0], sumI)
                ax.set_xlabel(motors[0])
                ax.set_ylabel("sum(Intensity)")
                ax.set_title(f"Summed intensity vs {motors[0]}")
            elif len(axes) == 2:
                ax = fig.add_subplot(111)
                X, Y = np.meshgrid(axes[0], axes[1], indexing="xy")
                pcm = ax.pcolormesh(X, Y, sumI.T, shading="auto")
                fig.colorbar(pcm, ax=ax, label="sum(Intensity)")
                ax.set_xlabel(motors[0])
                ax.set_ylabel(motors[1])
                ax.set_title(f"Summed intensity vs {motors[0]} and {motors[1]}")
            else:
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                ax = fig.add_subplot(111, projection="3d")
                A0, A1, A2 = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
                sc = ax.scatter(A0.ravel(), A1.ravel(), A2.ravel(), c=sumI.ravel(), s=10)
                fig.colorbar(sc, ax=ax, label="sum(Intensity)")
                ax.set_xlabel(motors[0]); ax.set_ylabel(motors[1]); ax.set_zlabel(motors[2])
                ax.set_title(f"Summed intensity vs {motors[0]}, {motors[1]}, {motors[2]}")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            return buf.getvalue()
        except Exception as e:
            print(f"[GUI|WARN] Could not render scan summary: {e}")
            return None


class RunSimulationPanel(QWidget):
    """Panel for running simulations and scans."""

    simulation_started = Signal()
    simulation_finished = Signal(object)
    simulation_error = Signal(str)

    def __init__(self, state: SimulationState, method_panel, parent=None):
        super().__init__(parent)
        self.state = state
        self.method_panel = method_panel
        self._worker = None
        self._thread = None
        self._scan_image_window = None
        self._setup_ui()
        self._register_observers()

    def _register_observers(self):
        """Register for state change notifications."""
        self.state.register_observer("stage_changed", self._on_stage_changed)
        self.state.register_observer("detector_changed", self._on_detector_changed)

    def _on_stage_changed(self, stage):
        """Handle stage configuration changes."""
        self._refresh_motor_dropdowns()

    def _on_detector_changed(self, detector):
        """Handle detector configuration changes."""
        self._refresh_motor_dropdowns()

    def _refresh_motor_dropdowns(self):
        """Refresh all motor comboboxes with current available motors."""
        available_motors = self._get_available_motors()

        # Refresh axes table motor combos
        for row in range(self.axes_table.rowCount()):
            motor_combo = self.axes_table.cellWidget(row, 0)
            if motor_combo and isinstance(motor_combo, QComboBox):
                current_motor = motor_combo.currentData()
                motor_combo.blockSignals(True)
                motor_combo.clear()

                if available_motors:
                    for motor in available_motors:
                        motor_combo.addItem(motor, motor)
                    motor_combo.setEnabled(True)
                else:
                    motor_combo.addItem("(no stage configured)", None)
                    motor_combo.setEnabled(False)

                # Restore selection if still available
                if current_motor:
                    idx = motor_combo.findData(current_motor)
                    if idx >= 0:
                        motor_combo.setCurrentIndex(idx)
                motor_combo.blockSignals(False)

        # Refresh couplings table motor combos (source and target columns)
        for row in range(self.couplings_table.rowCount()):
            for col in [0, 1]:
                combo = self.couplings_table.cellWidget(row, col)
                if combo and isinstance(combo, QComboBox):
                    current = combo.currentData()
                    combo.blockSignals(True)
                    combo.clear()

                    if available_motors:
                        for motor in available_motors:
                            combo.addItem(motor, motor)
                        combo.setEnabled(True)
                    else:
                        combo.addItem("(no stage configured)", None)
                        combo.setEnabled(False)

                    if current:
                        idx = combo.findData(current)
                        if idx >= 0:
                            combo.setCurrentIndex(idx)
                    combo.blockSignals(False)

    def _setup_ui(self):
        """Setup the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Title
        title = QLabel("Run Simulation")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title)

        # Tab widget for single run vs scan
        self.tabs = QTabWidget()

        # --- Single Run Tab ---
        single_tab = QWidget()
        single_layout = QVBoxLayout(single_tab)

        # Info about current settings
        info_group = QGroupBox("Current Configuration")
        info_layout = QVBoxLayout(info_group)

        self.config_info = QLabel("Configure simulation objects in the Object Browser")
        self.config_info.setWordWrap(True)
        self.config_info.setStyleSheet("color: #808080;")
        info_layout.addWidget(self.config_info)

        single_layout.addWidget(info_group)

        # Run button
        self.run_single_btn = QPushButton("Run Single Simulation")
        self.run_single_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a6a2a;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3a8a3a;
            }
            QPushButton:disabled {
                background-color: #555555;
            }
        """)
        self.run_single_btn.clicked.connect(self._on_run_single)
        single_layout.addWidget(self.run_single_btn)

        # Run Propagation Only button
        self.run_prop_btn = QPushButton("Run Propagation Only")
        self.run_prop_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a5a6a;
                color: white;
                font-weight: bold;
                padding: 8px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #3a7a8a;
            }
            QPushButton:disabled {
                background-color: #555555;
            }
        """)
        self.run_prop_btn.setToolTip(
            "Run wavefield propagation on current detector pixel values.\n"
            "Use this to propagate results from a previous simulation\n"
            "through the optics stack without re-running ADI."
        )
        self.run_prop_btn.clicked.connect(self._on_run_propagation_only)
        single_layout.addWidget(self.run_prop_btn)

        single_layout.addStretch()
        self.tabs.addTab(single_tab, "Single Run")

        # --- Scan Tab ---
        scan_tab = QWidget()
        scan_layout = QVBoxLayout(scan_tab)

        # Scroll area for scan options
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        scan_content = QWidget()
        scan_content_layout = QVBoxLayout(scan_content)
        scan_content_layout.setContentsMargins(0, 0, 0, 0)

        # Scan dimensions
        dim_group = QGroupBox("Scan Axes")
        dim_layout = QVBoxLayout(dim_group)

        # Number of axes
        axis_row = QWidget()
        axis_layout = QHBoxLayout(axis_row)
        axis_layout.setContentsMargins(0, 0, 0, 0)
        axis_layout.addWidget(QLabel("Number of Axes:"))
        self.num_axes = QSpinBox()
        self.num_axes.setRange(1, 3)
        self.num_axes.setValue(1)
        self.num_axes.valueChanged.connect(self._on_num_axes_changed)
        axis_layout.addWidget(self.num_axes)
        axis_layout.addStretch()
        dim_layout.addWidget(axis_row)

        # Axes table
        self.axes_table = QTableWidget(1, 4)
        self.axes_table.setHorizontalHeaderLabels(["Motor", "Start", "Stop", "Step Size"])
        self.axes_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.axes_table.verticalHeader().setVisible(False)
        self.axes_table.setMaximumHeight(150)
        self._setup_axes_table_row(0)
        dim_layout.addWidget(self.axes_table)

        scan_content_layout.addWidget(dim_group)

        # Motor couplings
        coupling_group = QGroupBox("Motor Couplings (Optional)")
        coupling_layout = QVBoxLayout(coupling_group)

        coupling_desc = QLabel("Define coupled motor movements (e.g., theta-2theta scan).\n"
                               "Ratio: for every 1 unit of source motor, target moves by ratio units.")
        coupling_desc.setWordWrap(True)
        coupling_desc.setStyleSheet("color: #808080; font-style: italic;")
        coupling_layout.addWidget(coupling_desc)

        # Enable couplings checkbox
        self.enable_couplings = QCheckBox("Enable Motor Couplings")
        self.enable_couplings.toggled.connect(self._on_couplings_toggled)
        coupling_layout.addWidget(self.enable_couplings)

        # Couplings table
        self.couplings_table = QTableWidget(1, 3)
        self.couplings_table.setHorizontalHeaderLabels(["Source Motor", "Target Motor", "Ratio"])
        self.couplings_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.couplings_table.verticalHeader().setVisible(False)
        self.couplings_table.setMaximumHeight(120)
        self.couplings_table.setEnabled(False)
        self._setup_coupling_table_row(0)
        coupling_layout.addWidget(self.couplings_table)

        # Add/Remove coupling buttons
        coupling_btn_row = QWidget()
        coupling_btn_layout = QHBoxLayout(coupling_btn_row)
        coupling_btn_layout.setContentsMargins(0, 0, 0, 0)

        self.add_coupling_btn = QPushButton("+ Add Coupling")
        self.add_coupling_btn.setEnabled(False)
        self.add_coupling_btn.clicked.connect(self._on_add_coupling)
        coupling_btn_layout.addWidget(self.add_coupling_btn)

        self.remove_coupling_btn = QPushButton("- Remove")
        self.remove_coupling_btn.setEnabled(False)
        self.remove_coupling_btn.clicked.connect(self._on_remove_coupling)
        coupling_btn_layout.addWidget(self.remove_coupling_btn)

        coupling_btn_layout.addStretch()
        coupling_layout.addWidget(coupling_btn_row)

        scan_content_layout.addWidget(coupling_group)

        # Scan options
        options_group = QGroupBox("Scan Options")
        options_layout = QFormLayout(options_group)

        # Scan mode
        self.scan_mode = QComboBox()
        self.scan_mode.addItem("Absolute", "absolute")
        self.scan_mode.addItem("Relative", "relative")
        options_layout.addRow("Mode:", self.scan_mode)

        # Degrees
        self.use_degrees = QCheckBox("Use Degrees")
        self.use_degrees.setChecked(True)
        options_layout.addRow(self.use_degrees)

        # Show plots
        self.show_plots = QCheckBox("Show Plots")
        self.show_plots.setChecked(True)
        options_layout.addRow(self.show_plots)

        # Per-step outputs
        self.output_intensity = QCheckBox("Intensity")
        self.output_intensity.setChecked(True)
        self.output_amplitude = QCheckBox("Amplitude")
        self.output_phase = QCheckBox("Phase")

        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_intensity)
        output_layout.addWidget(self.output_amplitude)
        output_layout.addWidget(self.output_phase)
        options_layout.addRow("Outputs:", output_row)

        scan_content_layout.addWidget(options_group)

        # Save directory
        save_group = QGroupBox("Save Options")
        save_layout = QFormLayout(save_group)

        dir_row = QWidget()
        dir_layout = QHBoxLayout(dir_row)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        self.save_dir_edit = QLineEdit()
        self.save_dir_edit.setPlaceholderText("Optional - leave empty for no save")
        dir_layout.addWidget(self.save_dir_edit)
        browse_btn = QPushButton("...")
        browse_btn.setFixedWidth(30)
        browse_btn.clicked.connect(self._on_browse_save_dir)
        dir_layout.addWidget(browse_btn)
        save_layout.addRow("Save Dir:", dir_row)

        scan_content_layout.addWidget(save_group)

        # Estimated steps
        self.steps_label = QLabel("Estimated steps: 0")
        self.steps_label.setStyleSheet("color: #808080;")
        scan_content_layout.addWidget(self.steps_label)

        scan_content_layout.addStretch()
        scroll.setWidget(scan_content)
        scan_layout.addWidget(scroll)

        # Run scan button
        self.run_scan_btn = QPushButton("Run Scan")
        self.run_scan_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a6a2a;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3a8a3a;
            }
            QPushButton:disabled {
                background-color: #555555;
            }
        """)
        self.run_scan_btn.clicked.connect(self._on_run_scan)
        scan_layout.addWidget(self.run_scan_btn)

        self.tabs.addTab(scan_tab, "Scan")

        layout.addWidget(self.tabs)

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("Ready")
        self.progress_label.setStyleSheet("color: #808080;")
        progress_layout.addWidget(self.progress_label)

        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        progress_layout.addWidget(self.cancel_btn)

        layout.addWidget(progress_group)

        # Connect table changes to step estimation
        self.axes_table.cellChanged.connect(self._update_step_estimate)

    def _get_available_motors(self) -> list:
        """Get list of available motors from stage and detector."""
        motors = []
        stage = self.state.stage
        if stage is not None:
            if hasattr(stage, 'motor_names'):
                motors.extend(stage.motor_names)
            elif hasattr(stage, '_motor_name') and stage._motor_name is not None:
                motors.extend([str(name) for name in stage._motor_name])

        detector = self.state.detector
        if detector is not None:
            motors.extend(["two_theta", "eta", "distance"])

        return motors

    def _validate_scan_motors(self, motors: list) -> tuple:
        """
        Validate that selected motors exist.

        Args:
            motors: List of motor names to validate.

        Returns:
            tuple: (is_valid: bool, error_messages: list)
        """
        errors = []
        available = set(self._get_available_motors())

        for motor in motors:
            if motor not in available:
                errors.append(f"Motor '{motor}' is not available")

        if len(motors) != len(set(motors)):
            errors.append("Duplicate motors in scan configuration")

        return (len(errors) == 0, errors)

    def _setup_axes_table_row(self, row):
        """Setup a row in the axes table with proper widgets."""
        # Motor combo box
        motor_combo = QComboBox()

        # Get motors dynamically from stage and detector
        available_motors = self._get_available_motors()
        if available_motors:
            for motor in available_motors:
                motor_combo.addItem(motor, motor)
        else:
            motor_combo.addItem("(no stage configured)", None)
            motor_combo.setEnabled(False)

        self.axes_table.setCellWidget(row, 0, motor_combo)

        # Start value
        start_spin = QDoubleSpinBox()
        start_spin.setDecimals(4)
        start_spin.setRange(-1e6, 1e6)
        start_spin.setValue(0)
        self.axes_table.setCellWidget(row, 1, start_spin)

        # Stop value
        stop_spin = QDoubleSpinBox()
        stop_spin.setDecimals(4)
        stop_spin.setRange(-1e6, 1e6)
        stop_spin.setValue(1)
        self.axes_table.setCellWidget(row, 2, stop_spin)

        # Step size
        step_spin = QDoubleSpinBox()
        step_spin.setDecimals(4)
        step_spin.setRange(1e-6, 1e6)
        step_spin.setValue(0.1)
        self.axes_table.setCellWidget(row, 3, step_spin)

    def _setup_coupling_table_row(self, row):
        """Setup a row in the couplings table."""
        # Get motors dynamically from stage and detector
        available_motors = self._get_available_motors()

        # Source motor combo box
        source_combo = QComboBox()
        if available_motors:
            for motor in available_motors:
                source_combo.addItem(motor, motor)
        else:
            source_combo.addItem("(no stage configured)", None)
            source_combo.setEnabled(False)
        self.couplings_table.setCellWidget(row, 0, source_combo)

        # Target motor combo box
        target_combo = QComboBox()
        if available_motors:
            for motor in available_motors:
                target_combo.addItem(motor, motor)
            # Default to two_theta if available, otherwise last item
            two_theta_idx = target_combo.findData("two_theta")
            if two_theta_idx >= 0:
                target_combo.setCurrentIndex(two_theta_idx)
        else:
            target_combo.addItem("(no stage configured)", None)
            target_combo.setEnabled(False)
        self.couplings_table.setCellWidget(row, 1, target_combo)

        # Ratio input (can be float like 2.0 or string like "1:2")
        ratio_edit = QLineEdit()
        ratio_edit.setText("2.0")
        ratio_edit.setPlaceholderText("e.g., 2.0 or 1:2")
        ratio_edit.setToolTip("Ratio: for each unit of source motor, target moves by this amount.\n"
                              "Can be a number (2.0) or ratio string (1:2)")
        self.couplings_table.setCellWidget(row, 2, ratio_edit)

    def _on_couplings_toggled(self, enabled):
        """Handle couplings enable/disable toggle."""
        self.couplings_table.setEnabled(enabled)
        self.add_coupling_btn.setEnabled(enabled)
        self.remove_coupling_btn.setEnabled(enabled and self.couplings_table.rowCount() > 1)

    def _on_add_coupling(self):
        """Add a new coupling row."""
        row = self.couplings_table.rowCount()
        self.couplings_table.insertRow(row)
        self._setup_coupling_table_row(row)
        self.remove_coupling_btn.setEnabled(True)

    def _on_remove_coupling(self):
        """Remove the last coupling row."""
        if self.couplings_table.rowCount() > 1:
            self.couplings_table.removeRow(self.couplings_table.rowCount() - 1)
        if self.couplings_table.rowCount() <= 1:
            self.remove_coupling_btn.setEnabled(False)

    def _on_num_axes_changed(self, value):
        """Handle change in number of scan axes."""
        current_rows = self.axes_table.rowCount()

        if value > current_rows:
            # Add rows
            for i in range(current_rows, value):
                self.axes_table.insertRow(i)
                self._setup_axes_table_row(i)
        elif value < current_rows:
            # Remove rows
            for i in range(current_rows - 1, value - 1, -1):
                self.axes_table.removeRow(i)

        self._update_step_estimate()

    def _update_step_estimate(self):
        """Update the estimated number of steps."""
        total_steps = 1
        for row in range(self.axes_table.rowCount()):
            start_widget = self.axes_table.cellWidget(row, 1)
            stop_widget = self.axes_table.cellWidget(row, 2)
            step_widget = self.axes_table.cellWidget(row, 3)

            if start_widget and stop_widget and step_widget:
                start = start_widget.value()
                stop = stop_widget.value()
                step = step_widget.value()

                if step > 0:
                    steps = int(abs(stop - start) / step) + 1
                    total_steps *= steps

        self.steps_label.setText(f"Estimated steps: {total_steps:,}")

    def _on_browse_save_dir(self):
        """Browse for save directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Save Directory",
            self.save_dir_edit.text() or str(Path.home())
        )
        if directory:
            self.save_dir_edit.setText(directory)

    def _get_scan_config(self) -> dict:
        """Build scan configuration from UI."""
        ranges = []
        stepsizes = []
        motors = []

        for row in range(self.axes_table.rowCount()):
            motor_combo = self.axes_table.cellWidget(row, 0)
            start_spin = self.axes_table.cellWidget(row, 1)
            stop_spin = self.axes_table.cellWidget(row, 2)
            step_spin = self.axes_table.cellWidget(row, 3)

            if motor_combo and start_spin and stop_spin and step_spin:
                motors.append(motor_combo.currentData())
                ranges.append((start_spin.value(), stop_spin.value()))
                stepsizes.append(step_spin.value())

        # Build per-step outputs
        outputs = []
        if self.output_intensity.isChecked():
            outputs.append("Intensity")
        if self.output_amplitude.isChecked():
            outputs.append("Amplitude")
        if self.output_phase.isChecked():
            outputs.append("Phase")

        # Build couplings dict if enabled
        couplings = None
        if self.enable_couplings.isChecked():
            couplings = {}
            for row in range(self.couplings_table.rowCount()):
                source_combo = self.couplings_table.cellWidget(row, 0)
                target_combo = self.couplings_table.cellWidget(row, 1)
                ratio_edit = self.couplings_table.cellWidget(row, 2)

                if source_combo and target_combo and ratio_edit:
                    source = source_combo.currentData()
                    target = target_combo.currentData()
                    ratio_str = ratio_edit.text().strip()

                    # Parse ratio - can be float or "a:b" string
                    try:
                        if ":" in ratio_str:
                            # Keep as string for scan_nD to parse
                            ratio = ratio_str
                        else:
                            ratio = float(ratio_str)
                    except ValueError:
                        QMessageBox.warning(
                            self, "Invalid Coupling Ratio",
                            f"Invalid ratio '{ratio_str}' for coupling {source} -> {target}.\n"
                            "Using default ratio 2.0."
                        )
                        ratio = 2.0

                    # Add to couplings dict
                    if source not in couplings:
                        couplings[source] = []
                    couplings[source].append((target, ratio))

        return {
            "motors": motors,
            "ranges": ranges,
            "stepsizes": stepsizes,
            "degrees": self.use_degrees.isChecked(),
            "scan_mode": self.scan_mode.currentData(),
            "couplings": couplings,
            "per_step_outputs": tuple(outputs) if outputs else ("Intensity",),
            "show_plots": self.show_plots.isChecked(),
            "save_dir": self.save_dir_edit.text() or None,
        }

    def _validate_simulation_objects(self) -> tuple:
        """Validate that required objects are configured and properly initialized."""
        missing = []

        # Check beam exists and is properly initialized
        beam = self.state.beam
        if beam is None:
            missing.append("Beam")
        else:
            # Check that beam has been defined (energy, wavelength set)
            if not hasattr(beam, '_energy') or beam._energy is None:
                missing.append("Beam (not initialized - click 'Create Beam')")
            elif not hasattr(beam, '_wavelength') or beam._wavelength is None:
                missing.append("Beam (wavelength not set)")

        # Check sample exists and has atoms
        sample = self.state.sample
        if sample is None:
            missing.append("Sample")
        else:
            # Check that sample has atoms loaded
            chunk_total = getattr(sample, 'chunk_total', None)
            if chunk_total is None or chunk_total == 0:
                missing.append("Sample (no atoms loaded - generate or import sample)")

        # Check detector exists and is properly initialized
        detector = self.state.detector
        if detector is None:
            missing.append("Detector")
        else:
            # Check that detector has been created with shape and pixel coordinates
            if not hasattr(detector, '_shape') or detector._shape is None:
                missing.append("Detector (not initialized - click 'Create Detector')")
            elif not hasattr(detector, '_pixel_coordinates') or detector._pixel_coordinates is None:
                missing.append("Detector (pixel coordinates not set - recreate detector)")

        # Check stage exists and is properly initialized
        stage = self.state.stage
        if stage is None:
            missing.append("Stage")
        else:
            # Check that stage has been created
            if not hasattr(stage, '_rotation') or stage._rotation is None:
                missing.append("Stage (not initialized - click 'Create Stage')")
            elif not hasattr(stage, '_translation') or stage._translation is None:
                missing.append("Stage (translation not set)")

        # Check optics if propagation is enabled
        if self.method_panel.is_propagation_enabled():
            optics = self.state.optics
            if optics is None:
                missing.append("Optics (required for propagation)")
            else:
                # Check that optics has components
                components = getattr(optics, 'components', None)
                if components is None or len(components) == 0:
                    missing.append("Optics (no components added - add at least one component)")

        return (len(missing) == 0, missing)

    def _on_run_single(self):
        """Handle run single simulation button."""
        valid, missing = self._validate_simulation_objects()
        if not valid:
            QMessageBox.warning(
                self, "Missing Objects",
                f"Please configure the following objects first:\n- " + "\n- ".join(missing)
            )
            return

        self._start_simulation("single")

    def _on_run_scan(self):
        """Handle run scan button."""
        valid, missing = self._validate_simulation_objects()
        if not valid:
            QMessageBox.warning(
                self, "Missing Objects",
                f"Please configure the following objects first:\n- " + "\n- ".join(missing)
            )
            return

        scan_config = self._get_scan_config()
        if not scan_config["motors"]:
            QMessageBox.warning(self, "No Motors", "Please configure at least one scan axis.")
            return

        # Validate motors before starting scan
        motors_valid, motor_errors = self._validate_scan_motors(scan_config["motors"])
        if not motors_valid:
            QMessageBox.warning(
                self, "Invalid Motors",
                "The following motor issues were found:\n- " + "\n- ".join(motor_errors)
            )
            return

        self._start_simulation("scan", scan_config)

    def _on_run_propagation_only(self):
        """Handle run propagation only button."""
        # Validate required objects for propagation
        missing = []

        beam = self.state.beam
        if beam is None:
            missing.append("Beam")
        elif not hasattr(beam, '_wavelength') or beam._wavelength is None:
            missing.append("Beam (wavelength not set)")

        detector = self.state.detector
        if detector is None:
            missing.append("Detector")
        elif not hasattr(detector, 'pixel_values') or detector.pixel_values is None:
            missing.append("Detector (no pixel values - run a simulation first)")

        optics = self.state.optics
        if optics is None:
            missing.append("Optics")
        elif not hasattr(optics, 'components') or not optics.components:
            missing.append("Optics (no components added)")

        if missing:
            QMessageBox.warning(
                self, "Missing Objects",
                f"Please configure the following objects first:\n- " + "\n- ".join(missing)
            )
            return

        self._start_simulation("propagation_only")

    def _start_simulation(self, mode, scan_config=None):
        """Start a simulation in a worker thread."""
        # Disable buttons
        self.run_single_btn.setEnabled(False)
        self.run_scan_btn.setEnabled(False)
        self.run_prop_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting...")

        # IMPORTANT: Gather all kwargs from method_panel in main thread!
        # Qt widgets are NOT thread-safe and must only be accessed from main thread.
        adi_kwargs = self.method_panel.get_adi_kwargs()
        propagation_enabled = self.method_panel.is_propagation_enabled()
        prop_kwargs = self.method_panel.get_propagation_kwargs()

        print(f"[GUI|INFO] ADI kwargs: {adi_kwargs}")
        print(f"[GUI|INFO] Propagation enabled: {propagation_enabled}")
        if propagation_enabled:
            print(f"[GUI|INFO] Propagation kwargs: {prop_kwargs}")

        # Create worker and thread with pre-gathered kwargs
        self._thread = QThread()
        self._worker = SimulationWorker(
            self.state, adi_kwargs, propagation_enabled, prop_kwargs, mode, scan_config
        )
        self._worker.moveToThread(self._thread)

        # Connect signals
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.scan_image.connect(self._on_scan_image)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)

        # Start
        self._thread.start()
        self.simulation_started.emit()

    def _on_progress(self, value, message):
        """Handle progress update."""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    def _on_scan_image(self, payload):
        """Show a per-step scan image rendered by the worker (main thread)."""
        self._show_scan_image(payload.get("title", "Scan"), payload.get("png"))

    def _show_scan_image(self, title, png_bytes):
        """Display PNG bytes in a persistent non-modal window."""
        if not png_bytes:
            return
        from PySide6.QtWidgets import QDialog
        from PySide6.QtGui import QPixmap
        if getattr(self, "_scan_image_window", None) is None:
            dlg = QDialog(self)
            dlg.setWindowTitle("Scan")
            layout = QVBoxLayout(dlg)
            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
            dlg._image_label = label
            self._scan_image_window = dlg
        dlg = self._scan_image_window
        pixmap = QPixmap()
        if pixmap.loadFromData(png_bytes, "PNG"):
            dlg._image_label.setPixmap(pixmap)
            dlg.setWindowTitle(title)
            dlg.adjustSize()
            dlg.show()

    def _on_finished(self, result):
        """Handle simulation finished."""
        self.run_single_btn.setEnabled(True)
        self.run_scan_btn.setEnabled(True)
        self.run_prop_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_label.setText("Complete")
        self.progress_label.setStyleSheet("color: #4ec94e;")

        if isinstance(result, dict) and result.get("summary_png"):
            self._show_scan_image("Scan summary", result["summary_png"])

        self.simulation_finished.emit(result)

        # Notify state that detector data has changed
        self.state.notify_object_modified("detector")

        # Show appropriate completion message
        mode = result.get("mode", "single") if isinstance(result, dict) else "single"
        if mode == "propagation_only":
            QMessageBox.information(self, "Complete", "Wavefield propagation completed successfully!")
        else:
            QMessageBox.information(self, "Complete", "Simulation completed successfully!")

    def _on_error(self, message):
        """Handle simulation error."""
        self.run_single_btn.setEnabled(True)
        self.run_scan_btn.setEnabled(True)
        self.run_prop_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"Error: {message}")
        self.progress_label.setStyleSheet("color: #e05050;")

        self.simulation_error.emit(message)

        QMessageBox.critical(self, "Error", f"Simulation failed:\n{message}")

    def _on_cancel(self):
        """Handle cancel button."""
        if self._worker:
            self._worker.cancel()
        self.progress_label.setText("Cancelling...")

    def _count_sample_atoms(self, sample) -> int:
        """
        Count total atoms in sample by reading chunk files.

        Args:
            sample: Sample object with directory and chunk_total

        Returns:
            int: Total atom count, or 0 if cannot be determined
        """
        import os

        if not hasattr(sample, 'directory') or not sample.directory:
            return 0

        chunk_total = getattr(sample, '_chunk_total', None)
        if chunk_total is None or chunk_total == 0:
            return 0

        total_atoms = 0
        for i in range(1, chunk_total + 1):
            chunk_file = os.path.join(sample.directory, f"atomic_positions_{i}.npy")
            if os.path.exists(chunk_file):
                try:
                    # Use mmap to avoid loading entire array into memory
                    positions = np.load(chunk_file, mmap_mode='r')
                    total_atoms += len(positions)
                except Exception:
                    pass
        return total_atoms

    def update_config_info(self):
        """Update the configuration info display."""
        parts = []

        if self.state.beam:
            beam = self.state.beam
            if hasattr(beam, '_energy') and beam._energy:
                parts.append(f"Beam: {beam._energy:.1f} eV")
            else:
                parts.append("Beam: configured")

        if self.state.sample:
            sample = self.state.sample
            atom_count = self._count_sample_atoms(sample)
            if atom_count > 0:
                parts.append(f"Sample: {atom_count:,} atoms")
            elif hasattr(sample, '_chunk_total') and sample._chunk_total:
                parts.append(f"Sample: {sample._chunk_total} chunks (atoms not counted)")
            else:
                parts.append("Sample: configured")

        if self.state.detector:
            detector = self.state.detector
            if hasattr(detector, 'shape') and detector.shape:
                parts.append(f"Detector: {detector.shape[0]}×{detector.shape[1]}")
            else:
                parts.append("Detector: configured")

        if self.state.stage:
            parts.append("Stage: configured")

        if self.state.optics:
            optics = self.state.optics
            if hasattr(optics, 'components'):
                n_comp = len(optics.components) if optics.components else 0
                parts.append(f"Optics: {n_comp} components")

        if parts:
            self.config_info.setText("\n".join(parts))
            self.config_info.setStyleSheet("color: #4ec94e;")
        else:
            self.config_info.setText("Configure simulation objects in the Object Browser")
            self.config_info.setStyleSheet("color: #808080;")

    def get_config(self) -> dict:
        """Get the current configuration."""
        return {
            "num_axes": self.num_axes.value(),
            "scan_mode": self.scan_mode.currentData(),
            "use_degrees": self.use_degrees.isChecked(),
            "show_plots": self.show_plots.isChecked(),
            "output_intensity": self.output_intensity.isChecked(),
            "output_amplitude": self.output_amplitude.isChecked(),
            "output_phase": self.output_phase.isChecked(),
            "save_dir": self.save_dir_edit.text(),
        }

    def set_config(self, config: dict):
        """Set the configuration."""
        if "num_axes" in config:
            self.num_axes.setValue(config["num_axes"])
        if "scan_mode" in config:
            idx = self.scan_mode.findData(config["scan_mode"])
            if idx >= 0:
                self.scan_mode.setCurrentIndex(idx)
        if "use_degrees" in config:
            self.use_degrees.setChecked(config["use_degrees"])
        if "show_plots" in config:
            self.show_plots.setChecked(config["show_plots"])
        if "output_intensity" in config:
            self.output_intensity.setChecked(config["output_intensity"])
        if "output_amplitude" in config:
            self.output_amplitude.setChecked(config["output_amplitude"])
        if "output_phase" in config:
            self.output_phase.setChecked(config["output_phase"])
        if "save_dir" in config:
            self.save_dir_edit.setText(config["save_dir"])
