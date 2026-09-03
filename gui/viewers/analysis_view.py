# -----------------------------------------------------------------------------
# Analysis View
# -----------------------------------------------------------------------------
"""
Analysis view widget with integration controls and preview.

Provides:
- Preview tab for viewing integration results
- Controls for integrate_detector_along_axis() API
- Support for both cartesian and angular coordinate systems
- Embedded matplotlib visualization
"""

import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QCheckBox,
    QPushButton,
    QFrame,
    QDoubleSpinBox,
    QSpinBox,
    QGroupBox,
    QFormLayout,
    QLineEdit,
    QTabWidget,
    QScrollArea,
    QMessageBox,
    QSplitter,
)
from PySide6.QtGui import QFont

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gui.state import SimulationState

# Try to import matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class AnalysisPreviewWidget(QWidget):
    """
    Widget for displaying analysis results with embedded matplotlib canvas.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._bin_centers = None
        self._integrated_values = None

    def _setup_ui(self):
        """Setup the preview UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if MATPLOTLIB_AVAILABLE:
            # Create matplotlib figure and canvas
            self.figure = Figure(figsize=(8, 6), dpi=100)
            self.figure.patch.set_facecolor('#1e1e1e')
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setStyleSheet("background-color: #1e1e1e;")
            self.ax = self.figure.add_subplot(111)
            self._style_axes()
            layout.addWidget(self.canvas)
        else:
            # Fallback message
            label = QLabel("Matplotlib not available for preview")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("color: #808080; background-color: #1e1e1e;")
            layout.addWidget(label)

    def _style_axes(self):
        """Apply dark theme styling to axes."""
        self.ax.set_facecolor('#1e1e1e')
        self.ax.tick_params(colors='#a0a0a0')
        self.ax.xaxis.label.set_color('#a0a0a0')
        self.ax.yaxis.label.set_color('#a0a0a0')
        self.ax.title.set_color('#d0d0d0')
        for spine in self.ax.spines.values():
            spine.set_color('#404040')

    def plot_integration(self, bin_centers: np.ndarray, integrated_values: np.ndarray,
                         title: str = "Integrated Detector Data",
                         xlabel: str = "Position",
                         ylabel: str = "Integrated Value"):
        """
        Plot integration results.

        Args:
            bin_centers: X-axis values (bin centers)
            integrated_values: Y-axis values (integrated data)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        self._bin_centers = bin_centers
        self._integrated_values = integrated_values

        self.ax.clear()
        self._style_axes()

        self.ax.plot(bin_centers, integrated_values, color='#4ec9b0', linewidth=1.5)
        self.ax.set_title(title, fontsize=12)
        self.ax.set_xlabel(xlabel, fontsize=10)
        self.ax.set_ylabel(ylabel, fontsize=10)
        self.ax.grid(True, alpha=0.3, color='#404040')

        self.figure.tight_layout()
        self.canvas.draw()

    def clear(self):
        """Clear the plot."""
        if MATPLOTLIB_AVAILABLE:
            self.ax.clear()
            self._style_axes()
            self.ax.set_title("No data", fontsize=12)
            self.canvas.draw()
        self._bin_centers = None
        self._integrated_values = None


class AnalysisView(QWidget):
    """
    Complete analysis view widget with integration controls.

    Provides controls for Analysis.integrate_detector_along_axis():
    - Data type (Intensity/Amplitude/Phase)
    - Axis selection (x/y/z for cartesian, eta/2theta/distance for angular)
    - Coordinate system (cartesian/angular)
    - Binning and aggregation options
    - Plot customization

    Signals:
        integration_complete: Emitted when integration is complete
    """

    integration_complete = Signal(object, object)  # bin_centers, integrated_values

    def __init__(self, state: SimulationState, parent=None):
        """
        Initialize the analysis view.

        Args:
            state: SimulationState instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.state = state
        self._last_bin_centers = None
        self._last_integrated_values = None
        self._setup_ui()
        self._register_observers()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Main content with tabs
        self.tabs = QTabWidget()

        # Tab 1: Preview
        preview_widget = self._create_preview_tab()
        self.tabs.addTab(preview_widget, "Preview")

        # Tab 2: Integration Controls
        controls_widget = self._create_controls_tab()
        self.tabs.addTab(controls_widget, "Integration Controls")

        layout.addWidget(self.tabs)

        # Status bar
        status_bar = QFrame()
        status_bar.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border-top: 1px solid #404040;
            }
        """)
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(8, 4, 8, 4)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #a0a0a0;")
        status_layout.addWidget(self.status_label)

        status_layout.addStretch()

        self.data_info = QLabel("No data")
        self.data_info.setStyleSheet("color: #808080;")
        status_layout.addWidget(self.data_info)

        layout.addWidget(status_bar)

    def _create_preview_tab(self) -> QWidget:
        """Create the preview tab with embedded matplotlib view."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QFrame()
        toolbar.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border-bottom: 1px solid #404040;
            }
        """)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 4, 8, 4)
        toolbar_layout.setSpacing(8)

        toolbar_layout.addWidget(QLabel("Analysis Preview"))
        toolbar_layout.addStretch()

        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._on_clear_preview)
        toolbar_layout.addWidget(clear_btn)

        # Save button
        save_btn = QPushButton("Save Plot")
        save_btn.clicked.connect(self._on_save_plot)
        toolbar_layout.addWidget(save_btn)

        # External plot button
        external_btn = QPushButton("External Window")
        external_btn.clicked.connect(self._on_external_plot)
        toolbar_layout.addWidget(external_btn)

        layout.addWidget(toolbar)

        # Preview widget
        self.preview = AnalysisPreviewWidget()
        layout.addWidget(self.preview, 1)

        return widget

    def _create_controls_tab(self) -> QWidget:
        """Create the integration controls tab."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ---- Data Selection ----
        data_group = QGroupBox("Data Selection")
        data_layout = QFormLayout(data_group)

        # Data type
        self.data_type = QComboBox()
        self.data_type.addItem("Intensity", "Intensity")
        self.data_type.addItem("Amplitude", "Amplitude")
        self.data_type.addItem("Phase", "Phase")
        data_layout.addRow("Data Type:", self.data_type)

        layout.addWidget(data_group)

        # ---- Coordinate System ----
        coord_group = QGroupBox("Coordinate System")
        coord_layout = QFormLayout(coord_group)

        # System selection
        self.coord_system = QComboBox()
        self.coord_system.addItem("Cartesian (x, y, z)", "cartesian")
        self.coord_system.addItem("Angular (eta, 2theta, distance)", "angular")
        self.coord_system.currentIndexChanged.connect(self._on_coord_system_changed)
        coord_layout.addRow("System:", self.coord_system)

        # Axis selection (cartesian)
        self.axis_cartesian = QComboBox()
        self.axis_cartesian.addItem("X", "x")
        self.axis_cartesian.addItem("Y", "y")
        self.axis_cartesian.addItem("Z", "z")
        coord_layout.addRow("Axis (Cartesian):", self.axis_cartesian)

        # Axis selection (angular)
        self.axis_angular = QComboBox()
        self.axis_angular.addItem("Eta (η)", "eta")
        self.axis_angular.addItem("2-Theta (2θ)", "2theta")
        self.axis_angular.addItem("Distance", "distance")
        self.axis_angular.setVisible(False)
        coord_layout.addRow("Axis (Angular):", self.axis_angular)

        # Degrees checkbox (for angular)
        self.degrees_check = QCheckBox("Output in degrees")
        self.degrees_check.setChecked(True)
        self.degrees_check.setVisible(False)
        coord_layout.addRow("", self.degrees_check)

        layout.addWidget(coord_group)

        # ---- Integration Parameters ----
        integration_group = QGroupBox("Integration Parameters")
        integration_layout = QFormLayout(integration_group)

        # Number of bins
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(10, 10000)
        self.bins_spin.setValue(200)
        integration_layout.addRow("Bins:", self.bins_spin)

        # Aggregator
        self.aggregator = QComboBox()
        self.aggregator.addItem("Mean", "mean")
        self.aggregator.addItem("Sum", "sum")
        integration_layout.addRow("Aggregator:", self.aggregator)

        layout.addWidget(integration_group)

        # ---- Plot Settings ----
        plot_group = QGroupBox("Plot Settings")
        plot_layout = QFormLayout(plot_group)

        # Title
        self.plot_title = QLineEdit()
        self.plot_title.setPlaceholderText("Integrated Detector Data")
        plot_layout.addRow("Title:", self.plot_title)

        # X-axis label
        self.xlabel = QLineEdit()
        self.xlabel.setPlaceholderText("Auto (based on axis)")
        plot_layout.addRow("X Label:", self.xlabel)

        # Y-axis label
        self.ylabel = QLineEdit()
        self.ylabel.setText("Integrated Value")
        plot_layout.addRow("Y Label:", self.ylabel)

        # Figure size
        figsize_widget = QWidget()
        figsize_layout = QHBoxLayout(figsize_widget)
        figsize_layout.setContentsMargins(0, 0, 0, 0)
        self.figsize_w = QDoubleSpinBox()
        self.figsize_w.setRange(2, 20)
        self.figsize_w.setValue(8)
        self.figsize_w.setSuffix(" in")
        figsize_layout.addWidget(self.figsize_w)
        figsize_layout.addWidget(QLabel("x"))
        self.figsize_h = QDoubleSpinBox()
        self.figsize_h.setRange(2, 20)
        self.figsize_h.setValue(6)
        self.figsize_h.setSuffix(" in")
        figsize_layout.addWidget(self.figsize_h)
        plot_layout.addRow("Figure Size:", figsize_widget)

        layout.addWidget(plot_group)

        # ---- Action Buttons ----
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout(action_group)

        # Generate integration button (preview in embedded view)
        integrate_preview_btn = QPushButton("Integrate && Preview")
        integrate_preview_btn.setStyleSheet("QPushButton { background-color: #2a5a2a; padding: 8px; }")
        integrate_preview_btn.clicked.connect(self._on_integrate_preview)
        action_layout.addWidget(integrate_preview_btn)

        # Generate integration button (external matplotlib window)
        integrate_external_btn = QPushButton("Integrate && External Plot")
        integrate_external_btn.setStyleSheet("QPushButton { background-color: #3a4a6a; padding: 8px; }")
        integrate_external_btn.clicked.connect(self._on_integrate_external)
        action_layout.addWidget(integrate_external_btn)

        # Save integration data
        save_data_btn = QPushButton("Save Integration Data...")
        save_data_btn.clicked.connect(self._on_save_data)
        action_layout.addWidget(save_data_btn)

        layout.addWidget(action_group)

        # Spacer
        layout.addStretch()

        scroll.setWidget(widget)
        return scroll

    def _on_coord_system_changed(self, index: int):
        """Handle coordinate system change."""
        is_angular = self.coord_system.currentData() == "angular"
        self.axis_cartesian.setVisible(not is_angular)
        self.axis_angular.setVisible(is_angular)
        self.degrees_check.setVisible(is_angular)

        # Update labels in parent form layout
        # Find the form layout and update row labels
        for widget in [self.axis_cartesian, self.axis_angular]:
            parent = widget.parent()
            if parent:
                layout = parent.layout()
                if isinstance(layout, QFormLayout):
                    # Update visibility of the row based on coordinate system
                    pass

    def _get_current_axis(self) -> str:
        """Get the currently selected axis based on coordinate system."""
        if self.coord_system.currentData() == "cartesian":
            return self.axis_cartesian.currentData()
        else:
            return self.axis_angular.currentData()

    def _get_integration_params(self) -> dict:
        """Get integration parameters from UI."""
        params = {
            'data_type': self.data_type.currentData(),
            'axis': self._get_current_axis(),
            'system': self.coord_system.currentData(),
            'degrees': self.degrees_check.isChecked(),
            'bins': self.bins_spin.value(),
            'aggregator': self.aggregator.currentData(),
            'figsize': (self.figsize_w.value(), self.figsize_h.value()),
        }

        # Title
        title = self.plot_title.text().strip()
        params['title'] = title if title else "Integrated Detector Data"

        # X label (None for auto)
        xlabel = self.xlabel.text().strip()
        params['xlabel'] = xlabel if xlabel else None

        # Y label
        ylabel = self.ylabel.text().strip()
        params['ylabel'] = ylabel if ylabel else "Integrated Value"

        return params

    def _check_prerequisites(self) -> bool:
        """Check if analysis can be performed."""
        # Check for analysis object
        if self.state.analysis is None:
            QMessageBox.warning(
                self, "No Analysis",
                "No analysis object available. Create one in the Object Browser first."
            )
            return False

        # Check for detector
        if self.state.detector is None:
            QMessageBox.warning(
                self, "No Detector",
                "No detector object available. Create and run a simulation first."
            )
            return False

        # Check for pixel values
        if not hasattr(self.state.detector, '_pixel_values') or self.state.detector._pixel_values is None:
            QMessageBox.warning(
                self, "No Data",
                "No pixel data available on detector. Run a simulation first."
            )
            return False

        # Check for pixel coordinates (required for integration)
        if not hasattr(self.state.detector, '_pixel_coordinates') or self.state.detector._pixel_coordinates is None:
            QMessageBox.warning(
                self, "No Coordinates",
                "Detector pixel coordinates not initialized. Position the detector first."
            )
            return False

        return True

    def _on_integrate_preview(self):
        """Perform integration and show in preview tab."""
        if not self._check_prerequisites():
            return

        try:
            params = self._get_integration_params()
            self.status_label.setText("Integrating...")

            # Perform integration without showing external plot
            bin_centers, integrated_values = self.state.analysis.integrate_detector_along_axis(
                detector=self.state.detector,
                data_type=params['data_type'],
                axis=params['axis'],
                system=params['system'],
                degrees=params['degrees'],
                bins=params['bins'],
                aggregator=params['aggregator'],
                plot=False,  # Don't show external plot
                save_plot=False,
                title=params['title'],
                xlabel=params['xlabel'],
                ylabel=params['ylabel'],
                figsize=params['figsize']
            )

            # Store results
            self._last_bin_centers = bin_centers
            self._last_integrated_values = integrated_values

            # Determine xlabel if auto
            xlabel = params['xlabel'] or params['axis']
            if params['system'] == 'angular' and params['degrees']:
                if params['axis'] in ['eta', '2theta']:
                    xlabel = f"{params['axis']} (degrees)"

            # Update preview
            self.preview.plot_integration(
                bin_centers, integrated_values,
                title=params['title'],
                xlabel=xlabel,
                ylabel=params['ylabel']
            )

            # Update status
            self.status_label.setText("Integration complete")
            self.data_info.setText(f"Bins: {len(bin_centers)}  Range: [{bin_centers.min():.2e}, {bin_centers.max():.2e}]")

            # Switch to preview tab
            self.tabs.setCurrentIndex(0)

            # Emit signal
            self.integration_complete.emit(bin_centers, integrated_values)

        except Exception as e:
            QMessageBox.critical(self, "Integration Error", f"Failed to integrate:\n{str(e)}")
            self.status_label.setText("Integration failed")

    def _on_integrate_external(self):
        """Perform integration and show in external matplotlib window."""
        if not self._check_prerequisites():
            return

        try:
            params = self._get_integration_params()
            self.status_label.setText("Integrating...")

            # Perform integration with external plot; the returned figure is
            # shown below and owned by this caller
            bin_centers, integrated_values, fig, ax = self.state.analysis.integrate_detector_along_axis(
                detector=self.state.detector,
                data_type=params['data_type'],
                axis=params['axis'],
                system=params['system'],
                degrees=params['degrees'],
                bins=params['bins'],
                aggregator=params['aggregator'],
                plot=True,  # Show external plot
                save_plot=False,
                title=params['title'],
                xlabel=params['xlabel'],
                ylabel=params['ylabel'],
                figsize=params['figsize'],
                return_figure=True
            )

            # Store results
            self._last_bin_centers = bin_centers
            self._last_integrated_values = integrated_values

            # Show the plot
            plt.show()

            # Update status
            self.status_label.setText("Integration complete")
            self.data_info.setText(f"Bins: {len(bin_centers)}")

            # Emit signal
            self.integration_complete.emit(bin_centers, integrated_values)

        except Exception as e:
            QMessageBox.critical(self, "Integration Error", f"Failed to integrate:\n{str(e)}")
            self.status_label.setText("Integration failed")

    def _on_clear_preview(self):
        """Clear the preview."""
        self.preview.clear()
        self._last_bin_centers = None
        self._last_integrated_values = None
        self.status_label.setText("Preview cleared")
        self.data_info.setText("No data")

    def _on_save_plot(self):
        """Save the preview plot to file."""
        if not MATPLOTLIB_AVAILABLE:
            return

        if self._last_bin_centers is None:
            QMessageBox.warning(self, "No Data", "No integration data to save. Run integration first.")
            return

        from PySide6.QtWidgets import QFileDialog

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Analysis Plot",
            "",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)"
        )

        if filename:
            try:
                self.preview.figure.savefig(filename, facecolor='#1e1e1e', edgecolor='none', dpi=150)
                self.status_label.setText(f"Saved: {Path(filename).name}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save plot:\n{str(e)}")

    def _on_external_plot(self):
        """Show current data in external matplotlib window."""
        if self._last_bin_centers is None or self._last_integrated_values is None:
            QMessageBox.warning(self, "No Data", "No integration data available. Run integration first.")
            return

        try:
            params = self._get_integration_params()
            xlabel = params['xlabel'] or params['axis']

            fig, ax = plt.subplots(figsize=params['figsize'])
            ax.plot(self._last_bin_centers, self._last_integrated_values)
            ax.set_title(params['title'])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(params['ylabel'])
            ax.grid(True, alpha=0.3)
            plt.show()

        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Failed to create plot:\n{str(e)}")

    def _on_save_data(self):
        """Save integration data to file."""
        if self._last_bin_centers is None or self._last_integrated_values is None:
            QMessageBox.warning(self, "No Data", "No integration data to save. Run integration first.")
            return

        from PySide6.QtWidgets import QFileDialog

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Integration Data",
            "",
            "NumPy (*.npz);;CSV (*.csv);;Text (*.txt)"
        )

        if filename:
            try:
                ext = Path(filename).suffix.lower()
                if ext == '.npz':
                    np.savez(filename,
                             bin_centers=self._last_bin_centers,
                             integrated_values=self._last_integrated_values)
                elif ext == '.csv':
                    data = np.column_stack((self._last_bin_centers, self._last_integrated_values))
                    np.savetxt(filename, data, delimiter=',', header='bin_center,integrated_value', comments='')
                else:
                    data = np.column_stack((self._last_bin_centers, self._last_integrated_values))
                    np.savetxt(filename, data, header='bin_center integrated_value')

                self.status_label.setText(f"Data saved: {Path(filename).name}")

            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save data:\n{str(e)}")

    def _register_observers(self):
        """Register for state change notifications."""
        self.state.register_observer("analysis_changed", self._on_analysis_changed)
        self.state.register_observer("detector_changed", self._on_detector_changed)

    def _on_analysis_changed(self, analysis):
        """Handle analysis state change."""
        if analysis is not None:
            self.status_label.setText("Analysis object available")
        else:
            self.status_label.setText("No analysis object")

    def _on_detector_changed(self, detector):
        """Handle detector state change."""
        if detector is not None:
            # Check if pixel data available
            if hasattr(detector, '_pixel_values') and detector._pixel_values is not None:
                self.data_info.setText(f"Detector ready: {detector.shape}")
            else:
                self.data_info.setText("Detector ready (no pixel data)")
        else:
            self.data_info.setText("No detector")

    def refresh(self):
        """Refresh the view from current state."""
        detector = self.state.detector
        analysis = self.state.analysis

        status_parts = []
        if analysis is not None:
            status_parts.append("Analysis ready")
        else:
            status_parts.append("No analysis")

        if detector is not None:
            if hasattr(detector, '_pixel_values') and detector._pixel_values is not None:
                status_parts.append(f"Detector: {detector.shape}")
            else:
                status_parts.append("Detector (no data)")
        else:
            status_parts.append("No detector")

        self.status_label.setText(" | ".join(status_parts))
