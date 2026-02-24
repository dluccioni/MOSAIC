# -----------------------------------------------------------------------------
# Detector View
# -----------------------------------------------------------------------------
"""
2D detector image display widget with full plotting controls.

Provides:
- Image display with colormap
- Zoom and pan controls
- Intensity/Amplitude/Phase mode switching
- Colorbar
- Pixel value inspection
- ROI selection
- Full controls for Detector plotting API (plot_detector, plot_detector_angles,
  plot_detector_position)
"""

import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from PySide6.QtCore import Qt, Signal, QPoint, QRectF
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QCheckBox,
    QPushButton,
    QFrame,
    QSlider,
    QDoubleSpinBox,
    QSpinBox,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QSizePolicy,
    QGroupBox,
    QFormLayout,
    QLineEdit,
    QTabWidget,
    QScrollArea,
    QMessageBox,
)
from PySide6.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QColor,
    QWheelEvent,
    QMouseEvent,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gui.state import SimulationState

# Try to import matplotlib for colormap
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, LogNorm
    import matplotlib.cm as cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class DetectorImageView(QGraphicsView):
    """
    Custom graphics view for detector image with zoom/pan.
    """

    pixel_hovered = Signal(int, int, float)  # x, y, value
    zoom_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._zoom = 1.0
        self._data = None
        self._colormap = 'viridis'
        self._log_scale = False
        self._vmin = None
        self._vmax = None
        self._extent = None  # [xmin, xmax, ymin, ymax] for coordinate mapping
        self._xlabel = None  # Label for x-axis (hover display)
        self._ylabel = None  # Label for y-axis (hover display)

        # Setup scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        # View settings
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)

        # Enable mouse tracking for hover
        self.setMouseTracking(True)

    def set_data(self, data: np.ndarray, colormap: str = 'viridis',
                 log_scale: bool = False, vmin: float = None, vmax: float = None,
                 extent: list = None, xlabel: str = None, ylabel: str = None):
        """
        Set the detector data to display.

        Args:
            data: 2D numpy array of detector values
            colormap: Matplotlib colormap name
            log_scale: Whether to use logarithmic scaling
            vmin: Minimum value for colormap
            vmax: Maximum value for colormap
            extent: [xmin, xmax, ymin, ymax] for coordinate mapping (hover display)
            xlabel: Label for x-axis (used in hover display)
            ylabel: Label for y-axis (used in hover display)
        """
        self._data = data
        self._colormap = colormap
        self._log_scale = log_scale
        self._vmin = vmin
        self._vmax = vmax
        self._extent = extent
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._update_image()

    def _update_image(self):
        """Update the displayed image."""
        if self._data is None:
            return

        data = self._data.copy()

        # Handle complex data
        if np.iscomplexobj(data):
            data = np.abs(data)

        # Determine min/max
        vmin = self._vmin if self._vmin is not None else np.nanmin(data)
        vmax = self._vmax if self._vmax is not None else np.nanmax(data)

        # Handle log scale
        if self._log_scale:
            data = np.where(data > 0, data, np.nan)
            if vmin <= 0:
                vmin = np.nanmin(data[data > 0]) if np.any(data > 0) else 1e-10
            data = np.log10(data)
            vmin = np.log10(vmin)
            vmax = np.log10(vmax) if vmax > 0 else 0

        # Normalize
        data = np.nan_to_num(data, nan=vmin)
        if vmax > vmin:
            normalized = (data - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(data)
        normalized = np.clip(normalized, 0, 1)

        # Apply colormap
        if MATPLOTLIB_AVAILABLE:
            cmap = cm.get_cmap(self._colormap)
            rgba = cmap(normalized)
            rgba = (rgba * 255).astype(np.uint8)
        else:
            # Grayscale fallback
            gray = (normalized * 255).astype(np.uint8)
            rgba = np.stack([gray, gray, gray, np.full_like(gray, 255)], axis=-1)

        # Create QImage
        height, width = rgba.shape[:2]
        bytes_per_line = 4 * width
        qimage = QImage(rgba.data, width, height, bytes_per_line, QImage.Format_RGBA8888)

        # Set pixmap
        pixmap = QPixmap.fromImage(qimage.copy())
        self.pixmap_item.setPixmap(pixmap)

        # Fit in view if first time
        if self._zoom == 1.0:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent):
        """Handle zoom with mouse wheel."""
        factor = 1.15 if event.angleDelta().y() > 0 else 1/1.15
        self._zoom *= factor
        self._zoom = max(0.1, min(10.0, self._zoom))
        self.scale(factor, factor)
        self.zoom_changed.emit(self._zoom)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for pixel inspection."""
        super().mouseMoveEvent(event)

        if self._data is None:
            return

        # Map to scene coordinates
        scene_pos = self.mapToScene(event.pos())
        x, y = int(scene_pos.x()), int(scene_pos.y())

        # Check bounds
        if 0 <= x < self._data.shape[1] and 0 <= y < self._data.shape[0]:
            value = self._data[y, x]
            if np.iscomplexobj(value):
                value = np.abs(value)
            self.pixel_hovered.emit(x, y, float(value))

    def fit_view(self):
        """Fit image to view."""
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self._zoom = 1.0

    def reset_zoom(self):
        """Reset zoom to 100%."""
        self.resetTransform()
        self._zoom = 1.0


class DetectorView(QWidget):
    """
    Complete detector view widget with full plotting controls.

    Provides controls for all Detector plotting API functions:
    - plot_detector(): 2D image in pixel space
    - plot_detector_angles(): Scatter plot in angle space
    - plot_detector_position(): 3D position visualization

    Signals:
        display_type_changed: Emitted when display type changes
        colormap_changed: Emitted when colormap changes
    """

    display_type_changed = Signal(str)
    colormap_changed = Signal(str)

    def __init__(self, state: SimulationState, parent=None):
        """
        Initialize the detector view.

        Args:
            state: SimulationState instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.state = state
        self._current_type = "Intensity"
        self._setup_ui()
        self._register_observers()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Main content with tabs
        self.tabs = QTabWidget()

        # Tab 1: Preview (embedded Qt view)
        preview_widget = self._create_preview_tab()
        self.tabs.addTab(preview_widget, "Preview")

        # Tab 2: Plot Controls
        plot_controls = self._create_plot_controls_tab()
        self.tabs.addTab(plot_controls, "Plot Controls")

        layout.addWidget(self.tabs)

        # Status bar (always visible)
        status_bar = QFrame()
        status_bar.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border-top: 1px solid #404040;
            }
        """)
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(8, 4, 8, 4)

        self.pixel_info = QLabel("Pixel: --, --  Value: --")
        self.pixel_info.setStyleSheet("color: #a0a0a0;")
        status_layout.addWidget(self.pixel_info)

        status_layout.addStretch()

        self.data_info = QLabel("No data")
        self.data_info.setStyleSheet("color: #808080;")
        status_layout.addWidget(self.data_info)

        layout.addWidget(status_bar)

    def _create_preview_tab(self) -> QWidget:
        """Create the preview tab with embedded Qt view."""
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

        # Display type
        toolbar_layout.addWidget(QLabel("Display:"))
        self.type_combo = QComboBox()
        self.type_combo.addItem("Intensity", "Intensity")
        self.type_combo.addItem("Amplitude", "Amplitude")
        self.type_combo.addItem("Phase", "Phase")
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        toolbar_layout.addWidget(self.type_combo)

        # Colormap
        toolbar_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        colormaps = ["viridis", "plasma", "inferno", "magma", "gray", "gist_gray", "hot", "jet", "turbo"]
        for cmap in colormaps:
            self.colormap_combo.addItem(cmap, cmap)
        self.colormap_combo.currentIndexChanged.connect(self._on_colormap_changed)
        toolbar_layout.addWidget(self.colormap_combo)

        # Log scale
        self.log_check = QCheckBox("Log")
        self.log_check.stateChanged.connect(self._on_display_options_changed)
        toolbar_layout.addWidget(self.log_check)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #404040;")
        toolbar_layout.addWidget(sep)

        # Display mode (spatial vs angular)
        toolbar_layout.addWidget(QLabel("Coords:"))
        self.display_mode = QComboBox()
        self.display_mode.addItem("Spatial (Y, Z)", "spatial")
        self.display_mode.addItem("Angular (η, 2θ)", "angular")
        self.display_mode.setToolTip(
            "Spatial: Physical coordinates in Angstroms\n"
            "Angular: Diffraction angles η (azimuthal) and 2θ (scattering)"
        )
        self.display_mode.currentIndexChanged.connect(self._on_display_mode_changed)
        toolbar_layout.addWidget(self.display_mode)

        # Degrees/Radians toggle for angular mode
        self.angular_degrees = QCheckBox("Degrees")
        self.angular_degrees.setChecked(True)
        self.angular_degrees.setToolTip("Show angles in degrees (unchecked = radians)")
        self.angular_degrees.stateChanged.connect(self._on_display_options_changed)
        self.angular_degrees.setVisible(False)  # Hidden until angular mode selected
        toolbar_layout.addWidget(self.angular_degrees)

        toolbar_layout.addStretch()

        # Zoom info
        self.zoom_label = QLabel("100%")
        self.zoom_label.setStyleSheet("color: #808080;")
        toolbar_layout.addWidget(self.zoom_label)

        # Fit button
        fit_btn = QPushButton("Fit")
        fit_btn.clicked.connect(self._on_fit_clicked)
        toolbar_layout.addWidget(fit_btn)

        # Save button
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._on_save_clicked)
        toolbar_layout.addWidget(save_btn)

        layout.addWidget(toolbar)

        # Image view
        self.image_view = DetectorImageView()
        self.image_view.pixel_hovered.connect(self._on_pixel_hovered)
        self.image_view.zoom_changed.connect(self._on_zoom_changed)
        layout.addWidget(self.image_view, 1)

        return widget

    def _create_plot_controls_tab(self) -> QWidget:
        """Create the plot controls tab with matplotlib plotting options."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ---- Common Plot Settings ----
        common_group = QGroupBox("Common Settings")
        common_layout = QFormLayout(common_group)

        # Plot type (Intensity/Amplitude/Phase)
        self.plot_type = QComboBox()
        self.plot_type.addItem("Intensity", "Intensity")
        self.plot_type.addItem("Amplitude", "Amplitude")
        self.plot_type.addItem("Phase", "Phase")
        common_layout.addRow("Data Type:", self.plot_type)

        # Scaling
        self.plot_scaling = QComboBox()
        self.plot_scaling.addItem("Linear", "linear")
        self.plot_scaling.addItem("Logarithmic", "log")
        common_layout.addRow("Scaling:", self.plot_scaling)

        # Colormap
        self.plot_colormap = QComboBox()
        colormaps = ["gist_gray", "viridis", "plasma", "inferno", "magma", "gray", "hot", "jet", "turbo", "coolwarm"]
        for cmap in colormaps:
            self.plot_colormap.addItem(cmap, cmap)
        common_layout.addRow("Colormap:", self.plot_colormap)

        # Title
        self.plot_title = QLineEdit()
        self.plot_title.setPlaceholderText("Leave blank for auto title")
        common_layout.addRow("Title:", self.plot_title)

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
        common_layout.addRow("Figure Size:", figsize_widget)

        layout.addWidget(common_group)

        # ---- Color Limits ----
        limits_group = QGroupBox("Value Limits")
        limits_layout = QFormLayout(limits_group)

        # Auto limits checkbox
        self.auto_limits = QCheckBox("Auto")
        self.auto_limits.setChecked(True)
        self.auto_limits.toggled.connect(self._on_auto_limits_toggled)
        limits_layout.addRow("", self.auto_limits)

        # vmin
        self.plot_vmin = QDoubleSpinBox()
        self.plot_vmin.setDecimals(6)
        self.plot_vmin.setRange(-1e20, 1e20)
        self.plot_vmin.setValue(0)
        self.plot_vmin.setEnabled(False)
        limits_layout.addRow("Min (vmin):", self.plot_vmin)

        # vmax
        self.plot_vmax = QDoubleSpinBox()
        self.plot_vmax.setDecimals(6)
        self.plot_vmax.setRange(-1e20, 1e20)
        self.plot_vmax.setValue(1)
        self.plot_vmax.setEnabled(False)
        limits_layout.addRow("Max (vmax):", self.plot_vmax)

        layout.addWidget(limits_group)

        # ---- Axis Limits ----
        axis_group = QGroupBox("Axis Limits")
        axis_layout = QFormLayout(axis_group)

        # Auto axis limits checkbox
        self.auto_axis = QCheckBox("Auto")
        self.auto_axis.setChecked(True)
        self.auto_axis.toggled.connect(self._on_auto_axis_toggled)
        axis_layout.addRow("", self.auto_axis)

        # X limits
        xlim_widget = QWidget()
        xlim_layout = QHBoxLayout(xlim_widget)
        xlim_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_xmin = QDoubleSpinBox()
        self.plot_xmin.setDecimals(2)
        self.plot_xmin.setRange(-1e12, 1e12)
        self.plot_xmin.setEnabled(False)
        xlim_layout.addWidget(self.plot_xmin)
        xlim_layout.addWidget(QLabel("to"))
        self.plot_xmax = QDoubleSpinBox()
        self.plot_xmax.setDecimals(2)
        self.plot_xmax.setRange(-1e12, 1e12)
        self.plot_xmax.setEnabled(False)
        xlim_layout.addWidget(self.plot_xmax)
        axis_layout.addRow("X Limits:", xlim_widget)

        # Y limits
        ylim_widget = QWidget()
        ylim_layout = QHBoxLayout(ylim_widget)
        ylim_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_ymin = QDoubleSpinBox()
        self.plot_ymin.setDecimals(2)
        self.plot_ymin.setRange(-1e12, 1e12)
        self.plot_ymin.setEnabled(False)
        ylim_layout.addWidget(self.plot_ymin)
        ylim_layout.addWidget(QLabel("to"))
        self.plot_ymax = QDoubleSpinBox()
        self.plot_ymax.setDecimals(2)
        self.plot_ymax.setRange(-1e12, 1e12)
        self.plot_ymax.setEnabled(False)
        ylim_layout.addWidget(self.plot_ymax)
        axis_layout.addRow("Y Limits:", ylim_widget)

        layout.addWidget(axis_group)

        # ---- plot_detector (2D Image) ----
        detector_group = QGroupBox("2D Detector Image (plot_detector)")
        detector_layout = QVBoxLayout(detector_group)

        detector_desc = QLabel("Displays pixel values on the detector plane using imshow.")
        detector_desc.setStyleSheet("color: #808080; font-style: italic;")
        detector_desc.setWordWrap(True)
        detector_layout.addWidget(detector_desc)

        plot_detector_btn = QPushButton("Generate 2D Plot")
        plot_detector_btn.setStyleSheet("QPushButton { background-color: #2a5a2a; padding: 8px; }")
        plot_detector_btn.clicked.connect(self._on_plot_detector)
        detector_layout.addWidget(plot_detector_btn)

        layout.addWidget(detector_group)

        # ---- plot_detector_angles (Angular Scatter) ----
        angles_group = QGroupBox("Angular Space Plot (plot_detector_angles)")
        angles_layout = QFormLayout(angles_group)

        angles_desc = QLabel("Scatter plot of pixel values in η and 2θ diffraction-angle space.")
        angles_desc.setStyleSheet("color: #808080; font-style: italic;")
        angles_desc.setWordWrap(True)
        angles_layout.addRow(angles_desc)

        # Degrees checkbox
        self.angles_degrees = QCheckBox("Use degrees")
        self.angles_degrees.setChecked(True)
        angles_layout.addRow("Units:", self.angles_degrees)

        # Marker size
        self.marker_size = QDoubleSpinBox()
        self.marker_size.setDecimals(1)
        self.marker_size.setRange(0.1, 50)
        self.marker_size.setValue(2)
        angles_layout.addRow("Marker Size:", self.marker_size)

        plot_angles_btn = QPushButton("Generate Angular Plot")
        plot_angles_btn.setStyleSheet("QPushButton { background-color: #2a5a2a; padding: 8px; }")
        plot_angles_btn.clicked.connect(self._on_plot_detector_angles)
        angles_layout.addRow(plot_angles_btn)

        layout.addWidget(angles_group)

        # ---- plot_detector_position (3D Position) ----
        position_group = QGroupBox("3D Position Plot (plot_detector_position)")
        position_layout = QFormLayout(position_group)

        position_desc = QLabel("3D scatter plot showing detector center and pixel positions in Cartesian space.")
        position_desc.setStyleSheet("color: #808080; font-style: italic;")
        position_desc.setWordWrap(True)
        position_layout.addRow(position_desc)

        # Elevation
        self.pos_elev = QDoubleSpinBox()
        self.pos_elev.setDecimals(1)
        self.pos_elev.setRange(-90, 90)
        self.pos_elev.setValue(0)
        self.pos_elev.setSuffix("°")
        position_layout.addRow("Elevation:", self.pos_elev)

        # Azimuth
        self.pos_azim = QDoubleSpinBox()
        self.pos_azim.setDecimals(1)
        self.pos_azim.setRange(-180, 180)
        self.pos_azim.setValue(90)
        self.pos_azim.setSuffix("°")
        position_layout.addRow("Azimuth:", self.pos_azim)

        # Position plot title
        self.pos_title = QLineEdit()
        self.pos_title.setPlaceholderText("Detector Position")
        position_layout.addRow("Title:", self.pos_title)

        plot_position_btn = QPushButton("Generate 3D Position Plot")
        plot_position_btn.setStyleSheet("QPushButton { background-color: #2a5a2a; padding: 8px; }")
        plot_position_btn.clicked.connect(self._on_plot_detector_position)
        position_layout.addRow(plot_position_btn)

        layout.addWidget(position_group)

        # Spacer
        layout.addStretch()

        scroll.setWidget(widget)
        return scroll

    def _on_auto_limits_toggled(self, checked: bool):
        """Handle auto limits toggle."""
        self.plot_vmin.setEnabled(not checked)
        self.plot_vmax.setEnabled(not checked)

    def _on_auto_axis_toggled(self, checked: bool):
        """Handle auto axis toggle."""
        self.plot_xmin.setEnabled(not checked)
        self.plot_xmax.setEnabled(not checked)
        self.plot_ymin.setEnabled(not checked)
        self.plot_ymax.setEnabled(not checked)

    def _get_plot_params(self) -> dict:
        """Get common plot parameters from UI."""
        params = {
            'type': self.plot_type.currentData(),
            'scaling': self.plot_scaling.currentData(),
            'cmap': self.plot_colormap.currentData(),
            'figsize': (self.figsize_w.value(), self.figsize_h.value()),
        }

        # Title (None for auto)
        title = self.plot_title.text().strip()
        params['title'] = title if title else None

        # Value limits
        if not self.auto_limits.isChecked():
            params['vmin'] = self.plot_vmin.value()
            params['vmax'] = self.plot_vmax.value()
        else:
            params['vmin'] = None
            params['vmax'] = None

        # Axis limits
        if not self.auto_axis.isChecked():
            params['xlim'] = (self.plot_xmin.value(), self.plot_xmax.value())
            params['ylim'] = (self.plot_ymin.value(), self.plot_ymax.value())
        else:
            params['xlim'] = None
            params['ylim'] = None

        return params

    def _on_plot_detector(self):
        """Generate 2D detector plot using plot_detector()."""
        detector = self.state.detector
        if detector is None:
            QMessageBox.warning(self, "No Detector", "No detector object available.")
            return

        # Check if pixel values exist
        if not hasattr(detector, '_pixel_values') or detector._pixel_values is None:
            QMessageBox.warning(self, "No Data", "No pixel data available. Run a simulation first.")
            return

        try:
            params = self._get_plot_params()

            fig, ax = detector.plot_detector(
                type=params['type'],
                title=params['title'],
                scaling=params['scaling'],
                vmin=params['vmin'],
                vmax=params['vmax'],
                xlim=params['xlim'],
                ylim=params['ylim'],
                figsize=params['figsize'],
                cmap=params['cmap']
            )
            plt.show()

        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Failed to generate plot:\n{str(e)}")

    def _on_plot_detector_angles(self):
        """Generate angular scatter plot using plot_detector_angles()."""
        detector = self.state.detector
        if detector is None:
            QMessageBox.warning(self, "No Detector", "No detector object available.")
            return

        # Check if pixel values exist
        if not hasattr(detector, '_pixel_values') or detector._pixel_values is None:
            QMessageBox.warning(self, "No Data", "No pixel data available. Run a simulation first.")
            return

        # Check if pixel coordinates exist (required for angular plot)
        if not hasattr(detector, '_pixel_coordinates') or detector._pixel_coordinates is None:
            QMessageBox.warning(self, "No Coordinates", "Detector pixel coordinates not initialized. Create/position detector first.")
            return

        try:
            params = self._get_plot_params()

            fig, ax = detector.plot_detector_angles(
                type=params['type'],
                title=params['title'],
                scaling=params['scaling'],
                degrees=self.angles_degrees.isChecked(),
                figsize=params['figsize'],
                cmap=params['cmap'],
                vmin=params['vmin'],
                vmax=params['vmax'],
                xlim=params['xlim'],
                ylim=params['ylim'],
                marker_size=self.marker_size.value()
            )
            plt.show()

        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Failed to generate plot:\n{str(e)}")

    def _on_plot_detector_position(self):
        """Generate 3D position plot using plot_detector_position()."""
        detector = self.state.detector
        if detector is None:
            QMessageBox.warning(self, "No Detector", "No detector object available.")
            return

        # Check if pixel coordinates exist
        if not hasattr(detector, '_pixel_coordinates') or detector._pixel_coordinates is None:
            QMessageBox.warning(self, "No Data", "Detector has no pixel coordinates. Create detector first.")
            return

        try:
            # Get title
            title = self.pos_title.text().strip()
            if not title:
                title = "Detector Position"

            fig, ax = detector.plot_detector_position(
                elev=self.pos_elev.value(),
                azim=self.pos_azim.value(),
                figsize=(self.figsize_w.value(), self.figsize_h.value()),
                title=title
            )
            plt.show()

        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Failed to generate plot:\n{str(e)}")

    def _register_observers(self):
        """Register for state change notifications."""
        self.state.register_observer("detector_changed", self._on_detector_changed)

    def _on_detector_changed(self, detector):
        """Handle detector state change."""
        self.refresh()

    def refresh(self):
        """Refresh the display from current detector state."""
        detector = self.state.detector
        if detector is None:
            self.data_info.setText("No detector")
            return

        # Get appropriate data
        data = None
        display_type = self.type_combo.currentData()

        # Check _pixel_values first to avoid triggering property getters that compute on None
        has_pixel_data = hasattr(detector, '_pixel_values') and detector._pixel_values is not None

        if has_pixel_data:
            if display_type == "Intensity":
                # Try computed property first, fall back to manual calculation
                try:
                    data = detector.pixel_intensity
                except (TypeError, AttributeError):
                    data = detector._pixel_values
                    data = np.abs(data) ** 2 if np.iscomplexobj(data) else data
            elif display_type == "Amplitude":
                try:
                    data = detector.pixel_amplitude
                except (TypeError, AttributeError):
                    data = detector._pixel_values
                    data = np.abs(data)
            elif display_type == "Phase":
                try:
                    data = detector.pixel_phase
                except (TypeError, AttributeError):
                    data = detector._pixel_values
                    data = np.angle(data) if np.iscomplexobj(data) else np.zeros_like(data)
            else:
                data = detector._pixel_values

        if data is None:
            self.data_info.setText("No data")
            return

        # Get display mode
        display_mode = self.display_mode.currentData()

        # Prepare data based on display mode
        if display_mode == 'angular':
            # Transform to angular coordinates
            data_2d, extent, xlabel, ylabel = self._prepare_angular_display(detector, data)
        else:
            # Spatial mode: reshape to 2D grid in physical coordinates
            data_2d = data.reshape(detector.shape) if len(data.shape) == 1 else data

            # Set extent in Angstroms (physical coordinates)
            pixel_size = getattr(detector, 'pixel_size', None)
            shape = detector.shape
            if pixel_size is not None:
                half_y = shape[0] * pixel_size[0] / 2
                half_z = shape[1] * pixel_size[1] / 2
                extent = [-half_y, half_y, -half_z, half_z]
            else:
                extent = None
            xlabel = "Y (Å)"
            ylabel = "Z (Å)"

        # Update display
        self.image_view.set_data(
            data_2d,
            colormap=self.colormap_combo.currentData(),
            log_scale=self.log_check.isChecked(),
            extent=extent,
            xlabel=xlabel,
            ylabel=ylabel
        )

        # Update info
        self.data_info.setText(
            f"Shape: {data_2d.shape[0]}x{data_2d.shape[1]}  "
            f"Min: {np.nanmin(data_2d):.2e}  Max: {np.nanmax(data_2d):.2e}"
        )

    def _prepare_angular_display(self, detector, data):
        """Prepare data for angular (η, 2θ) display.

        Args:
            detector: The detector object
            data: 1D or 2D array of pixel values

        Returns:
            tuple: (data_2d, extent, xlabel, ylabel)
        """
        # Get pixel coordinates
        coords = getattr(detector, 'pixel_coordinates', None)
        if coords is None:
            # Fallback to spatial if no coordinates
            return data.reshape(detector.shape), None, "Y (Å)", "Z (Å)"

        x, y, z = coords[0], coords[1], coords[2]

        # Calculate angular coordinates
        two_theta = np.arctan2(np.sqrt(y**2 + z**2), x)
        eta = np.arctan2(y, z)

        # Convert to degrees if checkbox is checked
        if self.angular_degrees.isChecked():
            two_theta = np.rad2deg(two_theta)
            eta = np.rad2deg(eta)
            xlabel = "η (°)"
            ylabel = "2θ (°)"
        else:
            xlabel = "η (rad)"
            ylabel = "2θ (rad)"

        # Check if detector forms a regular grid in angular space
        geometry = getattr(detector, '_geometry', 'rectangular')
        construction_mode = getattr(detector, '_construction_mode', 'plane')
        input_mode = getattr(detector, '_input_mode', 'spatial')

        is_regular_angular_grid = (
            (geometry == 'ring' and construction_mode == 'shell') or
            input_mode == 'angular'
        )

        if is_regular_angular_grid:
            # Direct reshape - pixels form a regular angular grid
            data_2d = data.reshape(detector.shape)
            eta_min, eta_max = eta.min(), eta.max()
            two_theta_min, two_theta_max = two_theta.min(), two_theta.max()
            extent = [eta_min, eta_max, two_theta_min, two_theta_max]
        else:
            # Need interpolation for non-regular grids
            data_2d, extent = self._interpolate_angular_data(
                eta, two_theta, data.ravel(), detector.shape
            )

        return data_2d, extent, xlabel, ylabel

    def _interpolate_angular_data(self, eta, two_theta, data, shape, resolution=256):
        """Interpolate non-regular grid data onto regular angular grid.

        Args:
            eta: Azimuthal angles (1D array)
            two_theta: Scattering angles (1D array)
            data: Pixel values (1D array)
            shape: Original detector shape
            resolution: Output grid resolution

        Returns:
            tuple: (data_2d, extent)
        """
        # Flatten arrays
        eta_flat = eta.ravel()
        two_theta_flat = two_theta.ravel()
        data_flat = data.ravel()

        # Determine grid bounds with padding
        eta_min, eta_max = eta_flat.min(), eta_flat.max()
        two_theta_min, two_theta_max = two_theta_flat.min(), two_theta_flat.max()

        eta_padding = (eta_max - eta_min) * 0.02
        two_theta_padding = (two_theta_max - two_theta_min) * 0.02

        eta_min -= eta_padding
        eta_max += eta_padding
        two_theta_min -= two_theta_padding
        two_theta_max += two_theta_padding

        extent = [eta_min, eta_max, two_theta_min, two_theta_max]

        try:
            from scipy.interpolate import griddata

            # Create regular grid
            grid_eta, grid_two_theta = np.mgrid[
                eta_min:eta_max:complex(0, resolution),
                two_theta_min:two_theta_max:complex(0, resolution)
            ]

            # Interpolate
            points = np.column_stack((eta_flat, two_theta_flat))
            data_2d = griddata(
                points, data_flat,
                (grid_eta, grid_two_theta),
                method='linear'
            )

            return data_2d.T, extent  # Transpose to match extent axis order (eta=x, two_theta=y)

        except ImportError:
            # Fallback: bin data into grid (less accurate but no scipy needed)
            print("Warning: scipy not available, using binned approximation")

            # Create bins
            eta_bins = np.linspace(eta_min, eta_max, resolution + 1)
            two_theta_bins = np.linspace(two_theta_min, two_theta_max, resolution + 1)

            # Bin the data
            data_2d = np.zeros((resolution, resolution))
            counts = np.zeros((resolution, resolution))

            eta_idx = np.clip(
                np.digitize(eta_flat, eta_bins) - 1, 0, resolution - 1
            )
            two_theta_idx = np.clip(
                np.digitize(two_theta_flat, two_theta_bins) - 1, 0, resolution - 1
            )

            for i, (ei, ti) in enumerate(zip(eta_idx, two_theta_idx)):
                data_2d[ei, ti] += data_flat[i]
                counts[ei, ti] += 1

            # Average where counts > 0
            mask = counts > 0
            data_2d[mask] /= counts[mask]
            data_2d[~mask] = np.nan

            return data_2d.T, extent  # Transpose to match extent axis order

    def _on_type_changed(self, index):
        """Handle display type change."""
        self.refresh()
        self.display_type_changed.emit(self.type_combo.currentData())

    def _on_colormap_changed(self, index):
        """Handle colormap change."""
        self.refresh()
        self.colormap_changed.emit(self.colormap_combo.currentData())

    def _on_display_options_changed(self):
        """Handle log scale toggle."""
        self.refresh()

    def _on_display_mode_changed(self, index):
        """Handle display mode toggle between spatial and angular."""
        mode = self.display_mode.currentData()

        # Show/hide angular-specific controls
        self.angular_degrees.setVisible(mode == 'angular')

        # Refresh display with new mode
        self.refresh()

    def _on_pixel_hovered(self, x: int, y: int, value: float):
        """Handle pixel hover."""
        # Check if we have extent info for coordinate mapping
        extent = getattr(self.image_view, '_extent', None)
        xlabel = getattr(self.image_view, '_xlabel', None)
        ylabel = getattr(self.image_view, '_ylabel', None)
        data = getattr(self.image_view, '_data', None)

        if extent is not None and data is not None:
            # Map pixel to coordinate using extent [xmin, xmax, ymin, ymax]
            xmin, xmax, ymin, ymax = extent
            height, width = data.shape[:2]

            if width > 0 and height > 0:
                coord_x = xmin + (x / width) * (xmax - xmin)
                coord_y = ymin + (y / height) * (ymax - ymin)

                x_label = xlabel if xlabel else "X"
                y_label = ylabel if ylabel else "Y"
                self.pixel_info.setText(f"{x_label}: {coord_x:.2f}, {y_label}: {coord_y:.2f}  Value: {value:.4e}")
                return

        # Fallback to pixel coordinates
        self.pixel_info.setText(f"Pixel: {x}, {y}  Value: {value:.4e}")

    def _on_zoom_changed(self, zoom: float):
        """Handle zoom change."""
        self.zoom_label.setText(f"{int(zoom * 100)}%")

    def _on_fit_clicked(self):
        """Handle fit button click."""
        self.image_view.fit_view()
        self.zoom_label.setText("Fit")

    def _on_save_clicked(self):
        """Handle save button click."""
        from PySide6.QtWidgets import QFileDialog

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Detector Image",
            "",
            "PNG (*.png);;TIFF (*.tiff);;NumPy (*.npy)"
        )
        if filename:
            try:
                detector = self.state.detector
                # Check _pixel_values directly to avoid warning print when not initialized
                if detector is None or not hasattr(detector, '_pixel_values'):
                    return

                ext = Path(filename).suffix.lower()
                data = detector._pixel_values

                if data is None:
                    return

                if ext == '.npy':
                    np.save(filename, data)
                elif ext == '.png':
                    # Get the displayed pixmap
                    pixmap = self.image_view.pixmap_item.pixmap()
                    pixmap.save(filename, "PNG")
                elif ext in ['.tiff', '.tif']:
                    from PIL import Image
                    if np.iscomplexobj(data):
                        data = np.abs(data)
                    img = Image.fromarray(data.astype(np.float32))
                    img.save(filename)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def set_display_type(self, display_type: str):
        """Set the display type programmatically."""
        idx = self.type_combo.findData(display_type)
        if idx >= 0:
            self.type_combo.setCurrentIndex(idx)

    def set_colormap(self, colormap: str):
        """Set the colormap programmatically."""
        idx = self.colormap_combo.findData(colormap)
        if idx >= 0:
            self.colormap_combo.setCurrentIndex(idx)

    def set_log_scale(self, enabled: bool):
        """Set log scale programmatically."""
        self.log_check.setChecked(enabled)
