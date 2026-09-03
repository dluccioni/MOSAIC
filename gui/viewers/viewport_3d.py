# -----------------------------------------------------------------------------
# 3D Viewport
# -----------------------------------------------------------------------------
"""
3D visualization viewport using VisPy for real-time experimental schematic.

Provides:
- Sample box visualization with stage rotations
- Detector plane at correct angles (2theta, eta)
- Beam visualization
- Optics components along beam path
- Uniform autoscaling (preserves relative sizes)
- Camera controls (rotate, pan, zoom)
- Coordinate axes
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QToolBar,
    QPushButton,
    QComboBox,
    QLabel,
    QFrame,
    QCheckBox,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gui.state import SimulationState
from gui.utils.diffraction_calc import DiffractionCalculator

# Try to import VisPy with PySide6 backend
VISPY_AVAILABLE = False
try:
    import vispy
    # Set the backend to PySide6 before any other vispy imports
    vispy.use('pyside6')
    from vispy import app, scene
    from vispy.scene import visuals
    from vispy.visuals.transforms import STTransform, MatrixTransform
    from vispy.geometry import create_box
    VISPY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: VisPy not available ({e}). 3D visualization will be limited.")
except RuntimeError as e:
    # Backend already set or other runtime issue
    try:
        from vispy import app, scene
        from vispy.scene import visuals
        from vispy.visuals.transforms import STTransform, MatrixTransform
        from vispy.geometry import create_box
        VISPY_AVAILABLE = True
    except ImportError:
        print(f"Warning: VisPy import failed ({e}). 3D visualization will be limited.")


class Viewport3D(QWidget):
    """
    3D viewport widget for visualizing the experimental setup as a schematic.

    Displays:
    - Sample bounding box (with stage rotations applied)
    - Detector plane at correct 2theta/eta position
    - Beam ray along X-axis
    - Optics components along beam path
    - Coordinate axes

    All objects are uniformly autoscaled to fit in the viewport while
    preserving their correct relative sizes and proportions.

    Signals:
        view_changed: Emitted when view changes (view_name)
        object_selected: Emitted when an object is clicked (object_type)
    """

    view_changed = Signal(str)
    object_selected = Signal(str)

    # Target viewport size in units - all objects are scaled to fit within this
    TARGET_VIEWPORT_SIZE = 40.0

    def __init__(self, state: SimulationState, parent=None):
        """
        Initialize the 3D viewport.

        Args:
            state: SimulationState instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.state = state
        self._scale_factor = 1.0  # Dynamic scale factor (uniform for all objects)
        self._last_max_dimension = 1.0  # Track for change detection
        self._visuals = {}  # Store visual objects
        self._vispy_active = False  # Track if VisPy is actually working

        # Diffraction peak visualization
        self._diff_calc = DiffractionCalculator()
        self._selected_peaks = []  # List of (h,k,l) tuples to display
        self._show_all_peaks = False  # Show all accessible peaks vs only selected

        self._setup_ui()
        self._register_observers()

        # Delayed refresh to catch initial state
        QTimer.singleShot(100, self.refresh)

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
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

        # View presets
        toolbar_layout.addWidget(QLabel("View:"))

        self.view_combo = QComboBox()
        self.view_combo.addItem("3D Perspective", "3d")
        self.view_combo.addItem("XY (Top)", "xy")
        self.view_combo.addItem("XZ (Front)", "xz")
        self.view_combo.addItem("YZ (Side)", "yz")
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        toolbar_layout.addWidget(self.view_combo)

        # Projection toggle
        self.ortho_btn = QPushButton("Ortho")
        self.ortho_btn.setCheckable(True)
        self.ortho_btn.clicked.connect(self._on_projection_changed)
        toolbar_layout.addWidget(self.ortho_btn)

        toolbar_layout.addStretch()

        # Show/hide toggles
        self.show_sample = QCheckBox("Sample")
        self.show_sample.setChecked(True)
        self.show_sample.toggled.connect(self._update_visibility)
        toolbar_layout.addWidget(self.show_sample)

        self.show_detector = QCheckBox("Detector")
        self.show_detector.setChecked(True)
        self.show_detector.toggled.connect(self._update_visibility)
        toolbar_layout.addWidget(self.show_detector)

        self.show_beam = QCheckBox("Beam")
        self.show_beam.setChecked(True)
        self.show_beam.toggled.connect(self._update_visibility)
        toolbar_layout.addWidget(self.show_beam)

        self.show_optics = QCheckBox("Optics")
        self.show_optics.setChecked(True)
        self.show_optics.toggled.connect(self._update_visibility)
        toolbar_layout.addWidget(self.show_optics)

        self.show_peaks = QCheckBox("Peaks")
        self.show_peaks.setChecked(False)
        self.show_peaks.toggled.connect(self._update_visibility)
        toolbar_layout.addWidget(self.show_peaks)

        self.show_q_vector = QCheckBox("Q Vector")
        self.show_q_vector.setChecked(False)
        self.show_q_vector.setToolTip("Show scattering vector Q = k_out - k_in based on detector position")
        self.show_q_vector.toggled.connect(self._on_q_vector_toggled)
        toolbar_layout.addWidget(self.show_q_vector)

        # Scale info label
        self.scale_label = QLabel("Scale: --")
        self.scale_label.setStyleSheet("color: #888888; font-size: 9px;")
        self.scale_label.setToolTip("Current viewport scale factor.\nAll objects use the same scale to preserve relative sizes.")
        toolbar_layout.addWidget(self.scale_label)

        # Auto-fit button
        fit_btn = QPushButton("Fit All")
        fit_btn.clicked.connect(self._fit_all)
        toolbar_layout.addWidget(fit_btn)

        # Reset view button
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._reset_view)
        toolbar_layout.addWidget(reset_btn)

        layout.addWidget(toolbar)

        # Create VisPy canvas or placeholder
        # _setup_vispy_canvas may fall back to placeholder on error
        if VISPY_AVAILABLE:
            self._setup_vispy_canvas()
        else:
            self._setup_placeholder()

        # canvas_widget should now be set by either method
        if hasattr(self, 'canvas_widget') and self.canvas_widget is not None:
            layout.addWidget(self.canvas_widget)
        else:
            # Final fallback
            self._setup_placeholder()
            layout.addWidget(self.canvas_widget)

    def _setup_vispy_canvas(self):
        """Setup VisPy canvas for 3D rendering."""
        try:
            # Create canvas with explicit parent to ensure Qt compatibility
            self.canvas = scene.SceneCanvas(
                keys='interactive',
                show=False,
                bgcolor='#1a1a1a',
                parent=self  # Set parent to ensure proper Qt integration
            )

            # Create view
            self.view = self.canvas.central_widget.add_view()
            self.view.camera = scene.TurntableCamera(
                fov=45,
                distance=100,  # Appropriate for TARGET_VIEWPORT_SIZE of 40
                center=(0, 0, 0),
                elevation=25,
                azimuth=-60
            )

            # Get the native widget - with PySide6 backend this should be a QWidget
            native_widget = self.canvas.native

            # Ensure it's a proper QWidget for the layout
            if isinstance(native_widget, QWidget):
                self.canvas_widget = native_widget
                self._vispy_active = True
            else:
                # Wrap in a container widget if needed
                self.canvas_widget = QWidget()
                container_layout = QVBoxLayout(self.canvas_widget)
                container_layout.setContentsMargins(0, 0, 0, 0)
                # Try to add the native widget
                try:
                    container_layout.addWidget(native_widget)
                    self._vispy_active = True
                except TypeError:
                    # If that fails, fall back to placeholder
                    raise RuntimeError("Cannot add VisPy canvas to Qt layout")

            # Create visual elements (after _vispy_active is set)
            self._create_grid()
            self._create_axes()
            self._create_origin_marker()

        except Exception as e:
            print(f"Error setting up VisPy canvas: {e}")
            # Fall back to placeholder
            self._vispy_active = False
            self._setup_placeholder()

    def _setup_placeholder(self):
        """Setup placeholder when VisPy is not available."""
        self.canvas_widget = QLabel(
            "3D Viewport\n\n"
            "VisPy not installed.\n"
            "Install with: pip install vispy pyopengl\n\n"
            "The GUI will still function,\n"
            "but 3D visualization is disabled."
        )
        self.canvas_widget.setAlignment(Qt.AlignCenter)
        self.canvas_widget.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                color: #808080;
                font-size: 14px;
            }
        """)

    def _create_grid(self):
        """Create ground plane grid."""
        if not self._vispy_active:
            return

        # XY grid plane at z=0
        grid = scene.visuals.GridLines(
            scale=(5, 5),
            color=(0.3, 0.3, 0.3, 0.5),
            parent=self.view.scene
        )
        self._visuals['grid'] = grid

    def _create_axes(self):
        """Create coordinate axes visual with labels."""
        if not self._vispy_active:
            return

        axis_length = 15

        # X axis (red) - beam direction
        x_axis = scene.visuals.Line(
            pos=np.array([[0, 0, 0], [axis_length, 0, 0]]),
            color='#ff4444',
            width=3,
            parent=self.view.scene
        )
        self._visuals['x_axis'] = x_axis

        # Y axis (green)
        y_axis = scene.visuals.Line(
            pos=np.array([[0, 0, 0], [0, axis_length, 0]]),
            color='#44ff44',
            width=3,
            parent=self.view.scene
        )
        self._visuals['y_axis'] = y_axis

        # Z axis (blue)
        z_axis = scene.visuals.Line(
            pos=np.array([[0, 0, 0], [0, 0, axis_length]]),
            color='#4444ff',
            width=3,
            parent=self.view.scene
        )
        self._visuals['z_axis'] = z_axis

        # Axis labels
        scene.visuals.Text(
            text='X (beam)', pos=[axis_length + 2, 0, 0], color='#ff4444',
            font_size=10, anchor_x='left', parent=self.view.scene
        )
        scene.visuals.Text(
            text='Y', pos=[0, axis_length + 2, 0], color='#44ff44',
            font_size=10, anchor_x='center', parent=self.view.scene
        )
        scene.visuals.Text(
            text='Z', pos=[0, 0, axis_length + 2], color='#4444ff',
            font_size=10, anchor_x='center', parent=self.view.scene
        )

    def _create_origin_marker(self):
        """Create origin marker sphere."""
        if not self._vispy_active:
            return

        origin = scene.visuals.Markers(parent=self.view.scene)
        origin.set_data(
            np.array([[0, 0, 0]]),
            face_color='white',
            size=8,
            edge_width=0
        )
        self._visuals['origin'] = origin

    def _calculate_scale_factor(self) -> bool:
        """
        Calculate a uniform scale factor to fit displayed objects in the viewport.

        This finds the maximum dimension across all VISIBLE objects and computes
        a single scale factor that maps everything to fit within TARGET_VIEWPORT_SIZE.
        Only objects with their visibility checkbox enabled are considered.

        Returns:
            bool: True if scale factor changed significantly (>1% change)
        """
        max_dimension = 1.0  # Minimum default (in Angstroms)
        dimension_sources = []  # Track what's contributing to max for debugging

        # Check sample dimensions (in Angstroms) - only if sample is visible
        sample = self.state.sample
        sample_visible = self.show_sample.isChecked() if hasattr(self, 'show_sample') else True
        if sample_visible and sample is not None and hasattr(sample, 'dimensions'):
            dims = sample.dimensions
            if dims is not None:
                try:
                    sample_max = float(max(dims))
                    if sample_max > 0:
                        dimension_sources.append(f"sample:{sample_max:.2e}Å")
                        max_dimension = max(max_dimension, sample_max)
                except (TypeError, ValueError):
                    pass

        # Check detector dimensions - only if detector is visible
        detector = self.state.detector
        detector_visible = self.show_detector.isChecked() if hasattr(self, 'show_detector') else True
        if detector_visible and detector is not None:
            # Check detector distance (in Angstroms) - this is usually the largest
            if hasattr(detector, 'distance'):
                dist = detector.distance
                if dist is not None and dist > 0:
                    try:
                        det_dist = float(dist)
                        dimension_sources.append(f"det_dist:{det_dist:.2e}Å")
                        max_dimension = max(max_dimension, det_dist)
                    except (TypeError, ValueError):
                        pass

            # Check detector size (in Angstroms)
            shape = getattr(detector, 'shape', None)
            pixel_size = getattr(detector, 'pixel_size', None)
            if shape is not None and pixel_size is not None:
                try:
                    det_size_y = shape[0] * pixel_size[0]
                    det_size_z = shape[1] * pixel_size[1]
                    det_size_max = max(det_size_y, det_size_z)
                    if det_size_max > 0:
                        dimension_sources.append(f"det_size:{det_size_max:.2e}Å")
                        max_dimension = max(max_dimension, det_size_max)
                except (TypeError, ValueError, IndexError):
                    pass

        # Check beam size - only if beam is visible
        beam = self.state.beam
        beam_visible = self.show_beam.isChecked() if hasattr(self, 'show_beam') else True
        if beam_visible and beam is not None:
            beam_size = getattr(beam, '_beam_size', None)
            if beam_size is not None:
                try:
                    beam_max = float(max(beam_size))
                    if beam_max > 0:
                        dimension_sources.append(f"beam:{beam_max:.2e}Å")
                        max_dimension = max(max_dimension, beam_max)
                except (TypeError, ValueError):
                    pass

        # Compute uniform scale factor: TARGET_VIEWPORT_SIZE / max_dimension
        # This ensures the largest object fits within the target size
        new_scale = self.TARGET_VIEWPORT_SIZE / max_dimension if max_dimension > 0 else 1.0

        # Check if scale changed significantly (>1% change)
        scale_changed = abs(new_scale - self._scale_factor) / max(self._scale_factor, 1e-10) > 0.01
        max_dim_changed = abs(max_dimension - self._last_max_dimension) / max(self._last_max_dimension, 1e-10) > 0.01

        if scale_changed or max_dim_changed:
            print(f"[Viewport3D] Scale updated: {self._scale_factor:.4e} -> {new_scale:.4e}")
            print(f"[Viewport3D] Max dimension: {max_dimension:.2e}Å from [{', '.join(dimension_sources)}]")

        self._scale_factor = new_scale
        self._last_max_dimension = max_dimension

        # Update scale label in toolbar
        self._update_scale_label(max_dimension, dimension_sources)

        return scale_changed or max_dim_changed

    def _update_scale_label(self, max_dimension: float, sources: list):
        """Update the scale info label in the toolbar."""
        if max_dimension >= 1e9:
            dim_str = f"{max_dimension/1e10:.1f} m"
        elif max_dimension >= 1e6:
            dim_str = f"{max_dimension/1e7:.1f} mm"
        elif max_dimension >= 1e3:
            dim_str = f"{max_dimension/1e4:.1f} μm"
        else:
            dim_str = f"{max_dimension:.0f} Å"

        # Identify the dominant source
        if sources:
            dominant = sources[-1].split(":")[0]  # Last source is usually largest
        else:
            dominant = "default"

        self.scale_label.setText(f"Max: {dim_str} ({dominant})")
        self.scale_label.setToolTip(
            f"Viewport scale: {self._scale_factor:.2e}\n"
            f"Max dimension: {max_dimension:.2e} Å\n"
            f"Sources: {', '.join(sources) if sources else 'none'}"
        )

    def _to_viewport(self, value):
        """Convert real value to viewport units."""
        return value * self._scale_factor

    def _create_or_update_sample(self):
        """Create or update sample visualization."""
        if not self._vispy_active:
            return

        sample = self.state.sample
        stage = self.state.stage

        # Remove existing sample visual
        if 'sample_box' in self._visuals:
            self._visuals['sample_box'].parent = None
            del self._visuals['sample_box']
        if 'sample_label' in self._visuals:
            self._visuals['sample_label'].parent = None
            del self._visuals['sample_label']

        if sample is None:
            return

        try:
            if hasattr(sample, 'dimensions'):
                dims = np.array(sample.dimensions, dtype=float)

                # Scale dimensions
                scaled_dims = dims * self._scale_factor

                # Create box
                box = scene.visuals.Box(
                    width=scaled_dims[1],   # Y dimension
                    height=scaled_dims[2],  # Z dimension
                    depth=scaled_dims[0],   # X dimension
                    color=(0.3, 0.5, 0.9, 0.4),
                    edge_color=(0.5, 0.7, 1.0, 1.0),
                    parent=self.view.scene
                )

                # Apply stage rotation if available
                if stage is not None and hasattr(stage, 'rotation'):
                    rotation = np.array(stage.rotation)
                    transform = MatrixTransform()

                    # Build 4x4 matrix from 3x3 rotation. VisPy applies
                    # row vectors (v @ M), so the column-convention stage
                    # rotation enters transposed.
                    mat4 = np.eye(4)
                    mat4[:3, :3] = rotation.T
                    transform.matrix = mat4
                    box.transform = transform

                self._visuals['sample_box'] = box

                # Add label
                label = scene.visuals.Text(
                    text='Sample',
                    pos=[0, 0, scaled_dims[2]/2 + 2],
                    color=(0.7, 0.8, 1.0, 1.0),
                    font_size=9,
                    anchor_x='center',
                    parent=self.view.scene
                )
                self._visuals['sample_label'] = label

        except Exception as e:
            print(f"Error creating sample visual: {e}")

    def _create_or_update_detector(self):
        """Create or update detector visualization."""
        if not self._vispy_active:
            return

        detector = self.state.detector

        # Remove existing detector visuals
        for key in ['detector_plane', 'detector_outline', 'detector_line', 'detector_label']:
            if key in self._visuals:
                self._visuals[key].parent = None
                del self._visuals[key]

        if detector is None:
            return

        try:
            # Get detector parameters (handle None values explicitly)
            distance = getattr(detector, 'distance', None)
            distance = distance if distance is not None else 1e10  # Default 1000mm in Angstroms

            two_theta = getattr(detector, 'two_theta', None)
            two_theta = two_theta if two_theta is not None else 0  # Radians

            eta = getattr(detector, 'eta', None)
            eta = eta if eta is not None else 0  # Radians

            # Get detector size (handle None values)
            shape = getattr(detector, 'shape', None)
            shape = shape if shape is not None else (256, 256)

            pixel_size = getattr(detector, 'pixel_size', None)
            pixel_size = pixel_size if pixel_size is not None else (5.5e5, 5.5e5)

            half_y = shape[0] * pixel_size[0] / 2
            half_z = shape[1] * pixel_size[1] / 2

            # Scale to viewport
            dist_scaled = distance * self._scale_factor
            size_y_scaled = half_y * self._scale_factor
            size_z_scaled = half_z * self._scale_factor

            # Calculate detector center position from spherical coords
            # Match Detector.py convention exactly:
            # x = distance * cos(two_theta)
            # y = distance * sin(two_theta) * sin(eta)
            # z = distance * sin(two_theta) * cos(eta)
            # When eta=0, positive two_theta rotates in XZ plane (about +Y axis)
            cx = dist_scaled * np.cos(two_theta)
            cy = dist_scaled * np.sin(two_theta) * np.sin(eta)
            cz = dist_scaled * np.sin(two_theta) * np.cos(eta)

            center = np.array([cx, cy, cz])

            # Check construction mode to determine visualization
            construction_mode = getattr(detector, '_construction_mode', 'plane')

            if construction_mode == 'shell':
                # Shell mode: visualize actual pixel coordinates as point cloud
                pixel_coords = getattr(detector, 'pixel_coordinates', None)
                if pixel_coords is not None and pixel_coords.shape[1] > 0:
                    # Scale to viewport
                    pix_scaled = pixel_coords * self._scale_factor

                    # Subsample for performance (every Nth pixel)
                    subsample = max(1, pixel_coords.shape[1] // 2000)
                    points = pix_scaled[:, ::subsample].T  # (N, 3)

                    # Create markers for shell pixels
                    detector_markers = scene.visuals.Markers(
                        pos=points,
                        size=5,
                        face_color=(0.9, 0.8, 0.3, 0.6),
                        edge_color=None,
                        parent=self.view.scene
                    )
                    self._visuals['detector_plane'] = detector_markers

                    # Draw line from origin to center (approximate center of shell)
                    line_pts = np.array([[0, 0, 0], center])
                    detector_line = scene.visuals.Line(
                        pos=line_pts,
                        color=(0.5, 0.5, 0.5, 0.5),
                        width=1,
                        parent=self.view.scene
                    )
                    self._visuals['detector_line'] = detector_line

                    # Add label with angle info
                    two_theta_deg = np.degrees(two_theta)
                    eta_deg = np.degrees(eta)
                    label_pos = center * 1.2  # Position label slightly beyond center
                    label = scene.visuals.Text(
                        text=f'Detector (shell)\n2θ={two_theta_deg:.1f}° η={eta_deg:.1f}°',
                        pos=label_pos,
                        color=(1.0, 0.9, 0.4, 1.0),
                        font_size=8,
                        anchor_x='center',
                        parent=self.view.scene
                    )
                    self._visuals['detector_label'] = label

                    return  # Skip plane visualization

            # Plane mode: standard flat rectangle visualization
            # Calculate detector normal (pointing back toward origin)
            normal = -center / (np.linalg.norm(center) + 1e-10)

            # Build orthonormal basis for detector plane
            up = np.array([0, 0, 1])
            if abs(np.dot(normal, up)) > 0.99:
                up = np.array([0, 1, 0])
            right = np.cross(normal, up)
            right = right / (np.linalg.norm(right) + 1e-10)
            up = np.cross(right, normal)
            up = up / (np.linalg.norm(up) + 1e-10)

            # Create detector corners
            corners = np.array([
                center - size_y_scaled * right - size_z_scaled * up,
                center + size_y_scaled * right - size_z_scaled * up,
                center + size_y_scaled * right + size_z_scaled * up,
                center - size_y_scaled * right + size_z_scaled * up,
            ])

            # Create filled rectangle for detector face
            # Using Mesh with two triangles
            vertices = np.vstack([corners, corners])  # 8 vertices for two-sided
            faces = np.array([
                [0, 1, 2], [0, 2, 3],  # Front
                [4, 6, 5], [4, 7, 6],  # Back
            ])
            colors = np.array([
                [0.9, 0.8, 0.3, 0.3],
                [0.9, 0.8, 0.3, 0.3],
                [0.9, 0.8, 0.3, 0.3],
                [0.9, 0.8, 0.3, 0.3],
                [0.9, 0.8, 0.3, 0.3],
                [0.9, 0.8, 0.3, 0.3],
                [0.9, 0.8, 0.3, 0.3],
                [0.9, 0.8, 0.3, 0.3],
            ])

            detector_plane = scene.visuals.Mesh(
                vertices=vertices[:4],
                faces=np.array([[0, 1, 2], [0, 2, 3]]),
                vertex_colors=colors[:4],
                parent=self.view.scene
            )
            self._visuals['detector_plane'] = detector_plane

            # Create outline
            outline_pts = np.vstack([corners, corners[0:1]])  # Close loop
            detector_outline = scene.visuals.Line(
                pos=outline_pts,
                color=(1.0, 0.9, 0.3, 1.0),
                width=2,
                parent=self.view.scene
            )
            self._visuals['detector_outline'] = detector_outline

            # Draw line from origin to detector center
            line_pts = np.array([[0, 0, 0], center])
            detector_line = scene.visuals.Line(
                pos=line_pts,
                color=(0.5, 0.5, 0.5, 0.5),
                width=1,
                parent=self.view.scene
            )
            self._visuals['detector_line'] = detector_line

            # Add label with angle info
            two_theta_deg = np.degrees(two_theta)
            eta_deg = np.degrees(eta)
            label_pos = center + 3 * up
            label = scene.visuals.Text(
                text=f'Detector\n2θ={two_theta_deg:.1f}° η={eta_deg:.1f}°',
                pos=label_pos,
                color=(1.0, 0.9, 0.4, 1.0),
                font_size=8,
                anchor_x='center',
                parent=self.view.scene
            )
            self._visuals['detector_label'] = label

        except Exception as e:
            print(f"Error creating detector visual: {e}")

    def _create_or_update_beam(self):
        """Create or update beam visualization."""
        if not self._vispy_active:
            return

        beam = self.state.beam
        detector = self.state.detector

        # Remove existing beam visuals
        for key in ['beam_line', 'beam_arrow', 'beam_label', 'diffracted_line', 'beam_outline']:
            if key in self._visuals:
                self._visuals[key].parent = None
                del self._visuals[key]

        if beam is None:
            return

        try:
            # Beam extends from -X to sample (at origin)
            # Pink color for incident beam
            BEAM_COLOR = (1.0, 0.4, 0.7, 0.9)  # Pink

            # Determine beam length based on detector distance
            det_dist = 20  # Default
            if detector is not None and hasattr(detector, 'distance'):
                dist = detector.distance
                if dist is not None:
                    det_dist = dist * self._scale_factor

            # Get beam size (in Angstroms) and shape
            beam_size = getattr(beam, '_beam_size', None)
            beam_shape = getattr(beam, '_beam_shape', 'rectangular')

            # Incoming beam start position
            incoming_start = np.array([-det_dist * 0.8, 0, 0])
            incoming_end = np.array([0, 0, 0])

            # Draw incident beam as a thick pink line
            beam_pts = np.array([incoming_start, incoming_end])
            beam_line = scene.visuals.Line(
                pos=beam_pts,
                color=BEAM_COLOR,
                width=6,
                parent=self.view.scene
            )
            self._visuals['beam_line'] = beam_line

            # Add arrow head at origin pointing in +X direction
            arrow_size = 2.0
            arrow_pts = np.array([
                [-arrow_size, arrow_size * 0.6, 0],
                [0, 0, 0],
                [-arrow_size, -arrow_size * 0.6, 0],
            ])
            beam_arrow = scene.visuals.Line(
                pos=arrow_pts,
                color=BEAM_COLOR,
                width=5,
                connect='strip',
                parent=self.view.scene
            )
            self._visuals['beam_arrow'] = beam_arrow

            # Add beam cross-section outline at sample position (origin, YZ plane)
            if beam_size is not None:
                # Scale beam size to viewport units
                half_y = beam_size[0] * self._scale_factor / 2
                half_z = beam_size[1] * self._scale_factor / 2

                if beam_shape == 'circular':
                    # Draw circular outline using multiple segments
                    n_segments = 32
                    angles = np.linspace(0, 2 * np.pi, n_segments + 1)
                    # Use average of Y and Z sizes for radius
                    radius = (half_y + half_z) / 2
                    outline_pts = np.zeros((n_segments + 1, 3))
                    outline_pts[:, 0] = 0  # X = 0 (at origin)
                    outline_pts[:, 1] = radius * np.cos(angles)  # Y
                    outline_pts[:, 2] = radius * np.sin(angles)  # Z
                else:
                    # Draw rectangular outline
                    outline_pts = np.array([
                        [0, -half_y, -half_z],
                        [0, +half_y, -half_z],
                        [0, +half_y, +half_z],
                        [0, -half_y, +half_z],
                        [0, -half_y, -half_z],  # Close the rectangle
                    ])

                beam_outline = scene.visuals.Line(
                    pos=outline_pts,
                    color=BEAM_COLOR,
                    width=2,
                    connect='strip',
                    parent=self.view.scene
                )
                self._visuals['beam_outline'] = beam_outline

            # Add diffracted beam line from origin to detector center (if detector exists)
            if detector is not None:
                two_theta = getattr(detector, 'two_theta', None)
                two_theta = two_theta if two_theta is not None else 0
                eta = getattr(detector, 'eta', None)
                eta = eta if eta is not None else 0

                # Calculate detector center position (same as in _create_or_update_detector)
                # Match Detector.py convention exactly:
                # x = distance * cos(two_theta)
                # y = distance * sin(two_theta) * sin(eta)
                # z = distance * sin(two_theta) * cos(eta)
                cx = det_dist * np.cos(two_theta)
                cy = det_dist * np.sin(two_theta) * np.sin(eta)
                cz = det_dist * np.sin(two_theta) * np.cos(eta)

                detector_center = np.array([cx, cy, cz])

                # Draw diffracted beam line (dashed effect via lighter color)
                diffracted_pts = np.array([[0, 0, 0], detector_center])
                diffracted_line = scene.visuals.Line(
                    pos=diffracted_pts,
                    color=(0.8, 0.8, 0.4, 0.6),  # Yellowish, semi-transparent
                    width=2,
                    parent=self.view.scene
                )
                self._visuals['diffracted_line'] = diffracted_line

            # Add label
            energy = getattr(beam, '_energy', None)
            if energy:
                label_text = f'Beam\n{energy:.0f} eV'
            else:
                label_text = 'Beam'

            label = scene.visuals.Text(
                text=label_text,
                pos=[incoming_start[0] + 3, 0, 2],
                color=(1.0, 0.7, 0.3, 1.0),
                font_size=8,
                anchor_x='left',
                parent=self.view.scene
            )
            self._visuals['beam_label'] = label

        except Exception as e:
            print(f"Error creating beam visual: {e}")

    def _create_or_update_optics(self):
        """Create or update optics visualization.

        Optics are placed along the line from sample to detector,
        positioned between the sample and detector (behind detector from sample's view).
        """
        if not self._vispy_active:
            return

        optics = self.state.optics

        # Remove existing optics visuals
        keys_to_remove = [k for k in self._visuals.keys() if k.startswith('optics_')]
        for key in keys_to_remove:
            self._visuals[key].parent = None
            del self._visuals[key]

        if optics is None:
            return

        try:
            components = getattr(optics, 'components', [])
            if not components:
                return

            # Get detector position to determine optics placement direction
            detector = self.state.detector
            if detector is not None:
                distance = getattr(detector, 'distance', None)
                distance = distance if distance is not None else 1e10
                two_theta = getattr(detector, 'two_theta', None)
                two_theta = two_theta if two_theta is not None else 0
                eta = getattr(detector, 'eta', None)
                eta = eta if eta is not None else 0

                # Calculate detector direction (unit vector from origin to detector)
                dist_scaled = distance * self._scale_factor
                det_x = dist_scaled * np.cos(two_theta)
                det_y = dist_scaled * np.sin(two_theta) * np.sin(eta)
                det_z = dist_scaled * np.sin(two_theta) * np.cos(eta)
                detector_pos = np.array([det_x, det_y, det_z])
                detector_dist = np.linalg.norm(detector_pos)

                if detector_dist > 1e-6:
                    direction = detector_pos / detector_dist
                else:
                    direction = np.array([1.0, 0.0, 0.0])
            else:
                # No detector, default to +X direction
                direction = np.array([1.0, 0.0, 0.0])
                detector_dist = 20.0  # Default distance in viewport units

            # Start optics placement just after sample, moving toward detector
            # Reserve some space before detector for visibility
            start_offset = 2.0  # Start just after sample
            end_offset = 3.0    # Stop before detector

            # Calculate available space for optics
            available_distance = max(detector_dist - start_offset - end_offset, 5.0)

            # Count non-free-space components to distribute them evenly
            physical_components = [c for c in components if c.get('kind', '') != 'free space']
            n_physical = len(physical_components)

            if n_physical == 0:
                return

            # Space components evenly along the path to detector
            spacing = available_distance / (n_physical + 1)
            comp_index = 0

            for i, comp in enumerate(components):
                kind = comp.get('kind', 'unknown')

                if kind == 'free space':
                    continue

                # Calculate position along direction to detector
                comp_index += 1
                t = start_offset + spacing * comp_index
                pos = direction * t

                # Draw component as a box or marker
                comp_size = 1.5  # Viewport units

                if kind == 'lens box':
                    # CRL - draw as a series of circles
                    n_lenses = comp.get('number', 1)
                    color = (0.3, 0.8, 0.3, 0.6)
                    label = f"CRL (N={n_lenses})"

                elif kind == 'aperture':
                    color = (0.8, 0.3, 0.3, 0.6)
                    label = "Aperture"

                elif kind == 'bragg magnifier 2b':
                    color = (0.8, 0.5, 0.8, 0.6)
                    mx = comp.get('magnification_x', 1)
                    my = comp.get('magnification_y', 1)
                    label = f"Bragg Mag\n({mx:.1f}x{my:.1f})"

                elif kind == 'angular filter':
                    color = (0.5, 0.8, 0.8, 0.6)
                    label = "Ang. Filter"

                else:
                    color = (0.5, 0.5, 0.5, 0.6)
                    label = kind

                # Create box for component - oriented perpendicular to beam direction
                box = scene.visuals.Box(
                    width=comp_size,
                    height=comp_size,
                    depth=comp_size * 0.3,
                    color=color,
                    edge_color=(1, 1, 1, 0.5),
                    parent=self.view.scene
                )

                # Calculate rotation to face the detector direction
                # We need to rotate the box so its depth axis aligns with 'direction'
                # Default box depth is along Z, we want it along 'direction'
                from vispy.visuals.transforms import MatrixTransform
                transform = MatrixTransform()

                # Build rotation matrix to align Z axis with direction
                z_axis = np.array([0, 0, 1])
                if abs(np.dot(z_axis, direction)) < 0.999:
                    # Need rotation
                    rot_axis = np.cross(z_axis, direction)
                    rot_axis = rot_axis / (np.linalg.norm(rot_axis) + 1e-10)
                    angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))

                    # Rodrigues' rotation formula for rotation matrix
                    c, s = np.cos(angle), np.sin(angle)
                    K = np.array([
                        [0, -rot_axis[2], rot_axis[1]],
                        [rot_axis[2], 0, -rot_axis[0]],
                        [-rot_axis[1], rot_axis[0], 0]
                    ])
                    R = np.eye(3) + s * K + (1 - c) * (K @ K)

                    # Build 4x4 transform matrix
                    mat = np.eye(4)
                    mat[:3, :3] = R
                    mat[:3, 3] = pos
                    transform.matrix = mat
                else:
                    # Direction is already along Z or -Z
                    transform.translate(pos)
                    if np.dot(z_axis, direction) < 0:
                        transform.rotate(180, (1, 0, 0))

                box.transform = transform
                self._visuals[f'optics_box_{i}'] = box

                # Add label - offset perpendicular to direction
                # Use Y-axis for offset if direction is not along Y
                if abs(direction[1]) < 0.9:
                    label_offset = np.array([0, comp_size + 0.5, 0])
                else:
                    label_offset = np.array([0, 0, comp_size + 0.5])

                label_pos = pos + label_offset
                text = scene.visuals.Text(
                    text=label,
                    pos=label_pos,
                    color=color[:3] + (1.0,),
                    font_size=7,
                    anchor_x='center',
                    parent=self.view.scene
                )
                self._visuals[f'optics_label_{i}'] = text

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error creating optics visual: {e}")

    def _register_observers(self):
        """Register for state change notifications."""
        self.state.register_observer("sample_changed", self._on_sample_changed)
        self.state.register_observer("detector_changed", self._on_detector_changed)
        self.state.register_observer("beam_changed", self._on_beam_changed)
        self.state.register_observer("stage_changed", self._on_stage_changed)
        self.state.register_observer("optics_changed", self._on_optics_changed)
        self.state.register_observer("crystal_changed", self._on_crystal_changed)

    def _rebuild_all_visuals(self):
        """Rebuild all visuals with current scale factor."""
        self._create_or_update_sample()
        self._create_or_update_detector()
        self._create_or_update_beam()
        self._create_or_update_optics()
        self._create_or_update_peaks()
        # Rebuild Q vector if it's enabled
        if self.show_q_vector.isChecked():
            self._create_or_update_q_vector()
        self._update_visibility()

    def _on_sample_changed(self, sample):
        """Handle sample change."""
        scale_changed = self._calculate_scale_factor()
        if scale_changed:
            # Scale changed - rebuild ALL visuals to maintain correct proportions
            self._rebuild_all_visuals()
        else:
            # Scale unchanged - only update affected visuals
            self._create_or_update_sample()
            self._create_or_update_beam()  # Beam depends on sample
            self._update_visibility()
        if self._vispy_active:
            self.canvas.update()

    def _on_detector_changed(self, detector):
        """Handle detector change."""
        scale_changed = self._calculate_scale_factor()
        if scale_changed:
            # Scale changed - rebuild ALL visuals to maintain correct proportions
            self._rebuild_all_visuals()
        else:
            # Scale unchanged - only update affected visuals
            self._create_or_update_detector()
            self._create_or_update_beam()  # Beam length depends on detector
            self._create_or_update_optics()  # Optics position depends on detector
            # Update Q vector if it's enabled (Q depends on detector position)
            if self.show_q_vector.isChecked():
                self._create_or_update_q_vector()
            self._update_visibility()
        if self._vispy_active:
            self.canvas.update()

    def _on_stage_changed(self, stage):
        """Handle stage change."""
        self._update_diff_calc_refs()
        self._create_or_update_sample()  # Sample rotation depends on stage
        self._create_or_update_peaks()  # Q-vectors in lab frame depend on stage
        self._update_visibility()
        if self._vispy_active:
            self.canvas.update()

    def _on_optics_changed(self, optics):
        """Handle optics change."""
        self._create_or_update_optics()
        self._update_visibility()
        if self._vispy_active:
            self.canvas.update()

    def _on_crystal_changed(self, crystal):
        """Handle crystal change (may affect sample and peaks display)."""
        self._update_diff_calc_refs()
        self._create_or_update_peaks()
        self._update_visibility()
        if self._vispy_active:
            self.canvas.update()

    def _on_beam_changed(self, beam):
        """Handle beam change."""
        self._update_diff_calc_refs()
        self._create_or_update_beam()
        self._create_or_update_peaks()  # Energy affects accessible peaks
        self._update_visibility()
        if self._vispy_active:
            self.canvas.update()

    def _update_diff_calc_refs(self):
        """Update diffraction calculator with current objects."""
        self._diff_calc.set_crystal(self.state.crystal)
        self._diff_calc.set_beam(self.state.beam)
        self._diff_calc.set_stage(self.state.stage)

    def set_selected_peaks(self, peaks: list):
        """Set the list of selected peaks to display.

        Args:
            peaks: List of (h,k,l) tuples to display
        """
        self._selected_peaks = list(peaks) if peaks else []
        self.show_peaks.setChecked(True)  # Auto-enable peaks visibility
        self._create_or_update_peaks()
        self._update_visibility()
        if self._vispy_active:
            self.canvas.update()

    def _create_or_update_peaks(self):
        """Create or update reciprocal lattice peak markers."""
        if not self._vispy_active:
            return

        # Remove existing peak visuals
        keys_to_remove = [k for k in list(self._visuals.keys()) if k.startswith('peak_')]
        for key in keys_to_remove:
            if self._visuals[key].parent is not None:
                self._visuals[key].parent = None
            del self._visuals[key]

        # Check if we have required objects
        crystal = self.state.crystal
        beam = self.state.beam

        if crystal is None or beam is None or beam._energy is None:
            return

        try:
            # Update calculator references
            self._update_diff_calc_refs()

            # Get Q-vectors for selected peaks and all accessible peaks
            selected_q_vectors = []  # Store (hkl, Q_lab) pairs
            accessible_q_vectors = []

            # Color definitions
            SELECTED_COLOR = np.array([1.0, 1.0, 0.0, 1.0])   # Yellow for selected
            ACCESSIBLE_COLOR = np.array([0.2, 0.8, 0.2, 0.7]) # Green for accessible

            # Calculate sample extent in viewport units for line scaling
            sample = self.state.sample
            sample_extent = 5.0  # Default minimum extent in viewport units
            if sample is not None and hasattr(sample, 'dimensions'):
                dims = sample.dimensions
                if dims is not None:
                    try:
                        # Get max dimension and scale to viewport
                        max_dim = float(max(dims))
                        scaled_extent = max_dim * self._scale_factor
                        sample_extent = max(sample_extent, scaled_extent * 1.5)  # 1.5x to extend past
                    except (TypeError, ValueError):
                        pass

            # Minimum line length to ensure visibility
            min_line_length = max(sample_extent, 8.0)

            # Collect Q-vectors for selected peaks
            for hkl in self._selected_peaks:
                try:
                    if not self._diff_calc.is_accessible(hkl):
                        continue
                    Q_lab = self._diff_calc.get_q_vector_lab(hkl)
                    selected_q_vectors.append((hkl, Q_lab))
                except Exception:
                    continue

            # Collect Q-vectors for accessible peaks (if showing all)
            if self._show_all_peaks:
                for hkl in self._diff_calc.enumerate_accessible_reflections(max_h=3, max_k=3, max_l=3):
                    if hkl in self._selected_peaks:
                        continue  # Already added as selected
                    try:
                        Q_lab = self._diff_calc.get_q_vector_lab(hkl)
                        accessible_q_vectors.append((hkl, Q_lab))
                    except Exception:
                        continue

            # Create visuals for selected peaks
            if selected_q_vectors:
                # Calculate positions for markers (extend to min_line_length)
                selected_positions = []
                for hkl, Q_lab in selected_q_vectors:
                    Q_norm = Q_lab / (np.linalg.norm(Q_lab) + 1e-10)
                    pos = Q_norm * min_line_length
                    selected_positions.append(pos)

                positions = np.array(selected_positions, dtype=np.float32)
                colors = np.array([SELECTED_COLOR] * len(selected_positions), dtype=np.float32)

                # Marker size scales with viewport
                marker_size = max(8, min(16, sample_extent * 0.8))

                markers = scene.visuals.Markers(parent=self.view.scene)
                markers.set_data(
                    positions,
                    face_color=colors,
                    size=marker_size,
                    edge_width=2,
                    edge_color='white',
                    symbol='o'
                )
                self._visuals['peak_selected_markers'] = markers

                # Add dotted yellow lines from origin to selected peaks
                origin = np.array([0.0, 0.0, 0.0])
                dot_spacing = max(0.5, sample_extent * 0.05)  # Spacing between dots
                dot_size = max(3, min(6, sample_extent * 0.3))  # Dot size

                all_dot_positions = []
                for pos in selected_positions:
                    # Calculate points along the line from origin to peak position
                    line_length = np.linalg.norm(pos - origin)
                    num_dots = max(3, int(line_length / dot_spacing))
                    for j in range(num_dots):
                        t = j / (num_dots - 1) if num_dots > 1 else 0
                        dot_pos = origin + t * (pos - origin)
                        all_dot_positions.append(dot_pos)

                if all_dot_positions:
                    dot_positions = np.array(all_dot_positions, dtype=np.float32)
                    dot_colors = np.array([SELECTED_COLOR] * len(all_dot_positions), dtype=np.float32)

                    line_dots = scene.visuals.Markers(parent=self.view.scene)
                    line_dots.set_data(
                        dot_positions,
                        face_color=dot_colors,
                        size=dot_size,
                        edge_width=0,
                        symbol='disc'
                    )
                    self._visuals['peak_selected_lines'] = line_dots

                # Add labels for selected peaks
                for i, ((hkl, Q_lab), pos) in enumerate(zip(selected_q_vectors, selected_positions)):
                    # Label for selected peak - scale offset based on sample extent
                    label_text = f"({hkl[0]},{hkl[1]},{hkl[2]})"
                    # Position label at end of line with offset proportional to extent
                    label_offset = sample_extent * 0.1
                    label_pos = pos + np.array([label_offset, label_offset, 0])
                    # Font size scales with viewport (8-12 range)
                    font_size = max(8, min(12, int(sample_extent * 0.6)))
                    label = scene.visuals.Text(
                        text=label_text,
                        pos=label_pos,
                        color=SELECTED_COLOR,
                        font_size=font_size,
                        anchor_x='left',
                        parent=self.view.scene
                    )
                    self._visuals[f'peak_label_{i}'] = label

            # Create markers for other accessible peaks (if enabled)
            if accessible_q_vectors:
                accessible_positions = []
                for hkl, Q_lab in accessible_q_vectors:
                    Q_norm = Q_lab / (np.linalg.norm(Q_lab) + 1e-10)
                    pos = Q_norm * min_line_length
                    accessible_positions.append(pos)

                positions = np.array(accessible_positions, dtype=np.float32)
                colors = np.array([ACCESSIBLE_COLOR] * len(accessible_positions), dtype=np.float32)

                # Smaller markers for non-selected peaks
                marker_size = max(4, min(10, sample_extent * 0.5))

                markers = scene.visuals.Markers(parent=self.view.scene)
                markers.set_data(
                    positions,
                    face_color=colors,
                    size=marker_size,
                    edge_width=0,
                    symbol='o'
                )
                self._visuals['peak_accessible_markers'] = markers

        except Exception as e:
            print(f"Error creating peak visuals: {e}")

    def toggle_show_all_peaks(self, show: bool):
        """Toggle showing all accessible peaks vs only selected.

        Args:
            show: True to show all accessible peaks, False for selected only
        """
        self._show_all_peaks = show
        self._create_or_update_peaks()
        self._update_visibility()
        if self._vispy_active:
            self.canvas.update()

    def _update_visibility(self):
        """Update visibility of all visuals based on checkboxes.

        Also recalculates the scale factor since it depends on which objects
        are currently visible.
        """
        if not self._vispy_active:
            return

        # Recalculate scale factor based on visible objects
        scale_changed = self._calculate_scale_factor()

        # If scale changed significantly, rebuild all visuals with new scale
        if scale_changed:
            self._rebuild_all_visuals()

        # Sample
        sample_visible = self.show_sample.isChecked() and self.state.sample is not None
        for key in ['sample_box', 'sample_label']:
            if key in self._visuals:
                self._visuals[key].visible = sample_visible

        # Detector
        detector_visible = self.show_detector.isChecked() and self.state.detector is not None
        for key in ['detector_plane', 'detector_outline', 'detector_line', 'detector_label']:
            if key in self._visuals:
                self._visuals[key].visible = detector_visible

        # Beam
        beam_visible = self.show_beam.isChecked() and self.state.beam is not None
        for key in ['beam_line', 'beam_arrow', 'beam_label', 'diffracted_line', 'beam_outline']:
            if key in self._visuals:
                self._visuals[key].visible = beam_visible

        # Optics
        optics_visible = self.show_optics.isChecked() and self.state.optics is not None
        for key in self._visuals.keys():
            if key.startswith('optics_'):
                self._visuals[key].visible = optics_visible

        # Peaks
        peaks_visible = self.show_peaks.isChecked()
        for key in list(self._visuals.keys()):
            if key.startswith('peak_'):
                self._visuals[key].visible = peaks_visible

        # Q vector
        self._update_q_vector_visibility()

        if self._vispy_active:
            self.canvas.update()

    def _on_view_changed(self, index):
        """Handle view preset change."""
        if not self._vispy_active:
            return

        view_type = self.view_combo.currentData()

        if view_type == "3d":
            self.view.camera.elevation = 25
            self.view.camera.azimuth = -60
        elif view_type == "xy":
            self.view.camera.elevation = 90
            self.view.camera.azimuth = 0
        elif view_type == "xz":
            self.view.camera.elevation = 0
            self.view.camera.azimuth = -90
        elif view_type == "yz":
            self.view.camera.elevation = 0
            self.view.camera.azimuth = 0

        self.canvas.update()
        self.view_changed.emit(view_type)

    def _on_projection_changed(self):
        """Handle projection toggle."""
        if not self._vispy_active:
            return

        if self.ortho_btn.isChecked():
            self.view.camera.fov = 0  # Orthographic
        else:
            self.view.camera.fov = 45  # Perspective

        self.canvas.update()

    def _on_q_vector_toggled(self, checked: bool):
        """Handle Q Vector checkbox toggle."""
        if checked:
            self._create_or_update_q_vector()
        self._update_q_vector_visibility()
        if self._vispy_active:
            self.canvas.update()

    def _update_q_vector_visibility(self):
        """Update visibility of Q vector visual."""
        if not self._vispy_active:
            return

        q_visible = self.show_q_vector.isChecked()
        for key in list(self._visuals.keys()):
            if key.startswith('q_vector_'):
                self._visuals[key].visible = q_visible

    def _create_or_update_q_vector(self):
        """Create or update the scattering vector Q = k_out - k_in visualization.

        This shows the actual scattering vector based on the current detector
        position, NOT the reciprocal lattice direction of selected peaks.

        The scattering vector Q is defined as:
            Q = k_out - k_in
        where k_in is the incident beam direction (+X) and k_out is the
        direction from sample to detector center.
        """
        if not self._vispy_active:
            return

        # Remove existing Q vector visuals
        keys_to_remove = [k for k in list(self._visuals.keys()) if k.startswith('q_vector_')]
        for key in keys_to_remove:
            if self._visuals[key].parent is not None:
                self._visuals[key].parent = None
            del self._visuals[key]

        # Check if we have required objects
        beam = self.state.beam
        detector = self.state.detector

        if beam is None or detector is None:
            return

        try:
            # Get detector position parameters
            two_theta = getattr(detector, 'two_theta', None)
            two_theta = two_theta if two_theta is not None else 0  # Radians

            eta = getattr(detector, 'eta', None)
            eta = eta if eta is not None else 0  # Radians

            # Calculate k_out direction (from sample at origin to detector center)
            # Using Detector.py convention:
            # x = distance * cos(two_theta)
            # y = distance * sin(two_theta) * sin(eta)
            # z = distance * sin(two_theta) * cos(eta)
            k_out = np.array([
                np.cos(two_theta),
                np.sin(two_theta) * np.sin(eta),
                np.sin(two_theta) * np.cos(eta)
            ])
            k_out = k_out / (np.linalg.norm(k_out) + 1e-10)

            # k_in is incident beam direction (+X axis)
            k_in = np.array([1.0, 0.0, 0.0])

            # Q = k_out - k_in (scattering vector)
            Q = k_out - k_in
            Q_mag = np.linalg.norm(Q)

            if Q_mag < 1e-10:
                # Q is essentially zero (detector at 2θ=0)
                return

            Q_norm = Q / Q_mag

            # Calculate line length based on sample extent
            sample = self.state.sample
            sample_extent = 5.0  # Default minimum extent in viewport units
            if sample is not None and hasattr(sample, 'dimensions'):
                dims = sample.dimensions
                if dims is not None:
                    try:
                        max_dim = float(max(dims))
                        scaled_extent = max_dim * self._scale_factor
                        sample_extent = max(sample_extent, scaled_extent * 1.5)
                    except (TypeError, ValueError):
                        pass

            # Line length for Q vector display
            line_length = max(sample_extent, 8.0)

            # Q vector endpoint
            Q_end = Q_norm * line_length

            # Color for Q vector - cyan to match peak arrows but distinct
            Q_COLOR = (0.0, 1.0, 1.0, 1.0)  # Cyan

            # Draw Q vector arrow from origin
            arrow_data = np.array([[0, 0, 0], Q_end], dtype=np.float32)
            q_arrow = scene.visuals.Line(
                pos=arrow_data,
                color=Q_COLOR,
                width=4,
                parent=self.view.scene
            )
            self._visuals['q_vector_arrow'] = q_arrow

            # Create arrowhead at the tip
            arrow_head_size = max(1.5, min(3.0, sample_extent * 0.15))

            # Find two perpendicular vectors to Q_norm for arrowhead
            if abs(Q_norm[2]) < 0.9:
                perp1 = np.cross(Q_norm, np.array([0, 0, 1]))
            else:
                perp1 = np.cross(Q_norm, np.array([0, 1, 0]))
            perp1 = perp1 / (np.linalg.norm(perp1) + 1e-10)
            perp2 = np.cross(Q_norm, perp1)
            perp2 = perp2 / (np.linalg.norm(perp2) + 1e-10)

            # Arrowhead: V-shape pointing along Q direction
            head_base = Q_end - Q_norm * arrow_head_size * 1.5
            head_pts = np.array([
                head_base + perp1 * arrow_head_size * 0.5,
                Q_end,
                head_base - perp1 * arrow_head_size * 0.5,
            ], dtype=np.float32)
            arrowhead1 = scene.visuals.Line(
                pos=head_pts,
                color=Q_COLOR,
                width=4,
                connect='strip',
                parent=self.view.scene
            )
            self._visuals['q_vector_arrowhead1'] = arrowhead1

            # Second V in perpendicular direction for 3D arrowhead
            head_pts2 = np.array([
                head_base + perp2 * arrow_head_size * 0.5,
                Q_end,
                head_base - perp2 * arrow_head_size * 0.5,
            ], dtype=np.float32)
            arrowhead2 = scene.visuals.Line(
                pos=head_pts2,
                color=Q_COLOR,
                width=4,
                connect='strip',
                parent=self.view.scene
            )
            self._visuals['q_vector_arrowhead2'] = arrowhead2

            # Add label showing "Q" and angles
            two_theta_deg = np.degrees(two_theta)
            eta_deg = np.degrees(eta)
            label_offset = sample_extent * 0.1
            label_pos = Q_end + np.array([label_offset, label_offset, 0])
            font_size = max(8, min(12, int(sample_extent * 0.6)))
            label = scene.visuals.Text(
                text=f"Q\n2θ={two_theta_deg:.1f}°",
                pos=label_pos,
                color=Q_COLOR,
                font_size=font_size,
                anchor_x='left',
                parent=self.view.scene
            )
            self._visuals['q_vector_label'] = label

        except Exception as e:
            print(f"Error creating Q vector visual: {e}")

    def _fit_all(self):
        """Fit all objects in view by recalculating scale and adjusting camera."""
        if not self._vispy_active:
            return

        # Force recalculate scale factor and rebuild all visuals
        self._last_max_dimension = 0  # Force recalculation
        self._calculate_scale_factor()
        self._rebuild_all_visuals()

        # Set camera distance appropriate for TARGET_VIEWPORT_SIZE
        # Objects are scaled to fit within ~40 units, so distance of 100 gives good view
        self.view.camera.distance = self.TARGET_VIEWPORT_SIZE * 2.5
        self.view.camera.center = (0, 0, 0)

        self.canvas.update()

    def _reset_view(self):
        """Reset camera to default view."""
        if not self._vispy_active:
            return

        self.view.camera.reset()
        # Default distance appropriate for TARGET_VIEWPORT_SIZE
        self.view.camera.distance = self.TARGET_VIEWPORT_SIZE * 2.5
        self.view.camera.elevation = 25
        self.view.camera.azimuth = -60
        self.view.camera.center = (0, 0, 0)
        self.view.camera.fov = 45
        self.ortho_btn.setChecked(False)
        self.view_combo.setCurrentIndex(0)
        self.canvas.update()

    def refresh(self):
        """Refresh all visuals from current state."""
        self._calculate_scale_factor()
        self._rebuild_all_visuals()

        if self._vispy_active:
            self.canvas.update()
