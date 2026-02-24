# -----------------------------------------------------------------------------
# Load Data Dialog
# -----------------------------------------------------------------------------
"""
Dialog for loading saved detector data and plots.

Supports:
- Detector pixel data: .npy, .npz, .h5, .tiff
- Saved plots: .png, .jpg, .svg, .pdf
"""

import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QPushButton, QLineEdit, QComboBox,
    QGroupBox, QFormLayout, QFileDialog, QMessageBox,
    QCheckBox, QTabWidget, QFrame, QSizePolicy,
)
from PySide6.QtGui import QPixmap, QImage

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gui.state import SimulationState


class ImagePreviewWidget(QLabel):
    """Widget for previewing images with scaling."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border: 1px solid #404040;
                border-radius: 4px;
            }
        """)
        self._pixmap = None
        self.setText("No preview")

    def set_image(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self._update_display()

    def set_array_info(self, shape, dtype, vmin, vmax):
        """Display array info instead of image."""
        self._pixmap = None
        info = f"Shape: {shape}\nDtype: {dtype}\nMin: {vmin:.4e}\nMax: {vmax:.4e}"
        self.setText(info)

    def _update_display(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()


class LoadDataDialog(QDialog):
    """Dialog for loading saved detector data and plots."""

    data_loaded = Signal(object, str)  # data, data_type

    def __init__(self, state: SimulationState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("Load Data")
        self.setMinimumSize(600, 500)

        self._loaded_data = None
        self._loaded_path = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Tabs for data type
        tabs = QTabWidget()

        # Detector pixels tab
        detector_tab = QWidget()
        self._setup_detector_tab(detector_tab)
        tabs.addTab(detector_tab, "Detector Pixels")

        # Plot image tab
        plot_tab = QWidget()
        self._setup_plot_tab(plot_tab)
        tabs.addTab(plot_tab, "Saved Plot")

        layout.addWidget(tabs)

    def _setup_detector_tab(self, widget):
        layout = QVBoxLayout(widget)

        # File selection
        file_group = QGroupBox("File Selection")
        file_layout = QFormLayout(file_group)

        file_widget = QWidget()
        file_hlayout = QHBoxLayout(file_widget)
        file_hlayout.setContentsMargins(0, 0, 0, 0)

        self.detector_file_edit = QLineEdit()
        self.detector_file_edit.setPlaceholderText("Select detector data file...")
        file_hlayout.addWidget(self.detector_file_edit, 1)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_detector_file)
        file_hlayout.addWidget(browse_btn)

        file_layout.addRow("File:", file_widget)

        self.detector_format_combo = QComboBox()
        self.detector_format_combo.addItem("Auto-detect", "auto")
        self.detector_format_combo.addItem("NumPy (.npy)", "npy")
        self.detector_format_combo.addItem("NumPy Archive (.npz)", "npz")
        self.detector_format_combo.addItem("HDF5 (.h5, .hdf5)", "hdf5")
        self.detector_format_combo.addItem("TIFF (.tiff, .tif)", "tiff")
        self.detector_format_combo.addItem("CBF (.cbf)", "cbf")
        file_layout.addRow("Format:", self.detector_format_combo)

        # Dataset key for npz/hdf5
        self.dataset_key = QLineEdit()
        self.dataset_key.setPlaceholderText("pixel_values (for npz/hdf5)")
        file_layout.addRow("Dataset key:", self.dataset_key)

        layout.addWidget(file_group)

        # Preview
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.detector_preview = ImagePreviewWidget()
        preview_layout.addWidget(self.detector_preview, 1)

        self.detector_info = QLabel("No file loaded")
        self.detector_info.setStyleSheet("color: #808080;")
        preview_layout.addWidget(self.detector_info)

        layout.addWidget(preview_group, 1)

        # Options
        options_group = QGroupBox("Load Options")
        options_layout = QVBoxLayout(options_group)

        self.replace_detector = QCheckBox("Replace current detector data")
        self.replace_detector.setChecked(True)
        options_layout.addWidget(self.replace_detector)

        self.apply_to_detector = QCheckBox("Apply to detector object")
        self.apply_to_detector.setChecked(True)
        options_layout.addWidget(self.apply_to_detector)

        self.as_overlay = QCheckBox("Load as comparison overlay")
        options_layout.addWidget(self.as_overlay)

        layout.addWidget(options_group)

        # Load button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(self._preview_detector_data)
        btn_layout.addWidget(preview_btn)

        self.load_detector_btn = QPushButton("Load")
        self.load_detector_btn.setEnabled(False)
        self.load_detector_btn.clicked.connect(self._load_detector_data)
        btn_layout.addWidget(self.load_detector_btn)

        layout.addLayout(btn_layout)

        # Connect signals
        self.detector_file_edit.textChanged.connect(self._on_detector_file_changed)

    def _setup_plot_tab(self, widget):
        layout = QVBoxLayout(widget)

        # File selection
        file_group = QGroupBox("File Selection")
        file_layout = QFormLayout(file_group)

        file_widget = QWidget()
        file_hlayout = QHBoxLayout(file_widget)
        file_hlayout.setContentsMargins(0, 0, 0, 0)

        self.plot_file_edit = QLineEdit()
        self.plot_file_edit.setPlaceholderText("Select plot image file...")
        file_hlayout.addWidget(self.plot_file_edit, 1)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_plot_file)
        file_hlayout.addWidget(browse_btn)

        file_layout.addRow("File:", file_widget)

        layout.addWidget(file_group)

        # Preview
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.plot_preview = ImagePreviewWidget()
        preview_layout.addWidget(self.plot_preview, 1)

        self.plot_info = QLabel("No file loaded")
        self.plot_info.setStyleSheet("color: #808080;")
        preview_layout.addWidget(self.plot_info)

        layout.addWidget(preview_group, 1)

        # Load button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.load_plot_btn = QPushButton("Load")
        self.load_plot_btn.setEnabled(False)
        self.load_plot_btn.clicked.connect(self._load_plot_image)
        btn_layout.addWidget(self.load_plot_btn)

        layout.addLayout(btn_layout)

        # Connect signals
        self.plot_file_edit.textChanged.connect(self._on_plot_file_changed)

    def _browse_detector_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Detector Data",
            "",
            "All Supported (*.npy *.npz *.h5 *.hdf5 *.tiff *.tif *.cbf);;"
            "NumPy (*.npy *.npz);;"
            "HDF5 (*.h5 *.hdf5);;"
            "TIFF (*.tiff *.tif);;"
            "CBF (*.cbf);;"
            "All Files (*)"
        )
        if filename:
            self.detector_file_edit.setText(filename)

    def _browse_plot_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Plot Image",
            "",
            "Images (*.png *.jpg *.jpeg *.svg *.pdf);;"
            "PNG (*.png);;"
            "JPEG (*.jpg *.jpeg);;"
            "SVG (*.svg);;"
            "PDF (*.pdf);;"
            "All Files (*)"
        )
        if filename:
            self.plot_file_edit.setText(filename)
            self._preview_plot_image()

    def _on_detector_file_changed(self, text):
        self.load_detector_btn.setEnabled(bool(text and Path(text).exists()))

    def _on_plot_file_changed(self, text):
        has_file = bool(text and Path(text).exists())
        self.load_plot_btn.setEnabled(has_file)
        if has_file:
            self._preview_plot_image()

    def _detect_format(self, path: Path) -> str:
        """Auto-detect file format from extension."""
        ext = path.suffix.lower()
        format_map = {
            '.npy': 'npy',
            '.npz': 'npz',
            '.h5': 'hdf5',
            '.hdf5': 'hdf5',
            '.tiff': 'tiff',
            '.tif': 'tiff',
            '.cbf': 'cbf',
        }
        return format_map.get(ext, 'npy')

    def _preview_detector_data(self):
        """Preview detector data file."""
        path = Path(self.detector_file_edit.text())
        if not path.exists():
            QMessageBox.warning(self, "Error", "File does not exist.")
            return

        try:
            data = self._load_detector_file(path)
            if data is None:
                return

            self._loaded_data = data
            self._loaded_path = path

            # Update preview info
            self.detector_preview.set_array_info(
                data.shape,
                str(data.dtype),
                np.nanmin(data),
                np.nanmax(data)
            )
            self.detector_info.setText(
                f"Loaded: {path.name} | Shape: {data.shape} | "
                f"Range: [{np.nanmin(data):.2e}, {np.nanmax(data):.2e}]"
            )
            self.load_detector_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {e}")
            self.detector_info.setText(f"Error: {e}")

    def _load_detector_file(self, path: Path) -> Optional[np.ndarray]:
        """Load detector data from file."""
        format_type = self.detector_format_combo.currentData()
        if format_type == "auto":
            format_type = self._detect_format(path)

        data = None
        key = self.dataset_key.text() or "pixel_values"

        if format_type == "npy":
            data = np.load(str(path))

        elif format_type == "npz":
            npz = np.load(str(path))
            if key in npz:
                data = npz[key]
            elif len(npz.files) == 1:
                data = npz[npz.files[0]]
            else:
                # Show available keys
                available = ", ".join(npz.files)
                QMessageBox.warning(
                    self, "Key Required",
                    f"NPZ file contains multiple arrays: {available}\n"
                    f"Please specify the dataset key."
                )
                return None

        elif format_type == "hdf5":
            try:
                import h5py
            except ImportError:
                QMessageBox.critical(
                    self, "Missing Dependency",
                    "h5py is required to load HDF5 files.\n"
                    "Install with: pip install h5py"
                )
                return None

            with h5py.File(str(path), 'r') as f:
                if key in f:
                    data = f[key][:]
                else:
                    # Try common paths
                    for try_key in ['detector/pixel_values', 'data', 'image']:
                        if try_key in f:
                            data = f[try_key][:]
                            break
                    if data is None:
                        QMessageBox.warning(
                            self, "Key Not Found",
                            f"Dataset '{key}' not found in HDF5 file."
                        )
                        return None

        elif format_type == "tiff":
            try:
                from PIL import Image
            except ImportError:
                QMessageBox.critical(
                    self, "Missing Dependency",
                    "Pillow is required to load TIFF files.\n"
                    "Install with: pip install Pillow"
                )
                return None

            img = Image.open(str(path))
            data = np.array(img)

        elif format_type == "cbf":
            try:
                import fabio
            except ImportError:
                QMessageBox.critical(
                    self, "Missing Dependency",
                    "fabio is required to load CBF files.\n"
                    "Install with: pip install fabio"
                )
                return None

            cbf = fabio.open(str(path))
            data = cbf.data

        return data

    def _load_detector_data(self):
        """Apply loaded detector data."""
        if self._loaded_data is None:
            self._preview_detector_data()
            if self._loaded_data is None:
                return

        try:
            if self.apply_to_detector.isChecked() and self.state.detector:
                detector = self.state.detector

                # Update detector shape if needed
                if self.replace_detector.isChecked():
                    if hasattr(detector, 'input_pixel_values'):
                        detector.input_pixel_values(self._loaded_data)
                    else:
                        # Access _pixel_values directly to avoid warning print
                        detector._pixel_values = self._loaded_data

                self.state.notify_observers("detector_changed", detector)

            self.data_loaded.emit(self._loaded_data, "detector")

            QMessageBox.information(
                self, "Success",
                f"Loaded detector data from {self._loaded_path.name}"
            )
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply data: {e}")

    def _preview_plot_image(self):
        """Preview plot image file."""
        path = Path(self.plot_file_edit.text())
        if not path.exists():
            return

        try:
            ext = path.suffix.lower()

            if ext in ['.png', '.jpg', '.jpeg']:
                pixmap = QPixmap(str(path))
                self.plot_preview.set_image(pixmap)
                self.plot_info.setText(
                    f"Loaded: {path.name} | Size: {pixmap.width()}x{pixmap.height()}"
                )

            elif ext == '.svg':
                from PySide6.QtSvg import QSvgRenderer
                from PySide6.QtGui import QPainter

                renderer = QSvgRenderer(str(path))
                size = renderer.defaultSize()

                image = QImage(size, QImage.Format_ARGB32)
                image.fill(Qt.transparent)

                painter = QPainter(image)
                renderer.render(painter)
                painter.end()

                pixmap = QPixmap.fromImage(image)
                self.plot_preview.set_image(pixmap)
                self.plot_info.setText(
                    f"Loaded: {path.name} | Size: {size.width()}x{size.height()}"
                )

            elif ext == '.pdf':
                self.plot_preview.setText("PDF preview not available")
                self.plot_info.setText(f"File: {path.name}")

        except Exception as e:
            self.plot_info.setText(f"Error loading preview: {e}")

    def _load_plot_image(self):
        """Load and display plot image."""
        path = Path(self.plot_file_edit.text())
        if not path.exists():
            QMessageBox.warning(self, "Error", "File does not exist.")
            return

        try:
            self.data_loaded.emit(str(path), "plot")
            QMessageBox.information(self, "Success", f"Loaded plot from {path.name}")
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load plot: {e}")


class ImportCIFDialog(QDialog):
    """Dialog for importing CIF crystal structure files."""

    cif_loaded = Signal(str)  # file path

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import CIF File")
        self.setMinimumSize(500, 300)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # File selection
        file_group = QGroupBox("CIF File")
        file_layout = QFormLayout(file_group)

        file_widget = QWidget()
        file_hlayout = QHBoxLayout(file_widget)
        file_hlayout.setContentsMargins(0, 0, 0, 0)

        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Select CIF file...")
        file_hlayout.addWidget(self.file_edit, 1)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        file_hlayout.addWidget(browse_btn)

        file_layout.addRow("File:", file_widget)
        layout.addWidget(file_group)

        # Preview
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)

        from PySide6.QtWidgets import QTextEdit
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setStyleSheet("font-family: monospace;")
        preview_layout.addWidget(self.preview_text)

        layout.addWidget(preview_group, 1)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        self.load_btn = QPushButton("Load")
        self.load_btn.setEnabled(False)
        self.load_btn.clicked.connect(self._load)
        btn_layout.addWidget(self.load_btn)

        layout.addLayout(btn_layout)

        # Signals
        self.file_edit.textChanged.connect(self._on_file_changed)

    def _browse(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open CIF File",
            "",
            "CIF Files (*.cif);;All Files (*)"
        )
        if filename:
            self.file_edit.setText(filename)

    def _on_file_changed(self, text):
        path = Path(text)
        has_file = path.exists() and path.suffix.lower() == '.cif'
        self.load_btn.setEnabled(has_file)

        if has_file:
            self._preview_cif(path)

    def _preview_cif(self, path: Path):
        """Preview CIF file contents."""
        try:
            with open(path, 'r') as f:
                content = f.read(5000)  # First 5KB

            if len(content) == 5000:
                content += "\n\n... (truncated)"

            self.preview_text.setText(content)

        except Exception as e:
            self.preview_text.setText(f"Error reading file: {e}")

    def _load(self):
        path = self.file_edit.text()
        if Path(path).exists():
            self.cif_loaded.emit(path)
            self.accept()


class ImportDeformationDialog(QDialog):
    """Dialog for importing deformation field files."""

    deformation_loaded = Signal(str, str)  # file path, field type

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Deformation Field")
        self.setMinimumSize(500, 400)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Import mode
        mode_group = QGroupBox("Import Mode")
        mode_layout = QVBoxLayout(mode_group)

        from PySide6.QtWidgets import QRadioButton, QButtonGroup

        self.mode_group = QButtonGroup()

        self.field_radio = QRadioButton("Displacement/Strain Field")
        self.field_radio.setChecked(True)
        self.mode_group.addButton(self.field_radio, 0)
        mode_layout.addWidget(self.field_radio)

        self.mesh_radio = QRadioButton("FE Mesh (Abaqus, etc.)")
        self.mode_group.addButton(self.mesh_radio, 1)
        mode_layout.addWidget(self.mesh_radio)

        layout.addWidget(mode_group)

        # File selection
        file_group = QGroupBox("File Selection")
        file_layout = QFormLayout(file_group)

        file_widget = QWidget()
        file_hlayout = QHBoxLayout(file_widget)
        file_hlayout.setContentsMargins(0, 0, 0, 0)

        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Select deformation file...")
        file_hlayout.addWidget(self.file_edit, 1)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        file_hlayout.addWidget(browse_btn)

        file_layout.addRow("File:", file_widget)

        self.field_type_combo = QComboBox()
        self.field_type_combo.addItem("Displacement (u)", "displacement")
        self.field_type_combo.addItem("Strain (ε)", "strain")
        self.field_type_combo.addItem("Rotation (ω)", "rotation")
        file_layout.addRow("Field Type:", self.field_type_combo)

        layout.addWidget(file_group)

        # Info
        info_group = QGroupBox("File Information")
        info_layout = QVBoxLayout(info_group)

        self.info_label = QLabel("No file selected")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)

        layout.addWidget(info_group, 1)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        self.load_btn = QPushButton("Load")
        self.load_btn.setEnabled(False)
        self.load_btn.clicked.connect(self._load)
        btn_layout.addWidget(self.load_btn)

        layout.addLayout(btn_layout)

        # Signals
        self.file_edit.textChanged.connect(self._on_file_changed)

    def _browse(self):
        if self.field_radio.isChecked():
            filter_str = "NumPy (*.npy *.npz);;HDF5 (*.h5 *.hdf5);;VTK (*.vtk *.vtu);;All Files (*)"
        else:
            filter_str = "Abaqus (*.inp *.odb);;ANSYS (*.rst);;All Files (*)"

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Deformation File",
            "",
            filter_str
        )
        if filename:
            self.file_edit.setText(filename)

    def _on_file_changed(self, text):
        path = Path(text)
        self.load_btn.setEnabled(path.exists())

        if path.exists():
            # Show file info
            size = path.stat().st_size
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f} MB"

            self.info_label.setText(
                f"File: {path.name}\n"
                f"Size: {size_str}\n"
                f"Format: {path.suffix}"
            )

    def _load(self):
        path = self.file_edit.text()
        field_type = self.field_type_combo.currentData()

        if Path(path).exists():
            self.deformation_loaded.emit(path, field_type)
            self.accept()
