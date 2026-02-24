# -----------------------------------------------------------------------------
# Bottom Tabs Panel
# -----------------------------------------------------------------------------
"""
Tabbed panel for detector view, analysis output, and scan progress.

The BottomTabs panel provides:
- Detector image view tab
- Analysis results tab
- Scan progress tab
- Log output tab
"""

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QLabel,
    QProgressBar,
    QPushButton,
    QFrame,
    QSplitter,
    QScrollArea,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gui.state import SimulationState


class BottomTabs(QWidget):
    """
    Tabbed panel for detector, analysis, and scan views.

    Signals:
        tab_changed: Emitted when tab selection changes (index)
    """

    tab_changed = Signal(int)

    def __init__(self, state: SimulationState, parent=None):
        """
        Initialize the bottom tabs panel.

        Args:
            state: SimulationState instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.state = state
        self._setup_ui()
        self._register_observers()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(self.tab_changed.emit)

        # Detector tab
        self.detector_tab = DetectorViewTab(self.state)
        self.tabs.addTab(self.detector_tab, "Detector")

        # Analysis tab
        self.analysis_tab = AnalysisViewTab(self.state)
        self.tabs.addTab(self.analysis_tab, "Analysis")

        # Scan tab
        self.scan_tab = ScanProgressTab(self.state)
        self.tabs.addTab(self.scan_tab, "Scan")

        layout.addWidget(self.tabs)

    def _register_observers(self):
        """Register for state change notifications."""
        self.state.register_observer("detector_changed", self._on_detector_changed)
        self.state.register_observer("simulation_finished", self._on_simulation_finished)
        self.state.register_observer("simulation_progress", self._on_simulation_progress)

    def _on_detector_changed(self, detector):
        """Handle detector change."""
        self.detector_tab.refresh()

    def _on_simulation_finished(self, result):
        """Handle simulation finished."""
        self.detector_tab.refresh()
        self.scan_tab.on_finished()

    def _on_simulation_progress(self, data):
        """Handle simulation progress."""
        if data:
            self.scan_tab.update_progress(
                data.get("progress", 0),
                data.get("message", "")
            )

    def set_detector_widget(self, widget: QWidget):
        """Replace the detector view widget."""
        self.detector_tab.set_view_widget(widget)

    def set_analysis_widget(self, widget: QWidget):
        """Replace the analysis view widget."""
        self.analysis_tab.set_view_widget(widget)


class DetectorViewTab(QWidget):
    """Tab for displaying detector image."""

    def __init__(self, state: SimulationState, parent=None):
        super().__init__(parent)
        self.state = state
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QFrame()
        toolbar.setStyleSheet("background-color: #2d2d2d; border-bottom: 1px solid #404040;")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 4, 8, 4)
        toolbar_layout.setSpacing(8)

        toolbar_layout.addWidget(QLabel("Display:"))

        self.display_combo = None  # Will be a combo box for intensity/phase/amplitude
        # Placeholder for now
        toolbar_layout.addWidget(QLabel("Intensity"))

        toolbar_layout.addStretch()

        # Save button
        save_btn = QPushButton("Save Image")
        save_btn.clicked.connect(self._on_save)
        toolbar_layout.addWidget(save_btn)

        layout.addWidget(toolbar)

        # View area (placeholder)
        self.view_container = QWidget()
        self.view_layout = QVBoxLayout(self.view_container)
        self.view_layout.setContentsMargins(0, 0, 0, 0)

        self.placeholder = QLabel("Detector View\n\nRun a simulation to see detector output")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet("color: #808080; font-size: 14px;")
        self.view_layout.addWidget(self.placeholder)

        layout.addWidget(self.view_container)

    def set_view_widget(self, widget: QWidget):
        """Replace the view widget."""
        # Clear existing
        while self.view_layout.count():
            item = self.view_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.view_layout.addWidget(widget)

    def refresh(self):
        """Refresh the detector view."""
        # Will be implemented with actual detector visualization
        pass

    def _on_save(self):
        """Handle save button click."""
        from PySide6.QtWidgets import QFileDialog
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Detector Image",
            "",
            "PNG (*.png);;TIFF (*.tiff);;NumPy (*.npy)"
        )
        if filename:
            # TODO: Implement save
            pass


class AnalysisViewTab(QWidget):
    """Tab for displaying analysis results."""

    def __init__(self, state: SimulationState, parent=None):
        super().__init__(parent)
        self.state = state
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QFrame()
        toolbar.setStyleSheet("background-color: #2d2d2d; border-bottom: 1px solid #404040;")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 4, 8, 4)
        toolbar_layout.setSpacing(8)

        toolbar_layout.addWidget(QLabel("Analysis Results"))
        toolbar_layout.addStretch()

        # Export button
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self._on_export)
        toolbar_layout.addWidget(export_btn)

        layout.addWidget(toolbar)

        # View area (placeholder for matplotlib)
        self.view_container = QWidget()
        self.view_layout = QVBoxLayout(self.view_container)
        self.view_layout.setContentsMargins(0, 0, 0, 0)

        self.placeholder = QLabel("Analysis View\n\nRun an analysis to see results")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet("color: #808080; font-size: 14px;")
        self.view_layout.addWidget(self.placeholder)

        layout.addWidget(self.view_container)

    def set_view_widget(self, widget: QWidget):
        """Replace the view widget."""
        while self.view_layout.count():
            item = self.view_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.view_layout.addWidget(widget)

    def _on_export(self):
        """Handle export button click."""
        from PySide6.QtWidgets import QFileDialog
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Analysis",
            "",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)"
        )
        if filename:
            # TODO: Implement export
            pass


class ScanProgressTab(QWidget):
    """Tab for displaying scan progress."""

    # Signals
    pause_requested = Signal()
    stop_requested = Signal()

    def __init__(self, state: SimulationState, parent=None):
        super().__init__(parent)
        self.state = state
        self._is_running = False
        self._is_paused = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Title
        title = QLabel("Scan Progress")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        # Status message
        self.status_label = QLabel("No scan running")
        self.status_label.setStyleSheet("color: #808080;")
        layout.addWidget(self.status_label)

        # Statistics frame
        stats_frame = QFrame()
        stats_frame.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 10px;
            }
        """)
        stats_layout = QVBoxLayout(stats_frame)
        stats_layout.setSpacing(8)

        # Stats labels
        self.elapsed_label = QLabel("Elapsed: --:--:--")
        self.remaining_label = QLabel("Remaining: --:--:--")
        self.points_label = QLabel("Points: 0 / 0")
        self.rate_label = QLabel("Rate: -- points/sec")

        for label in [self.elapsed_label, self.remaining_label, self.points_label, self.rate_label]:
            label.setStyleSheet("color: #a0a0a0;")
            stats_layout.addWidget(label)

        layout.addWidget(stats_frame)

        # Control buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._on_pause)
        btn_layout.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        btn_layout.addWidget(self.stop_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        layout.addStretch()

    def update_progress(self, progress: float, message: str = ""):
        """
        Update the progress display.

        Args:
            progress: Progress value (0.0 to 1.0)
            message: Status message
        """
        self._is_running = True
        self.progress_bar.setValue(int(progress * 100))
        if message:
            self.status_label.setText(message)

        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)

    def on_started(self, total_points: int = 0):
        """Handle scan started."""
        self._is_running = True
        self._is_paused = False
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting scan...")
        self.points_label.setText(f"Points: 0 / {total_points}")
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.pause_btn.setText("Pause")

    def on_finished(self):
        """Handle scan finished."""
        self._is_running = False
        self._is_paused = False
        self.status_label.setText("Scan complete")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setText("Pause")

    def _on_pause(self):
        """Handle pause button click."""
        if self._is_paused:
            self._is_paused = False
            self.pause_btn.setText("Pause")
            self.status_label.setText("Resuming...")
        else:
            self._is_paused = True
            self.pause_btn.setText("Resume")
            self.status_label.setText("Paused")

        self.pause_requested.emit()

    def _on_stop(self):
        """Handle stop button click."""
        self.stop_requested.emit()
        self.status_label.setText("Stopping...")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
