# -----------------------------------------------------------------------------
# Main Window
# -----------------------------------------------------------------------------
"""
Main application window for the X-ray simulator GUI.

Provides:
- Dockable panel layout
- Menu bar and toolbar
- Central 3D viewport
- Status bar with GPU monitoring
"""

import sys
import os
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports when running directly
_gui_dir = Path(__file__).parent
_project_dir = _gui_dir.parent
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

from PySide6.QtCore import Qt, QSettings, QSize, Signal, Slot
from PySide6.QtWidgets import (
    QMainWindow,
    QDockWidget,
    QToolBar,
    QStatusBar,
    QMenuBar,
    QMenu,
    QFileDialog,
    QMessageBox,
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
    QProgressBar,
    QSplitter,
    QTabWidget,
    QStackedWidget,
)
from PySide6.QtGui import QAction, QKeySequence, QIcon

# Use absolute imports to support both package and direct execution
try:
    from gui.state import SimulationState
    from gui.panels.object_browser import ObjectBrowser
    from gui.panels.console_panel import ConsolePanel
    from gui.panels.simulation_method_panel import SimulationMethodPanel
    from gui.panels.run_simulation_panel import RunSimulationPanel
    from gui.panels.global_working_directory_panel import GlobalWorkingDirectoryPanel
    from gui.inspectors.crystal_inspector import CrystalInspector
    from gui.inspectors.sample_inspector import SampleInspector
    from gui.inspectors.beam_inspector import BeamInspector
    from gui.inspectors.detector_inspector import DetectorInspector
    from gui.inspectors.stage_inspector import StageInspector
    from gui.inspectors.optics_inspector import OpticsInspector
    from gui.inspectors.defects_inspector import DefectsInspector
    from gui.inspectors.deformation_inspector import DeformationInspector
    from gui.inspectors.analysis_inspector import AnalysisInspector
    from gui.viewers.detector_view import DetectorView
    from gui.viewers.analysis_view import AnalysisView
    from gui.viewers.viewport_3d import Viewport3D
    from gui.dialogs.alignment_dialog import AlignmentDialog
except ImportError:
    from state import SimulationState
    from panels.object_browser import ObjectBrowser
    from panels.console_panel import ConsolePanel
    from panels.simulation_method_panel import SimulationMethodPanel
    from panels.run_simulation_panel import RunSimulationPanel
    from panels.global_working_directory_panel import GlobalWorkingDirectoryPanel
    from inspectors.crystal_inspector import CrystalInspector
    from inspectors.sample_inspector import SampleInspector
    from inspectors.beam_inspector import BeamInspector
    from inspectors.detector_inspector import DetectorInspector
    from inspectors.stage_inspector import StageInspector
    from inspectors.optics_inspector import OpticsInspector
    from inspectors.defects_inspector import DefectsInspector
    from inspectors.deformation_inspector import DeformationInspector
    from inspectors.analysis_inspector import AnalysisInspector
    from viewers.detector_view import DetectorView
    from viewers.analysis_view import AnalysisView
    from viewers.viewport_3d import Viewport3D
    from dialogs.alignment_dialog import AlignmentDialog


class MainWindow(QMainWindow):
    """
    Main application window with dockable panels.

    Layout:
    +-----------------------------------------------------------------------------+
    |  Menu Bar: File | Simulation | View | Tools | Help                          |
    +-----------------------------------------------------------------------------+
    |  Toolbar: [Run] [Stop] [Save] [Load] | GPU Status                           |
    +-------------+-----------------------------------------------+---------------+
    |             |                                               |               |
    |  Object     |         Central 3D Viewport                   |  Inspector    |
    |  Browser    |         (VisPy Canvas)                        |  Panel        |
    |             |                                               |               |
    +-------------+-----------------------------------------------+---------------+
    |             |                                               |               |
    |             |  Bottom Tabs: Detector | Analysis | Log       |  Console      |
    +-------------+-----------------------------------------------+---------------+
    """

    # Signals
    simulation_requested = Signal()
    stop_requested = Signal()

    def __init__(self, state: SimulationState = None):
        """
        Initialize the main window.

        Args:
            state: SimulationState instance (creates new one if None)
        """
        super().__init__()

        # State management
        self.state = state if state is not None else SimulationState()

        # Window properties
        self.setWindowTitle("X-ray Diffraction Simulator")
        self.setMinimumSize(1200, 800)

        # Initialize components
        self._setup_central_widget()
        self._setup_docks()
        self._setup_menus()
        self._setup_toolbar()
        self._setup_statusbar()

        # Register state observers
        self._register_observers()

        # Restore window state
        self._restore_settings()

    # -------------------------------------------------------------------------
    # Setup Methods
    # -------------------------------------------------------------------------

    def _setup_central_widget(self):
        """Setup the central widget with 3D viewport and bottom tabs."""
        # Create a splitter for viewport and bottom tabs
        self.central_splitter = QSplitter(Qt.Vertical)

        # 3D Viewport for experimental schematic visualization
        self.viewport_3d = Viewport3D(self.state)

        # Bottom tab widget for detector view, analysis, etc.
        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.setMinimumHeight(200)

        # Detector View - full plotting controls
        self.detector_view = DetectorView(self.state)

        # Analysis View - integration controls
        self.analysis_view = AnalysisView(self.state)

        # Scan progress tab placeholder
        scan_tab = QLabel("Scan Progress")
        scan_tab.setAlignment(Qt.AlignCenter)
        scan_tab.setStyleSheet("background-color: #2d2d2d; color: #808080;")

        self.bottom_tabs.addTab(self.detector_view, "Detector")
        self.bottom_tabs.addTab(self.analysis_view, "Analysis")
        self.bottom_tabs.addTab(scan_tab, "Scan")

        # Add to splitter
        self.central_splitter.addWidget(self.viewport_3d)
        self.central_splitter.addWidget(self.bottom_tabs)
        self.central_splitter.setSizes([400, 400])

        self.setCentralWidget(self.central_splitter)

    def _setup_docks(self):
        """Setup dockable panels."""
        # Global Working Directory Panel (top of left panel)
        self.global_dir_dock = QDockWidget("Working Directory", self)
        self.global_dir_dock.setObjectName("GlobalWorkingDirectoryDock")
        self.global_dir_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.global_dir_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )

        self.global_dir_panel = GlobalWorkingDirectoryPanel(self.state)
        self.global_dir_panel.setMinimumWidth(180)
        self.global_dir_panel.setMaximumHeight(90)
        self.global_dir_dock.setWidget(self.global_dir_panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.global_dir_dock)

        # Object Browser (left, below global directory)
        self.object_browser_dock = QDockWidget("Object Browser", self)
        self.object_browser_dock.setObjectName("ObjectBrowserDock")
        self.object_browser_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # Create real ObjectBrowser
        self.object_browser = ObjectBrowser(self.state)
        self.object_browser.setMinimumWidth(180)
        self.object_browser_dock.setWidget(self.object_browser)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.object_browser_dock)

        # Stack object browser below global directory
        self.splitDockWidget(self.global_dir_dock, self.object_browser_dock, Qt.Vertical)

        # Connect ObjectBrowser signals
        self.object_browser.object_selected.connect(self._on_object_selected)
        self.object_browser.object_activated.connect(self._on_object_activated)
        self.object_browser.create_requested.connect(self._on_create_object_requested)
        self.object_browser.delete_requested.connect(self._on_delete_object_requested)

        # Simulation Method Panel (below Object Browser)
        self.sim_method_dock = QDockWidget("Simulation Method", self)
        self.sim_method_dock.setObjectName("SimulationMethodDock")
        self.sim_method_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.simulation_method_panel = SimulationMethodPanel(self.state)
        self.simulation_method_panel.setMinimumWidth(180)
        self.sim_method_dock.setWidget(self.simulation_method_panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.sim_method_dock)

        # Stack simulation method below object browser
        self.splitDockWidget(self.object_browser_dock, self.sim_method_dock, Qt.Vertical)

        # Run Simulation Panel (below Simulation Method)
        self.run_sim_dock = QDockWidget("Run Simulation", self)
        self.run_sim_dock.setObjectName("RunSimulationDock")
        self.run_sim_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.run_simulation_panel = RunSimulationPanel(self.state, self.simulation_method_panel)
        self.run_simulation_panel.setMinimumWidth(180)
        self.run_sim_dock.setWidget(self.run_simulation_panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.run_sim_dock)

        # Stack run simulation below simulation method
        self.splitDockWidget(self.sim_method_dock, self.run_sim_dock, Qt.Vertical)

        # Connect simulation panel signals
        self.run_simulation_panel.simulation_started.connect(self._on_simulation_started)
        self.run_simulation_panel.simulation_finished.connect(self._on_simulation_finished)
        self.run_simulation_panel.simulation_error.connect(self._on_simulation_error)

        # Inspector Panel (right) - using stacked widget for multiple inspectors
        self.inspector_dock = QDockWidget("Inspector", self)
        self.inspector_dock.setObjectName("InspectorDock")
        self.inspector_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # Create stacked widget to hold all inspectors
        self.inspector_stack = QStackedWidget()
        self.inspector_stack.setMinimumWidth(280)

        # Create placeholder for when nothing is selected
        self.inspector_placeholder = QLabel("Select an object to view properties")
        self.inspector_placeholder.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.inspector_placeholder.setStyleSheet("padding: 20px; background-color: #2d2d2d; color: #808080;")
        self.inspector_stack.addWidget(self.inspector_placeholder)  # Index 0

        # Create all inspector panels
        self.inspectors = {}
        self.inspectors["crystal"] = CrystalInspector(self.state)
        self.inspectors["sample"] = SampleInspector(self.state)
        self.inspectors["beam"] = BeamInspector(self.state)
        self.inspectors["detector"] = DetectorInspector(self.state)
        self.inspectors["stage"] = StageInspector(self.state)
        self.inspectors["optics"] = OpticsInspector(self.state)
        self.inspectors["defects"] = DefectsInspector(self.state)
        self.inspectors["deformation"] = DeformationInspector(self.state)
        self.inspectors["analysis"] = AnalysisInspector(self.state)

        # Add inspectors to stack (indices 1-9)
        for name, inspector in self.inspectors.items():
            self.inspector_stack.addWidget(inspector)

        self.inspector_dock.setWidget(self.inspector_stack)
        self.addDockWidget(Qt.RightDockWidgetArea, self.inspector_dock)

        # Connect Crystal Inspector signals for diffraction peaks
        crystal_inspector = self.inspectors.get("crystal")
        if crystal_inspector:
            crystal_inspector.peak_selection_changed.connect(self._on_peak_selection_changed)
            crystal_inspector.align_to_peak_requested.connect(self._on_align_to_peak_requested)

        # Console Panel (bottom right)
        self.console_dock = QDockWidget("Console", self)
        self.console_dock.setObjectName("ConsoleDock")
        self.console_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea)

        # Create real ConsolePanel
        try:
            self.console_panel = ConsolePanel(self.state)
            self.console_dock.setWidget(self.console_panel)
        except Exception:
            # Fallback to placeholder if ConsolePanel fails to load
            console_placeholder = QLabel("Console Output")
            console_placeholder.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            console_placeholder.setStyleSheet("padding: 10px; background-color: #1e1e1e; color: #808080; font-family: monospace;")
            self.console_dock.setWidget(console_placeholder)

        self.addDockWidget(Qt.RightDockWidgetArea, self.console_dock)

        # Stack console under inspector
        self.splitDockWidget(self.inspector_dock, self.console_dock, Qt.Vertical)

    def _setup_menus(self):
        """Setup the menu bar."""
        menubar = self.menuBar()

        # ----- File Menu -----
        file_menu = menubar.addMenu("&File")

        new_action = QAction("&New Simulation", self)
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self._on_new)
        file_menu.addAction(new_action)

        open_action = QAction("&Open Project...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._on_open)
        file_menu.addAction(open_action)

        save_action = QAction("&Save Project", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self._on_save)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save Project &As...", self)
        save_as_action.setShortcut(QKeySequence.SaveAs)
        save_as_action.triggered.connect(self._on_save_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        # Load Data submenu
        load_menu = file_menu.addMenu("&Load Data")
        load_detector_action = QAction("Detector Pixels...", self)
        load_detector_action.triggered.connect(self._on_load_detector_pixels)
        load_menu.addAction(load_detector_action)

        load_plot_action = QAction("Saved Plot...", self)
        load_plot_action.triggered.connect(self._on_load_saved_plot)
        load_menu.addAction(load_plot_action)

        # Export submenu
        export_menu = file_menu.addMenu("&Export")
        export_detector_action = QAction("Detector Image...", self)
        export_detector_action.triggered.connect(self._on_export_detector)
        export_menu.addAction(export_detector_action)

        export_plot_action = QAction("Current Plot...", self)
        export_plot_action.triggered.connect(self._on_export_plot)
        export_menu.addAction(export_plot_action)

        export_script_action = QAction("Python Script...", self)
        export_script_action.triggered.connect(self._on_export_script)
        export_menu.addAction(export_script_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # ----- Simulation Menu -----
        sim_menu = menubar.addMenu("&Simulation")

        run_action = QAction("&Run Simulation", self)
        run_action.setShortcut("F5")
        run_action.triggered.connect(self._on_run_simulation)
        sim_menu.addAction(run_action)

        stop_action = QAction("&Stop", self)
        stop_action.setShortcut("Shift+F5")
        stop_action.triggered.connect(self._on_stop_simulation)
        sim_menu.addAction(stop_action)

        sim_menu.addSeparator()

        scan_wizard_action = QAction("Scan &Wizard...", self)
        scan_wizard_action.triggered.connect(self._on_scan_wizard)
        sim_menu.addAction(scan_wizard_action)

        # ----- View Menu -----
        view_menu = menubar.addMenu("&View")

        view_menu.addAction(self.global_dir_dock.toggleViewAction())
        view_menu.addAction(self.object_browser_dock.toggleViewAction())
        view_menu.addAction(self.inspector_dock.toggleViewAction())
        view_menu.addAction(self.console_dock.toggleViewAction())

        view_menu.addSeparator()

        reset_layout_action = QAction("&Reset Layout", self)
        reset_layout_action.triggered.connect(self._on_reset_layout)
        view_menu.addAction(reset_layout_action)

        # ----- Tools Menu -----
        tools_menu = menubar.addMenu("&Tools")

        presets_action = QAction("&Presets...", self)
        presets_action.triggered.connect(self._on_open_presets)
        tools_menu.addAction(presets_action)

        tools_menu.addSeparator()

        settings_action = QAction("&Settings...", self)
        settings_action.triggered.connect(self._on_settings)
        tools_menu.addAction(settings_action)

        # ----- Help Menu -----
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _setup_toolbar(self):
        """Setup the main toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setObjectName("MainToolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        # Run button
        self.run_action = QAction("Run", self)
        self.run_action.setToolTip("Run simulation (F5)")
        self.run_action.triggered.connect(self._on_run_simulation)
        toolbar.addAction(self.run_action)

        # Stop button
        self.stop_action = QAction("Stop", self)
        self.stop_action.setToolTip("Stop simulation (Shift+F5)")
        self.stop_action.setEnabled(False)
        self.stop_action.triggered.connect(self._on_stop_simulation)
        toolbar.addAction(self.stop_action)

        toolbar.addSeparator()

        # Save button
        save_action = QAction("Save", self)
        save_action.setToolTip("Save project (Ctrl+S)")
        save_action.triggered.connect(self._on_save)
        toolbar.addAction(save_action)

        # Load button
        load_action = QAction("Load", self)
        load_action.setToolTip("Open project (Ctrl+O)")
        load_action.triggered.connect(self._on_open)
        toolbar.addAction(load_action)

        toolbar.addSeparator()

        # Preset button
        preset_action = QAction("Presets", self)
        preset_action.setToolTip("Manage presets")
        preset_action.triggered.connect(self._on_open_presets)
        toolbar.addAction(preset_action)

        # Add spacer
        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy(), spacer.sizePolicy().verticalPolicy())
        spacer.setMinimumWidth(20)
        toolbar.addWidget(spacer)

        # GPU status label (will be replaced with actual GPU monitor widget)
        self.gpu_label = QLabel("GPU: --")
        self.gpu_label.setStyleSheet("padding: 0 10px; color: #808080;")
        toolbar.addWidget(self.gpu_label)

    def _setup_statusbar(self):
        """Setup the status bar."""
        statusbar = QStatusBar()
        self.setStatusBar(statusbar)

        # Status message
        self.status_label = QLabel("Ready")
        statusbar.addWidget(self.status_label, 1)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        statusbar.addPermanentWidget(self.progress_bar)

    def _register_observers(self):
        """Register for state change notifications."""
        self.state.register_observer("simulation_started", self._on_simulation_started)
        self.state.register_observer("simulation_finished", self._on_simulation_finished)
        self.state.register_observer("simulation_progress", self._on_simulation_progress)
        self.state.register_observer("error_occurred", self._on_error)
        self.state.register_observer("project_changed", self._on_project_changed)

    # -------------------------------------------------------------------------
    # Settings Persistence
    # -------------------------------------------------------------------------

    def _save_settings(self):
        """Save window state and geometry."""
        settings = QSettings("XraySimulator", "GUI")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())

    def _restore_settings(self):
        """Restore window state and geometry."""
        settings = QSettings("XraySimulator", "GUI")
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        state = settings.value("windowState")
        if state:
            self.restoreState(state)

    # -------------------------------------------------------------------------
    # Menu Action Handlers
    # -------------------------------------------------------------------------

    def _on_new(self):
        """Handle new simulation action."""
        if self.state.is_dirty:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "There are unsaved changes. Do you want to save before creating a new simulation?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            if reply == QMessageBox.Save:
                self._on_save()
            elif reply == QMessageBox.Cancel:
                return

        self.state.new_project()
        self.status_label.setText("New simulation created")
        self._update_window_title()

    def _on_open(self):
        """Handle open project action."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Project",
            self.state.working_directory,
            "X-ray Simulator Project (*.xrp);;JSON (*.json);;All Files (*)"
        )
        if filename:
            try:
                self.state.load_from_file(filename)
                self.state.working_directory = str(Path(filename).parent)
                self.status_label.setText(f"Opened: {Path(filename).name}")
                self._update_window_title()
            except Exception as e:
                QMessageBox.critical(
                    self, "Error Loading Project",
                    f"Failed to load project:\n{str(e)}"
                )

    def _on_save(self):
        """Handle save project action."""
        if self.state.metadata.file_path:
            try:
                self.state.save_to_file(self.state.metadata.file_path)
                self.status_label.setText("Project saved")
                self._update_window_title()
            except Exception as e:
                QMessageBox.critical(
                    self, "Error Saving Project",
                    f"Failed to save project:\n{str(e)}"
                )
        else:
            self._on_save_as()

    def _on_save_as(self):
        """Handle save as action."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Project As",
            self.state.working_directory,
            "X-ray Simulator Project (*.xrp);;JSON (*.json)"
        )
        if filename:
            # Ensure file has extension
            if not filename.endswith('.xrp') and not filename.endswith('.json'):
                filename += '.xrp'
            try:
                self.state.save_to_file(filename)
                self.state.working_directory = str(Path(filename).parent)
                self.status_label.setText(f"Saved: {Path(filename).name}")
                self._update_window_title()
            except Exception as e:
                QMessageBox.critical(
                    self, "Error Saving Project",
                    f"Failed to save project:\n{str(e)}"
                )

    def _on_load_detector_pixels(self):
        """Handle load detector pixels action."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Detector Pixels",
            self.state.working_directory,
            "NumPy (*.npy *.npz);;HDF5 (*.h5 *.hdf5);;TIFF (*.tiff *.tif);;All Files (*)"
        )
        if filename:
            # TODO: Implement detector pixel loading
            self.status_label.setText(f"Loaded detector data: {Path(filename).name}")

    def _on_load_saved_plot(self):
        """Handle load saved plot action."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Saved Plot",
            self.state.working_directory,
            "Images (*.png *.jpg *.svg *.pdf);;All Files (*)"
        )
        if filename:
            # TODO: Implement plot loading
            self.status_label.setText(f"Loaded plot: {Path(filename).name}")

    def _on_export_detector(self):
        """Handle export detector image action."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Detector Image",
            self.state.working_directory,
            "PNG (*.png);;TIFF (*.tiff);;NumPy (*.npy)"
        )
        if filename:
            # TODO: Implement detector export
            self.status_label.setText(f"Exported: {Path(filename).name}")

    def _on_export_plot(self):
        """Handle export current plot action."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Plot",
            self.state.working_directory,
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)"
        )
        if filename:
            # TODO: Implement plot export
            self.status_label.setText(f"Exported: {Path(filename).name}")

    def _on_export_script(self):
        """Handle export Python script action."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Python Script",
            self.state.working_directory,
            "Python (*.py)"
        )
        if filename:
            # TODO: Implement script export
            self.status_label.setText(f"Exported script: {Path(filename).name}")

    def _on_run_simulation(self):
        """Handle run simulation action."""
        self.simulation_requested.emit()
        self.run_action.setEnabled(False)
        self.stop_action.setEnabled(True)
        self.state.notify_simulation_started()

    def _on_stop_simulation(self):
        """Handle stop simulation action."""
        self.stop_requested.emit()
        self.stop_action.setEnabled(False)

    def _on_scan_wizard(self):
        """Handle scan wizard action."""
        try:
            from gui.dialogs.scan_wizard import ScanWizard
            wizard = ScanWizard(self.state, parent=self)
            wizard.exec()
        except ImportError as e:
            QMessageBox.warning(
                self, "Import Error",
                f"Could not load scan wizard:\n{str(e)}"
            )
            self.status_label.setText("Scan wizard failed to load")
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to open scan wizard:\n{str(e)}"
            )

    def _on_reset_layout(self):
        """Reset window layout to defaults."""
        self.global_dir_dock.setVisible(True)
        self.object_browser_dock.setVisible(True)
        self.inspector_dock.setVisible(True)
        self.console_dock.setVisible(True)

        self.addDockWidget(Qt.LeftDockWidgetArea, self.global_dir_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.object_browser_dock)
        self.splitDockWidget(self.global_dir_dock, self.object_browser_dock, Qt.Vertical)
        self.addDockWidget(Qt.RightDockWidgetArea, self.inspector_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.console_dock)
        self.splitDockWidget(self.inspector_dock, self.console_dock, Qt.Vertical)

    def _on_open_presets(self):
        """Handle open presets dialog action."""
        # TODO: Show presets dialog
        self.status_label.setText("Presets dialog not yet implemented")

    def _on_settings(self):
        """Handle settings dialog action."""
        # TODO: Show settings dialog
        self.status_label.setText("Settings not yet implemented")

    def _on_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About X-ray Simulator",
            "X-ray Diffraction Simulator\n\n"
            "A comprehensive GUI for X-ray diffraction simulations.\n\n"
            "Version 1.0.0"
        )

    # -------------------------------------------------------------------------
    # State Observer Callbacks
    # -------------------------------------------------------------------------

    def _on_simulation_started(self, data=None):
        """Handle simulation started event."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Simulation running...")

    def _on_simulation_finished(self, result=None):
        """Handle simulation finished event."""
        self.run_action.setEnabled(True)
        self.stop_action.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Simulation complete")

    def _on_simulation_progress(self, data=None):
        """Handle simulation progress event."""
        if data:
            progress = int(data.get("progress", 0) * 100)
            message = data.get("message", "")
            self.progress_bar.setValue(progress)
            if message:
                self.status_label.setText(message)

    def _on_error(self, error=None):
        """Handle error event."""
        self.run_action.setEnabled(True)
        self.stop_action.setEnabled(False)
        self.progress_bar.setVisible(False)
        if error:
            QMessageBox.critical(self, "Error", str(error))
            self.status_label.setText(f"Error: {error}")

    def _on_project_changed(self, data=None):
        """Handle project changed event."""
        self._update_window_title()

    # -------------------------------------------------------------------------
    # Object Browser Signal Handlers
    # -------------------------------------------------------------------------

    def _on_object_selected(self, obj_type: str, obj):
        """
        Handle object selection in the browser.

        Args:
            obj_type: Type of object selected (e.g., "crystal", "beam")
            obj: The actual object instance (may be None if not created)
        """
        # Switch to the appropriate inspector
        if obj_type in self.inspectors:
            inspector = self.inspectors[obj_type]
            self.inspector_stack.setCurrentWidget(inspector)

            # Update the dock title
            self.inspector_dock.setWindowTitle(f"Inspector - {obj_type.title()}")

            # Update the inspector with the current object
            if hasattr(inspector, 'set_object'):
                inspector.set_object(obj)

            self.status_label.setText(f"Selected: {obj_type.title()}")
        else:
            # Show placeholder if no matching inspector
            self.inspector_stack.setCurrentIndex(0)
            self.inspector_dock.setWindowTitle("Inspector")

    def _on_object_activated(self, obj_type: str, obj):
        """
        Handle object double-click (activation) in the browser.

        Args:
            obj_type: Type of object activated
            obj: The object instance
        """
        # Same as selection for now - shows the inspector
        self._on_object_selected(obj_type, obj)

    def _on_create_object_requested(self, obj_type: str):
        """
        Handle request to create a new object.

        Args:
            obj_type: Type of object to create
        """
        self.status_label.setText(f"Creating {obj_type}...")

        try:
            if obj_type == "crystal":
                # For crystal, show the inspector and let user load CIF
                self._on_object_selected(obj_type, None)
                self.status_label.setText("Select a CIF file to load crystal")

            elif obj_type == "sample":
                # For sample, show the inspector and let user select directory first
                # Sample requires a directory for chunk files - don't create until directory selected
                self._on_object_selected(obj_type, None)
                self.status_label.setText("Select a working directory for sample files")

            elif obj_type == "beam":
                # For beam, show the inspector and let user select directory first
                # Beam requires a directory for metadata files - don't create until directory selected
                self._on_object_selected(obj_type, None)
                self.status_label.setText("Select a working directory for beam files")

            elif obj_type == "detector":
                # For detector, show the inspector and let user select directory first
                # Detector requires a directory for metadata files - don't create until directory selected
                self._on_object_selected(obj_type, None)
                self.status_label.setText("Select a working directory for detector files")

            elif obj_type == "stage":
                # For stage, show the inspector and let user select directory first
                # Stage requires a directory for metadata files - don't create until directory selected
                self._on_object_selected(obj_type, None)
                self.status_label.setText("Select a working directory for stage files")

            elif obj_type == "optics":
                # For optics, show the inspector and let user select directory first
                # Optics requires a directory for metadata files - don't create until directory selected
                self._on_object_selected(obj_type, None)
                self.status_label.setText("Select a working directory for optics files")

            elif obj_type == "defects":
                # For defects, show the inspector and let user select directory first
                # Defects requires a directory for metadata files - don't create until directory selected
                self._on_object_selected(obj_type, None)
                self.status_label.setText("Select a working directory for defects files")

            elif obj_type == "deformation":
                # For deformation, show the inspector and let user select directory first
                # Deformation requires a directory for files - don't create until directory selected
                self._on_object_selected(obj_type, None)
                self.status_label.setText("Select a working directory for deformation files")

            elif obj_type == "analysis":
                # Create analysis object
                from Analysis import analysis
                new_analysis = analysis(directory=self.state.working_directory)
                self.state.analysis = new_analysis
                self._on_object_selected(obj_type, new_analysis)
                self.status_label.setText("Analysis object created")

            else:
                self.status_label.setText(f"Unknown object type: {obj_type}")

        except ImportError as e:
            QMessageBox.warning(
                self, "Import Error",
                f"Could not import {obj_type} module:\n{str(e)}\n\n"
                "Make sure the X-ray simulator modules are properly installed."
            )
            self.status_label.setText(f"Failed to create {obj_type}")
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to create {obj_type}:\n{str(e)}"
            )
            self.status_label.setText(f"Error creating {obj_type}")

    def _on_delete_object_requested(self, obj_type: str):
        """
        Handle request to delete an object.

        Args:
            obj_type: Type of object deleted
        """
        # Object has already been set to None by ObjectBrowser
        self.status_label.setText(f"{obj_type.title()} deleted")

        # Show placeholder in inspector
        self.inspector_stack.setCurrentIndex(0)
        self.inspector_dock.setWindowTitle("Inspector")

    def _update_window_title(self):
        """Update window title based on project state."""
        title = "X-ray Diffraction Simulator"

        # Show file name if project has been saved
        if self.state.metadata.file_path:
            filename = Path(self.state.metadata.file_path).name
            title = f"{filename} - {title}"
        elif self.state.metadata.name != "Untitled":
            title = f"{self.state.metadata.name} - {title}"

        if self.state.is_dirty:
            title = f"*{title}"
        self.setWindowTitle(title)

    # -------------------------------------------------------------------------
    # Simulation Signal Handlers
    # -------------------------------------------------------------------------

    def _on_simulation_started(self):
        """Handle simulation started event."""
        self.status_label.setText("Simulation running...")
        self.status_label.setStyleSheet("color: #d4a94e;")
        # Update run simulation panel config info
        if hasattr(self, 'run_simulation_panel'):
            self.run_simulation_panel.update_config_info()
        # Log to console
        if hasattr(self, 'console_panel'):
            self.console_panel.log_info("Simulation started")

    def _on_simulation_finished(self, result):
        """Handle simulation finished event."""
        mode = result.get("mode", "unknown")
        self.status_label.setText(f"Simulation complete ({mode})")
        self.status_label.setStyleSheet("color: #4ec94e;")

        # Log to console
        if hasattr(self, 'console_panel'):
            self.console_panel.log_info(f"Simulation complete ({mode})")

        # Refresh detector view in bottom tabs if it exists
        if hasattr(self, 'detector_view') and self.state.detector is not None:
            try:
                self.detector_view.refresh()
            except Exception:
                pass

        # Refresh analysis view in bottom tabs if it exists
        if hasattr(self, 'analysis_view'):
            try:
                self.analysis_view.refresh()
            except Exception:
                pass

        # Refresh 3D viewport
        if hasattr(self, 'viewport_3d'):
            try:
                self.viewport_3d.refresh()
            except Exception:
                pass

    def _on_simulation_error(self, message):
        """Handle simulation error event."""
        self.status_label.setText(f"Simulation error: {message[:50]}...")
        self.status_label.setStyleSheet("color: #e05050;")
        # Log to console
        if hasattr(self, 'console_panel'):
            self.console_panel.log_error(f"Simulation error: {message}")

    # -------------------------------------------------------------------------
    # Public Methods for Panel Updates
    # -------------------------------------------------------------------------

    def set_object_browser(self, widget: QWidget):
        """Replace the object browser with the given widget."""
        self.object_browser_dock.setWidget(widget)

    def set_inspector(self, widget: QWidget):
        """Replace the inspector with the given widget."""
        self.inspector_dock.setWidget(widget)

    def set_console(self, widget: QWidget):
        """Replace the console with the given widget."""
        self.console_dock.setWidget(widget)

    def set_viewport(self, widget: QWidget):
        """Replace the 3D viewport with the given widget."""
        self.central_splitter.replaceWidget(0, widget)

    def set_bottom_tab(self, index: int, widget: QWidget, title: str):
        """
        Replace or add a bottom tab.

        Args:
            index: Tab index
            widget: Widget to add/replace
            title: Tab title
        """
        if index < self.bottom_tabs.count():
            self.bottom_tabs.removeTab(index)
        self.bottom_tabs.insertTab(index, widget, title)

    def update_gpu_status(self, memory_used: float, memory_total: float):
        """
        Update the GPU status display.

        Args:
            memory_used: Used GPU memory in GB
            memory_total: Total GPU memory in GB
        """
        self.gpu_label.setText(f"GPU: {memory_used:.1f}/{memory_total:.1f} GB")

    # -------------------------------------------------------------------------
    # Event Overrides
    # -------------------------------------------------------------------------

    def closeEvent(self, event):
        """Handle window close event."""
        if self.state.is_dirty:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "There are unsaved changes. Do you want to save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            if reply == QMessageBox.Save:
                self._on_save()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return

        self._save_settings()
        event.accept()

    # -------------------------------------------------------------------------
    # Diffraction Peak Signal Handlers
    # -------------------------------------------------------------------------

    def _on_peak_selection_changed(self, selected_peaks: list):
        """
        Handle peak selection change from Crystal Inspector.

        Updates the 3D viewport to show selected peaks.

        Args:
            selected_peaks: List of (h,k,l) tuples for selected peaks
        """
        if hasattr(self, 'viewport_3d'):
            self.viewport_3d.set_selected_peaks(selected_peaks)

            if selected_peaks:
                self.status_label.setText(f"Showing {len(selected_peaks)} peaks in 3D view")
            else:
                self.status_label.setText("Peak visualization cleared")

    def _on_align_to_peak_requested(self, hkl: tuple):
        """
        Handle alignment request from Crystal Inspector.

        Opens the alignment dialog for the selected peak.

        Args:
            hkl: Miller indices (h,k,l) of the peak to align to
        """
        # Check prerequisites
        if self.state.crystal is None:
            QMessageBox.warning(
                self, "No Crystal",
                "Please load a crystal structure first."
            )
            return

        if self.state.beam is None or self.state.beam._energy is None:
            QMessageBox.warning(
                self, "No Beam",
                "Please configure the beam energy first."
            )
            return

        try:
            # Open alignment dialog
            dialog = AlignmentDialog(self.state, hkl, parent=self)
            dialog.alignment_applied.connect(self._on_alignment_applied)
            dialog.exec()

            # Show result message AFTER dialog fully closes
            # (avoids VisPy OpenGL context issues with QMessageBox during draw)
            if hasattr(dialog, '_result_message') and dialog._result_message:
                title, message = dialog._result_message
                QMessageBox.information(self, title, message)

        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to open alignment dialog:\n{str(e)}"
            )

    def _on_alignment_applied(self, alignment_info: dict):
        """
        Handle successful alignment from the dialog.

        Args:
            alignment_info: Dict with alignment details
        """
        hkl = alignment_info.get('hkl', (0, 0, 0))
        changes = alignment_info.get('changes', [])

        self.status_label.setText(
            f"Aligned to ({hkl[0]},{hkl[1]},{hkl[2]})"
        )

        # Log to console
        if hasattr(self, 'console_panel'):
            self.console_panel.log_info(
                f"Aligned to ({hkl[0]},{hkl[1]},{hkl[2]}): {', '.join(changes)}"
            )

        # Refresh 3D viewport to show new positions
        if hasattr(self, 'viewport_3d'):
            self.viewport_3d.refresh()
