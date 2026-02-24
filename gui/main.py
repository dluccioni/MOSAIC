# -----------------------------------------------------------------------------
# GUI Entry Point
# -----------------------------------------------------------------------------
"""
Entry point for the X-ray simulator GUI application.

Usage:
    python -m gui

Or run directly:
    python gui/main.py

Or in a notebook/script:
    import sys
    sys.path.insert(0, 'path/to/Xray-Simulator')
    from gui.main import run
    run()
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports (for running as script)
try:
    _gui_dir = Path(__file__).parent
    _project_dir = _gui_dir.parent
except NameError:
    # __file__ is not defined in interactive environments (e.g., VS Code Interactive Window)
    _project_dir = Path.cwd()
    _gui_dir = _project_dir / "gui"

if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))
if str(_gui_dir) not in sys.path:
    sys.path.insert(0, str(_gui_dir))


def check_dependencies():
    """
    Check that all required dependencies are installed.

    Returns:
        tuple: (success, missing_packages)
    """
    missing = []

    # Check PySide6
    try:
        import PySide6
        from PySide6.QtCore import __version__ as qt_version
        print(f"PySide6 version: {qt_version}")
    except ImportError:
        missing.append("pyside6")

    # Check VisPy
    try:
        import vispy
        print(f"VisPy version: {vispy.__version__}")
    except ImportError:
        missing.append("vispy")

    # Check NumPy
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except ImportError:
        missing.append("numpy")

    # Check CuPy (optional but recommended)
    try:
        import cupy as cp
        print(f"CuPy version: {cp.__version__}")
    except ImportError:
        print("CuPy not found (GPU acceleration will be limited)")

    return len(missing) == 0, missing


def setup_high_dpi():
    """Configure high DPI scaling for Qt."""
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        # Enable high DPI scaling
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass  # Ignore if not supported


def setup_dark_theme(app):
    """
    Apply a dark theme to the application.

    Args:
        app: QApplication instance
    """
    from PySide6.QtGui import QPalette, QColor
    from PySide6.QtCore import Qt

    # Create dark palette
    dark_palette = QPalette()

    # Base colors
    dark_palette.setColor(QPalette.Window, QColor(45, 45, 45))
    dark_palette.setColor(QPalette.WindowText, QColor(208, 208, 208))
    dark_palette.setColor(QPalette.Base, QColor(30, 30, 30))
    dark_palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(45, 45, 45))
    dark_palette.setColor(QPalette.ToolTipText, QColor(208, 208, 208))
    dark_palette.setColor(QPalette.Text, QColor(208, 208, 208))
    dark_palette.setColor(QPalette.Button, QColor(45, 45, 45))
    dark_palette.setColor(QPalette.ButtonText, QColor(208, 208, 208))
    dark_palette.setColor(QPalette.BrightText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))

    # Disabled colors
    dark_palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(127, 127, 127))
    dark_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
    dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))
    dark_palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(80, 80, 80))
    dark_palette.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor(127, 127, 127))

    app.setPalette(dark_palette)

    # Additional stylesheet for finer control
    app.setStyleSheet("""
        QToolTip {
            color: #d0d0d0;
            background-color: #2d2d2d;
            border: 1px solid #505050;
            padding: 4px;
        }

        QMenuBar {
            background-color: #2d2d2d;
            padding: 2px;
        }

        QMenuBar::item {
            padding: 4px 8px;
            background: transparent;
        }

        QMenuBar::item:selected {
            background: #404040;
        }

        QMenu {
            background-color: #2d2d2d;
            border: 1px solid #404040;
        }

        QMenu::item {
            padding: 4px 24px 4px 24px;
        }

        QMenu::item:selected {
            background-color: #404040;
        }

        QMenu::separator {
            height: 1px;
            background: #404040;
            margin: 4px 8px;
        }

        QToolBar {
            background-color: #2d2d2d;
            border: none;
            spacing: 4px;
            padding: 4px;
        }

        QToolButton {
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: 4px;
            padding: 4px 8px;
        }

        QToolButton:hover {
            background-color: #404040;
            border: 1px solid #505050;
        }

        QToolButton:pressed {
            background-color: #353535;
        }

        QDockWidget {
            titlebar-close-icon: url(close.png);
            titlebar-normal-icon: url(float.png);
        }

        QDockWidget::title {
            background-color: #2d2d2d;
            padding: 6px;
            text-align: left;
        }

        QTabWidget::pane {
            border: 1px solid #404040;
            background-color: #2d2d2d;
        }

        QTabBar::tab {
            background-color: #252525;
            border: 1px solid #404040;
            padding: 6px 12px;
            margin-right: 2px;
        }

        QTabBar::tab:selected {
            background-color: #2d2d2d;
            border-bottom: none;
        }

        QTabBar::tab:hover:!selected {
            background-color: #353535;
        }

        QScrollBar:vertical {
            background-color: #1e1e1e;
            width: 12px;
            margin: 0;
        }

        QScrollBar::handle:vertical {
            background-color: #505050;
            min-height: 20px;
            border-radius: 6px;
            margin: 2px;
        }

        QScrollBar::handle:vertical:hover {
            background-color: #606060;
        }

        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0;
        }

        QScrollBar:horizontal {
            background-color: #1e1e1e;
            height: 12px;
            margin: 0;
        }

        QScrollBar::handle:horizontal {
            background-color: #505050;
            min-width: 20px;
            border-radius: 6px;
            margin: 2px;
        }

        QScrollBar::handle:horizontal:hover {
            background-color: #606060;
        }

        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0;
        }

        QProgressBar {
            background-color: #1e1e1e;
            border: 1px solid #404040;
            border-radius: 4px;
            text-align: center;
        }

        QProgressBar::chunk {
            background-color: #2a82da;
            border-radius: 3px;
        }

        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #1e1e1e;
            border: 1px solid #404040;
            border-radius: 4px;
            padding: 4px 8px;
            selection-background-color: #2a82da;
        }

        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border: 1px solid #2a82da;
        }

        QComboBox::drop-down {
            border: none;
            padding-right: 8px;
        }

        QComboBox::down-arrow {
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 6px solid #808080;
        }

        QComboBox QAbstractItemView {
            background-color: #2d2d2d;
            border: 1px solid #404040;
            selection-background-color: #2a82da;
        }

        QPushButton {
            background-color: #404040;
            border: 1px solid #505050;
            border-radius: 4px;
            padding: 6px 16px;
        }

        QPushButton:hover {
            background-color: #4a4a4a;
            border: 1px solid #606060;
        }

        QPushButton:pressed {
            background-color: #353535;
        }

        QPushButton:disabled {
            background-color: #303030;
            color: #606060;
        }

        QGroupBox {
            border: 1px solid #404040;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 8px;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }

        QSplitter::handle {
            background-color: #404040;
        }

        QSplitter::handle:horizontal {
            width: 2px;
        }

        QSplitter::handle:vertical {
            height: 2px;
        }

        QStatusBar {
            background-color: #2d2d2d;
            border-top: 1px solid #404040;
        }

        QTreeView, QListView, QTableView {
            background-color: #1e1e1e;
            border: 1px solid #404040;
            alternate-background-color: #252525;
        }

        QTreeView::item:selected, QListView::item:selected, QTableView::item:selected {
            background-color: #2a82da;
        }

        QTreeView::item:hover, QListView::item:hover, QTableView::item:hover {
            background-color: #353535;
        }

        QHeaderView::section {
            background-color: #2d2d2d;
            border: none;
            border-right: 1px solid #404040;
            border-bottom: 1px solid #404040;
            padding: 4px 8px;
        }

        QCheckBox::indicator, QRadioButton::indicator {
            width: 16px;
            height: 16px;
        }

        QCheckBox::indicator:unchecked, QRadioButton::indicator:unchecked {
            border: 1px solid #505050;
            background-color: #1e1e1e;
        }

        QCheckBox::indicator:checked {
            border: 1px solid #2a82da;
            background-color: #2a82da;
        }

        QRadioButton::indicator:checked {
            border: 1px solid #2a82da;
            background-color: #2a82da;
            border-radius: 8px;
        }

        QSlider::groove:horizontal {
            border: 1px solid #404040;
            background: #1e1e1e;
            height: 6px;
            border-radius: 3px;
        }

        QSlider::handle:horizontal {
            background: #505050;
            border: 1px solid #606060;
            width: 14px;
            margin: -4px 0;
            border-radius: 7px;
        }

        QSlider::handle:horizontal:hover {
            background: #606060;
        }

        QSlider::sub-page:horizontal {
            background: #2a82da;
            border-radius: 3px;
        }
    """)


def run():
    """Run the GUI application."""
    print("=" * 60)
    print("X-ray Diffraction Simulator GUI")
    print("=" * 60)

    # Check dependencies
    success, missing = check_dependencies()
    if not success:
        print(f"\nError: Missing required packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        sys.exit(1)

    print()

    # Setup high DPI before creating QApplication
    setup_high_dpi()

    # Create or get existing application
    from PySide6.QtWidgets import QApplication

    # Check if QApplication already exists (e.g., from %gui qt in notebook)
    app = QApplication.instance()
    app_created = False

    if app is None:
        app = QApplication(sys.argv)
        app_created = True
        app.setApplicationName("X-ray Simulator")
        app.setOrganizationName("XraySimulator")
        app.setOrganizationDomain("xraysimulator.local")
        # Apply dark theme only for new app
        setup_dark_theme(app)
    else:
        print("Using existing QApplication instance (notebook mode)")

    # Create and show main window
    # Use absolute imports to support both package and direct execution
    try:
        from gui.main_window import MainWindow
        from gui.state import SimulationState
    except ImportError:
        # Fallback for running from gui directory
        from main_window import MainWindow
        from state import SimulationState

    state = SimulationState()
    window = MainWindow(state)
    window.show()

    print("GUI started. Close the window to exit.")

    # Run event loop only if we created the app
    # If using %gui qt, the event loop is already running
    if app_created:
        return app.exec()
    else:
        # Return the window so it can be accessed from notebook
        return window


def main():
    """Main entry point."""
    run()


def run_from_notebook():
    """
    Run the GUI from a Jupyter notebook.

    Usage in notebook cell:
    ```python
    import sys
    sys.path.insert(0, r'x:\\Dresselhaus Lab\\Code\\Xray-Simulator')

    # Enable Qt event loop integration (run this BEFORE importing the GUI)
    %gui qt

    from gui.main import run_from_notebook
    run_from_notebook()
    ```

    Note: The %gui qt magic enables non-blocking GUI execution.
    Without it, the GUI will block the notebook until closed.
    """
    run()


if __name__ == "__main__":
    main()
