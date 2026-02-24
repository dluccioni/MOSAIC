# -----------------------------------------------------------------------------
# Global Working Directory Panel
# -----------------------------------------------------------------------------
"""
Panel for setting the global working directory.

The global working directory serves as the default starting directory for all
directory options throughout the GUI. Individual objects can still override
with their own directories.
"""

import sys
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gui.state import SimulationState


class GlobalWorkingDirectoryPanel(QWidget):
    """
    Panel for selecting and displaying the global working directory.

    This directory is used as the default for all directory options in the GUI
    when no specific directory has been set for an individual object.

    Signals:
        directory_changed: Emitted when the directory is changed (new_dir)
    """

    directory_changed = Signal(str)

    def __init__(self, state: SimulationState, parent=None):
        """
        Initialize the global working directory panel.

        Args:
            state: SimulationState instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.state = state
        self._setup_ui()
        self._register_observers()
        self._refresh_display()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Label
        label = QLabel("Global Working Directory")
        label.setStyleSheet("font-weight: bold; color: #c0c0c0;")
        layout.addWidget(label)

        # Directory row
        dir_row = QWidget()
        dir_layout = QHBoxLayout(dir_row)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        dir_layout.setSpacing(4)

        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("Not set - using current directory")
        self.dir_edit.editingFinished.connect(self._on_edit_finished)
        dir_layout.addWidget(self.dir_edit, 1)

        browse_btn = QPushButton("...")
        browse_btn.setMaximumWidth(30)
        browse_btn.setToolTip("Browse for directory")
        browse_btn.clicked.connect(self._on_browse)
        dir_layout.addWidget(browse_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setMaximumWidth(50)
        clear_btn.setToolTip("Clear global directory (use current directory)")
        clear_btn.clicked.connect(self._on_clear)
        dir_layout.addWidget(clear_btn)

        layout.addWidget(dir_row)

        # Info label
        info_label = QLabel("Default directory for all objects")
        info_label.setStyleSheet("color: #808080; font-style: italic; font-size: 11px;")
        layout.addWidget(info_label)

        # Style
        self.setStyleSheet("""
            QWidget {
                background-color: #252525;
            }
            QLineEdit {
                background-color: #1e1e1e;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 4px 6px;
                color: #e0e0e0;
            }
            QLineEdit:focus {
                border-color: #2a82da;
            }
            QPushButton {
                background-color: #3d3d3d;
                border: 1px solid #4d4d4d;
                border-radius: 3px;
                padding: 4px 8px;
                color: #e0e0e0;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
            QPushButton:pressed {
                background-color: #2d2d2d;
            }
        """)

    def _register_observers(self):
        """Register for state change notifications."""
        self.state.register_observer(
            "global_working_directory_changed",
            self._on_global_dir_changed
        )

    def _on_global_dir_changed(self, directory):
        """Handle global working directory change from state."""
        self._refresh_display()

    def _refresh_display(self):
        """Refresh the display to show current global directory."""
        global_dir = self.state.global_working_directory
        if global_dir:
            self.dir_edit.setText(global_dir)
            self.dir_edit.setToolTip(global_dir)
        else:
            self.dir_edit.setText("")
            self.dir_edit.setToolTip("Not set - using current directory")

    def _on_browse(self):
        """Handle browse button click."""
        # Start from current global dir or current working directory
        start_dir = self.state.get_default_directory()

        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Global Working Directory", start_dir
        )
        if dir_path:
            self.dir_edit.setText(dir_path)
            self.state.global_working_directory = dir_path
            self.directory_changed.emit(dir_path)

    def _on_edit_finished(self):
        """Handle manual edit of directory path."""
        new_dir = self.dir_edit.text().strip()
        if new_dir != self.state.global_working_directory:
            self.state.global_working_directory = new_dir
            self.directory_changed.emit(new_dir)

    def _on_clear(self):
        """Handle clear button click."""
        self.dir_edit.setText("")
        self.state.global_working_directory = ""
        self.directory_changed.emit("")

    def get_directory(self) -> str:
        """
        Get the current global working directory.

        Returns:
            Directory path or empty string if not set
        """
        return self.state.global_working_directory

    def set_directory(self, directory: str):
        """
        Set the global working directory.

        Args:
            directory: Directory path (empty string to clear)
        """
        self.state.global_working_directory = directory
