# -----------------------------------------------------------------------------
# Analysis Inspector
# -----------------------------------------------------------------------------
"""
Inspector panel for creating the Analysis object.

Provides controls for:
- Working directory selection
- Analysis object creation

Note: Analysis functions (FFT, integration, etc.) are handled in the
Analysis View pane under the 3D viewport.
"""

import sys
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox, QLineEdit,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gui.state import SimulationState
from gui.panels.inspector import InspectorPanel


class AnalysisInspector(InspectorPanel):
    """
    Inspector for creating and managing the Analysis object.

    This inspector only handles object creation. All analysis functions
    (FFT distance dependence, integration, etc.) are available in the
    Analysis View tab in the bottom panel.
    """

    analysis_created = Signal(object)

    def __init__(self, state: SimulationState, parent=None):
        super().__init__(state, parent)
        self.set_title("Analysis")
        self._setup_analysis_ui()
        self._register_observers()

    def _setup_analysis_ui(self):
        """Setup analysis-specific UI elements."""
        # Directory Group
        dir_group = self.add_group("Directory")
        dir_layout = dir_group.layout()

        # Directory selection row
        dir_row = QWidget()
        dir_row_layout = QHBoxLayout(dir_row)
        dir_row_layout.setContentsMargins(0, 0, 0, 0)

        self.working_dir = QLineEdit()
        self.working_dir.setReadOnly(True)
        # Set placeholder based on global directory
        global_dir = self.state.global_working_directory
        if global_dir:
            self.working_dir.setPlaceholderText(f"Using global: {global_dir}")
        else:
            self.working_dir.setPlaceholderText("Select working directory...")
        dir_row_layout.addWidget(self.working_dir, 1)

        browse_btn = QPushButton("Browse...")
        browse_btn.setToolTip("Browse for directory")
        browse_btn.clicked.connect(self._on_browse_dir)
        dir_row_layout.addWidget(browse_btn)

        dir_layout.addRow(dir_row)

        # Load existing button (only enabled when metadata found)
        self.load_existing_btn = QPushButton("Load Existing Analysis")
        self.load_existing_btn.clicked.connect(self._on_load_existing)
        self.load_existing_btn.setEnabled(False)
        dir_layout.addRow(self.load_existing_btn)

        # Info label
        dir_info = QLabel("Directory for saving analysis outputs\n(plots, data files, etc.)")
        dir_info.setStyleSheet("color: #808080; font-style: italic;")
        dir_info.setWordWrap(True)
        dir_layout.addRow("", dir_info)

        # Create Object Group
        create_group = self.add_group("Create Object")
        create_layout = create_group.layout()

        # Status label
        self.status_label = QLabel("Not created")
        self.status_label.setStyleSheet("color: #808080;")
        create_layout.addRow("Status:", self.status_label)

        # Create button
        create_btn = QPushButton("Create Analysis Object")
        create_btn.clicked.connect(self._on_create_analysis)
        create_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a6a2a;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a8a3a;
            }
        """)
        create_layout.addRow("", create_btn)

        # Info Group
        info_group = self.add_group("Info")
        info_layout = info_group.layout()

        info_text = QLabel(
            "Analysis functions are available in the\n"
            "Analysis tab (bottom panel) after creating\n"
            "the analysis object.\n\n"
            "Available functions:\n"
            "• FFT Distance Dependence\n"
            "• Detector Integration\n"
            "• Line Profiles"
        )
        info_text.setStyleSheet("color: #a0a0a0;")
        info_layout.addRow(info_text)

    def _register_observers(self):
        """Register for state change notifications."""
        self.state.register_observer("analysis_changed", self._on_analysis_state_changed)
        self.state.register_observer("global_working_directory_changed", self._on_global_dir_changed)

    def _on_global_dir_changed(self, directory):
        """Handle global working directory change."""
        # Update the working directory field if it's empty
        if not self.working_dir.text():
            self.working_dir.setPlaceholderText(
                f"Using global: {directory}" if directory
                else "Select working directory..."
            )

    def _on_analysis_state_changed(self, analysis):
        """Handle analysis state change."""
        self._refresh_display()

    def _refresh_display(self):
        """Refresh the display based on current state."""
        analysis = self.state.analysis

        if analysis is not None:
            self.status_label.setText("Ready")
            self.status_label.setStyleSheet("color: #4ec94e;")

            # Update directory display if available
            if hasattr(analysis, 'directory') and analysis.directory:
                self.working_dir.setText(str(analysis.directory))
        else:
            self.status_label.setText("Not created")
            self.status_label.setStyleSheet("color: #808080;")

    def _on_browse_dir(self):
        """Handle browse directory button."""
        # Use current text, or global directory, or current working directory
        current_dir = (
            self.working_dir.text()
            or self.state.get_default_directory()
        )
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Working Directory", current_dir
        )
        if dir_path:
            self.working_dir.setText(dir_path)
            # Check if analysis metadata exists
            metadata_path = Path(dir_path) / "analysis_metadata.json"
            if metadata_path.exists():
                # Enable and style button green when metadata found
                self.load_existing_btn.setEnabled(True)
                self.load_existing_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #2a6a2a;
                        color: white;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #3a8a3a;
                    }
                """)
            else:
                # Disable button and reset style when no metadata
                self.load_existing_btn.setEnabled(False)
                self.load_existing_btn.setStyleSheet("")

    def _on_load_existing(self):
        """Load existing analysis from selected directory."""
        directory = self.working_dir.text()
        if not directory:
            QMessageBox.warning(self, "No Directory",
                              "Please select a working directory first.")
            return

        try:
            from Analysis import analysis
            existing_analysis = analysis(directory=directory)
            # Try to load metadata
            if hasattr(existing_analysis, 'read_analysis_metadata'):
                existing_analysis.read_analysis_metadata()
            self.state.analysis = existing_analysis
            self.analysis_created.emit(existing_analysis)
            self._refresh_display()
            QMessageBox.information(self, "Success", "Analysis object loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load analysis:\n{str(e)}")

    def _on_create_analysis(self):
        """Handle create analysis object button."""
        try:
            from Analysis import analysis

            # Get directory (use global directory if not specified, then current working dir)
            output_dir = self.working_dir.text() or self.state.get_default_directory()

            # Create the analysis object
            new_analysis = analysis(directory=output_dir)
            self.state.analysis = new_analysis

            # Update display
            self._refresh_display()

            # Emit signal
            self.analysis_created.emit(new_analysis)

            QMessageBox.information(
                self, "Success",
                f"Analysis object created.\n\n"
                f"Working directory: {output_dir}\n\n"
                f"Analysis functions are now available in the\n"
                f"Analysis tab (bottom panel)."
            )

        except ImportError as e:
            QMessageBox.critical(
                self, "Import Error",
                f"Could not import Analysis module:\n{str(e)}\n\n"
                "Make sure the X-ray simulator modules are properly installed."
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to create analysis object:\n{str(e)}"
            )

    def get_config(self) -> dict:
        """Get current configuration as a dictionary."""
        return {
            "working_dir": self.working_dir.text(),
        }

    def set_config(self, config: dict):
        """Apply configuration from a dictionary."""
        if "working_dir" in config:
            self.working_dir.setText(config["working_dir"])
