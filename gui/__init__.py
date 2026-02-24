# -----------------------------------------------------------------------------
# X-ray Simulator GUI Package
# -----------------------------------------------------------------------------
"""
GUI package for the X-ray diffraction simulator.

This package provides a comprehensive graphical user interface for controlling
all aspects of X-ray diffraction simulations, including:
- Crystal structure definition and manipulation
- Sample generation and visualization
- Beam configuration
- Detector setup and positioning
- Stage/goniometer control
- Optical components
- Defect modeling
- Deformation field application
- Experiment orchestration and scanning
- Analysis tools

Usage:
    from gui import main
    main.run()

Or from command line:
    python -m gui
"""

import sys
from pathlib import Path

__version__ = "1.0.0"
__author__ = "X-ray Simulator Team"

# Add parent directory to path for imports
_gui_dir = Path(__file__).parent
_project_dir = _gui_dir.parent
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

# Use try/except to handle both package and direct import scenarios
try:
    from gui.state import SimulationState
    from gui.main_window import MainWindow
except ImportError:
    from .state import SimulationState
    from .main_window import MainWindow

__all__ = ["SimulationState", "MainWindow"]
