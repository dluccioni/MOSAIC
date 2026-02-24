# -----------------------------------------------------------------------------
# Panels Package
# -----------------------------------------------------------------------------
"""
GUI panels for the X-ray simulator.

This package contains the main dockable panels:
- ObjectBrowser: Tree view of simulation objects
- InspectorPanel: Base class for property editors
- ConsolePanel: Logging output display
- BottomTabs: Tabbed panel for detector/analysis views
"""

from .object_browser import ObjectBrowser
from .inspector import InspectorPanel
from .console_panel import ConsolePanel
from .bottom_tabs import BottomTabs

__all__ = ["ObjectBrowser", "InspectorPanel", "ConsolePanel", "BottomTabs"]
