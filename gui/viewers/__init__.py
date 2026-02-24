# -----------------------------------------------------------------------------
# Viewers Package
# -----------------------------------------------------------------------------
"""
Visualization widgets for the X-ray simulator GUI.

This package contains:
- Viewport3D: Main 3D scene viewer using VisPy
- DetectorView: 2D detector image display
- AtomCloudViewer: GPU-accelerated atom visualization with LOD
"""

from .viewport_3d import Viewport3D
from .detector_view import DetectorView

__all__ = ["Viewport3D", "DetectorView"]
