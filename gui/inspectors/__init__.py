# -----------------------------------------------------------------------------
# Inspectors Package
# -----------------------------------------------------------------------------
"""
Inspector panels for editing simulation object properties.

Each inspector provides a specialized UI for its corresponding module:
- CrystalInspector: Crystal structure and orientation
- SampleInspector: Sample dimensions and generation
- BeamInspector: X-ray beam configuration
- DetectorInspector: Detector setup and positioning
- StageInspector: Goniometer/motor control
- OpticsInspector: Optical components stack
- DefectsInspector: Defect definitions
- DeformationInspector: Deformation field import
- AnalysisInspector: Analysis tools
"""

from .crystal_inspector import CrystalInspector
from .sample_inspector import SampleInspector
from .beam_inspector import BeamInspector
from .detector_inspector import DetectorInspector
from .stage_inspector import StageInspector
from .optics_inspector import OpticsInspector
from .defects_inspector import DefectsInspector
from .deformation_inspector import DeformationInspector
from .analysis_inspector import AnalysisInspector

__all__ = [
    "CrystalInspector",
    "SampleInspector",
    "BeamInspector",
    "DetectorInspector",
    "StageInspector",
    "OpticsInspector",
    "DefectsInspector",
    "DeformationInspector",
    "AnalysisInspector",
]
