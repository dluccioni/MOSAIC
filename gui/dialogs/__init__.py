# -----------------------------------------------------------------------------
# Dialogs Package
# -----------------------------------------------------------------------------
"""
Dialog windows for the X-ray simulator GUI.

This package contains:
- ScanWizard: Multi-step scan configuration wizard
- PresetDialog: Preset browser and save dialog
- LoadDataDialog: Load saved plots and detector pixels
- ImportCIFDialog: CIF crystal structure import
- ImportDeformationDialog: Deformation field import
- AlignmentDialog: Diffraction peak alignment wizard
"""

from .scan_wizard import ScanWizard
from .preset_dialog import PresetDialog
from .load_data_dialog import LoadDataDialog, ImportCIFDialog, ImportDeformationDialog
from .alignment_dialog import AlignmentDialog

__all__ = [
    "ScanWizard",
    "PresetDialog",
    "LoadDataDialog",
    "ImportCIFDialog",
    "ImportDeformationDialog",
    "AlignmentDialog",
]
