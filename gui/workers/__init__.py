# -----------------------------------------------------------------------------
# Workers Package
# -----------------------------------------------------------------------------
"""
Background worker threads for the X-ray simulator GUI.

This package contains:
- SimulationWorker: Runs heavy simulation computations off the main thread
- ScanWorker: Executes multi-point scans with progress reporting
"""

from .simulation_worker import SimulationWorker
from .scan_worker import ScanWorker

__all__ = ["SimulationWorker", "ScanWorker"]
