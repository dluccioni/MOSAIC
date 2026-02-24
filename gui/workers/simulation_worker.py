# -----------------------------------------------------------------------------
# Simulation Worker
# -----------------------------------------------------------------------------
"""
Background worker for running X-ray simulations.

Provides:
- Threaded execution of heavy computations
- Progress reporting
- Cancellation support
- Error handling
"""

import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

from PySide6.QtCore import QObject, QThread, Signal, Slot, QMutex, QWaitCondition

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gui.state import SimulationState


class SimulationMode(Enum):
    """Simulation computation modes."""
    SCATTERING_KINEMATIC = "scattering_kinematic"
    SCATTERING_DYNAMICAL = "scattering_dynamical"
    TRANSMISSION = "transmission"
    COMBINED = "combined"  # Scattering + transmission
    WAVEFIELD = "wavefield"


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""
    mode: SimulationMode = SimulationMode.COMBINED
    use_gpu: bool = True
    scattering_params: Optional[Dict] = None
    transmission_params: Optional[Dict] = None
    wavefield_params: Optional[Dict] = None


class SimulationWorker(QObject):
    """
    Worker object for running simulations in a background thread.

    Signals:
        started: Emitted when simulation starts
        progress: Emitted with progress updates (progress, message)
        finished: Emitted when simulation completes (result_dict)
        error: Emitted when an error occurs (exception)
        cancelled: Emitted when simulation is cancelled
    """

    started = Signal()
    progress = Signal(float, str)
    finished = Signal(object)
    error = Signal(Exception)
    cancelled = Signal()

    def __init__(self, state: SimulationState, parent=None):
        """
        Initialize the simulation worker.

        Args:
            state: SimulationState instance
            parent: Parent QObject
        """
        super().__init__(parent)
        self.state = state
        self._config = SimulationConfig()
        self._cancel_requested = False
        self._paused = False
        self._mutex = QMutex()
        self._pause_condition = QWaitCondition()

    def set_config(self, config: SimulationConfig):
        """Set the simulation configuration."""
        self._config = config

    def request_cancel(self):
        """Request cancellation of the current simulation."""
        self._cancel_requested = True

    def request_pause(self):
        """Request pause of the current simulation."""
        self._paused = True

    def request_resume(self):
        """Request resume of a paused simulation."""
        self._paused = False
        self._pause_condition.wakeAll()

    def _check_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancel_requested

    def _check_paused(self):
        """Check if pause was requested and wait if so."""
        self._mutex.lock()
        while self._paused and not self._cancel_requested:
            self._pause_condition.wait(self._mutex)
        self._mutex.unlock()

    @Slot()
    def run(self):
        """Run the simulation (called in worker thread)."""
        self._cancel_requested = False
        self._paused = False

        self.started.emit()
        self.progress.emit(0.0, "Initializing simulation...")

        try:
            # Validate required objects
            if not self._validate_state():
                return

            result = self._run_simulation()

            if self._cancel_requested:
                self.cancelled.emit()
            else:
                self.finished.emit(result)

        except Exception as e:
            traceback.print_exc()
            self.error.emit(e)

    def _validate_state(self) -> bool:
        """Validate that all required objects are available."""
        missing = []

        if self.state.sample is None:
            missing.append("Sample")
        if self.state.beam is None:
            missing.append("Beam")
        if self.state.detector is None:
            missing.append("Detector")
        if self.state.stage is None:
            missing.append("Stage")

        if missing:
            error_msg = f"Missing required objects: {', '.join(missing)}"
            self.error.emit(ValueError(error_msg))
            return False

        return True

    def _run_simulation(self) -> Dict[str, Any]:
        """Run the actual simulation."""
        result = {}

        mode = self._config.mode
        beam = self.state.beam
        sample = self.state.sample
        detector = self.state.detector
        stage = self.state.stage

        self.progress.emit(0.1, f"Running {mode.value} simulation...")

        # Run appropriate simulation type
        if mode == SimulationMode.SCATTERING_KINEMATIC:
            result = self._run_kinematic_scattering()

        elif mode == SimulationMode.SCATTERING_DYNAMICAL:
            result = self._run_dynamical_scattering()

        elif mode == SimulationMode.TRANSMISSION:
            result = self._run_transmission()

        elif mode == SimulationMode.COMBINED:
            result = self._run_combined()

        elif mode == SimulationMode.WAVEFIELD:
            result = self._run_wavefield()

        self.progress.emit(1.0, "Simulation complete")
        return result

    def _run_kinematic_scattering(self) -> Dict[str, Any]:
        """Run kinematic scattering simulation."""
        self.progress.emit(0.2, "Computing kinematic scattering...")

        beam = self.state.beam
        sample = self.state.sample
        detector = self.state.detector
        stage = self.state.stage

        if hasattr(beam, 'atomic_scattering_kinematic'):
            beam.atomic_scattering_kinematic(
                sample, detector, stage,
                use_gpu=self._config.use_gpu
            )

        self._check_paused()
        if self._check_cancelled():
            return {}

        self.progress.emit(0.9, "Processing results...")
        return {"mode": "kinematic", "success": True}

    def _run_dynamical_scattering(self) -> Dict[str, Any]:
        """Run dynamical scattering simulation."""
        self.progress.emit(0.2, "Computing dynamical scattering...")

        beam = self.state.beam
        sample = self.state.sample
        detector = self.state.detector
        stage = self.state.stage

        if hasattr(beam, 'atomic_scattering_dynamical'):
            beam.atomic_scattering_dynamical(
                sample, detector, stage,
                use_gpu=self._config.use_gpu
            )

        self._check_paused()
        if self._check_cancelled():
            return {}

        self.progress.emit(0.9, "Processing results...")
        return {"mode": "dynamical", "success": True}

    def _run_transmission(self) -> Dict[str, Any]:
        """Run transmission simulation."""
        self.progress.emit(0.2, "Computing transmission...")

        beam = self.state.beam
        sample = self.state.sample
        detector = self.state.detector
        stage = self.state.stage

        params = self._config.transmission_params or {}

        if hasattr(beam, 'atomic_transmission'):
            beam.atomic_transmission(
                sample, detector, stage,
                use_gpu=self._config.use_gpu,
                **params
            )

        self._check_paused()
        if self._check_cancelled():
            return {}

        self.progress.emit(0.9, "Processing results...")
        return {"mode": "transmission", "success": True}

    def _run_combined(self) -> Dict[str, Any]:
        """Run combined scattering + transmission simulation."""
        self.progress.emit(0.2, "Computing scattering and transmission...")

        beam = self.state.beam
        sample = self.state.sample
        detector = self.state.detector
        stage = self.state.stage

        scattering_params = self._config.scattering_params or {}
        transmission_params = self._config.transmission_params or {}

        if hasattr(beam, 'atomic_direct_interaction'):
            beam.atomic_direct_interaction(
                sample, detector, stage,
                scattering=True,
                scattering_params=scattering_params.get('params', [None]),
                transmission=transmission_params.get('enabled', False),
                transmission_params=transmission_params.get('params', [1.7, 1.0]),
                use_gpu=self._config.use_gpu
            )

        self._check_paused()
        if self._check_cancelled():
            return {}

        self.progress.emit(0.9, "Processing results...")
        return {"mode": "combined", "success": True}

    def _run_wavefield(self) -> Dict[str, Any]:
        """Run wavefield propagation through optics."""
        self.progress.emit(0.2, "Computing wavefield propagation...")

        beam = self.state.beam
        optics = self.state.optics

        if optics is None:
            self.error.emit(ValueError("No optics stack defined"))
            return {}

        params = self._config.wavefield_params or {}

        if hasattr(beam, 'wavefield_propagation'):
            beam.wavefield_propagation(
                optics,
                use_gpu=self._config.use_gpu,
                **params
            )

        self._check_paused()
        if self._check_cancelled():
            return {}

        self.progress.emit(0.9, "Processing results...")
        return {"mode": "wavefield", "success": True}


class SimulationThread(QThread):
    """
    Thread wrapper for SimulationWorker.

    Provides a simple interface for running simulations in a background thread.
    """

    def __init__(self, worker: SimulationWorker, parent=None):
        """
        Initialize the simulation thread.

        Args:
            worker: SimulationWorker instance
            parent: Parent QObject
        """
        super().__init__(parent)
        self.worker = worker
        self.worker.moveToThread(self)
        self.started.connect(self.worker.run)

    def stop(self):
        """Request the simulation to stop."""
        self.worker.request_cancel()
        self.quit()
        self.wait(5000)  # Wait up to 5 seconds
