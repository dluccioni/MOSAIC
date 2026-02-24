# -----------------------------------------------------------------------------
# Scan Worker
# -----------------------------------------------------------------------------
"""
Background worker for running multi-point scans.

Provides:
- nD scan execution with progress reporting
- Live mode (GUI updates) or batch mode
- Pause/resume support
- Result collection and saving
"""

import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from PySide6.QtCore import QObject, QThread, Signal, Slot, QMutex, QWaitCondition

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gui.state import SimulationState


class ScanMode(Enum):
    """Scan execution modes."""
    LIVE = "live"  # Update GUI after each point
    BATCH = "batch"  # Run all points, update at end


@dataclass
class ScanConfig:
    """Configuration for a scan."""
    motors: List[str] = field(default_factory=list)  # Motor names to scan
    ranges: List[Tuple[float, float]] = field(default_factory=list)  # (start, stop) for each motor
    steps: List[int] = field(default_factory=list)  # Number of steps for each motor
    mode: ScanMode = ScanMode.LIVE
    save_intermediate: bool = True
    output_directory: str = ""
    plot_prefix: str = "scan"
    use_gpu: bool = True


@dataclass
class ScanPoint:
    """Result from a single scan point."""
    index: int
    motor_values: Dict[str, float]
    detector_data: Optional[np.ndarray] = None
    timestamp: float = 0.0


class ScanWorker(QObject):
    """
    Worker object for running scans in a background thread.

    Signals:
        started: Emitted when scan starts (total_points)
        progress: Emitted with progress updates (current_point, total_points, message)
        point_completed: Emitted after each scan point (ScanPoint)
        finished: Emitted when scan completes (results_list)
        error: Emitted when an error occurs (exception)
        cancelled: Emitted when scan is cancelled
        paused: Emitted when scan is paused
        resumed: Emitted when scan is resumed
    """

    started = Signal(int)  # total_points
    progress = Signal(int, int, str)  # current, total, message
    point_completed = Signal(object)  # ScanPoint
    finished = Signal(object)  # List[ScanPoint]
    error = Signal(Exception)
    cancelled = Signal()
    paused = Signal()
    resumed = Signal()

    def __init__(self, state: SimulationState, parent=None):
        """
        Initialize the scan worker.

        Args:
            state: SimulationState instance
            parent: Parent QObject
        """
        super().__init__(parent)
        self.state = state
        self._config = ScanConfig()
        self._cancel_requested = False
        self._paused = False
        self._mutex = QMutex()
        self._pause_condition = QWaitCondition()

    def set_config(self, config: ScanConfig):
        """Set the scan configuration."""
        self._config = config

    def request_cancel(self):
        """Request cancellation of the current scan."""
        self._cancel_requested = True
        self._paused = False  # Unpause to allow cancellation
        self._pause_condition.wakeAll()

    def request_pause(self):
        """Request pause of the current scan."""
        if not self._paused:
            self._paused = True
            self.paused.emit()

    def request_resume(self):
        """Request resume of a paused scan."""
        if self._paused:
            self._paused = False
            self._pause_condition.wakeAll()
            self.resumed.emit()

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
        """Run the scan (called in worker thread)."""
        self._cancel_requested = False
        self._paused = False

        try:
            # Validate configuration
            if not self._validate_config():
                return

            # Calculate total points
            total_points = self._calculate_total_points()
            self.started.emit(total_points)

            # Run the scan
            results = self._run_scan(total_points)

            if self._cancel_requested:
                self.cancelled.emit()
            else:
                self.finished.emit(results)

        except Exception as e:
            traceback.print_exc()
            self.error.emit(e)

    def _validate_config(self) -> bool:
        """Validate the scan configuration."""
        if not self._config.motors:
            self.error.emit(ValueError("No motors specified for scan"))
            return False

        if len(self._config.motors) != len(self._config.ranges):
            self.error.emit(ValueError("Motors and ranges count mismatch"))
            return False

        if len(self._config.motors) != len(self._config.steps):
            self.error.emit(ValueError("Motors and steps count mismatch"))
            return False

        # Check that required objects exist
        if self.state.beam is None:
            self.error.emit(ValueError("No beam configured"))
            return False

        if self.state.sample is None:
            self.error.emit(ValueError("No sample configured"))
            return False

        if self.state.detector is None:
            self.error.emit(ValueError("No detector configured"))
            return False

        if self.state.stage is None:
            self.error.emit(ValueError("No stage configured"))
            return False

        return True

    def _calculate_total_points(self) -> int:
        """Calculate total number of scan points."""
        total = 1
        for steps in self._config.steps:
            total *= steps
        return total

    def _run_scan(self, total_points: int) -> List[ScanPoint]:
        """Run the actual scan."""
        results = []

        # Generate scan positions
        positions = self._generate_positions()

        start_time = time.time()

        for idx, motor_values in enumerate(positions):
            # Check for cancellation
            if self._check_cancelled():
                break

            # Check for pause
            self._check_paused()
            if self._check_cancelled():
                break

            # Update progress
            elapsed = time.time() - start_time
            if idx > 0:
                eta = (elapsed / idx) * (total_points - idx)
                eta_str = self._format_time(eta)
            else:
                eta_str = "--:--"

            self.progress.emit(
                idx + 1,
                total_points,
                f"Point {idx + 1}/{total_points} (ETA: {eta_str})"
            )

            # Move motors
            self._move_motors(motor_values)

            # Run simulation at this point
            self._run_single_simulation()

            # Collect result
            point = ScanPoint(
                index=idx,
                motor_values=motor_values.copy(),
                detector_data=self._get_detector_data(),
                timestamp=time.time()
            )
            results.append(point)

            # Emit point completed
            self.point_completed.emit(point)

            # Save intermediate if configured
            if self._config.save_intermediate and self._config.output_directory:
                self._save_point(point)

        return results

    def _generate_positions(self) -> List[Dict[str, float]]:
        """Generate all motor positions for the scan."""
        positions = []

        # Generate arrays for each motor
        motor_arrays = []
        for i, (motor, (start, stop), steps) in enumerate(
            zip(self._config.motors, self._config.ranges, self._config.steps)
        ):
            motor_arrays.append(np.linspace(start, stop, steps))

        # Create meshgrid and flatten
        if len(motor_arrays) == 1:
            for val in motor_arrays[0]:
                positions.append({self._config.motors[0]: val})
        else:
            grids = np.meshgrid(*motor_arrays, indexing='ij')
            flat_grids = [g.flatten() for g in grids]

            for i in range(len(flat_grids[0])):
                pos = {}
                for j, motor in enumerate(self._config.motors):
                    pos[motor] = flat_grids[j][i]
                positions.append(pos)

        return positions

    def _move_motors(self, motor_values: Dict[str, float]):
        """Move motors to specified positions."""
        stage = self.state.stage
        if stage is None:
            return

        for motor_name, value in motor_values.items():
            if hasattr(stage, 'set_motor_value_absolute'):
                try:
                    stage.set_motor_value_absolute(motor_name, value)
                except Exception as e:
                    print(f"Warning: Failed to move motor {motor_name}: {e}")

    def _run_single_simulation(self):
        """Run simulation at current position."""
        beam = self.state.beam
        sample = self.state.sample
        detector = self.state.detector
        stage = self.state.stage

        if hasattr(beam, 'atomic_direct_interaction'):
            beam.atomic_direct_interaction(
                sample, detector, stage,
                scattering=True,
                scattering_params=[None],
                transmission=False,
                transmission_params=[1.7, 1.0],
                use_gpu=self._config.use_gpu
            )

    def _get_detector_data(self) -> Optional[np.ndarray]:
        """Get current detector data."""
        detector = self.state.detector
        if detector is None:
            return None

        # Access _pixel_values directly to avoid warning print when not initialized
        if hasattr(detector, '_pixel_values') and detector._pixel_values is not None:
            return detector._pixel_values.copy()

        return None

    def _save_point(self, point: ScanPoint):
        """Save a single scan point to file."""
        if not self._config.output_directory:
            return

        output_dir = Path(self._config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detector data
        if point.detector_data is not None:
            filename = output_dir / f"{self._config.plot_prefix}_point_{point.index:04d}.npy"
            np.save(filename, point.detector_data)

    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS or MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"


class ScanThread(QThread):
    """
    Thread wrapper for ScanWorker.

    Provides a simple interface for running scans in a background thread.
    """

    def __init__(self, worker: ScanWorker, parent=None):
        """
        Initialize the scan thread.

        Args:
            worker: ScanWorker instance
            parent: Parent QObject
        """
        super().__init__(parent)
        self.worker = worker
        self.worker.moveToThread(self)
        self.started.connect(self.worker.run)

    def stop(self):
        """Request the scan to stop."""
        self.worker.request_cancel()
        self.quit()
        self.wait(5000)  # Wait up to 5 seconds

    def pause(self):
        """Request the scan to pause."""
        self.worker.request_pause()

    def resume(self):
        """Request the scan to resume."""
        self.worker.request_resume()
