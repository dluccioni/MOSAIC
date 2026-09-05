# -----------------------------------------------------------------------------
# Hardware Panel
# -----------------------------------------------------------------------------
"""
Panel showing the machine profile the simulation adapts to.

Displays hardware.report() (host RAM, CPU, every CUDA device with its free
memory and watchdog state), the live host and device budgets from the memory
governors, the calibration on file for the current device, and a button that
runs a quick calibration in a worker thread.
"""
import os
import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QPlainTextEdit,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import hardware


_GIB = 1024 ** 3


class _CalibrateWorker(QThread):
    """Runs hardware.calibrate(quick=True) off the GUI thread."""
    finished_with = Signal(str)

    def run(self):
        """Run the calibration and report its outcome as text."""
        try:
            data = hardware.calibrate(quick=True, log=lambda m: None)
            fast = data.get("throughput", {}).get("fast", {}).get("best")
            msg = "calibration written"
            if fast:
                msg += f": fast kernel {fast:.2e} atom.px/s"
            self.finished_with.emit(msg)
        except Exception as exc:
            self.finished_with.emit(f"calibration failed: {exc}")


class HardwarePanel(QWidget):
    """
    Machine profile, memory budgets and calibration state.

    The text is refreshed every two seconds while the panel is visible, so
    the device and host figures track a running simulation.
    """

    def __init__(self, state=None, parent=None):
        """
        Initialize the hardware panel.

        Args:
            state: SimulationState instance (kept for symmetry with the other
                panels; the profile itself is process-wide)
            parent: Parent widget
        """
        super().__init__(parent)
        self.state = state
        self._worker = None
        self._setup_ui()
        self._timer = QTimer(self)
        self._timer.setInterval(2000)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()
        self._refresh()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.text = QPlainTextEdit(self)
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.text.setMinimumHeight(140)
        layout.addWidget(self.text)

        row = QHBoxLayout()
        self.calibrate_button = QPushButton("Calibrate (quick)", self)
        self.calibrate_button.setToolTip(
            "Measure kernel throughput, bytes per atom and transfer rates on this "
            "machine (about a minute, GPU must be idle). Seeds the launch sizes and "
            "the run-time estimate shown before a scan.")
        self.calibrate_button.clicked.connect(self._calibrate)
        row.addWidget(self.calibrate_button)
        self.status = QLabel("", self)
        self.status.setWordWrap(True)
        row.addWidget(self.status, 1)
        layout.addLayout(row)

        hint = QLabel(
            "Overrides (environment): MOSAIC_DEVICES, MOSAIC_GPU_MEM_LIMIT, "
            "MOSAIC_HOST_MEM_LIMIT, MOSAIC_LAUNCH_CAP_S, MOSAIC_CPU_THREADS, "
            "MOSAIC_CHUNK_CACHE=0. See hardware.py.", self)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(hint)

    # ------------------------------------------------------------------ text
    def _report(self):
        """Assemble the profile, budget, cache and calibration text."""
        lines = [hardware.report()]
        try:
            hg = hardware.host_governor()
            lines.append(f"Host budget now: {hg.budget() / _GIB:.1f} GB "
                         f"(allowance {hg.allowance() / _GIB:.1f} GB after working sets)")
        except Exception:
            pass
        try:
            import Sample
            cache = Sample._chunk_cache()
            if cache is not None:
                st = cache.stats()
                lines.append(f"Chunk cache: {st['entries']} entries, {st['bytes'] / _GIB:.2f} GB, "
                             f"{st['hits']} hits / {st['misses']} misses")
        except Exception:
            pass
        try:
            if hardware.probe().gpus:
                for g in hardware.probe().gpus:
                    dg = hardware.device_governor(g.index)
                    lines.append(f"GPU {g.index} budget now: {dg.budget() / _GIB:.1f} GB")
                cal = hardware.load_calibration()
                if cal:
                    fast = cal.get("throughput", {}).get("fast", {})
                    rate = fast.get("best") or fast.get("live")
                    lines.append(f"Calibration: {cal.get('written', 'unknown date')}"
                                 + (f", fast kernel {rate:.2e} atom.px/s" if rate else "")
                                 + (f", CPU {cal['cpu']['atom_px_per_s']:.2e} atom.px/s"
                                    if cal.get("cpu", {}).get("atom_px_per_s") else ""))
                else:
                    lines.append("Calibration: none on file (defaults in use; press Calibrate)")
        except Exception:
            pass
        return "\n".join(lines)

    def _refresh(self):
        """Refresh the text while the panel is visible."""
        if not self.isVisible():
            return
        try:
            self.text.setPlainText(self._report())
        except Exception as exc:
            self.text.setPlainText(f"hardware profile unavailable: {exc}")

    # ------------------------------------------------------------- calibrate
    def _calibrate(self):
        """Handle the Calibrate button: run a quick calibration in a worker thread."""
        if self._worker is not None and self._worker.isRunning():
            return
        self.calibrate_button.setEnabled(False)
        self.status.setText("calibrating...")
        self._worker = _CalibrateWorker(self)
        self._worker.finished_with.connect(self._calibrated)
        self._worker.start()

    def _calibrated(self, msg):
        """Show the worker's result and re-enable the button."""
        self.status.setText(msg)
        self.calibrate_button.setEnabled(True)
        self._refresh()
