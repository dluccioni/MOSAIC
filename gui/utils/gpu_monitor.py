# -----------------------------------------------------------------------------
# GPU Monitor
# -----------------------------------------------------------------------------
"""
Widget for monitoring GPU memory and status.

Provides:
- Real-time GPU memory usage display
- Memory warning indicators
- Multi-GPU support
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QProgressBar,
    QToolButton,
    QMenu,
    QFrame,
)
from PySide6.QtGui import QAction

# Try to import CuPy for GPU info
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


@dataclass
class GPUInfo:
    """Information about a GPU device."""
    device_id: int
    name: str
    memory_total: int  # bytes
    memory_used: int  # bytes
    memory_free: int  # bytes


class GPUMonitor(QWidget):
    """
    Widget for displaying GPU memory status.

    Updates periodically and shows:
    - Current memory usage
    - Total memory available
    - Usage bar with color coding

    Signals:
        memory_warning: Emitted when memory usage exceeds threshold
    """

    memory_warning = Signal(float)  # usage percentage

    WARNING_THRESHOLD = 0.8  # 80% memory usage
    CRITICAL_THRESHOLD = 0.95  # 95% memory usage

    def __init__(self, update_interval: int = 2000, parent=None):
        """
        Initialize the GPU monitor.

        Args:
            update_interval: Update interval in milliseconds
            parent: Parent widget
        """
        super().__init__(parent)
        self._update_interval = update_interval
        self._current_device = 0
        self._setup_ui()
        self._setup_timer()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(8)

        # GPU label with device selector
        self.gpu_button = QToolButton()
        self.gpu_button.setText("GPU:")
        self.gpu_button.setPopupMode(QToolButton.InstantPopup)
        self._setup_device_menu()
        layout.addWidget(self.gpu_button)

        # Memory bar
        self.memory_bar = QProgressBar()
        self.memory_bar.setMaximumWidth(100)
        self.memory_bar.setMaximumHeight(16)
        self.memory_bar.setTextVisible(False)
        self.memory_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1e1e1e;
                border: 1px solid #404040;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background-color: #2a82da;
                border-radius: 2px;
            }
        """)
        layout.addWidget(self.memory_bar)

        # Memory text
        self.memory_label = QLabel("--/-- GB")
        self.memory_label.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        layout.addWidget(self.memory_label)

    def _setup_device_menu(self):
        """Setup the GPU device selection menu."""
        menu = QMenu(self)

        if not CUPY_AVAILABLE:
            action = QAction("No GPU available", self)
            action.setEnabled(False)
            menu.addAction(action)
        else:
            try:
                device_count = cp.cuda.runtime.getDeviceCount()
                for i in range(device_count):
                    with cp.cuda.Device(i):
                        props = cp.cuda.runtime.getDeviceProperties(i)
                        name = props.get('name', b'Unknown').decode('utf-8')

                    action = QAction(f"GPU {i}: {name}", self)
                    action.setData(i)
                    action.triggered.connect(lambda checked, dev=i: self._select_device(dev))
                    menu.addAction(action)

                    if i == 0:
                        action.setCheckable(True)
                        action.setChecked(True)
            except Exception as e:
                action = QAction(f"Error: {e}", self)
                action.setEnabled(False)
                menu.addAction(action)

        self.gpu_button.setMenu(menu)

    def _setup_timer(self):
        """Setup the update timer."""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_display)
        self._timer.start(self._update_interval)

        # Initial update
        self._update_display()

    def _select_device(self, device_id: int):
        """Select a GPU device to monitor."""
        self._current_device = device_id

        # Update menu checkmarks
        menu = self.gpu_button.menu()
        for action in menu.actions():
            if action.data() is not None:
                action.setChecked(action.data() == device_id)

        self._update_display()

    def _update_display(self):
        """Update the display with current GPU info."""
        info = self._get_gpu_info()

        if info is None:
            self.memory_bar.setValue(0)
            self.memory_label.setText("No GPU")
            return

        # Calculate usage
        usage = info.memory_used / info.memory_total if info.memory_total > 0 else 0

        # Update progress bar
        self.memory_bar.setValue(int(usage * 100))

        # Update color based on usage
        if usage >= self.CRITICAL_THRESHOLD:
            color = "#ff4444"
        elif usage >= self.WARNING_THRESHOLD:
            color = "#ffa500"
        else:
            color = "#2a82da"

        self.memory_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: #1e1e1e;
                border: 1px solid #404040;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 2px;
            }}
        """)

        # Update label
        used_gb = info.memory_used / (1024 ** 3)
        total_gb = info.memory_total / (1024 ** 3)
        self.memory_label.setText(f"{used_gb:.1f}/{total_gb:.1f} GB")

        # Emit warning if threshold exceeded
        if usage >= self.WARNING_THRESHOLD:
            self.memory_warning.emit(usage)

    def _get_gpu_info(self) -> Optional[GPUInfo]:
        """Get current GPU information."""
        if not CUPY_AVAILABLE:
            return None

        try:
            with cp.cuda.Device(self._current_device):
                meminfo = cp.cuda.runtime.memGetInfo()
                free_memory = meminfo[0]
                total_memory = meminfo[1]
                used_memory = total_memory - free_memory

                props = cp.cuda.runtime.getDeviceProperties(self._current_device)
                name = props.get('name', b'Unknown').decode('utf-8')

                return GPUInfo(
                    device_id=self._current_device,
                    name=name,
                    memory_total=total_memory,
                    memory_used=used_memory,
                    memory_free=free_memory
                )
        except Exception as e:
            print(f"Error getting GPU info: {e}")
            return None

    def get_memory_usage(self) -> Tuple[float, float]:
        """
        Get current memory usage.

        Returns:
            Tuple of (used_gb, total_gb)
        """
        info = self._get_gpu_info()
        if info is None:
            return (0.0, 0.0)

        used_gb = info.memory_used / (1024 ** 3)
        total_gb = info.memory_total / (1024 ** 3)
        return (used_gb, total_gb)

    def set_update_interval(self, interval_ms: int):
        """Set the update interval in milliseconds."""
        self._update_interval = interval_ms
        self._timer.setInterval(interval_ms)

    def start_monitoring(self):
        """Start the monitoring timer."""
        if not self._timer.isActive():
            self._timer.start()

    def stop_monitoring(self):
        """Stop the monitoring timer."""
        self._timer.stop()


class GPUMemoryBar(QFrame):
    """
    Simple GPU memory bar without device selection.

    Compact widget for embedding in status bars.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._setup_timer()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.label = QLabel("GPU: --")
        self.label.setStyleSheet("color: #808080; font-size: 11px;")
        layout.addWidget(self.label)

    def _setup_timer(self):
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update)
        self._timer.start(2000)
        self._update()

    def _update(self):
        if not CUPY_AVAILABLE:
            self.label.setText("GPU: N/A")
            return

        try:
            meminfo = cp.cuda.runtime.memGetInfo()
            free_memory = meminfo[0]
            total_memory = meminfo[1]
            used_memory = total_memory - free_memory

            used_gb = used_memory / (1024 ** 3)
            total_gb = total_memory / (1024 ** 3)

            self.label.setText(f"GPU: {used_gb:.1f}/{total_gb:.1f} GB")
        except Exception:
            self.label.setText("GPU: Error")
