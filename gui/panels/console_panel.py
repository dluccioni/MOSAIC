# -----------------------------------------------------------------------------
# Console Panel
# -----------------------------------------------------------------------------
"""
Console panel for displaying log messages and output.

The ConsolePanel provides:
- Scrolling log view with timestamp
- Log level filtering
- Search functionality
- Copy/Clear actions
- Integration with Python logging
- Capture of stdout/stderr for simulator print output
"""

import sys
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import deque

from PySide6.QtCore import Qt, Signal, Slot, QObject, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QComboBox,
    QLineEdit,
    QPushButton,
    QLabel,
    QFrame,
)
from PySide6.QtGui import QTextCursor, QTextCharFormat, QColor, QFont

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gui.state import SimulationState


class QTextEditHandler(logging.Handler, QObject):
    """
    Logging handler that emits signals for Qt integration.
    """

    log_message = Signal(str, int)  # message, level

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_message.emit(msg, record.levelno)
        except Exception:
            self.handleError(record)


class StreamRedirector(QObject):
    """
    Redirects stdout/stderr to emit signals for Qt integration.
    Captures print() output from the simulator's custom logging class.
    """

    text_written = Signal(str, int)  # text, level (INFO for stdout, ERROR for stderr)

    def __init__(self, original_stream, is_stderr=False):
        super().__init__()
        self._original_stream = original_stream
        self._is_stderr = is_stderr
        self._buffer = ""

    def write(self, text):
        # Always write to original stream as well
        if self._original_stream:
            self._original_stream.write(text)
            self._original_stream.flush()

        # Skip empty strings
        if not text or text.isspace():
            return

        # Determine log level based on content and stream type
        level = logging.ERROR if self._is_stderr else logging.INFO

        # Parse simulator's custom logging format: [name|LEVEL] message
        # This helps categorize output properly
        match = re.match(r'\[([^\|]+)\|([A-Z]+)\]\s*(.*)', text.strip())
        if match:
            log_name, level_str, message = match.groups()
            level_map = {
                'DEBUG': logging.DEBUG,
                'NORMAL': logging.INFO,
                'VERBOSE': logging.DEBUG,
                'WARNING': logging.WARNING,
                'ERROR': logging.ERROR,
            }
            level = level_map.get(level_str, logging.INFO)

        # Emit the text
        self.text_written.emit(text.rstrip(), level)

    def flush(self):
        if self._original_stream:
            self._original_stream.flush()

    def fileno(self):
        if self._original_stream:
            return self._original_stream.fileno()
        raise OSError("StreamRedirector does not have a fileno")

    def isatty(self):
        return False


class ConsolePanel(QWidget):
    """
    Panel for displaying log messages and simulator output.

    Features:
    - Color-coded log levels
    - Level filtering
    - Text search
    - Copy to clipboard
    - Clear functionality
    - Message limit to prevent memory issues
    - Captures both Python logging and print() output

    Signals:
        message_logged: Emitted when a message is logged (message, level)
    """

    message_logged = Signal(str, int)

    # Log level colors
    LEVEL_COLORS = {
        logging.DEBUG: "#808080",     # Gray
        logging.INFO: "#d0d0d0",      # Light gray
        logging.WARNING: "#ffa500",   # Orange
        logging.ERROR: "#ff4444",     # Red
        logging.CRITICAL: "#ff0000",  # Bright red
    }

    # Log level names
    LEVEL_NAMES = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRITICAL",
    }

    def __init__(self, state: SimulationState = None, max_messages: int = 1000, parent=None):
        """
        Initialize the console panel.

        Args:
            state: SimulationState instance
            max_messages: Maximum messages to keep in buffer
            parent: Parent widget
        """
        super().__init__(parent)
        self.state = state
        self._max_messages = max_messages
        self._messages = deque(maxlen=max_messages)
        self._filter_level = logging.DEBUG
        self._search_text = ""

        # Store original streams
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._stdout_redirector = None
        self._stderr_redirector = None

        self._setup_ui()
        self._setup_logging()
        self._setup_stream_capture()

        # Log initial message
        self.log_info("Console initialized - capturing simulator output")

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QFrame()
        toolbar.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border-bottom: 1px solid #404040;
            }
        """)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 4, 8, 4)
        toolbar_layout.setSpacing(8)

        # Level filter
        level_label = QLabel("Level:")
        toolbar_layout.addWidget(level_label)

        self.level_combo = QComboBox()
        self.level_combo.addItem("All", logging.DEBUG)
        self.level_combo.addItem("Info+", logging.INFO)
        self.level_combo.addItem("Warning+", logging.WARNING)
        self.level_combo.addItem("Error+", logging.ERROR)
        self.level_combo.currentIndexChanged.connect(self._on_level_changed)
        toolbar_layout.addWidget(self.level_combo)

        # Search
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search...")
        self.search_edit.setMaximumWidth(200)
        self.search_edit.textChanged.connect(self._on_search_changed)
        toolbar_layout.addWidget(self.search_edit)

        toolbar_layout.addStretch()

        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear)
        toolbar_layout.addWidget(clear_btn)

        # Copy button
        copy_btn = QPushButton("Copy")
        copy_btn.clicked.connect(self._copy_to_clipboard)
        toolbar_layout.addWidget(copy_btn)

        layout.addWidget(toolbar)

        # Text area
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Consolas", 10))
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d0d0d0;
                border: none;
                selection-background-color: #2a82da;
            }
        """)
        layout.addWidget(self.text_edit)

        # Status bar
        self.status_label = QLabel("0 messages")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #252525;
                color: #808080;
                padding: 4px 8px;
                font-size: 11px;
            }
        """)
        layout.addWidget(self.status_label)

    def _setup_logging(self):
        """Setup Python logging integration."""
        # Create and configure handler
        self._handler = QTextEditHandler()
        self._handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        ))
        self._handler.log_message.connect(self._on_log_message)

        # Add handler to root logger and set level to capture everything
        root_logger = logging.getLogger()
        root_logger.addHandler(self._handler)

        # Ensure root logger captures all levels (default is WARNING)
        if root_logger.level > logging.DEBUG:
            root_logger.setLevel(logging.DEBUG)

    def _setup_stream_capture(self):
        """Setup stdout/stderr capture for print() statements."""
        # Create redirectors
        self._stdout_redirector = StreamRedirector(self._original_stdout, is_stderr=False)
        self._stderr_redirector = StreamRedirector(self._original_stderr, is_stderr=True)

        # Connect signals
        self._stdout_redirector.text_written.connect(self._on_stream_output)
        self._stderr_redirector.text_written.connect(self._on_stream_output)

        # Redirect streams
        sys.stdout = self._stdout_redirector
        sys.stderr = self._stderr_redirector

    def _restore_streams(self):
        """Restore original stdout/stderr."""
        if self._original_stdout:
            sys.stdout = self._original_stdout
        if self._original_stderr:
            sys.stderr = self._original_stderr

    @Slot(str, int)
    def _on_stream_output(self, text: str, level: int):
        """
        Handle captured stdout/stderr output.

        Args:
            text: Output text
            level: Log level
        """
        if not text.strip():
            return

        # Format with timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        level_name = self.LEVEL_NAMES.get(level, "INFO")
        formatted = f"{timestamp} [{level_name}] {text}"

        # Store and display
        self._messages.append((formatted, level))

        if level >= self._filter_level:
            if not self._search_text or self._search_text.lower() in formatted.lower():
                self._append_message(formatted, level)

        self._update_status()
        self.message_logged.emit(formatted, level)

    def _on_log_message(self, message: str, level: int):
        """
        Handle incoming log message from Python logging.

        Args:
            message: Log message
            level: Log level
        """
        # Store message
        self._messages.append((message, level))

        # Display if passes filter
        if level >= self._filter_level:
            if not self._search_text or self._search_text.lower() in message.lower():
                self._append_message(message, level)

        # Update status
        self._update_status()

        # Emit signal
        self.message_logged.emit(message, level)

    def _append_message(self, message: str, level: int):
        """
        Append a message to the text area.

        Args:
            message: Message text
            level: Log level
        """
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)

        # Set color based on level
        fmt = QTextCharFormat()
        color = self.LEVEL_COLORS.get(level, "#d0d0d0")
        fmt.setForeground(QColor(color))

        cursor.setCharFormat(fmt)
        cursor.insertText(message + "\n")

        # Scroll to bottom
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()

    def _on_level_changed(self, index: int):
        """Handle level filter change."""
        self._filter_level = self.level_combo.currentData()
        self._refresh_display()

    def _on_search_changed(self, text: str):
        """Handle search text change."""
        self._search_text = text
        self._refresh_display()

    def _refresh_display(self):
        """Refresh the display based on current filters."""
        self.text_edit.clear()

        for message, level in self._messages:
            if level >= self._filter_level:
                if not self._search_text or self._search_text.lower() in message.lower():
                    self._append_message(message, level)

    def _update_status(self):
        """Update the status bar."""
        total = len(self._messages)
        visible = sum(1 for msg, lvl in self._messages
                     if lvl >= self._filter_level
                     and (not self._search_text or self._search_text.lower() in msg.lower()))
        self.status_label.setText(f"{visible}/{total} messages")

    def _copy_to_clipboard(self):
        """Copy visible messages to clipboard."""
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(self.text_edit.toPlainText())

    def clear(self):
        """Clear all messages."""
        self._messages.clear()
        self.text_edit.clear()
        self._update_status()

    def log(self, message: str, level: int = logging.INFO):
        """
        Log a message directly.

        Args:
            message: Message text
            level: Log level
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        level_name = self.LEVEL_NAMES.get(level, "INFO")
        formatted = f"{timestamp} [{level_name}] GUI: {message}"
        self._on_log_message(formatted, level)

    def log_info(self, message: str):
        """Log an info message."""
        self.log(message, logging.INFO)

    def log_warning(self, message: str):
        """Log a warning message."""
        self.log(message, logging.WARNING)

    def log_error(self, message: str):
        """Log an error message."""
        self.log(message, logging.ERROR)

    def log_debug(self, message: str):
        """Log a debug message."""
        self.log(message, logging.DEBUG)

    def write_to_console(self, text: str, level: int = logging.INFO):
        """
        Write text directly to the console (for external use).

        Args:
            text: Text to write
            level: Log level for coloring
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        level_name = self.LEVEL_NAMES.get(level, "INFO")
        formatted = f"{timestamp} [{level_name}] {text}"

        self._messages.append((formatted, level))

        if level >= self._filter_level:
            if not self._search_text or self._search_text.lower() in formatted.lower():
                self._append_message(formatted, level)

        self._update_status()

    def closeEvent(self, event):
        """Handle close event - remove logging handler and restore streams."""
        # Remove logging handler
        root_logger = logging.getLogger()
        if self._handler in root_logger.handlers:
            root_logger.removeHandler(self._handler)

        # Restore original streams
        self._restore_streams()

        super().closeEvent(event)

    def __del__(self):
        """Cleanup when object is deleted."""
        try:
            self._restore_streams()
        except Exception:
            pass
