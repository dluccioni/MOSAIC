# -----------------------------------------------------------------------------
# Inspector Panel
# -----------------------------------------------------------------------------
"""
Base inspector panel for editing simulation object properties.

The InspectorPanel provides:
- Dynamic property editing widgets
- Value validation and conversion
- Change notifications to state
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List, Tuple, Union
from dataclasses import dataclass
import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QScrollArea,
    QFrame,
    QLabel,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
    QPushButton,
    QGroupBox,
    QFileDialog,
    QSizePolicy,
    QSlider,
)
from PySide6.QtGui import QDoubleValidator, QIntValidator

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gui.state import SimulationState


@dataclass
class PropertyDef:
    """Definition of an editable property."""
    name: str  # Internal name
    label: str  # Display label
    type: str  # "float", "int", "str", "bool", "choice", "vector3", "file", "slider"
    default: Any = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[Tuple[str, Any]]] = None  # For "choice" type: [(display, value), ...]
    suffix: str = ""  # Unit suffix (e.g., "eV", "Å")
    tooltip: str = ""
    readonly: bool = False
    decimals: int = 4  # For float type
    file_filter: str = "All Files (*)"  # For file type


class InspectorPanel(QWidget):
    """
    Base panel for editing object properties.

    Subclass this for specific object types (CrystalInspector, BeamInspector, etc.)

    Signals:
        property_changed: Emitted when a property value changes (name, value)
        apply_requested: Emitted when user clicks Apply button
    """

    property_changed = Signal(str, object)
    apply_requested = Signal()

    def __init__(self, state: SimulationState, parent=None):
        """
        Initialize the inspector panel.

        Args:
            state: SimulationState instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.state = state
        self._widgets: Dict[str, QWidget] = {}
        self._property_defs: Dict[str, PropertyDef] = {}
        self._current_object = None
        self._block_signals = False

        self._setup_ui()

    def _setup_ui(self):
        """Setup the base UI structure."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        self.header = QLabel("Inspector")
        self.header.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                border-bottom: 1px solid #404040;
            }
        """)
        layout.addWidget(self.header)

        # Scroll area for properties
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Content widget
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        self.content_layout.setSpacing(8)
        self.content_layout.addStretch()

        scroll.setWidget(self.content)
        layout.addWidget(scroll)

        # Apply button (hidden by default)
        self.apply_btn = QPushButton("Apply Changes")
        self.apply_btn.clicked.connect(self._on_apply)
        self.apply_btn.setVisible(False)
        layout.addWidget(self.apply_btn)

    def set_title(self, title: str):
        """Set the inspector title."""
        self.header.setText(title)

    def set_object(self, obj):
        """
        Set the object being edited.

        Args:
            obj: The simulation object
        """
        self._current_object = obj
        self._refresh_values()

    def show_apply_button(self, show: bool = True):
        """Show or hide the Apply button."""
        self.apply_btn.setVisible(show)

    def add_group(self, title: str) -> QGroupBox:
        """
        Add a property group.

        Args:
            title: Group title

        Returns:
            The group box widget
        """
        group = QGroupBox(title)
        group_layout = QFormLayout(group)
        group_layout.setContentsMargins(10, 15, 10, 10)
        group_layout.setSpacing(8)
        group_layout.setLabelAlignment(Qt.AlignRight)

        # Insert before the stretch
        self.content_layout.insertWidget(self.content_layout.count() - 1, group)
        return group

    def add_property(self, prop: PropertyDef, group: QGroupBox = None) -> QWidget:
        """
        Add a property editor widget.

        Args:
            prop: Property definition
            group: Optional group to add to

        Returns:
            The created widget
        """
        self._property_defs[prop.name] = prop
        widget = self._create_widget(prop)
        self._widgets[prop.name] = widget

        # Determine layout to add to
        if group:
            layout = group.layout()
        else:
            layout = self.content_layout
            # Insert before stretch
            idx = layout.count() - 1

        # Add to layout
        if isinstance(layout, QFormLayout):
            label = QLabel(prop.label)
            if prop.tooltip:
                label.setToolTip(prop.tooltip)
                widget.setToolTip(prop.tooltip)
            layout.addRow(label, widget)
        else:
            container = QWidget()
            container_layout = QHBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            label = QLabel(prop.label)
            label.setMinimumWidth(100)
            if prop.tooltip:
                label.setToolTip(prop.tooltip)
                widget.setToolTip(prop.tooltip)
            container_layout.addWidget(label)
            container_layout.addWidget(widget, 1)
            layout.insertWidget(idx, container)

        return widget

    def _create_widget(self, prop: PropertyDef) -> QWidget:
        """
        Create the appropriate widget for a property type.

        Args:
            prop: Property definition

        Returns:
            The created widget
        """
        if prop.type == "float":
            widget = QDoubleSpinBox()
            widget.setDecimals(prop.decimals)
            if prop.min_val is not None:
                widget.setMinimum(prop.min_val)
            else:
                widget.setMinimum(-1e9)
            if prop.max_val is not None:
                widget.setMaximum(prop.max_val)
            else:
                widget.setMaximum(1e9)
            if prop.step is not None:
                widget.setSingleStep(prop.step)
            if prop.suffix:
                widget.setSuffix(f" {prop.suffix}")
            if prop.default is not None:
                widget.setValue(prop.default)
            widget.setReadOnly(prop.readonly)
            widget.valueChanged.connect(lambda v, n=prop.name: self._on_value_changed(n, v))

        elif prop.type == "int":
            widget = QSpinBox()
            if prop.min_val is not None:
                widget.setMinimum(int(prop.min_val))
            else:
                widget.setMinimum(-999999999)
            if prop.max_val is not None:
                widget.setMaximum(int(prop.max_val))
            else:
                widget.setMaximum(999999999)
            if prop.step is not None:
                widget.setSingleStep(int(prop.step))
            if prop.suffix:
                widget.setSuffix(f" {prop.suffix}")
            if prop.default is not None:
                widget.setValue(prop.default)
            widget.setReadOnly(prop.readonly)
            widget.valueChanged.connect(lambda v, n=prop.name: self._on_value_changed(n, v))

        elif prop.type == "str":
            widget = QLineEdit()
            if prop.default is not None:
                widget.setText(str(prop.default))
            widget.setReadOnly(prop.readonly)
            widget.textChanged.connect(lambda v, n=prop.name: self._on_value_changed(n, v))

        elif prop.type == "bool":
            widget = QCheckBox()
            if prop.default is not None:
                widget.setChecked(prop.default)
            widget.setEnabled(not prop.readonly)
            widget.stateChanged.connect(lambda v, n=prop.name: self._on_value_changed(n, v == Qt.Checked))

        elif prop.type == "choice":
            widget = QComboBox()
            if prop.choices:
                for display, value in prop.choices:
                    widget.addItem(display, value)
            if prop.default is not None:
                idx = widget.findData(prop.default)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            widget.setEnabled(not prop.readonly)
            widget.currentIndexChanged.connect(
                lambda idx, n=prop.name, w=widget: self._on_value_changed(n, w.currentData())
            )

        elif prop.type == "vector3":
            widget = Vector3Widget(prop.default, prop.decimals, prop.suffix, prop.readonly)
            widget.value_changed.connect(lambda v, n=prop.name: self._on_value_changed(n, v))

        elif prop.type == "file":
            widget = FilePathWidget(prop.file_filter, prop.default, prop.readonly)
            widget.path_changed.connect(lambda v, n=prop.name: self._on_value_changed(n, v))

        elif prop.type == "slider":
            widget = SliderWidget(
                prop.min_val or 0,
                prop.max_val or 100,
                prop.default or 0,
                prop.decimals,
                prop.suffix,
                prop.readonly
            )
            widget.value_changed.connect(lambda v, n=prop.name: self._on_value_changed(n, v))

        else:
            # Fallback to line edit
            widget = QLineEdit()
            widget.setReadOnly(True)
            widget.setText(f"Unknown type: {prop.type}")

        return widget

    def _on_value_changed(self, name: str, value: Any):
        """
        Handle property value change.

        Args:
            name: Property name
            value: New value
        """
        if self._block_signals:
            return

        self.property_changed.emit(name, value)

        # Update object attribute if set
        if self._current_object is not None:
            if hasattr(self._current_object, name):
                try:
                    setattr(self._current_object, name, value)
                except Exception:
                    pass

    def _on_apply(self):
        """Handle Apply button click."""
        self.apply_requested.emit()

    def _refresh_values(self):
        """Refresh all widget values from the current object."""
        if self._current_object is None:
            return

        self._block_signals = True
        try:
            for name, widget in self._widgets.items():
                if hasattr(self._current_object, name):
                    value = getattr(self._current_object, name)
                    self._set_widget_value(name, value)
        finally:
            self._block_signals = False

    def _set_widget_value(self, name: str, value: Any):
        """
        Set a widget's value.

        Args:
            name: Property name
            value: Value to set
        """
        widget = self._widgets.get(name)
        prop = self._property_defs.get(name)
        if widget is None or prop is None:
            return

        try:
            if prop.type == "float":
                widget.setValue(float(value) if value is not None else 0.0)
            elif prop.type == "int":
                widget.setValue(int(value) if value is not None else 0)
            elif prop.type == "str":
                widget.setText(str(value) if value is not None else "")
            elif prop.type == "bool":
                widget.setChecked(bool(value) if value is not None else False)
            elif prop.type == "choice":
                idx = widget.findData(value)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            elif prop.type == "vector3":
                widget.set_value(value)
            elif prop.type == "file":
                widget.set_path(str(value) if value else "")
            elif prop.type == "slider":
                widget.set_value(float(value) if value is not None else 0.0)
        except Exception:
            pass

    def get_value(self, name: str) -> Any:
        """
        Get the current value of a property.

        Args:
            name: Property name

        Returns:
            Current value
        """
        widget = self._widgets.get(name)
        prop = self._property_defs.get(name)
        if widget is None or prop is None:
            return None

        try:
            if prop.type == "float":
                return widget.value()
            elif prop.type == "int":
                return widget.value()
            elif prop.type == "str":
                return widget.text()
            elif prop.type == "bool":
                return widget.isChecked()
            elif prop.type == "choice":
                return widget.currentData()
            elif prop.type == "vector3":
                return widget.get_value()
            elif prop.type == "file":
                return widget.get_path()
            elif prop.type == "slider":
                return widget.get_value()
        except Exception:
            pass

        return None

    def set_value(self, name: str, value: Any):
        """
        Set a property value programmatically.

        Args:
            name: Property name
            value: Value to set
        """
        self._block_signals = True
        try:
            self._set_widget_value(name, value)
        finally:
            self._block_signals = False

    def clear(self):
        """Clear all properties."""
        self._widgets.clear()
        self._property_defs.clear()
        self._current_object = None

        # Remove all widgets except header and apply button
        while self.content_layout.count() > 1:
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


class Vector3Widget(QWidget):
    """Widget for editing 3D vectors."""

    value_changed = Signal(object)

    def __init__(self, default=None, decimals=4, suffix="", readonly=False, parent=None):
        super().__init__(parent)
        self._decimals = decimals
        self._setup_ui(default, decimals, suffix, readonly)

    def _setup_ui(self, default, decimals, suffix, readonly):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.spinboxes = []
        labels = ["X:", "Y:", "Z:"]
        defaults = default if default is not None else [0, 0, 0]

        for i, label in enumerate(labels):
            lbl = QLabel(label)
            lbl.setFixedWidth(16)
            layout.addWidget(lbl)

            sb = QDoubleSpinBox()
            sb.setDecimals(decimals)
            sb.setMinimum(-1e9)
            sb.setMaximum(1e9)
            sb.setValue(defaults[i] if i < len(defaults) else 0)
            sb.setReadOnly(readonly)
            if suffix:
                sb.setSuffix(f" {suffix}")
            sb.valueChanged.connect(self._emit_value)
            layout.addWidget(sb, 1)
            self.spinboxes.append(sb)

    def _emit_value(self):
        self.value_changed.emit(self.get_value())

    def get_value(self) -> np.ndarray:
        return np.array([sb.value() for sb in self.spinboxes])

    def set_value(self, value):
        if value is None:
            value = [0, 0, 0]
        for i, sb in enumerate(self.spinboxes):
            if i < len(value):
                sb.blockSignals(True)
                sb.setValue(float(value[i]))
                sb.blockSignals(False)


class FilePathWidget(QWidget):
    """Widget for file path selection."""

    path_changed = Signal(str)

    def __init__(self, file_filter="All Files (*)", default="", readonly=False, parent=None):
        super().__init__(parent)
        self._filter = file_filter
        self._setup_ui(default, readonly)

    def _setup_ui(self, default, readonly):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.line_edit = QLineEdit()
        self.line_edit.setText(default or "")
        self.line_edit.setReadOnly(readonly)
        self.line_edit.textChanged.connect(self.path_changed.emit)
        layout.addWidget(self.line_edit, 1)

        self.browse_btn = QPushButton("...")
        self.browse_btn.setFixedWidth(30)
        self.browse_btn.setEnabled(not readonly)
        self.browse_btn.clicked.connect(self._browse)
        layout.addWidget(self.browse_btn)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select File", "", self._filter)
        if path:
            self.line_edit.setText(path)

    def get_path(self) -> str:
        return self.line_edit.text()

    def set_path(self, path: str):
        self.line_edit.blockSignals(True)
        self.line_edit.setText(path)
        self.line_edit.blockSignals(False)


class SliderWidget(QWidget):
    """Widget combining slider and spinbox."""

    value_changed = Signal(float)

    def __init__(self, min_val=0, max_val=100, default=0, decimals=2, suffix="", readonly=False, parent=None):
        super().__init__(parent)
        self._min = min_val
        self._max = max_val
        self._decimals = decimals
        self._scale = 10 ** decimals
        self._setup_ui(min_val, max_val, default, decimals, suffix, readonly)

    def _setup_ui(self, min_val, max_val, default, decimals, suffix, readonly):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(min_val * self._scale))
        self.slider.setMaximum(int(max_val * self._scale))
        self.slider.setValue(int(default * self._scale))
        self.slider.setEnabled(not readonly)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider, 1)

        self.spinbox = QDoubleSpinBox()
        self.spinbox.setDecimals(decimals)
        self.spinbox.setMinimum(min_val)
        self.spinbox.setMaximum(max_val)
        self.spinbox.setValue(default)
        self.spinbox.setReadOnly(readonly)
        if suffix:
            self.spinbox.setSuffix(f" {suffix}")
        self.spinbox.valueChanged.connect(self._on_spinbox_changed)
        layout.addWidget(self.spinbox)

    def _on_slider_changed(self, value):
        float_val = value / self._scale
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(float_val)
        self.spinbox.blockSignals(False)
        self.value_changed.emit(float_val)

    def _on_spinbox_changed(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(int(value * self._scale))
        self.slider.blockSignals(False)
        self.value_changed.emit(value)

    def get_value(self) -> float:
        return self.spinbox.value()

    def set_value(self, value: float):
        self.spinbox.blockSignals(True)
        self.slider.blockSignals(True)
        self.spinbox.setValue(value)
        self.slider.setValue(int(value * self._scale))
        self.spinbox.blockSignals(False)
        self.slider.blockSignals(False)
