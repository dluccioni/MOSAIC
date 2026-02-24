# -----------------------------------------------------------------------------
# Object Browser Panel
# -----------------------------------------------------------------------------
"""
Tree view panel showing all simulation objects.

The ObjectBrowser displays a hierarchical view of:
- Crystal (lattice structure)
- Sample (atomic positions)
- Beam (X-ray source)
- Detector (measurement)
- Stage (goniometer)
- Optics (optical components)
- Defects (defect definitions)
- Deformation (strain fields)
- Analysis (analysis tools)
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

from PySide6.QtCore import Qt, Signal, QModelIndex
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QMenu,
    QInputDialog,
    QMessageBox,
    QHeaderView,
)
from PySide6.QtGui import QAction, QIcon, QColor, QBrush

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gui.state import SimulationState


class ObjectBrowser(QWidget):
    """
    Tree view panel for browsing simulation objects.

    Signals:
        object_selected: Emitted when an object is selected (object_name, object)
        object_activated: Emitted when an object is double-clicked (object_name, object)
        create_requested: Emitted when user requests to create an object (object_type)
        delete_requested: Emitted when user requests to delete an object (object_name)
    """

    object_selected = Signal(str, object)
    object_activated = Signal(str, object)
    create_requested = Signal(str)
    delete_requested = Signal(str)

    # Object types with their display names and icons
    OBJECT_TYPES = [
        ("crystal", "Crystal", "Structure and orientation"),
        ("sample", "Sample", "Atomic positions"),
        ("beam", "Beam", "X-ray source"),
        ("detector", "Detector", "Measurement device"),
        ("stage", "Stage", "Goniometer/motors"),
        ("optics", "Optics", "Optical components"),
        ("defects", "Defects", "Defect definitions"),
        ("deformation", "Deformation", "Strain fields"),
        ("analysis", "Analysis", "Analysis tools"),
    ]

    def __init__(self, state: SimulationState, parent=None):
        """
        Initialize the object browser.

        Args:
            state: SimulationState instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.state = state
        self._setup_ui()
        self._register_observers()
        self._refresh_tree()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create tree widget
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Object", "Status"])
        self.tree.setColumnCount(2)
        self.tree.setAlternatingRowColors(True)
        self.tree.setAnimated(True)
        self.tree.setIndentation(20)
        self.tree.setRootIsDecorated(False)
        self.tree.setExpandsOnDoubleClick(False)

        # Configure header
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)

        # Enable context menu
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)

        # Connect signals
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)

        layout.addWidget(self.tree)

        # Style
        self.setStyleSheet("""
            QTreeWidget {
                background-color: #1e1e1e;
                border: none;
                font-size: 13px;
            }
            QTreeWidget::item {
                padding: 6px 4px;
                border-bottom: 1px solid #2d2d2d;
            }
            QTreeWidget::item:selected {
                background-color: #2a82da;
            }
            QTreeWidget::item:hover:!selected {
                background-color: #353535;
            }
        """)

    def _register_observers(self):
        """Register for state change notifications."""
        for obj_type, _, _ in self.OBJECT_TYPES:
            event = f"{obj_type}_changed"
            self.state.register_observer(event, lambda data, t=obj_type: self._on_object_changed(t))

        self.state.register_observer("project_changed", lambda _: self._refresh_tree())

    def _refresh_tree(self):
        """Refresh the entire tree."""
        self.tree.clear()

        for obj_type, display_name, description in self.OBJECT_TYPES:
            item = QTreeWidgetItem([display_name, ""])
            item.setData(0, Qt.UserRole, obj_type)
            item.setToolTip(0, description)

            # Set status based on whether object exists
            obj = getattr(self.state, obj_type, None)
            self._update_item_status(item, obj)

            self.tree.addTopLevelItem(item)

    def _update_item_status(self, item: QTreeWidgetItem, obj):
        """
        Update the status column for an item.

        Args:
            item: Tree widget item
            obj: The simulation object (or None)
        """
        if obj is None:
            item.setText(1, "Not created")
            item.setForeground(1, QBrush(QColor("#808080")))
        else:
            item.setText(1, "Ready")
            item.setForeground(1, QBrush(QColor("#4ec94e")))

            # Add additional info based on object type
            obj_type = item.data(0, Qt.UserRole)
            info = self._get_object_info(obj_type, obj)
            if info:
                item.setToolTip(1, info)

    def _get_object_info(self, obj_type: str, obj) -> str:
        """
        Get a brief info string for an object.

        Args:
            obj_type: Type of object
            obj: The object instance

        Returns:
            Info string for tooltip
        """
        info_parts = []

        try:
            if obj_type == "crystal":
                if hasattr(obj, "symbol"):
                    info_parts.append(f"Symbol: {obj.symbol}")
                if hasattr(obj, "space_group"):
                    info_parts.append(f"Space group: {obj.space_group}")

            elif obj_type == "sample":
                if hasattr(obj, "dimensions"):
                    dims = obj.dimensions
                    info_parts.append(f"Size: {dims[0]:.0f} x {dims[1]:.0f} x {dims[2]:.0f} Å")
                if hasattr(obj, "atom_count"):
                    info_parts.append(f"Atoms: {obj.atom_count:,}")

            elif obj_type == "beam":
                if hasattr(obj, "energy"):
                    info_parts.append(f"Energy: {obj.energy:.1f} eV")
                if hasattr(obj, "wavelength"):
                    info_parts.append(f"λ: {obj.wavelength:.4f} Å")

            elif obj_type == "detector":
                if hasattr(obj, "shape"):
                    info_parts.append(f"Shape: {obj.shape[0]} x {obj.shape[1]} px")
                if hasattr(obj, "distance"):
                    info_parts.append(f"Distance: {obj.distance:.1f} mm")

            elif obj_type == "stage":
                if hasattr(obj, "motors"):
                    info_parts.append(f"Motors: {len(obj.motors)}")

            elif obj_type == "optics":
                if hasattr(obj, "stack"):
                    info_parts.append(f"Components: {len(obj.stack)}")

            elif obj_type == "defects":
                count = 0
                if hasattr(obj, "stacking_faults"):
                    count += len(getattr(obj, "stacking_faults", []))
                if hasattr(obj, "cracks"):
                    count += len(getattr(obj, "cracks", []))
                info_parts.append(f"Defects: {count}")

            elif obj_type == "deformation":
                if hasattr(obj, "field_type"):
                    info_parts.append(f"Type: {obj.field_type}")

            elif obj_type == "analysis":
                if hasattr(obj, "directory"):
                    info_parts.append(f"Directory: {obj.directory}")

        except Exception:
            pass

        return "\n".join(info_parts)

    def _on_object_changed(self, obj_type: str):
        """
        Handle object change notification.

        Args:
            obj_type: Type of object that changed
        """
        # Find and update the corresponding item
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item.data(0, Qt.UserRole) == obj_type:
                obj = getattr(self.state, obj_type, None)
                self._update_item_status(item, obj)
                break

    def _on_selection_changed(self):
        """Handle tree selection change."""
        items = self.tree.selectedItems()
        if items:
            item = items[0]
            obj_type = item.data(0, Qt.UserRole)
            obj = getattr(self.state, obj_type, None)
            self.object_selected.emit(obj_type, obj)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """
        Handle item double-click.

        Args:
            item: Clicked item
            column: Clicked column
        """
        obj_type = item.data(0, Qt.UserRole)
        obj = getattr(self.state, obj_type, None)

        if obj is None:
            # Request creation if object doesn't exist
            self.create_requested.emit(obj_type)
        else:
            # Emit activation signal
            self.object_activated.emit(obj_type, obj)

    def _show_context_menu(self, position):
        """
        Show context menu for the selected item.

        Args:
            position: Menu position
        """
        item = self.tree.itemAt(position)
        if item is None:
            return

        obj_type = item.data(0, Qt.UserRole)
        obj = getattr(self.state, obj_type, None)

        menu = QMenu(self)

        if obj is None:
            create_action = QAction(f"Create {obj_type.title()}", self)
            create_action.triggered.connect(lambda: self.create_requested.emit(obj_type))
            menu.addAction(create_action)
        else:
            edit_action = QAction(f"Edit {obj_type.title()}", self)
            edit_action.triggered.connect(lambda: self.object_activated.emit(obj_type, obj))
            menu.addAction(edit_action)

            menu.addSeparator()

            delete_action = QAction(f"Delete {obj_type.title()}", self)
            delete_action.triggered.connect(lambda: self._confirm_delete(obj_type))
            menu.addAction(delete_action)

        menu.exec_(self.tree.viewport().mapToGlobal(position))

    def _confirm_delete(self, obj_type: str):
        """
        Confirm and handle object deletion.

        Args:
            obj_type: Type of object to delete
        """
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete the {obj_type}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            setattr(self.state, obj_type, None)
            self.delete_requested.emit(obj_type)

    def select_object(self, obj_type: str):
        """
        Programmatically select an object in the tree.

        Args:
            obj_type: Type of object to select
        """
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item.data(0, Qt.UserRole) == obj_type:
                self.tree.setCurrentItem(item)
                break

    def get_selected_object_type(self) -> Optional[str]:
        """
        Get the currently selected object type.

        Returns:
            Object type string or None
        """
        items = self.tree.selectedItems()
        if items:
            return items[0].data(0, Qt.UserRole)
        return None
