"""Identity-based scene hierarchy; duplicate names remain independently selectable."""
from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem


class SceneOutliner(QTreeWidget):
    move_requested = Signal(object)
    MIME_TYPE = 'application/x-proximap-outliner-move'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setIndentation(16)
        self.setSelectionMode(self.SelectionMode.ExtendedSelection)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        # Our event handlers enforce same-tree drops; Qt supplies the indicators
        # without owning row removal or model mutation.
        self.setDragDropMode(self.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setAutoExpandDelay(650)
        self.items_by_object = {}
        self._drag_objects = None
        self._scene = None

    def rebuild(self, scene):
        self._scene = scene
        expanded = {obj: item.isExpanded() for obj, item in self.items_by_object.items()}
        self.blockSignals(True)
        self.clear()
        self.items_by_object = {}
        pending = set(scene.objects)
        def insert(obj):
            if obj in self.items_by_object:
                return self.items_by_object[obj]
            parent = insert(obj.parent) if obj.parent in pending else self
            item = QTreeWidgetItem(parent, [obj.name])
            item.setData(0, Qt.ItemDataRole.UserRole, obj)
            item.setToolTip(0, 'Empty / Parent object' if obj.mesh is None else 'Mesh object')
            self.items_by_object[obj] = item
            item.setExpanded(expanded.get(obj, obj.mesh is None))
            item.setSelected(obj in scene.selected_objects)
            return item
        for obj in scene.objects:
            insert(obj)
        if scene.active_object in self.items_by_object:
            self.setCurrentItem(self.items_by_object[scene.active_object], 0, self.selectionModel().SelectionFlag.NoUpdate)
        self.blockSignals(False)

    def mimeTypes(self):
        return [self.MIME_TYPE]

    def mimeData(self, items):
        # Qt's default MIME data serializes item roles, including our Python
        # objects and their mesh arrays. Keep large assets entirely in memory.
        data = QMimeData()
        data.setData(self.MIME_TYPE, b'move')
        return data

    def supportedDropActions(self):
        return Qt.DropAction.MoveAction

    def _selected_roots(self):
        from mesh_editor.import_options import selection_roots
        def item_path(obj):
            item, path = self.items_by_object[obj], []
            while item is not None:
                parent = item.parent()
                path.insert(0, parent.indexOfChild(item) if parent else self.indexOfTopLevelItem(item))
                item = parent
            return path
        selected = [self.object_for(item) for item in self.selectedItems()]
        return sorted(selection_roots(selected), key=item_path)

    def startDrag(self, supported_actions):
        self._drag_objects = self._selected_roots()
        if not self._drag_objects:
            self._drag_objects = None
            return
        drag = QDrag(self)
        drag.setMimeData(self.mimeData(self.selectedItems()))
        rect = self.visualItemRect(self.items_by_object[self._drag_objects[0]])
        if rect.isValid():
            drag.setPixmap(self.viewport().grab(rect))
        try:
            # QAbstractItemView.startDrag may delete source rows after a move.
            # The scene coordinator already rebuilds them, so own the QDrag.
            drag.exec(Qt.DropAction.MoveAction, Qt.DropAction.MoveAction)
        finally:
            self._drag_objects = None
            drag.deleteLater()

    def _drop_location(self, point):
        target = self.object_for(self.itemAt(point))
        indicator = self.dropIndicatorPosition()
        if target is None or indicator == self.DropIndicatorPosition.OnViewport:
            return None, None
        if indicator == self.DropIndicatorPosition.OnItem:
            return target, None
        parent = target.parent
        if indicator == self.DropIndicatorPosition.AboveItem:
            return parent, target
        siblings = parent.children if parent is not None else [obj for obj in self._scene.objects if obj.parent is None]
        index = siblings.index(target) + 1
        return parent, siblings[index] if index < len(siblings) else None

    def _valid_drop(self, objects, parent):
        return bool(objects) and all(obj in self.items_by_object for obj in objects) and (
            parent is None or (parent not in objects and not any(obj in parent.ancestors() for obj in objects)))

    def dragEnterEvent(self, event):
        if event.source() is self and event.mimeData().hasFormat(self.MIME_TYPE):
            super().dragEnterEvent(event)
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.source() is not self or not event.mimeData().hasFormat(self.MIME_TYPE):
            event.ignore()
            return
        super().dragMoveEvent(event)
        parent, _ = self._drop_location(event.position().toPoint())
        if not self._valid_drop(self._drag_objects or self._selected_roots(), parent):
            event.ignore()

    def dropEvent(self, event):
        if event.source() is not self or not event.mimeData().hasFormat(self.MIME_TYPE):
            event.ignore()
            return
        objects = self._drag_objects or self._selected_roots()
        parent, before = self._drop_location(event.position().toPoint())
        if not self._valid_drop(objects, parent):
            event.ignore()
            return
        # The coordinator changes the scene and rebuilds this tree atomically.
        # Never call Qt's default row mover: it would only move UI items.
        self.move_requested.emit((objects, parent, before))
        event.setDropAction(Qt.DropAction.MoveAction)
        event.accept()

    @staticmethod
    def object_for(item):
        return item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None
