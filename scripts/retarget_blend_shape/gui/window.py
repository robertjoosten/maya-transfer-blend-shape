from maya import cmds
from PySide2 import QtWidgets, QtGui, QtCore

from retarget_blend_shape import transfer
from retarget_blend_shape.gui import dcc
from retarget_blend_shape.gui import icon
from retarget_blend_shape.gui import common
from retarget_blend_shape.gui import widgets
from retarget_blend_shape.utils import undo
from retarget_blend_shape.utils import naming


WINDOW_TITLE = "Retarget Blend Shape"
WINDOW_ICON = icon.get_icon_file_path("RB_icon.png")
__all__ = [
    "RetargetBlendShapeWidget",
    "show",
]


class RetargetBlendShapeWidget(QtWidgets.QWidget):
    def __init__(self, parent):
        super(RetargetBlendShapeWidget, self).__init__(parent)

        # variables
        self._transfer = transfer.Transfer()
        scale_factor = self.logicalDpiX() / 96.0
        label_size = QtCore.QSize(75 * scale_factor, 18 * scale_factor)
        button_size = QtCore.QSize(100 * scale_factor, 18 * scale_factor)

        # set window
        self.setWindowFlags(QtCore.Qt.Window)
        self.setWindowTitle(WINDOW_TITLE)
        self.setWindowIcon(QtGui.QIcon(WINDOW_ICON))
        self.resize(400 * scale_factor, 25 * scale_factor)

        # create layout
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # create source and target widgets
        source_text = QtWidgets.QLabel(self)
        source_text.setText("Source:")
        source_text.setFixedSize(label_size)
        layout.addWidget(source_text, 0, 0)

        self.source = QtWidgets.QLineEdit(self)
        self.source.setReadOnly(True)
        layout.addWidget(self.source, 0, 1)

        source_button = QtWidgets.QPushButton(self)
        source_button.setText("Set source")
        source_button.setFixedSize(button_size)
        source_button.released.connect(self.set_source_from_selection)
        layout.addWidget(source_button, 0, 2)

        target_text = QtWidgets.QLabel(self)
        target_text.setText("Target:")
        target_text.setFixedSize(label_size)
        layout.addWidget(target_text, 1, 0)

        self.target = QtWidgets.QLineEdit(self)
        self.target.setReadOnly(True)
        layout.addWidget(self.target, 1, 1)

        target_button = QtWidgets.QPushButton(self)
        target_button.setText("Set target")
        target_button.setFixedSize(button_size)
        target_button.released.connect(self.set_target_from_selection)
        layout.addWidget(target_button, 1, 2)

        # create threshold widgets
        threshold_text = QtWidgets.QLabel(self)
        threshold_text.setText("Threshold:")
        layout.addWidget(threshold_text, 2, 0)

        self.threshold = QtWidgets.QDoubleSpinBox(self)
        self.threshold.setDecimals(3)
        self.threshold.setSingleStep(0.001)
        self.threshold.setValue(0.001)
        layout.addWidget(self.threshold, 2, 1, 1, 2)

        div = widgets.DividerWidget(self)
        layout.addWidget(div, 3, 0, 1, 3)

        # create transfer widgets
        self.transfer_selection = QtWidgets.QPushButton(self)
        self.transfer_selection.setText("Transfer selection")
        self.transfer_selection.released.connect(self.transfer_from_selection)
        layout.addWidget(self.transfer_selection, 4, 0, 1, 3)

        self.transfer_blend_shape = QtWidgets.QPushButton(self)
        self.transfer_blend_shape.setText("Transfer from blend shape")
        self.transfer_blend_shape.released.connect(self.transfer_from_blend_shape)
        layout.addWidget(self.transfer_blend_shape, 5, 0, 1, 3)

        self.reset()

    # ------------------------------------------------------------------------

    @property
    def transfer(self):
        """
        :return: Transfer object
        :rtype: transfer.Transfer
        """
        return self._transfer

    @common.display_error
    def set_source_from_selection(self):
        """
        :raise RuntimeError: When nothing is selected.
        """
        selection = cmds.ls(selection=True)
        if not selection:
            raise RuntimeError("Unable to set source, nothing selected.")

        with common.WaitCursor():
            self.transfer.set_source(selection[0])
            self.source.setText(naming.get_name(selection[0]))
            self.reset()

    @common.display_error
    def set_target_from_selection(self):
        """
        :raise RuntimeError: When nothing is selected.
        """
        selection = cmds.ls(selection=True)
        if not selection:
            raise RuntimeError("Unable to set target, nothing selected.")

        with common.WaitCursor():
            self.transfer.set_target(selection[0])
            self.target.setText(naming.get_name(selection[0]))
            self.reset()

    # ------------------------------------------------------------------------

    @common.display_error
    def transfer_from_selection(self):
        with common.WaitCursor():
            with undo.UndoChunk():
                for node in cmds.ls(selection=True):
                    self.transfer.execute_from_mesh(node, threshold=self.threshold.value())

    @common.display_error
    def transfer_from_blend_shape(self):
        with common.WaitCursor():
            with undo.UndoChunk():
                self.transfer.execute_from_blend_shape(threshold=self.threshold.value())

    def reset(self):
        """
        Enable the conversion buttons depending on the valid state of the
        transfer object and if the source has a blend shape attached.
        """
        is_valid = self.transfer.is_valid()
        self.transfer_selection.setEnabled(is_valid)
        self.transfer_blend_shape.setEnabled(bool(is_valid and self.transfer.has_blend_shape()))


def show():
    parent = dcc.get_main_window()
    widget = RetargetBlendShapeWidget(parent)
    widget.show()
