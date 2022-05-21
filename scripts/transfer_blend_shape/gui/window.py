from maya import cmds
from PySide2 import QtWidgets, QtGui, QtCore

from transfer_blend_shape import transfer
from transfer_blend_shape.gui import dcc
from transfer_blend_shape.gui import icon
from transfer_blend_shape.gui import common
from transfer_blend_shape.gui import widgets
from transfer_blend_shape.utils import undo
from transfer_blend_shape.utils import naming


__all__ = [
    "TransferBlendShapeWidget",
    "show",
]
WINDOW_TITLE = "Transfer Blend Shape"
WINDOW_ICON = icon.get_icon_file_path("TBS_icon.png")


class TransferBlendShapeWidget(QtWidgets.QWidget):
    def __init__(self, parent):
        super(TransferBlendShapeWidget, self).__init__(parent)

        # variables
        self._transfer = transfer.Transfer()
        scale_factor = self.logicalDpiX() / 96.0
        label_size = QtCore.QSize(85 * scale_factor, 18 * scale_factor)
        button_size = QtCore.QSize(120 * scale_factor, 18 * scale_factor)

        # set window
        self.setWindowFlags(QtCore.Qt.Window)
        self.setWindowTitle(WINDOW_TITLE)
        self.setWindowIcon(QtGui.QIcon(WINDOW_ICON))
        self.resize(450 * scale_factor, 25 * scale_factor)

        # create layout
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # create source, target and virtual widgets
        source_text = QtWidgets.QLabel(self)
        source_text.setText("Source mesh:")
        source_text.setFixedSize(label_size)
        layout.addWidget(source_text, 0, 0)

        self.source = QtWidgets.QLineEdit(self)
        self.source.setReadOnly(True)
        layout.addWidget(self.source, 0, 1)

        source_button = QtWidgets.QPushButton(self)
        source_button.setText("Set source mesh")
        source_button.setFixedSize(button_size)
        source_button.released.connect(self.set_source_from_selection)
        layout.addWidget(source_button, 0, 2)

        target_text = QtWidgets.QLabel(self)
        target_text.setText("Target mesh:")
        target_text.setFixedSize(label_size)
        layout.addWidget(target_text, 1, 0)

        self.target = QtWidgets.QLineEdit(self)
        self.target.setReadOnly(True)
        layout.addWidget(self.target, 1, 1)

        target_button = QtWidgets.QPushButton(self)
        target_button.setText("Set target mesh")
        target_button.setFixedSize(button_size)
        target_button.released.connect(self.set_target_from_selection)
        layout.addWidget(target_button, 1, 2)

        virtual_text = QtWidgets.QLabel(self)
        virtual_text.setText("Virtual mesh:")
        virtual_text.setFixedSize(label_size)
        layout.addWidget(virtual_text, 2, 0)

        self.virtual = QtWidgets.QLineEdit(self)
        self.virtual.setReadOnly(True)
        self.virtual.setPlaceholderText("optional...")
        layout.addWidget(self.virtual, 2, 1)

        virtual_button = QtWidgets.QPushButton(self)
        virtual_button.setText("Set virtual mesh")
        virtual_button.setFixedSize(button_size)
        virtual_button.released.connect(self.set_virtual_from_selection)
        layout.addWidget(virtual_button, 2, 2)

        div = widgets.DividerWidget(self)
        layout.addWidget(div, 3, 0, 1, 3)

        # create threshold widgets
        threshold_text = QtWidgets.QLabel(self)
        threshold_text.setText("Threshold:")
        layout.addWidget(threshold_text, 4, 0)

        self.threshold = QtWidgets.QDoubleSpinBox(self)
        self.threshold.setDecimals(3)
        self.threshold.setSingleStep(0.001)
        self.threshold.setValue(0.001)
        self.threshold.valueChanged.connect(self.set_threshold)
        self.threshold.setToolTip("The threshold determines the threshold where "
                                  "vertices are considered to be static.")
        layout.addWidget(self.threshold, 4, 1, 1, 2)

        # create iterations widgets
        iterations_text = QtWidgets.QLabel(self)
        iterations_text.setText("Iterations:")
        layout.addWidget(iterations_text, 5, 0)

        self.iterations = QtWidgets.QSpinBox(self)
        self.iterations.setValue(3)
        self.iterations.setMinimum(0)
        self.iterations.valueChanged.connect(self.set_iterations)
        self.iterations.setToolTip("The iterations determine the amount of smoothing "
                                   "operations applied to the deformed vertices.")
        layout.addWidget(self.iterations, 5, 1, 1, 2)

        # create colour set widgets
        colour_sets_text = QtWidgets.QLabel(self)
        colour_sets_text.setText("Colour sets:")
        layout.addWidget(colour_sets_text, 6, 0)

        self.create_colour_sets = QtWidgets.QCheckBox(self)
        self.create_colour_sets.stateChanged.connect(self.set_create_colour_sets)
        self.create_colour_sets.setToolTip("Colour sets will be created that will visualize "
                                           "the deformed vertices and the smoothing weights.")
        layout.addWidget(self.create_colour_sets, 6, 1, 1, 2)

        div = widgets.DividerWidget(self)
        layout.addWidget(div, 7, 0, 1, 3)

        # create transfer widgets
        self.transfer_selection = QtWidgets.QPushButton(self)
        self.transfer_selection.setText("Transfer selection")
        self.transfer_selection.released.connect(self.transfer_from_selection)
        layout.addWidget(self.transfer_selection, 8, 0, 1, 3)

        self.transfer_blend_shape = QtWidgets.QPushButton(self)
        self.transfer_blend_shape.setText("Transfer from blend shape")
        self.transfer_blend_shape.released.connect(self.transfer_from_blend_shape)
        layout.addWidget(self.transfer_blend_shape, 9, 0, 1, 3)

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
            raise RuntimeError("Unable to set source mesh, nothing selected.")

        self.transfer.set_source_mesh(selection[0])
        self.source.setText(naming.get_name(selection[0]))
        self.reset()

    @common.display_error
    def set_target_from_selection(self):
        """
        :raise RuntimeError: When nothing is selected.
        """
        selection = cmds.ls(selection=True)
        if not selection:
            raise RuntimeError("Unable to set target mesh, nothing selected.")

        self.transfer.set_target_mesh(selection[0])
        self.target.setText(naming.get_name(selection[0]))
        self.reset()

    @common.display_error
    def set_virtual_from_selection(self):
        """
        """
        selection = cmds.ls(selection=True)
        virtual_mesh = selection[0] if selection else None
        virtual_mesh_name = naming.get_name(selection[0]) if selection else None
        self.transfer.set_virtual_mesh(virtual_mesh)
        self.virtual.setText(virtual_mesh_name)
        self.reset()

    def set_iterations(self, iterations):
        """
        :param int iterations:
        """
        self.transfer.set_iterations(iterations)

    def set_threshold(self, threshold):
        """
        :param float threshold:
        """
        self.transfer.set_threshold(threshold)

    def set_create_colour_sets(self, state):
        """
        :param int state:
        """
        self.transfer.set_create_colour_sets(bool(state))

    # ------------------------------------------------------------------------

    @common.display_error
    def transfer_from_selection(self):
        with common.WaitCursor():
            with undo.UndoChunk():
                for node in cmds.ls(selection=True):
                    self.transfer.execute_from_mesh(node)

    @common.display_error
    def transfer_from_blend_shape(self):
        with common.WaitCursor():
            with undo.UndoChunk():
                self.transfer.execute_from_blend_shape()

    def reset(self):
        """
        Enable the conversion buttons depending on the valid state of the
        transfer object and if the source has a blend shape attached.
        """
        is_valid = self.transfer.is_valid()
        self.transfer_selection.setEnabled(is_valid)
        self.transfer_blend_shape.setEnabled(bool(is_valid and self.transfer.is_valid_with_blend_shape()))


def show():
    parent = dcc.get_main_window()
    widget = TransferBlendShapeWidget(parent)
    widget.show()
