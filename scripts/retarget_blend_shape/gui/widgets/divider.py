from PySide2 import QtCore, QtWidgets, QtGui


__all__ = [
    "DividerWidget",
]


class DividerWidget(QtWidgets.QFrame):
    def __init__(self, parent, horizontal=True):
        super(DividerWidget, self).__init__(parent)
        line = QtWidgets.QFrame.HLine if horizontal else QtWidgets.QFrame.VLine
        self.setFrameShape(line)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
