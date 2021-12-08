from __future__ import absolute_import
import shiboken2
from six import integer_types
from maya import OpenMayaUI
from PySide2 import QtWidgets, QtCore


__all__ = [
    "get_main_window",
    "maya_to_qt",
    "qt_to_maya",
]


def get_main_window():
    """
    :return: Maya main window
    :raise RuntimeError: When the main window cannot be obtained.
    """
    ptr = OpenMayaUI.MQtUtil.mainWindow()
    ptr = integer_types[-1](ptr)
    if ptr:
        return shiboken2.wrapInstance(ptr, QtWidgets.QMainWindow)

    raise RuntimeError("Failed to obtain a handle on the Maya main window.")


# ----------------------------------------------------------------------------


def maya_to_qt(name, type_=QtWidgets.QWidget):
    """
    :param str name: Maya path of an ui object
    :param cls type_:
    :return: QWidget of parsed Maya path
    :rtype: QWidget
    :raise RuntimeError: When no handle could be obtained
    """
    ptr = OpenMayaUI.MQtUtil.findControl(name)
    if ptr is None:
        ptr = OpenMayaUI.MQtUtil.findLayout(name)
    if ptr is None:
        ptr = OpenMayaUI.MQtUtil.findMenuItem(name)
    if ptr is not None:
        ptr = integer_types[-1](ptr)
        return shiboken2.wrapInstance(ptr, type_)

    raise RuntimeError("Failed to obtain a handle to '{}'.".format(name))


def qt_to_maya(widget):
    """
    :param QWidget widget: QWidget of a maya ui object
    :return: Maya path of parsed QWidget
    :rtype: str
    """
    ptr = shiboken2.getCppPointer(widget)[0]
    ptr = integer_types[-1](ptr)
    return OpenMayaUI.MQtUtil.fullName(ptr)