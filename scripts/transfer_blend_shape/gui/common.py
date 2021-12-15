import six
import sys
import shiboken2
from PySide2 import QtGui, QtCore, QtWidgets
from functools import wraps

from transfer_blend_shape.utils import decorator


__all__ = [
    "WaitCursor",
    "get_application",
    "display_error",
]


class WaitCursor(object):
    """
    Display a wait cursor for the duration of the engine. This will indicate
    to the user a task is being calculated.
    """
    def __enter__(self):
        app = get_application()
        app.setOverrideCursor(QtCore.Qt.WaitCursor)

    def __exit__(self, exc_type, exc_val, exc_tb):
        app = get_application()
        app.restoreOverrideCursor()


@decorator.memoize
def get_application():
    """
    Get the application making sure it is returned as a QtWidgets.QApplication
    where the instance sometimes is returned as a QtWidgets.QCoreApplication,
    which misses vital methods. At first it is attempted to create a new
    application which is a bit backwards, but unfortunately nessecary due to
    to fact that it crashes if checked when the instance is None.

    :return: Application
    :rtype: QtWidgets.QApplication
    """
    try:
        app = QtWidgets.QApplication()
    except (TypeError, RuntimeError):
        app = QtWidgets.QApplication.instance()

    if not isinstance(app, QtWidgets.QApplication):
        app_pointer = shiboken2.getCppPointer(app)[0]
        return shiboken2.wrapInstance(app_pointer, QtWidgets.QApplication)
    else:
        return app


def display_error(func):
    """
    The display error function will catch the error of a function and then
    create a dialog window that displays the error. This way it is not
    necessary to keep an eye out for the script editor.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            ret = func(*args, **kwargs)
            return ret
        except Exception as e:
            # get parent
            parent = args[0] if args and isinstance(args[0], QtWidgets.QWidget) else None

            # create message box
            message_box = QtWidgets.QMessageBox(parent)
            message_box.setIcon(QtWidgets.QMessageBox.Critical)
            message_box.setText(str(e))
            message_box.setWindowTitle(e.__class__.__name__)
            message_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
            message_box.exec_()

            # re-raise error
            t, v, tb = sys.exc_info()
            try:
                # catch type error, noticed some custom error classes take
                # more than one argument in the init. If that happens we
                # resort to a RuntimeError instead.
                raise six.reraise(t, v, tb)
            except TypeError:
                raise six.reraise(RuntimeError, v, tb)

    return wrapper
