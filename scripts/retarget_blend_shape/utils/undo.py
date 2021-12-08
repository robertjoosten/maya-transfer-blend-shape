import maya.cmds as cmds


class UndoChunk(object):
    """
    When using QT to trigger commands, it is a known bug that the undo is
    split into individual cmds commands. Wrapping the command in this context
    will enforce that the entire action is undoable with one click.
    """
    def __enter__(self):
        cmds.undoInfo(openChunk=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        cmds.undoInfo(closeChunk=True)
