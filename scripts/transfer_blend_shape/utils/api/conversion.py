import logging
from maya.api import OpenMaya


log = logging.getLogger(__name__)


def get_dag(node):
    """
    :param str node:
    :return: Maya dag path node
    :rtype: OpenMaya.MDagPath
    """
    sel = OpenMaya.MSelectionList()
    sel.add(node)
    return sel.getDagPath(0)
