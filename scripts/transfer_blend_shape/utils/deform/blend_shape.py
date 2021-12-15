from maya import cmds
from maya.api import OpenMaya

from transfer_blend_shape.utils import api


def get_blend_shape(node):
    """
    :param str node:
    :return: Blend shape
    :rtype: str
    """
    nodes = cmds.listRelatives(node, shapes=True) or []
    nodes.append(node)

    for history in cmds.listHistory(nodes):
        if cmds.nodeType(history) == "blendShape":
            return history


def get_blend_shape_targets(blend_shape):
    """
    :param str blend_shape:
    :return: Blend shape targets
    :rtype: list[str]
    """
    return cmds.listAttr("{}.w".format(blend_shape), multi=True) or []
