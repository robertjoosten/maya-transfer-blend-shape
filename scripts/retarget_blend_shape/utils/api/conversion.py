import logging
from maya.api import OpenMaya


log = logging.getLogger(__name__)


def get_object(node):
    """
    :param str node:
    :return: Maya object node
    :rtype: OpenMaya.MObject
    """
    sel = OpenMaya.MSelectionList()
    sel.add(node)
    return sel.getDependNode(0)


def get_dependency(node):
    """
    :param str node:
    :return: Maya dependency node
    :rtype: OpenMaya.MFnDependencyNode
    """
    obj = get_object(node)
    return OpenMaya.MFnDependencyNode(obj)


def get_dag(node):
    """
    :param str node:
    :return: Maya dag path node
    :rtype: OpenMaya.MDagPath
    """
    sel = OpenMaya.MSelectionList()
    sel.add(node)
    return sel.getDagPath(0)


def get_component(node):
    """
    We extend the component function to force components on objects of a
    certain type. These can be extended if need be. This means all vertex
    components will be provided if a mesh shape is parsed as the node.

    :param str node:
    :return: Maya dag path node and components
    :rtype: tuple[OpenMaya.MDagPath, OpenMaya.MObject]
    """
    if not node.count("."):
        dag = get_dag(node)
        dag.extendToShape()

        if dag.hasFn(OpenMaya.MFn.kMesh):
            node += ".vtx[*]"
        elif dag.hasFn(OpenMaya.MFn.kNurbsSurface) or dag.hasFn(OpenMaya.MFn.kNurbsSurface):
            node += ".cv[*]"
        else:
            log.warning("No component conversion found for node '{}'.".format(node))

    sel = OpenMaya.MSelectionList()
    sel.add(node)
    return sel.getComponent(0)


def get_plug(node):
    """
    :param str node:
    :return: Maya plug node
    :rtype: OpenMaya.MPlug
    """
    sel = OpenMaya.MSelectionList()
    sel.add(node)
    return sel.getPlug(0)

