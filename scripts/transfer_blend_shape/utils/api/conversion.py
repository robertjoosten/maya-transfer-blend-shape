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


def get_mesh_fn(node):
    """
    :param str node:
    :return: Mesh fn
    :rtype: OpenMaya.MFnMesh
    :raise RuntimeError: When provided node is not a mesh.
    """
    dag = get_dag(node)
    dag.extendToShape()

    if not dag.hasFn(OpenMaya.MFn.kMesh):
        raise RuntimeError("Node '{}' is not a mesh.".format(node))

    return OpenMaya.MFnMesh(dag)
