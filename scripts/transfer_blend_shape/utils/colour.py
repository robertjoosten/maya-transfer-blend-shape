from maya import cmds
from maya.api import OpenMaya
from transfer_blend_shape.utils import api


def create_colour_set(mesh, name, colours):
    """
    :param str mesh:
    :param str name:
    :param list colours:
    """
    dag = api.conversion.get_dag(mesh)
    dag.extendToShape()
    mesh_fn = OpenMaya.MFnMesh(dag)
    vertices = range(mesh_fn.numVertices)

    cmds.polyColorSet(mesh, create=True, colorSet=name, clamped=True, representation="RGB")
    mesh_fn.setVertexColors(colours, vertices)
