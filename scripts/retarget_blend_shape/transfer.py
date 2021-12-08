import numpy
import scipy.linalg
from maya import cmds
from maya.api import OpenMaya

from retarget_blend_shape.utils import api
from retarget_blend_shape.utils import naming
from retarget_blend_shape.utils import conversion
from retarget_blend_shape.utils.deform import blend_shape


class Transfer(object):
    """
    Deformation transfer applies the deformation exhibited by a source mesh
    onto a different target mesh.
    """
    def __init__(self, source=None, target=None):
        # variables
        self._source = None
        self._source_fn = OpenMaya.MFnMesh()
        self._source_points = OpenMaya.MPointArray()

        self._target = None
        self._target_fn = OpenMaya.MFnMesh()
        self._target_points = OpenMaya.MPointArray()

        self.triangle_indices = OpenMaya.MIntArray()
        self.target_matrix = numpy.empty(shape=0)
        self.target_matrix_transpose = numpy.empty(shape=0)
        self.lu = numpy.empty(shape=0)
        self.piv = numpy.empty(shape=0)

        if source is not None:
            self.set_source(source)
        if target is not None:
            self.set_target(target)

    # ------------------------------------------------------------------------

    @property
    def source(self):
        """
        :return: Source
        :rtype: str
        """
        return self._source

    @property
    def source_fn(self):
        """
        :return: Source fn
        :rtype: OpenMaya.MFnMesh
        """
        return self._source_fn

    @property
    def source_points(self):
        """
        :return: Source points
        :rtype: OpenMaya.MPointArray[OpenMaya.MPoint]
        """
        return self._source_points

    def set_source(self, source):
        """
        :param str source:
        :raise RuntimeError: When source is not a mesh.
        :raise RuntimeError: When source vertex count doesn't match target.
        """
        name = naming.get_name(source)
        dag = api.conversion.get_dag(source)
        dag.extendToShape()

        if not dag.hasFn(OpenMaya.MFn.kMesh):

            raise RuntimeError("Source '{}' is not a mesh.".format(name))

        mesh_fn = OpenMaya.MFnMesh(dag)
        if self.target and mesh_fn.numVertices != self.target_fn.numVertices:
            raise RuntimeError("Unable to set source '{}', doesn't match target vertex count.".format(name))

        self._source = source
        self._source_fn = mesh_fn
        self._source_points = mesh_fn.getPoints(OpenMaya.MSpace.kObject)
        _, self.triangle_indices = mesh_fn.getTriangles()

    # ------------------------------------------------------------------------

    @property
    def target(self):
        """
        :return: Target
        :rtype: str
        """
        return self._target

    @property
    def target_fn(self):
        """
        :return: Target fn
        :rtype: OpenMaya.MFnMesh
        """
        return self._target_fn

    @property
    def target_points(self):
        """
        :return: Target points
        :rtype: OpenMaya.MPointArray[OpenMaya.MPoint]
        """
        return self._target_points

    def set_target(self, target):
        """
        :param str target:
        :raise RuntimeError: When source is not a mesh.
        :raise RuntimeError: When source vertex count doesn't match target.
        """
        name = naming.get_name(target)
        dag = api.conversion.get_dag(target)
        dag.extendToShape()

        if not dag.hasFn(OpenMaya.MFn.kMesh):
            raise RuntimeError("Target '{}' is not a mesh.".format(name))

        mesh_fn = OpenMaya.MFnMesh(dag)
        if self.source and mesh_fn.numVertices != self.source_fn.numVertices:
            raise RuntimeError("Unable to set target '{}', doesn't match source vertex count.".format(name))

        self._target = target
        self._target_fn = mesh_fn
        self._target_points = mesh_fn.getPoints(OpenMaya.MSpace.kObject)
        self.target_matrix = self.calculate_target_matrix()
        self.target_matrix_transpose = self.target_matrix.transpose()
        self.lu, self.piv = scipy.linalg.lu_factor(numpy.dot(self.target_matrix_transpose, self.target_matrix))

    # ------------------------------------------------------------------------

    def is_valid(self):
        """
        :return: Valid state
        :rtype: bool
        """
        return bool(self.source and self.target)

    def has_blend_shape(self):
        """
        :return: Source blend shape state
        :rtype: bool
        """
        return bool(blend_shape.get_blend_shape(self.source)) if self.is_valid() else False

    # ------------------------------------------------------------------------

    @staticmethod
    def calculate_edge_matrix(point1, point2, point3):
        """
        :param OpenMaya.MPoint point1:
        :param OpenMaya.MPoint point2:
        :param OpenMaya.MPoint point3:
        :return: Edge matrix
        :rtype: numpy.Array
        """
        e0 = OpenMaya.MVector(point2 - point1)
        e1 = OpenMaya.MVector(point3 - point1)
        e2 = e0 ^ e1

        return numpy.array([e0, e1, e2]).transpose()

    def calculate_target_matrix(self):
        """
        :return: Target matrix
        :rtype: numpy.Array
        """
        matrix = numpy.zeros((len(self.triangle_indices), self.target_fn.numVertices))
        for i, (i0, i1, i2) in enumerate(conversion.as_chunks(self.triangle_indices, 3)):
            e0 = OpenMaya.MVector(self.target_points[i1] - self.target_points[i0])
            e1 = OpenMaya.MVector(self.target_points[i2] - self.target_points[i0])

            va = numpy.array([e0, e1]).transpose()

            q, r = numpy.linalg.qr(va)
            inv_rqt = numpy.dot(numpy.linalg.inv(r), q.transpose())

            for j in range(3):
                matrix[i * 3 + j][i0] = - inv_rqt[0][j] - inv_rqt[1][j]
                matrix[i * 3 + j][i1] = inv_rqt[0][j]
                matrix[i * 3 + j][i2] = inv_rqt[1][j]

        return matrix

    # ------------------------------------------------------------------------

    def get_static_vertices(self, points, threshold=0.001):
        """
        :param OpenMaya.MPointArray points:
        :param float threshold:
        :return: Static vertices
        :rtype: set[int]
        """
        vertices = set()
        for i, (point1, point2) in enumerate(zip(self.source_points, points)):
            distance = OpenMaya.MVector(point2 - point1).length()
            if distance < threshold:
                vertices.add(i)

        return vertices

    def get_displacement(self, points, static_vertices):
        """
        :param OpenMaya.MPointArray points:
        :param set[int] static_vertices:
        :return: Centroid
        :rtype: numpy.Array
        """
        centroid = OpenMaya.MVector()
        for i in static_vertices:
            centroid += OpenMaya.MVector(points[i] - OpenMaya.MVector(self.target_points[i]))
        centroid /= len(static_vertices)
        return numpy.array(centroid)

    def calculate_source_gradient(self, points):
        """
        :param OpenMaya.MPointArray points:
        :return: Source gradient
        :rtype: numpy.Array
        """
        matrix = numpy.zeros((len(self.triangle_indices), 3))
        for i, (i0, i1, i2) in enumerate(conversion.as_chunks(self.triangle_indices, 3)):
            va = self.calculate_edge_matrix(self.source_points[i0], self.source_points[i1], self.source_points[i2])
            vb = self.calculate_edge_matrix(points[i0], points[i1], points[i2])

            q, r = numpy.linalg.qr(va)
            inv_rqt = numpy.dot(numpy.linalg.inv(r), q.transpose())

            sa = numpy.dot(vb, inv_rqt)
            sat = sa.transpose()

            for row in range(sat.shape[0]):
                for column in range(sat.shape[1]):
                    matrix[i * 3 + row][column] = sat[row][column]

        return matrix

    # ------------------------------------------------------------------------

    def execute(self, points, name, threshold=0.001):
        """
        :param OpenMaya.MPointArray points:
        :param str name:
        :param float threshold:
        :return: Target
        :rtype: str
        :raise RuntimeError: When transfer is invalid.
        :raise RuntimeError: When no static vertices are found.
        """
        if not self.is_valid():
            raise RuntimeError("Invalid transfer, set source and target.")

        static_vertices = self.get_static_vertices(points, threshold=threshold)
        if not static_vertices:
            raise RuntimeError("No static vertices found for target '{}', "
                               "try increasing the threshold".format(name))

        source_gradient = self.calculate_source_gradient(points)
        uts = numpy.dot(self.target_matrix_transpose, source_gradient)

        target_points = scipy.linalg.lu_solve((self.lu, self.piv), uts)
        displacement = self.get_displacement(target_points, static_vertices)

        target_points = [
            self.target_points[i] if i in static_vertices else OpenMaya.MPoint(point - displacement)
            for i, point in enumerate(target_points)
        ]

        target = cmds.duplicate(self.target, name=name)[0]
        target_dag = api.conversion.get_dag(target)
        target_dag.extendToShape()
        target_fn = OpenMaya.MFnMesh(target_dag)
        target_fn.setPoints(target_points, OpenMaya.MSpace.kObject)

        return target

    def execute_from_mesh(self, mesh, name=None, threshold=0.001):
        """
        :param str mesh:
        :param str/None name:
        :param float threshold:
        :return: Target
        :rtype: str
        :raise RuntimeError: When transfer is invalid.
        :raise RuntimeError: When provided mesh is not a mesh.
        :raise RuntimeError: When mesh vertex count doesn't match source.
        :raise RuntimeError: When no static vertices are found.
        """
        if not self.is_valid():
            raise RuntimeError("Invalid transfer, set source and target.")

        mesh_name = naming.get_leaf_name(mesh)
        mesh_dag = api.conversion.get_dag(mesh)
        mesh_dag.extendToShape()
        if not mesh_dag.hasFn(OpenMaya.MFn.kMesh):
            raise RuntimeError("Mesh '{}' is not a mesh.".format(mesh_name))

        mesh_fn = OpenMaya.MFnMesh(mesh_dag)
        if self.source_fn.numVertices != mesh_fn.numVertices:
            raise RuntimeError("Mesh '{}' vertex count doesn't match with source.".format(mesh_name))

        name = name if name is not None else "{}_TGT".format(mesh_name)
        points = mesh_fn.getPoints(OpenMaya.MSpace.kObject)
        return self.execute(points, name, threshold)

    def execute_from_blend_shape(self, threshold):
        """
        :param float threshold:
        :return: Targets
        :rtype: list[str]
        :raise RuntimeError: When transfer is invalid.
        :raise RuntimeError: When no blend shape is connected to the source.
        :raise RuntimeError: When no static vertices are found.
        """
        if not self.is_valid():
            raise RuntimeError("Invalid transfer, set source and target.")

        if not self.has_blend_shape():
            source_name = naming.get_name(self.source)
            raise RuntimeError("Source '{}' doesn't contain a blend shape node connection.".format(source_name))

        bs = blend_shape.get_blend_shape(self.source)
        cmds.setAttr("{}.envelope".format(bs), 1)
        for name in blend_shape.get_blend_shape_targets(bs):
            cmds.setAttr("{}.{}".format(bs, name), 0)

        targets = []
        for name in blend_shape.get_blend_shape_targets(bs):
            cmds.setAttr("{}.{}".format(bs, name), 1)
            points = self.source_fn.getPoints(OpenMaya.MSpace.kObject)
            cmds.setAttr("{}.{}".format(bs, name), 0)

            target = self.execute(points, name, threshold)
            targets.append(target)

        return targets
