import numpy
import scipy.linalg
from maya import cmds
from maya.api import OpenMaya

from retarget_blend_shape.utils import api
from retarget_blend_shape.utils import naming
from retarget_blend_shape.utils import conversion
from retarget_blend_shape.utils import blend_shape


class Transfer(object):
    """
    Deformation transfer applies the deformation exhibited by a source mesh
    onto a different target mesh.
    """
    def __init__(self, source, target):
        # variables
        self.triangle_indices = None
        self.source_points = None
        self.target_points = None

        # initialize
        self.source = naming.get_name(source)
        self.source_dag = api.conversion.get_dag(source)
        self.source_dag.extendToShape()
        if not self.source_dag.hasFn(OpenMaya.MFn.kMesh):
            raise RuntimeError("Source '{}' is not a mesh.".format(self.source))

        self.target = naming.get_name(target)
        self.target_dag = api.conversion.get_dag(target)
        self.target_dag.extendToShape()
        if not self.source_dag.hasFn(OpenMaya.MFn.kMesh):
            raise RuntimeError("Target '{}' is not a mesh.".format(self.target))

        self.source_fn = OpenMaya.MFnMesh(self.source_dag)
        self.target_fn = OpenMaya.MFnMesh(self.target_dag)
        if self.source_fn.numVertices != self.target_fn.numVertices:
            raise RuntimeError("Source and target vertex count don't match.")

        _, self.triangle_indices = self.source_fn.getTriangles()
        self.source_points = self.source_fn.getPoints(OpenMaya.MSpace.kObject)
        self.target_points = self.target_fn.getPoints(OpenMaya.MSpace.kObject)

        # calculate target matrix
        self.target_matrix = self.calculate_target_matrix()
        self.target_matrix_transpose = self.target_matrix.transpose()
        self.lu = scipy.linalg.lu_factor(numpy.dot(self.target_matrix_transpose, self.target_matrix))

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
        :raise RuntimeError: When no static vertices are found.
        """
        static_vertices = self.get_static_vertices(points, threshold=threshold)
        if not static_vertices:
            raise RuntimeError("No static vertices found for target '{}', "
                               "try increasing the threshold".format(name))

        source_gradient = self.calculate_source_gradient(points)
        uts = numpy.dot(self.target_matrix_transpose, source_gradient)

        target_points = scipy.linalg.lu_solve(self.lu, uts)
        displacement = self.get_displacement(target_points, static_vertices)

        target_points = [
            self.target_points[i] if i in static_vertices else OpenMaya.MPoint(point - displacement)
            for i, point in enumerate(target_points)
        ]

        target = cmds.duplicate(self.target_dag.fullPathName(), name=name)[0]
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
        :raise RuntimeError: When provided mesh is not a mesh.
        :raise RuntimeError: When mesh vertex count doesn't match source.
        :raise RuntimeError: When no static vertices are found.
        """
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
        :raise RuntimeError: When no blend shape is connected to the source.
        :raise RuntimeError: When no static vertices are found.
        """
        bs = blend_shape.get_blend_shape(self.source_dag.fullPathName())
        if bs is None:
            raise RuntimeError("Source '{}' doesn't contain a blend shape node connection.".format(self.source))

        cmds.setAttr("{}.envelope".format(bs), 1)
        for name in blend_shape.get_blend_shape_targets(bs):
            cmds.setAttr("{}.{}".format(bs, name), 0)

        targets = []
        for name in blend_shape.get_blend_shape_targets(bs):
            cmds.setAttr("{}.{}".format(bs, name), 1)
            points = self.source_fn.getPoints(OpenMaya.MSpace.kObject)
            cmds.setAttr("{}.{}".format(bs, name), 0)
            targets.append(self.execute(points, name, threshold))

        return targets
