import numpy
import scipy.linalg
from maya import cmds
from maya.api import OpenMaya

from retarget_blend_shape.utils import api
from retarget_blend_shape.utils import conversion
from retarget_blend_shape.utils import blend_shape


class Transfer(object):
    def __init__(self, source, target):
        # variables
        self.triangle_indices = None
        self.source_points = None
        self.target_points = None

        # initialize
        self.source_dag = api.conversion.get_dag(source)
        self.source_dag.extendToShape()
        if not self.source_dag.hasFn(OpenMaya.MFn.kMesh):
            raise RuntimeError("Source '{}' is not a mesh.".format(source))

        self.target_dag = api.conversion.get_dag(target)
        self.target_dag.extendToShape()
        if not self.source_dag.hasFn(OpenMaya.MFn.kMesh):
            raise RuntimeError("Target '{}' is not a mesh.".format(source))

        self.source_fn = OpenMaya.MFnMesh(self.source_dag)
        self.target_fn = OpenMaya.MFnMesh(self.target_dag)
        if self.source_fn.numVertices != self.target_fn.numVertices:
            raise RuntimeError("Source and targets vertex count doesn't match.")

        self.blend_shape = blend_shape.get_blend_shape(source)
        if self.blend_shape is None:
            raise RuntimeError("Source '{}' has no blend shape attached.".format(source))

        _, self.triangle_indices = self.source_fn.getTriangles()
        self.source_points = self.source_fn.getPoints(OpenMaya.MSpace.kObject)
        self.target_points = self.target_fn.getPoints(OpenMaya.MSpace.kObject)

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

    def get_anchored_vertices(self, deformed_points, threshold=0.001):
        """
        :param OpenMaya.MPointArray deformed_points:
        :param float threshold:
        :return: Anchored vertices
        :rtype: set[int]
        """
        vertices = set()
        for i, (point1, point2) in enumerate(zip(self.source_points, deformed_points)):
            distance = OpenMaya.MVector(point2 - point1).length()
            if distance < threshold:
                vertices.add(i)

        return vertices

    def get_centroid(self, deformed_points, anchor_indices):
        """
        :param OpenMaya.MPointArray deformed_points:
        :param set[int] anchor_indices:
        :return: Centroid
        :rtype: numpy.Array
        """
        centroid = OpenMaya.MVector()
        for i in anchor_indices:
            centroid += OpenMaya.MVector(deformed_points[i] - OpenMaya.MVector(self.target_points[i]))
        centroid /= len(anchor_indices)
        return numpy.array(centroid)

    def calculate_source_gradient(self, deformed_points):
        """
        :param OpenMaya.MPointArray deformed_points:
        :return: Source gradient
        :rtype: numpy.Array
        """
        matrix = numpy.zeros((len(self.triangle_indices), 3))
        for i, (i0, i1, i2) in enumerate(conversion.as_chunks(self.triangle_indices, 3)):
            va = self.calculate_edge_matrix(self.source_points[i0], self.source_points[i1], self.source_points[i2])
            vb = self.calculate_edge_matrix(deformed_points[i0], deformed_points[i1], deformed_points[i2])

            q, r = numpy.linalg.qr(va)
            inv_rqt = numpy.dot(numpy.linalg.inv(r), q.transpose())

            sa = numpy.dot(vb, inv_rqt)
            sat = sa.transpose()

            for row in range(sat.shape[0]):
                for column in range(sat.shape[1]):
                    matrix[i * 3 + row][column] = sat[row][column]

        return matrix

    # ------------------------------------------------------------------------

    def execute(self, threshold=0.001):
        # reset blend shape
        cmds.setAttr("{}.envelope".format(self.blend_shape), 1)
        for name in blend_shape.get_blend_shape_targets(self.blend_shape):
            cmds.setAttr("{}.{}".format(self.blend_shape, name), 0)

        # calculate target matrix
        target_matrix = self.calculate_target_matrix()
        target_matrix_transpose = target_matrix.transpose()
        lu = scipy.linalg.lu_factor(numpy.dot(target_matrix_transpose, target_matrix))

        # calculate retarget meshes
        for name in blend_shape.get_blend_shape_targets(self.blend_shape):
            cmds.setAttr("{}.{}".format(self.blend_shape, name), 1)
            source_deformed_points = self.source_fn.getPoints(OpenMaya.MSpace.kObject)
            cmds.setAttr("{}.{}".format(self.blend_shape, name), 0)

            anchored_vertices = self.get_anchored_vertices(source_deformed_points, threshold=threshold)
            if not anchored_vertices:
                raise RuntimeError("No anchored vertices found for blend shape '{}', "
                                   "try increasing the threshold".format(name))

            source_gradient = self.calculate_source_gradient(source_deformed_points)
            uts = numpy.dot(target_matrix_transpose, source_gradient)
            target_deformed_points = scipy.linalg.lu_solve(lu, uts)
            target_centroid = self.get_centroid(target_deformed_points, anchored_vertices)
            target_deformed_points = [
                self.target_points[i] if i in anchored_vertices else OpenMaya.MPoint(point - target_centroid)
                for i, point in enumerate(target_deformed_points)
            ]

            target_mesh = cmds.duplicate(self.target_dag.fullPathName(), name=name)[0]
            target_dag = api.conversion.get_dag(target_mesh)
            target_dag.extendToShape()
            target_fn = OpenMaya.MFnMesh(target_dag)
            target_fn.setPoints(target_deformed_points, OpenMaya.MSpace.kObject)
