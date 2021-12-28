import time
import numpy
import logging
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from maya import cmds
from maya.api import OpenMaya

from transfer_blend_shape.utils import api
from transfer_blend_shape.utils import naming
from transfer_blend_shape.utils import conversion
from transfer_blend_shape.utils.deform import blend_shape


log = logging.getLogger(__name__)


class Transfer(object):
    """
    Deformation transfer applies the deformation exhibited by a source mesh
    onto a different target mesh.
    """
    def __init__(self, source=None, target=None):
        # variables
        self._source = None
        self._source_fn = OpenMaya.MFnMesh()
        self._source_area = numpy.empty(shape=0)
        self._source_points = numpy.empty(shape=0)

        self._target = None
        self._target_fn = OpenMaya.MFnMesh()
        self._target_points = numpy.empty(shape=0)

        self.triangle_indices = OpenMaya.MIntArray()
        self.connectivity_indices = []
        self.target_matrix = numpy.empty(shape=0)

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
        :rtype: numpy.Array
        """
        return self._source_points

    @property
    def source_area(self):
        """
        :return: Source area
        :rtype: numpy.Array
        """
        return self._source_area

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

        if not bool(self.triangle_indices):
            _, self.triangle_indices = mesh_fn.getTriangles()

        self._source_points = numpy.array(mesh_fn.getPoints(OpenMaya.MSpace.kObject))[:, :-1]
        self._source_area = self.calculate_area(self._source_points)

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
        :rtype: numpy.Array
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
        mesh_iter = OpenMaya.MItMeshVertex(dag)
        if self.source and mesh_fn.numVertices != self.source_fn.numVertices:
            raise RuntimeError("Unable to set target '{}', doesn't match source vertex count.".format(name))

        self._target = target
        self._target_fn = mesh_fn

        if not bool(self.triangle_indices):
            _, self.triangle_indices = mesh_fn.getTriangles()

        self.connectivity_indices = []

        while not mesh_iter.isDone():
            indices = list(mesh_iter.getConnectedVertices())
            self.connectivity_indices.append(indices)
            mesh_iter.next()

        self._target_points = numpy.array(mesh_fn.getPoints(OpenMaya.MSpace.kObject))[:, :-1]
        self.target_matrix = self.calculate_target_matrix()

    # ------------------------------------------------------------------------

    def is_valid(self):
        """
        :return: Valid state
        :rtype: bool
        """
        return bool(self.source and self.target)

    def is_valid_with_blend_shape(self):
        """
        :return: Valid state + blend shape
        :rtype: bool
        """
        return bool(blend_shape.get_blend_shape(self.source)) if self.is_valid() else False

    # ------------------------------------------------------------------------

    @staticmethod
    def calculate_edge_matrix(point1, point2, point3):
        """
        :param numpy.Array point1:
        :param numpy.Array point2:
        :param numpy.Array point3:
        :return: Edge matrix
        :rtype: numpy.Array
        """
        e0 = point2 - point1
        e1 = point3 - point1
        e2 = numpy.cross(e0, e1)
        return numpy.array([e0, e1, e2]).transpose()

    def calculate_target_matrix(self):
        """
        :return: Target matrix
        :rtype: numpy.Array
        """
        matrix = numpy.zeros((len(self.triangle_indices), self.target_fn.numVertices))
        for i, (i0, i1, i2) in enumerate(conversion.as_chunks(self.triangle_indices, 3)):
            e0 = self.target_points[i1] - self.target_points[i0]
            e1 = self.target_points[i2] - self.target_points[i0]
            va = numpy.array([e0, e1]).transpose()

            q, r = numpy.linalg.qr(va)
            inv_rqt = numpy.dot(numpy.linalg.inv(r), q.transpose())

            for j in range(3):
                matrix[i * 3 + j][i0] = - inv_rqt[0][j] - inv_rqt[1][j]
                matrix[i * 3 + j][i1] = inv_rqt[0][j]
                matrix[i * 3 + j][i2] = inv_rqt[1][j]

        return matrix

    # ------------------------------------------------------------------------

    def filter_vertices(self, points, threshold=0.001):
        """
        :param numpy.Array points:
        :param float threshold:
        :return: Static/Dynamic vertices
        :rtype: numpy.Array, numpy.Array
        """
        lengths = scipy.linalg.norm(self.source_points - points, axis=1)
        return numpy.nonzero(lengths <= threshold)[0], numpy.nonzero(lengths > threshold)[0]

    def calculate_deformation_gradient(self, points):
        """
        :param numpy.Array points:
        :return: Deformation gradient
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
            matrix[i * 3: i * 3 + 3] = sat

        return matrix

    # ------------------------------------------------------------------------

    def calculate_area(self, points):
        """
        :param numpy.Array points:
        :return: Triangle areas
        :rtype: numpy.Array
        """
        vertex_area = numpy.zeros(shape=(len(points), ))
        triangle_points = numpy.take(points, self.triangle_indices, axis=0)
        triangle_points = triangle_points.reshape((len(triangle_points) / 3, 3, 3))

        length = triangle_points - triangle_points[:, [1, 2, 0], :]
        length = scipy.linalg.norm(length, axis=2)

        s = numpy.sum(length, axis=1) / 2.0
        areas = numpy.sqrt(s*(s-length[:, 0])*(s-length[:, 1])*(s-length[:, 2]))

        for indices, area in zip(conversion.as_chunks(self.triangle_indices, 3), areas):
            for index in indices:
                vertex_area[index] += area

        return vertex_area

    def calculate_laplacian_weights(self, points, ignore, iterations=3):
        """
        Calculate the laplacian weights depending on the change in per vertex
        area between the source and target points. The calculated weights are
        smoothed a number of times defined by the iterations, this will even
        out the smooth.

        :param numpy.Array points:
        :param numpy.Array ignore:
        :param int iterations:
        :return: Laplacian weights
        :rtype: numpy.Array
        """
        area = self.calculate_area(points)
        weights = numpy.dstack((self.source_area, area))
        weights = numpy.max(weights.transpose(), axis=0) / numpy.min(weights.transpose(), axis=0) - 1
        smoothing_matrix = self.calculate_laplacian_matrix(numpy.ones(len(points)), ignore)

        for _ in range(iterations):
            diff = numpy.array(smoothing_matrix.dot(weights))
            weights = weights - diff

        return weights.reshape(len(points))

    def calculate_laplacian_matrix(self, weights, ignore):
        """
        Create a laplacian smoothing matrix based on the weights, for the
        smoothing the number of vertices and vertex connectivity is used
        together with the provided weights, the weights are clamped to a
        maximum of 1.

        :param numpy.Array weights:
        :param numpy.Array ignore:
        :return: Laplacian smoothing matrix
        :rtype: scipy.sparse.csr.csr_matrix
        """
        # TODO: look into preserving mesh curvature
        num = self.target_fn.numVertices
        weights[ignore] = 0
        data, rows, columns = [], [], []

        for i, weight in enumerate(weights):
            weight = min([weights[i], 1])
            indices = self.connectivity_indices[i]
            z = len(indices)
            data = data + ([i] * (z + 1))
            rows = rows + indices + [i]
            columns = columns + ([-weight / float(z)] * z) + [weight]

        return scipy.sparse.coo_matrix((columns, (data, rows)), shape=(num, num)).tocsr()

    # ------------------------------------------------------------------------

    def execute(self, points, name, iterations=3, threshold=0.001):
        """
        :param numpy.Array points:
        :param str name:
        :param int iterations:
        :param float threshold:
        :return: Target
        :rtype: str
        :raise RuntimeError: When transfer is invalid.
        :raise RuntimeError: When no static vertices are found.
        """
        t = time.time()

        if not self.is_valid():
            raise RuntimeError("Invalid transfer, set source and target.")

        static_vertices, dynamic_vertices = self.filter_vertices(points, threshold=threshold)
        if not len(static_vertices):
            raise RuntimeError("No static vertices found for target '{}', "
                               "try increasing the threshold".format(name))
        elif not len(dynamic_vertices):
            target = cmds.duplicate(self.target, name=name)[0]
            log.info("Transferred '{}' as a static mesh.".format(name))
            return target

        static_matrix = self.target_matrix[:, static_vertices]
        static_points = self.target_points[static_vertices, :]
        static_gradient = numpy.dot(static_matrix, static_points)
        deformation_gradient = self.calculate_deformation_gradient(points) - static_gradient

        dynamic_matrix = self.target_matrix[:, dynamic_vertices]
        dynamic_matrix_transpose = dynamic_matrix.transpose()

        lu, piv = scipy.linalg.lu_factor(numpy.dot(dynamic_matrix_transpose, dynamic_matrix))
        uts = numpy.dot(dynamic_matrix_transpose, deformation_gradient)

        dynamic_points = scipy.linalg.lu_solve((lu, piv), uts)
        target_points = self.target_points.copy()
        target_points[dynamic_vertices, :] = dynamic_points

        smoothing_weights = self.calculate_laplacian_weights(points, static_vertices, iterations)
        smoothing_matrix = self.calculate_laplacian_matrix(smoothing_weights, static_vertices)
        for _ in range(iterations):
            diff = numpy.array(smoothing_matrix.dot(target_points))
            target_points = target_points - diff

        target_points = [OpenMaya.MPoint(point) for point in target_points]

        target = cmds.duplicate(self.target, name=name)[0]
        target_dag = api.conversion.get_dag(target)
        target_dag.extendToShape()
        target_fn = OpenMaya.MFnMesh(target_dag)
        target_fn.setPoints(target_points, OpenMaya.MSpace.kObject)
        log.info("Transferred '{}' in {:.3f} seconds.".format(name, time.time() - t))

        return target

    def execute_from_mesh(self, mesh, name=None, iterations=3, threshold=0.001):
        """
        :param str mesh:
        :param str/None name:
        :param int iterations:
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
        return self.execute(points, name, iterations, threshold)

    def execute_from_blend_shape(self, iterations=3, threshold=0.001):
        """
        :param int iterations:
        :param float threshold:
        :return: Targets
        :rtype: list[str]
        :raise RuntimeError: When transfer is invalid.
        :raise RuntimeError: When no blend shape is connected to the source.
        :raise RuntimeError: When no static vertices are found.
        """
        if not self.is_valid_with_blend_shape():
            raise RuntimeError("Invalid transfer, set source with blend shape and target.")

        bs = blend_shape.get_blend_shape(self.source)
        cmds.setAttr("{}.envelope".format(bs), 1)
        for name in blend_shape.get_blend_shape_targets(bs):
            cmds.setAttr("{}.{}".format(bs, name), 0)

        targets = []
        for name in blend_shape.get_blend_shape_targets(bs):
            cmds.setAttr("{}.{}".format(bs, name), 1)
            points = numpy.array(self.source_fn.getPoints(OpenMaya.MSpace.kObject))[:, :-1]
            cmds.setAttr("{}.{}".format(bs, name), 0)

            target = self.execute(points, name, iterations, threshold)
            targets.append(target)

        return targets
