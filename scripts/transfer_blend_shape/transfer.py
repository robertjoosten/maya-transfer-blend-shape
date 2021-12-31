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
from transfer_blend_shape.utils import colour
from transfer_blend_shape.utils import conversion
from transfer_blend_shape.utils.deform import blend_shape


log = logging.getLogger(__name__)


class Transfer(object):
    """
    Deformation transfer applies the deformation exhibited by a source mesh
    onto a different target mesh.
    """
    def __init__(
            self,
            source=None,
            target=None,
            iterations=3,
            threshold=0.001,
            create_colour_sets=False
    ):
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

        self._threshold = 0.001
        self._iterations = 3
        self._create_colour_sets = False

        self.set_iterations(iterations)
        self.set_threshold(threshold)
        self.set_create_colour_sets(create_colour_sets)

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

    @property
    def iterations(self):
        """
        :return: Iterations
        :rtype: int
        """
        return self._iterations

    def set_iterations(self, iterations):
        """
        :param int iterations:
        :raise TypeError: When iterations is not a int.
        :raise ValueError: When iterations is lower than 0.
        """
        if not isinstance(iterations, int):
            raise TypeError("Unable to set iterations, should be of type int.")
        elif iterations < 0:
            raise ValueError("Num iterations are not allowed to be lower than 0.")

        self._iterations = iterations

    @property
    def threshold(self):
        """
        :return: Threshold
        :rtype: float
        """
        return self._threshold

    def set_threshold(self, threshold):
        """
        :param float threshold:
        :raise TypeError: When threshold is not a float or int.
        :raise ValueError: When threshold is lower or equal to 0.
        """
        if not isinstance(threshold, (float, int)):
            raise TypeError("Unable to set threshold, should be of type int/float.")
        elif threshold <= 0.0:
            raise ValueError("Threshold is not allowed to be 0.0 or lower.")

        self._threshold = threshold

    @property
    def create_colour_sets(self):
        """
        :return: Create colour sets state
        :rtype: bool
        """
        return self._create_colour_sets

    def set_create_colour_sets(self, state):
        """
        :param bool state:
        """
        if not isinstance(state, bool):
            raise TypeError("Unable to set colour set creation state, should be of type bool.")

        self._create_colour_sets = state

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

    def filter_vertices(self, points):
        """
        :param numpy.Array points:
        :return: Static/Dynamic vertices
        :rtype: numpy.Array, numpy.Array
        """
        lengths = scipy.linalg.norm(self.source_points - points, axis=1)
        return numpy.nonzero(lengths <= self.threshold)[0], numpy.nonzero(lengths > self.threshold)[0]

    def calculate_area(self, points):
        """
        :param numpy.Array points:
        :return: Triangle areas
        :rtype: numpy.Array
        """
        vertex_area = numpy.zeros(shape=(len(points),))
        triangle_points = numpy.take(points, self.triangle_indices, axis=0)
        triangle_points = triangle_points.reshape((len(triangle_points) / 3, 3, 3))

        length = triangle_points - triangle_points[:, [1, 2, 0], :]
        length = scipy.linalg.norm(length, axis=2)

        s = numpy.sum(length, axis=1) / 2.0
        areas = numpy.sqrt(s * (s - length[:, 0]) * (s - length[:, 1]) * (s - length[:, 2]))

        for indices, area in zip(conversion.as_chunks(self.triangle_indices, 3), areas):
            for index in indices:
                vertex_area[index] += area

        return vertex_area

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

    def calculate_laplacian_weights(self, points, ignore):
        """
        Calculate the laplacian weights depending on the change in per vertex
        area between the source and target points. The calculated weights are
        smoothed a number of times defined by the iterations, this will even
        out the smooth.

        :param numpy.Array points:
        :param numpy.Array ignore:
        :return: Laplacian weights
        :rtype: numpy.Array
        """
        area = self.calculate_area(points)
        weights = numpy.dstack((self.source_area, area))
        weights = numpy.max(weights.transpose(), axis=0) / numpy.min(weights.transpose(), axis=0) - 1
        smoothing_matrix = self.calculate_laplacian_matrix(numpy.ones(len(points)), ignore)

        for _ in range(self.iterations):
            diff = numpy.array(smoothing_matrix.dot(weights))
            weights = weights - diff

        return weights.reshape(len(points))

    def calculate_laplacian_matrix(self, weights, ignore):
        """
        Create a laplacian smoothing matrix based on the weights, for the
        smoothing the number of vertices and vertex connectivity is used
        together with the provided weights, the weights are clamped to a
        maximum of 1. Any ignore indices will have their weights set to 0.

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
            data += ([i] * (z + 1))
            rows += indices + [i]
            columns += ([-weight / float(z)] * z) + [weight]

        return scipy.sparse.coo_matrix((columns, (data, rows)), shape=(num, num)).tocsr()

    # ------------------------------------------------------------------------

    def execute(self, points, name):
        """
        :param numpy.Array points:
        :param str name:
        :return: Target
        :rtype: str
        :raise RuntimeError: When transfer is invalid.
        :raise RuntimeError: When no static vertices are found.
        """
        t = time.time()

        if not self.is_valid():
            raise RuntimeError("Invalid transfer, set source and target.")

        static_vertices, deformed_vertices = self.filter_vertices(points)
        if not len(static_vertices):
            raise RuntimeError("No static vertices found for target '{}', "
                               "try increasing the threshold".format(name))
        elif not len(deformed_vertices):
            target = cmds.duplicate(self.target, name=name)[0]
            log.info("Transferred '{}' as a static mesh.".format(name))
            return target

        # calculate deformation gradient, the static vertices are used to
        # anchor the static vertices in place.
        static_matrix = self.target_matrix[:, static_vertices]
        static_points = self.target_points[static_vertices, :]
        static_gradient = numpy.dot(static_matrix, static_points)
        deformation_gradient = self.calculate_deformation_gradient(points) - static_gradient

        # isolate dynamic vertices and solve their position. As it is quicker
        # to set all points rather than individual ones the entire target
        # point list is constructed.
        deformed_matrix = self.target_matrix[:, deformed_vertices]
        deformed_matrix_transpose = deformed_matrix.transpose()
        lu, piv = scipy.linalg.lu_factor(numpy.dot(deformed_matrix_transpose, deformed_matrix))
        uts = numpy.dot(deformed_matrix_transpose, deformation_gradient)
        deformed_points = scipy.linalg.lu_solve((lu, piv), uts)

        target_points = self.target_points.copy()
        target_points[deformed_vertices, :] = deformed_points

        # calculate the laplacian smoothing weights/matrix using the
        # per-vertex area difference, this will ensure area's with most
        # highest difference receive the most smoothing, these are applied
        # to the calculated points
        smoothing_weights = self.calculate_laplacian_weights(points, static_vertices)
        smoothing_matrix = self.calculate_laplacian_matrix(smoothing_weights, static_vertices)
        for _ in range(self.iterations):
            diff = numpy.array(smoothing_matrix.dot(target_points))
            target_points = target_points - diff

        # duplicate the original target and update its points
        target = cmds.duplicate(self.target, name=name)[0]
        target_dag = api.conversion.get_dag(target)
        target_dag.extendToShape()
        target_fn = OpenMaya.MFnMesh(target_dag)
        target_fn.setPoints([OpenMaya.MPoint(point) for point in target_points], OpenMaya.MSpace.kObject)

        # create an deformed vertices and weight map colour set on the target
        # that can be used for debugging reasons.
        if self.create_colour_sets:
            vertices = set(deformed_vertices)
            vertices_colour = [[int(index in vertices)] * 3 for index in range(target_fn.numVertices)]
            weights_colour = [[weight] * 3 for weight in smoothing_weights]
            colour.create_colour_set(target, "deformed", vertices_colour)
            colour.create_colour_set(target, "weights", weights_colour)

        log.info("Transferred '{}' in {:.3f} seconds.".format(name, time.time() - t))

        return target

    def execute_from_mesh(self, mesh, name=None):
        """
        :param str mesh:
        :param str/None name:
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
        return self.execute(points, name)

    def execute_from_blend_shape(self):
        """
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

            target = self.execute(points, name)
            targets.append(target)

        return targets
