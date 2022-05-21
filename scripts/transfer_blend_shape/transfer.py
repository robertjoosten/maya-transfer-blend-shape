import time
import numpy
import logging
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from maya import cmds
from maya.api import OpenMaya

from transfer_blend_shape.utils import api
from transfer_blend_shape.utils import colour
from transfer_blend_shape.utils import conversion
from transfer_blend_shape.utils import decorator
from transfer_blend_shape.utils import naming
from transfer_blend_shape.utils.deform import blend_shape

log = logging.getLogger(__name__)


class Transfer(object):
    """
    Deformation transfer applies the deformation exhibited by a source mesh
    onto a different target mesh. The transfer can be aided by a virtual
    mesh that creates additional triangles.
    """

    def __init__(
            self,
            source_mesh=None,
            target_mesh=None,
            virtual_mesh=None,
            iterations=3,
            threshold=0.001,
            create_colour_sets=False
    ):
        self._source_mesh = None
        self._target_mesh = None
        self._virtual_mesh = None
        self._threshold = 0.001
        self._iterations = 3
        self._create_colour_sets = False

        self.set_source_mesh(source_mesh)
        self.set_virtual_mesh(virtual_mesh)
        self.set_target_mesh(target_mesh)
        self.set_iterations(iterations)
        self.set_threshold(threshold)
        self.set_create_colour_sets(create_colour_sets)

    # ------------------------------------------------------------------------

    @property
    def source_mesh(self):
        """
        :return: Source mesh
        :rtype: str
        """
        return self._source_mesh

    @decorator.memoize
    def get_source_points(self):
        """
        :return: Source points
        :rtype: numpy.Array
        :raise RuntimeError: When source is not defined.
        """
        if self.source_mesh is None:
            raise RuntimeError("Source mesh has not been defined, unable to query points.")

        mesh_fn = api.conversion.get_mesh_fn(self.source_mesh)
        return numpy.array(mesh_fn.getPoints(OpenMaya.MSpace.kObject))[:, :-1]

    @decorator.memoize
    def get_source_triangles(self):
        """
        :return: Source triangles
        :rtype: list[int]
        :raise RuntimeError: When source is not defined.
        """
        if self.source_mesh is None:
            raise RuntimeError("Source mesh has not been defined, unable to query triangle indices.")

        mesh_fn = api.conversion.get_mesh_fn(self.source_mesh)
        _, triangles = mesh_fn.getTriangles()
        return list(triangles)

    @decorator.memoize
    def get_source_area(self):
        """
        :return: Source triangle area
        :rtype: numpy.Array
        :raise RuntimeError: When source is not defined.
        """
        if self.source_mesh is None:
            raise RuntimeError("Source mesh has not been defined, unable to query area.")

        source_points = self.get_source_points()
        return self.calculate_area(source_points)

    def set_source_mesh(self, source_mesh):
        """
        :param str source_mesh:
        """
        self._source_mesh = source_mesh
        self.get_source_points.clear()
        self.get_source_triangles.clear()
        self.get_source_area.clear()
        self.get_virtual_triangles.clear()

    # ------------------------------------------------------------------------

    @property
    def target_mesh(self):
        """
        :return: Target mesh
        :rtype: str
        """
        return self._target_mesh

    @decorator.memoize
    def get_target_points(self):
        """
        :return: Target points
        :rtype: numpy.Array
        :raise RuntimeError: When target is not defined.
        """
        if self.target_mesh is None:
            raise RuntimeError("Target mesh has not been defined, unable to query points.")

        mesh_fn = api.conversion.get_mesh_fn(self.target_mesh)
        return numpy.array(mesh_fn.getPoints(OpenMaya.MSpace.kObject))[:, :-1]

    @decorator.memoize
    def get_target_connectivity(self):
        """
        :return: Target connectivity
        :rtype: list[list[int]]
        :raise RuntimeError: When target is not defined.
        """
        if self.target_mesh is None:
            raise RuntimeError("Target mesh has not been defined, unable to query connectivity.")

        connectivity = []
        mesh_dag = api.conversion.get_dag(self.target_mesh)
        mesh_iter = OpenMaya.MItMeshVertex(mesh_dag)

        while not mesh_iter.isDone():
            indices = list(mesh_iter.getConnectedVertices())
            connectivity.append(indices)
            mesh_iter.next()

        return connectivity

    @decorator.memoize
    def get_target_matrix(self):
        """
        :return: Target matrix
        :rtype: numpy.Array
        :raise RuntimeError: When target is not defined.
        """
        if self.target_mesh is None:
            raise RuntimeError("Target mesh has not been defined, unable to query matrix.")

        return self.calculate_target_matrix()

    def set_target_mesh(self, target_mesh):
        """
        :param str target_mesh:
        """
        self._target_mesh = target_mesh
        self.get_target_points.clear()
        self.get_target_connectivity.clear()
        self.get_target_matrix.clear()

    # ------------------------------------------------------------------------

    @property
    def virtual_mesh(self):
        """
        :return: Virtual
        :rtype: str
        """
        return self._virtual_mesh

    @decorator.memoize
    def get_virtual_triangles(self, threshold=0.001):
        """
        :param float threshold:
        :return: Virtual triangles
        :rtype: list[int]
        :raise RuntimeError: When minimum length surpasses threshold.
        """
        if self.virtual_mesh is None:
            return []

        idx = {}
        mesh_fn = api.conversion.get_mesh_fn(self.virtual_mesh)

        source_points = self.get_source_points()
        virtual_points = numpy.array(mesh_fn.getPoints(OpenMaya.MSpace.kObject))[:, :-1]
        _, virtual_triangles = mesh_fn.getTriangles()

        for i, point in enumerate(virtual_points):
            lengths = scipy.linalg.norm(source_points - point, axis=1)
            index = lengths.argmin()

            if lengths[index] > threshold:
                raise RuntimeError("Unable to map vertex {} if the virtual mesh "
                                   "to the source mesh.".format(index))

            idx[i] = index

        return [idx[vertex] for vertex in virtual_triangles]

    def set_virtual_mesh(self, virtual_mesh):
        """
        :param str virtual_mesh:
        """
        self._virtual_mesh = virtual_mesh
        self.get_target_matrix.clear()
        self.get_virtual_triangles.clear()

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
        is_source_valid = self.source_mesh and cmds.objExists(self.source_mesh)
        is_target_valid = self.target_mesh and cmds.objExists(self.target_mesh)
        is_virtual_valid = not self.virtual_mesh or cmds.objExists(self.virtual_mesh)
        return bool(is_source_valid and is_target_valid and is_virtual_valid)

    def is_valid_with_blend_shape(self):
        """
        :return: Valid state + blend shape
        :rtype: bool
        """
        if not self.is_valid():
            return False

        return bool(blend_shape.get_blend_shape(self.source_mesh))

    # ------------------------------------------------------------------------

    def filter_vertices(self, points):
        """
        :param numpy.Array points:
        :return: Static/Dynamic vertices
        :rtype: numpy.Array, numpy.Array
        """
        source_points = self.get_source_points()
        lengths = scipy.linalg.norm(source_points - points, axis=1)
        return numpy.nonzero(lengths <= self.threshold)[0], numpy.nonzero(lengths > self.threshold)[0]

    def calculate_area(self, points):
        """
        :param numpy.Array points:
        :return: Triangle areas
        :rtype: numpy.Array
        """
        vertex_area = numpy.zeros(shape=(len(points),))
        source_triangles = self.get_source_triangles()
        triangle_points = numpy.take(points, source_triangles, axis=0)
        triangle_points = triangle_points.reshape((len(triangle_points) // 3, 3, 3))

        length = triangle_points - triangle_points[:, [1, 2, 0], :]
        length = scipy.linalg.norm(length, axis=2)

        s = numpy.sum(length, axis=1) / 2.0
        areas = numpy.sqrt(s * (s - length[:, 0]) * (s - length[:, 1]) * (s - length[:, 2]))

        for indices, area in zip(conversion.as_chunks(source_triangles, 3), areas):
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
        triangles = self.get_source_triangles() + self.get_virtual_triangles()
        target_points = self.get_target_points()

        matrix = numpy.zeros((len(triangles), target_points.shape[0]))
        for i, (i0, i1, i2) in enumerate(conversion.as_chunks(triangles, 3)):
            e0 = target_points[i1] - target_points[i0]
            e1 = target_points[i2] - target_points[i0]
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
        triangles = self.get_source_triangles() + self.get_virtual_triangles()
        source_points = self.get_source_points()

        matrix = numpy.zeros((len(triangles), 3))
        for i, (i0, i1, i2) in enumerate(conversion.as_chunks(triangles, 3)):
            va = self.calculate_edge_matrix(source_points[i0], source_points[i1], source_points[i2])
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
        source_area = self.get_source_area()
        target_area = self.calculate_area(points)
        weights = numpy.dstack((source_area, target_area))
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
        num = self.get_target_points().shape[0]
        connectivity = self.get_target_connectivity()

        weights[ignore] = 0
        data, rows, columns = [], [], []

        for i, weight in enumerate(weights):
            weight = min([weights[i], 1])
            indices = connectivity[i]
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
        :raise RuntimeError: When vertex count doesn't match between source and target.
        :raise RuntimeError: When no static vertices are found.
        """
        t = time.time()

        if not self.is_valid():
            raise RuntimeError("Invalid transfer, set at least source and target.")

        source_points = self.get_source_points()
        target_points = self.get_target_points()
        if source_points.shape[0] != target_points.shape[0]:
            raise RuntimeError("Vertex count between source mesh '{}' and target mesh '{}' "
                               "do not match.".format(self.source_mesh, self.target_mesh))

        static_vertices, deformed_vertices = self.filter_vertices(points)
        if not len(static_vertices):
            raise RuntimeError("No static vertices found for target '{}', "
                               "try increasing the threshold".format(name))
        elif not len(deformed_vertices):
            target = cmds.duplicate(self.target_mesh, name=name)[0]
            log.info("Transferred '{}' as a static mesh.".format(name))
            return target

        target_matrix = self.get_target_matrix()

        # calculate deformation gradient, the static vertices are used to
        # anchor the static vertices in place.
        static_matrix = target_matrix[:, static_vertices]
        static_points = target_points[static_vertices, :]
        static_gradient = numpy.dot(static_matrix, static_points)
        deformation_gradient = self.calculate_deformation_gradient(points) - static_gradient

        # isolate dynamic vertices and solve their position. As it is quicker
        # to set all points rather than individual ones the entire target
        # point list is constructed.
        deformed_matrix = target_matrix[:, deformed_vertices]
        deformed_matrix_transpose = deformed_matrix.transpose()
        lu, piv = scipy.linalg.lu_factor(numpy.dot(deformed_matrix_transpose, deformed_matrix))
        uts = numpy.dot(deformed_matrix_transpose, deformation_gradient)
        deformed_points = scipy.linalg.lu_solve((lu, piv), uts)

        target_points = target_points.copy()
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
        target = cmds.duplicate(self.target_mesh, name=name)[0]
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
        :raise RuntimeError: When vertex count doesn't match between source and target.
        :raise RuntimeError: When provided mesh is not a mesh.
        :raise RuntimeError: When mesh vertex count doesn't match source.
        :raise RuntimeError: When no static vertices are found.
        """
        if not self.is_valid():
            raise RuntimeError("Invalid transfer, set source and target.")

        mesh_name = naming.get_leaf_name(mesh)
        mesh_fn = api.conversion.get_mesh_fn(mesh)
        name = name if name is not None else "{}_TGT".format(mesh_name)
        points = numpy.array(mesh_fn.getPoints(OpenMaya.MSpace.kObject))[:, :-1]
        return self.execute(points, name)

    def execute_from_blend_shape(self):
        """
        :return: Targets
        :rtype: list[str]
        :raise RuntimeError: When transfer is invalid.
        :raise RuntimeError: When vertex count doesn't match between source and target.
        :raise RuntimeError: When no blend shape is connected to the source.
        :raise RuntimeError: When no static vertices are found.
        """
        if not self.is_valid_with_blend_shape():
            raise RuntimeError("Invalid transfer, set at least source with blend shape and target.")

        bs = blend_shape.get_blend_shape(self.source_mesh)
        mesh_fn = api.conversion.get_mesh_fn(self.source_mesh)

        cmds.setAttr("{}.envelope".format(bs), 1)
        for name in blend_shape.get_blend_shape_targets(bs):
            cmds.setAttr("{}.{}".format(bs, name), 0)

        targets = []
        for name in blend_shape.get_blend_shape_targets(bs):
            cmds.setAttr("{}.{}".format(bs, name), 1)
            points = numpy.array(mesh_fn.getPoints(OpenMaya.MSpace.kObject))[:, :-1]
            cmds.setAttr("{}.{}".format(bs, name), 0)

            target = self.execute(points, name)
            targets.append(target)

        return targets
