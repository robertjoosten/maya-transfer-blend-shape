import os
from maya import cmds
from unittest import TestCase

from transfer_blend_shape import transfer


class TransferTestCase(TestCase):
    """
    The transfer test case load the saved example scenes and simply runs the
    blend shape conversion node. Linked with CI on github it will provide a
    test case for all supported Maya versions.
    """

    def setUp(self):
        file_path = os.path.join(os.path.dirname(__file__), "bin", "scene.ma")
        cmds.file(file_path, open=True, force=True, prompt=False)

    def tearDown(self):
        cmds.file(newFile=True, force=True)

    # ------------------------------------------------------------------------

    def test_execute_from_mesh(self):
        """
        Execute the transfer from a provided mesh and validate the name and
        colour sets on the created target mesh.
        """
        t = transfer.Transfer("source_MESH", "target_MESH")
        t.set_create_colour_sets(True)
        self.assertTrue(t.is_valid())

        mesh = t.execute_from_mesh("jawOpen_MESH", name="jawOpen")
        mesh_colour_sets = cmds.polyColorSet(mesh, query=True, allColorSets=True) or []

        self.assertEqual(mesh, "jawOpen")
        self.assertIn("deformed", mesh_colour_sets)
        self.assertIn("weights", mesh_colour_sets)

    def test_execute_from_blend_shape(self):
        """
        Execute the transfer from the attached blend shape on the source mesh
        and validate the name of the created target mesh.
        """
        t = transfer.Transfer("source_MESH", "target_MESH")
        self.assertTrue(t.is_valid_with_blend_shape())

        meshes = t.execute_from_blend_shape()
        self.assertIn("jawOpen", meshes)
