import os
import logging
from maya import cmds
from mayaunittest import MayaTestCase

from transfer_blend_shape import transfer


class TestTransfer(MayaTestCase):
    """
    The transfer test case load the saved example scenes and simply runs the
    blend shape conversion node. Linked with CI on github it will provide a
    test case for all supported Maya versions.
    """
    file_new = MayaTestCase.FILE_NEW_ALWAYS
    logging_level = logging.CRITICAL

    def setUp(self):
        super(TestTransfer, self).setUp()

        self.load_plugin("fbxmaya")
        file_path = os.path.join(os.path.dirname(__file__), "bin", "scene.fbx")
        cmds.file(file_path, i=True, force=True, prompt=False, ignoreVersion=True)

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
        cmds.blendShape("jawOpen_MESH", "source_MESH")
        cmds.delete("jawOpen_MESH")

        t = transfer.Transfer("source_MESH", "target_MESH")
        self.assertTrue(t.is_valid_with_blend_shape())

        meshes = t.execute_from_blend_shape()
        self.assertIn("jawOpen_MESH", meshes)
