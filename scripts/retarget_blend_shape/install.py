import logging
from maya import cmds, mel

log = logging.getLogger(__name__)
ROOT_PACKAGE = __name__.rsplit(".", 1)[0]

SHELF_NAME = "MiscTools"
SHELF_TOOL = {
    "label": "retargetBlendshape",
    "command": "import {0}.ui; {0}.ui.show()".format(ROOT_PACKAGE),
    "annotation": "Retarget blendshapes between meshes with the same topology.",
    "image1": "RB_icon.png",
    "sourceType": "python"
}


def execute():
    """
    Add a new shelf in Maya with all the tools that are provided in the
    SHELF_TOOLS variable. If the tab exists it will be deleted and re-created
    from scratch.
    """
    shelf_main = mel.eval("$tmpVar=$gShelfTopLevel")
    shelves = cmds.tabLayout(shelf_main, query=True, childArray=True)

    if SHELF_NAME in shelves:
        cmds.deleteUI(SHELF_NAME)

    cmds.shelfLayout(SHELF_NAME, parent=shelf_main)
    if SHELF_TOOL.get("image1"):
        cmds.shelfButton(style="iconOnly", parent=SHELF_NAME, **SHELF_TOOL)
    else:
        cmds.shelfButton(style="textOnly", parent=SHELF_NAME, **SHELF_TOOL)

    log.info("retarget-blend-shape installed successfully.")

