import logging
from maya import cmds, mel

log = logging.getLogger(__name__)
ROOT_PACKAGE = __name__.rsplit(".", 1)[0]

SHELF_NAME = "MiscTools"
SHELF_TOOL = {
    "label": "transfer-blend-shape",
    "command": "import {0}.gui; {0}.gui.show()".format(ROOT_PACKAGE),
    "annotation": "Transfer blend shapes between meshes with the same topology.",
    "image1": "TBS_icon.png",
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

    if SHELF_NAME not in shelves:
        cmds.shelfLayout(SHELF_NAME, parent=shelf_main)

    names = cmds.shelfLayout(SHELF_NAME, query=True, childArray=True) or []
    labels = [cmds.shelfButton(n, query=True, label=True) for n in names]

    if SHELF_TOOL.get("label") in labels:
        index = labels.index(SHELF_TOOL.get("label"))
        cmds.deleteUI(names[index])

    if SHELF_TOOL.get("image1"):
        cmds.shelfButton(style="iconOnly", parent=SHELF_NAME, **SHELF_TOOL)
    else:
        cmds.shelfButton(style="textOnly", parent=SHELF_NAME, **SHELF_TOOL)

    log.info("transfer-blend-shape installed successfully.")

