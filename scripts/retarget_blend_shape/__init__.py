"""		
Retarget your blend shapes between meshes with the same topology.

Installation
============
* Extract the content of the .rar file anywhere on disk.
* Drag the retarget-blend-shape.mel file in Maya to permanently install the script.

Usage
=====
A button on the MiscTools shelf will be created that will allow easy access to
the ui, this way the user doesn't need to worry about any of the code. If user
wishes to not use the shelf button the following commands can be used.

Command line
::
    import retarget_blend_shape
    transfer = retarget_blend_shape.Transfer(source, target)
    transfer.execute(threshold=0.001)
    
Display UI
::
    import retarget_blend_shape.ui
    retarget_blend_shape.ui.show()
"""
from retarget_blend_shape.transfer import Transfer

__author__ = "Robert Joosten"
__version__ = "1.0.1"
__email__ = "rwm.joosten@gmail.com"