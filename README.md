# maya-retarget-blendshape
Retarget your blend shapes between meshes with the same topology.

## Installation
* Extract the content of the .rar file anywhere on disk.
* Drag the retarget-blend-shape.mel file in Maya to permanently install the script.

## Usage
A button on the MiscTools shelf will be created that will allow easy access to the ui, this way the user doesn't need to worry about any of the code. If user wishes to not use the shelf button the following commands can be used. The transfer will only work if at least one vertex has no delta, these fixed vertices are used to transfer the solution to the correct position in object space, the threshold can be increased to make sure vertices are linked.

Command line:
```python
import retarget_blend_shape
transfer = retarget_blend_shape.Transfer(source, target)
transfer.execute_from_mesh(mesh, name, threshold=0.001)
```

Display UI:
```python
import retarget_blend_shape.gui
retarget_blend_shape.gui.show()
```

## Note
This tool requires *numpy* and *scipy* to be installed to your environment. Using linux or Maya 2022+ on windows this can be done via a simple pip install. For older windows versions a custom version will have to be compiled against the correct VS version. 