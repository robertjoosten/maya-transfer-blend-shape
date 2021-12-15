# maya-transfer-blend-shape
Retarget your blend shapes between meshes with the same topology.

<p align="center"><img src="docs/_images/transfer-blend-shape-ui.png?raw=true"></p>

## Installation
* Extract the content of the .rar file anywhere on disk.
* Drag the transfer-blend-shape.mel file in Maya to permanently install the script.

## Usage
A button on the MiscTools shelf will be created that will allow easy access to 
the ui, this way the user doesn't need to worry about any of the code. If user 
wishes to not use the shelf button the following commands can be used. The 
transfer will only work if at least one vertex has no delta, these fixed 
vertices are used to transfer the solution to the correct position in object 
space, the threshold can be increased to make sure vertices are linked.

When the target is set/changed either by initializing the Transfer object or 
via the UI the LU factor is calculated, this can be quite time consuming.

Command line:
```python
import transfer_blend_shape
transfer = transfer_blend_shape.Transfer(source, target)
transfer.execute_from_mesh(mesh, name, threshold=0.001)
transfer.execute_from_blend_shape(threshold=0.001)
```

Display UI:
```python
import transfer_blend_shape.gui
transfer_blend_shape.gui.show()
```

## Note
This tool requires *numpy* and *scipy* to be installed to your environment. 
Using linux or Maya 2022+ on windows this can be done via a simple pip 
install. For older windows versions a custom version will have to be compiled 
against the correct VS version. 

The tool is an implementation of <a href="http://people.csail.mit.edu/sumner/research/deftransfer/" target="_blank">Deformation Transfer for Triangle Meshes</a>, 
the code was converted to python from c++ library <a href="https://github.com/JavongChang/ofxDeformationTransfer" target="_blank">ofxDeformationTransfer</a> 
from <a href="https://github.com/JavongChang" target="_blank">JavongChang</a> 
and made functional for Maya.