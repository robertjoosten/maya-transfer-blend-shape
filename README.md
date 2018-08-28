# maya-retarget-blendshape
Retarget your blendshapes between meshes with the same topology.

<p align="center"><img src="docs/_images/retargetBlendshapeExample.png?raw=true"></p>
<a href="https://vimeo.com/170360738" target="_blank"><p align="center">Click for video</p></a>

## Installation
* Extract the content of the .rar file anywhere on disk.
* Drag the retargetBlendshape.mel file in Maya to permanently install the script.

## Usage
A button on the MiscTools shelf will be created that will allow easy access to the ui, this way the user doesn't need to worry about any of the code.
If user wishes to not use the shelf button the following commands can be used.

Command line:
```python
import retargetBlendshape
retargetBlendshape.convert(
  source,
  blendshape,
  target,
  scale=True, 
  rotate=True, 
  smooth=0, 
  smoothIterations=0,
  space=OpenMaya.MSpace.kObject,
)
```

Display UI:
```python
import retargetBlendshape.ui
retargetBlendshape.ui.show()
```

## Note
Retarget your blendshapes between meshes with the same topology. There are a few options that can be helpful to achieve the desired results. 

* Scaling your delta depending on the size difference between the source and the target vertex. 
* Rotating the delta depending on the normal difference between the source and the target vertex. 
* Smoothing based on the vertex size between the retarget mesh and the blendshape mesh.
