"""					
I N S T A L L A T I O N:
    Copy the "rjRetargetBlendshape" folder to your Maya scripts directory:
        C:\Users\<USER>\Documents\maya\scripts

U S A G E:
    Display the UI with the following code:
        import rjRetargetBlendshape.ui
        rjRetargetBlendshape.ui.show()
        
    Command line:
        import rjRetargetBlendshape
        rjRetargetBlendshape.convert(
            source,
            blendshape,
            target,
            scale=True, 
            rotate=True, 
            smooth=0, 
            smoothIterations=0,
            space=OpenMaya.MSpace.kObject,
        )
        
N O T E:
    Retarget your blendshapes between meshes with the same topology.
    There are a few options that can be helpful to achieve the desired
    results. 

        - Scaling your delta depending on the size difference between
          the source and the target vertex. 
          
        - Rotating the delta depending on the normal difference between 
          the source and the target vertex. 
          
        - Smoothing based on the vertex size between the retarget mesh
          and the blendshape mesh.
"""

import math
from maya import OpenMaya, cmds

__author__    = "Robert Joosten"
__version__   = "0.8.2"
__email__     = "rwm.joosten@gmail.com"

# ----------------------------------------------------------------------------

def convert(
    source, 
    blendshape, 
    target, 
    scale=True, 
    rotate=True, 
    smooth=0, 
    smoothIterations=2,
    space=OpenMaya.MSpace.kObject
):
    """
    Create a new target based on the difference between the source and 
    blendshape. The target mesh is duplicated and the difference between 
    source and target is transfered onto the duplicated mesh. When 
    transfering both scale and rotation of the delta vectors can be taken 
    into account. Once the data is transfered an Laplacian smoothing algorithm
    can be applied onto the newly created target to create a more desired 
    result.
    
    :param str source:
    :param str blendshape:
    :param str target:
    :param bool scale: Take scale between delta vectors into account
    :param bool rotate: Take rotation between normal vectors into account
    :param float smooth: Smoothing strength
    :param int smoothIterations: Times the smooth algorithm repeats
    :param OpenMaya.MSpace space:
    :raises RuntimeError: If the vertex count doesn't match
    :return: Name of transfered blendshape mesh
    :rtype: str
    """
    # convert objects to dag
    dags = []
    meshes = []
    
    for i, name in enumerate([source, target, blendshape]):
        dags.append(asMDagPath(asMObject(name)))
        meshes.append(asMFnMesh(dags[i]))
            
    sourceD, targetD, blendshapeD = dags
    sourceM, targetM, blendshapeM = meshes

    # compare vertex count
    count = set([m.numVertices() for m in meshes])
   
    if len(count) != 1:
        raise RuntimeError(
            "Input geometry doesn't have matching vertex counts!"
        )
        
    # duplicate target to manipulate mesh
    targetB = getBasename(target)
    blendshapeB = getBasename(blendshape)

    target = cmds.duplicate(
        target, 
        rr=True, 
        n="{0}_{1}".format(targetB, blendshapeB)
    )[0]
    
    # parent duplicated target
    if cmds.listRelatives(target, p=True):
        target = cmds.parent(target, world=True)[0]
    
    targetD = asMDagPath(asMObject(target))
    targetM = asMFnMesh(targetD)
    
    # iterate vertices
    count = next(iter(count))
    positions = OpenMaya.MPointArray()
    
    # variables
    dags = [sourceD, targetD, blendshapeD]
    
    points = [OpenMaya.MPoint(), OpenMaya.MPoint(), OpenMaya.MPoint()]
    normals = [OpenMaya.MVector(), OpenMaya.MVector()]
    
    length = [0,0,0]
    lengths = [[],[],[]]

    # get new positions
    for i in range(count):  
        # initialize component
        component = asComponent(i)

        # loop meshes
        for j, dag in enumerate(dags):
            # get points
            meshes[j].getPoint(i, points[j], space)
            
            # get length
            l = getAverageLength(dag, component, space)
            
            length[j] = l
            lengths[j].append(l)
        
        # variables
        sourceP, targetP, blendshapeP = points
        sourceL, targetL, blendshapeL = length
        
        # difference vector
        vector = blendshapeP - sourceP
        
        # handle scale
        if scale: 
            scaleFactor = targetL/sourceL if sourceL and targetL else 1
            
            # scale vector
            vector = vector * scaleFactor 
        
        # handle normal rotation
        if rotate:
            # get normals
            for j, mesh in enumerate(meshes[:-1]):
                mesh.getVertexNormal(i, True, normals[j], space)
               
            # get quaternion between normals
            sourceN, targetN = normals
            quaternion = sourceN.rotateTo(targetN)
            
            # rotate vector
            vector = vector.rotateBy(quaternion)
            
        positions.append(targetP + vector)
    
    # set all points
    targetM.setPoints(positions, space)
    
    # loop over smooth iterations
    for _ in range(smoothIterations):
        
        # loop over vertices
        for i, sourceL, targetL, blendshapeL in zip(range(count), *lengths):
        
            # smooth vertex
            setLaplacianSmooth(
                targetD, 
                i, 
                space, 
                sourceL, 
                targetL, 
                blendshapeL, 
                smooth
            )
            
    return target
            
# ----------------------------------------------------------------------------
        
def setLaplacianSmooth(
    dag, 
    index, 
    space, 
    sourceL, 
    targetL, 
    blendshapeL, 
    smooth
):
    """
    Laplacian smoothing algorithm, where the average of the neighbouring points
    are used to smooth the target vertices, based on the factor between the 
    average length of both the source and the target mesh.
    
    :param OpenMaya.MDagPath dag: 
    :param int index: Component index
    :param OpenMaya.MSpace space:
    :param float sourceL: 
    :param float targetL: 
    :param float blendshapeL:
    :param float smooth:
    """
    # calculate factor
    component = asComponent(index)
    avgL = getAverageLength(dag, component, space)
    
    targetF = blendshapeL/sourceL if sourceL and targetL else 1
    blendshapeF = avgL/targetL if sourceL and blendshapeL else 1
                
    factor = abs((1-targetF/blendshapeF)*smooth)
    factor = max(min(factor, 1), 0)
    
    # ignore if there is not smooth factor
    if not factor:
        return

    # get average position
    component = asComponent(index)
    vtx = OpenMaya.MItMeshVertex(dag, component)
    
    origP, avgP = getAveragePosition(dag, component, space)
    
    # get new position
    avgP = avgP * factor
    origP = origP * (1-factor)
    newP = avgP + OpenMaya.MVector(origP)
    
    # set new position
    vtx.setPosition(newP, space)
    
def getAveragePosition(dag, component, space):
    """
    Get average position of connected vertices.
    
    :param OpenMaya.MDagPath dag: 
    :param OpenMaya.MFnSingleIndexedComponent component: 
    :param OpenMaya.MSpace space:
    :return: Average length of the connected edges
    :rtype: OpenMaya.MPoint
    """
    averagePos = OpenMaya.MPoint()

    # get connected vertices
    connected = OpenMaya.MIntArray()
    
    iterate = OpenMaya.MItMeshVertex(dag, component)
    iterate.getConnectedVertices(connected)
    
    # get original position
    originalPos = iterate.position(space)
      
    # ignore if no vertices are connected
    if not connected.length():
        return averagePos
    
    # get average 
    component = asComponent(connected)
    iterate = OpenMaya.MItMeshVertex(dag, component)
    while not iterate.isDone():
        averagePos += OpenMaya.MVector(iterate.position(space))
        iterate.next()
        
    averagePos = averagePos/connected.length()
    return originalPos, averagePos

def getAverageLength(dag, component, space):
    """
    Get average length of connected edges.
    
    :param OpenMaya.MDagPath dag: 
    :param OpenMaya.MFnSingleIndexedComponent component: 
    :param OpenMaya.MSpace space:
    :return: Average length of the connected edges
    :rtype: float
    """
    total = 0
    
    lengthUtil = OpenMaya.MScriptUtil()
    lengthPtr = lengthUtil.asDoublePtr()

    # get connected edges
    connected = OpenMaya.MIntArray()

    iterate = OpenMaya.MItMeshVertex(dag, component)
    iterate.getConnectedEdges(connected)
    
    # ignore if no edges are connected
    if not connected.length():
        return 0
    
    # get average
    component = asComponent(connected, OpenMaya.MFn.kMeshEdgeComponent)
    iterate = OpenMaya.MItMeshEdge(dag, component)
    while not iterate.isDone():
        iterate.getLength(lengthPtr, space)
        total += lengthUtil.getDouble(lengthPtr)
        
        iterate.next()
        
    return total/connected.length()
    
# ----------------------------------------------------------------------------

def getBasename(name):
    """
    Strip the parent and namespace data of the provided string.
    
    :param str name: 
    :return: Base name of parsed object
    :rtype: str
    """
    return name.split("|")[-1].split(":")[-1]

# ----------------------------------------------------------------------------

def asMIntArray(index):
    """
    index -> OpenMaya.MIntArray
    
    :param int/OpenMaya.MIntArray index: indices
    :return: Array of indices
    :rtype: OpenMaya.MIntArray
    """
    if type(index) != OpenMaya.MIntArray:
        array = OpenMaya.MIntArray()
        array.append(index)
        return array

    return index

def asComponent(index, t=OpenMaya.MFn.kMeshVertComponent):
    """
    index -> OpenMaya.MFn.kComponent
    Based on the input type it will create a component type for this tool
    the following components are being used.
    
    * OpenMaya.MFn.kMeshVertComponent
    * OpenMaya.MFn.kMeshEdgeComponent
    
    :param int/OpenMaya.MIntArray index: indices to create component for
    :param OpenMaya.MFn.kComponent t: can be all of OpenMaya component types.
    :return: Initialized components
    :rtype: OpenMaya.MFnSingleIndexedComponent
    """
    # convert input to an MIntArray if it not already is one
    array = asMIntArray(index)

    # initialize component
    component = OpenMaya.MFnSingleIndexedComponent().create(t)
    OpenMaya.MFnSingleIndexedComponent(component).addElements(array)
    return component
    
# ----------------------------------------------------------------------------
     
def asMObject(path):
    """
    str -> OpenMaya.MObject

    :param str path: Path to Maya object
    :rtype: OpenMaya.MObject
    """
    selectionList = OpenMaya.MSelectionList()
    selectionList.add(path)
    
    obj = OpenMaya.MObject()
    selectionList.getDependNode(0, obj)
    return obj
    
def asMDagPath(obj):
    """
    OpenMaya.MObject -> OpenMaya.MDagPath

    :param OpenMaya.MObject obj:
    :rtype: OpenMaya.MDagPath
    """
    return OpenMaya.MDagPath.getAPathTo(obj)
    
def asMFnMesh(dag):
    """
    OpenMaya.MDagPath -> OpenMaya.MfnMesh

    :param OpenMaya.MDagPath dag:
    :rtype: OpenMaya.MfnMesh
    """
    
    return OpenMaya.MFnMesh(dag)
