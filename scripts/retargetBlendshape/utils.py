import math
from maya import OpenMaya, cmds


# ----------------------------------------------------------------------------


def getSelectedMeshes():
    """
    Get all selected meshes, the current selection will be looped and checked
    if any of the selected transforms contain a mesh node. If this is the case
    the transform will be added to the selection list.

    :return: Parents nodes of all selected meshes
    :rtype: list
    """
    # get selection
    selection = cmds.ls(sl=True, l=True)
    extendedSelection = []

    # extend selection
    for sel in selection:
        extendedSelection.extend(
            cmds.listRelatives(sel, s=True, ni=True, f=True)
        )

    # return parent of meshes
    return list(set([
        cmds.listRelatives(m, p=True, f=True)[0]
        for m in extendedSelection
        if cmds.nodeType(m) == "mesh"
    ]))


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
