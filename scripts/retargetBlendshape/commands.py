from maya import OpenMaya, cmds
from . import utils


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
        dags.append(utils.asMDagPath(utils.asMObject(name)))
        meshes.append(utils.asMFnMesh(dags[i]))
            
    sourceD, targetD, blendshapeD = dags
    sourceM, targetM, blendshapeM = meshes

    # compare vertex count
    count = set([m.numVertices() for m in meshes])
   
    if len(count) != 1:
        raise RuntimeError(
            "Input geometry doesn't have matching vertex counts!"
        )
        
    # duplicate target to manipulate mesh
    targetB = utils.getBasename(target)
    blendshapeB = utils.getBasename(blendshape)

    target = cmds.duplicate(
        target, 
        rr=True, 
        n="{0}_{1}".format(targetB, blendshapeB)
    )[0]
    
    # parent duplicated target
    if cmds.listRelatives(target, p=True):
        target = cmds.parent(target, world=True)[0]
    
    targetD = utils.asMDagPath(utils.asMObject(target))
    targetM = utils.asMFnMesh(targetD)
    
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
        component = utils.asComponent(i)

        # loop meshes
        for j, dag in enumerate(dags):
            # get points
            meshes[j].getPoint(i, points[j], space)
            
            # get length
            l = utils.getAverageLength(dag, component, space)
            
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
            utils.setLaplacianSmooth(
                targetD, 
                i, 
                space, 
                sourceL, 
                targetL, 
                blendshapeL, 
                smooth
            )
            
    return target
