from functools import partial
from maya import cmds, OpenMaya, OpenMayaUI

# import pyside, do qt version check for maya 2017 >
qtVersion = cmds.about(qtVersion=True)
if qtVersion.startswith("4"):
    from PySide.QtGui import *
    from PySide.QtCore import *
    import shiboken
else:
    from PySide2.QtGui import *
    from PySide2.QtCore import *
    from PySide2.QtWidgets import *
    import shiboken2 as shiboken
    
# import command line convert
from . import convert

# ----------------------------------------------------------------------------

SPACE = []
SPACE.append(OpenMaya.MSpace.kObject)
SPACE.append(OpenMaya.MSpace.kWorld)
    
# ----------------------------------------------------------------------------
  
FONT = QFont()
FONT.setFamily("Consolas")

BOLT_FONT = QFont()
BOLT_FONT.setFamily("Consolas")
BOLT_FONT.setWeight(100)    

# ---------------------------------------------------------------------------- 
        
def mayaWindow():
    """
    Get Maya's main window.
    
    :rtype: QMainWindow
    """
    window = OpenMayaUI.MQtUtil.mainWindow()
    window = shiboken.wrapInstance(long(window), QMainWindow)
    
    return window  
    
# ----------------------------------------------------------------------------
    
def title(parent, name):
    title = QLabel(parent)
    title.setText(name)
    title.setFont(BOLT_FONT)
    return title
    
def divider(parent):
    line = QFrame(parent)
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line

# ----------------------------------------------------------------------------

def getSelectedMeshes():
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

class SelectionWidget(QWidget):
    signal = Signal()
    def __init__(self, parent, label, selectionMode="single"):
        QWidget.__init__(self, parent)
        
        # selection
        self._selection = []

        # create layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(3, 0, 3, 0)
        layout.setSpacing(0)
        
        # create label
        self.label = QLabel(self)
        self.label.setText("( 0 ) Mesh(es)")
        self.label.setFont(FONT)
        layout.addWidget(self.label)
        
        # create button
        button = QPushButton(self)
        button.setText(label)
        button.setFont(FONT)
        button.released.connect(
            partial(
                self.setSelection, 
                label, 
                selectionMode
            )
        )
        
        layout.addWidget(button)
        
    # ------------------------------------------------------------------------
    
    @property
    def selection(self):
        return self._selection
        
    # ------------------------------------------------------------------------
        
    def setSelection(self, label, selectionMode="single"):
        # get selection
        meshes = getSelectedMeshes()

        # update ui
        self.label.setText("( {0} ) Mesh(es)".format(len(meshes)))
        self.label.setToolTip("\n".join(meshes))
        
        # process selection mode
        if selectionMode == "single" and meshes:
            meshes = meshes[0]
            
        self._selection = meshes
        self.signal.emit()
        
# ----------------------------------------------------------------------------

class CheckBoxWidget(QWidget):
    def __init__(self, parent, label, toolTip):
        QWidget.__init__(self, parent)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(3, 0, 3, 0)
        layout.setSpacing(0)
        
        # create checkbox
        self.checkBox = QCheckBox(self)
        self.checkBox.setText(label)
        self.checkBox.setFont(FONT)
        self.checkBox.setChecked(True)
        self.checkBox.setToolTip(toolTip)
        layout.addWidget(self.checkBox) 
        
    # ------------------------------------------------------------------------
        
    def isChecked(self):
        return self.checkBox.isChecked()
        
class SpinBoxWidget(QWidget):
    def __init__(
        self, 
        parent, 
        widget, 
        label, 
        toolTip, 
        value, 
        minimum, 
        maximum, 
        step
    ):
        QWidget.__init__(self, parent)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(3, 0, 3, 0)
        layout.setSpacing(0)
        
        # create label
        l = QLabel(self)
        l.setText(label)
        l.setFont(FONT)
        layout.addWidget(l)
        
        # create spinbox
        self.spinBox = widget(self)
        self.spinBox.setFont(FONT)
        self.spinBox.setToolTip(toolTip)
        self.spinBox.setValue(value)
        self.spinBox.setSingleStep(step)
        self.spinBox.setMinimum(minimum)
        self.spinBox.setMaximum(maximum)
        layout.addWidget(self.spinBox) 

    # ------------------------------------------------------------------------
        
    def value(self):
        return self.spinBox.value()
 
class RetargetBlendshapeWidget(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent)
           
        # set ui
        self.setParent(parent)        
        self.setWindowFlags(Qt.Window)   
        self.setWindowIcon(QIcon(":/blendShape.png"))

        self.setWindowTitle("Retarget Blendshapes")           
        self.setObjectName("RetargetUI")
        self.resize(300, 200)
                
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
                
        # create title
        t = title(self, "Retarget Blendshapes")
        layout.addWidget(t) 
        
        # create selection widget
        self.sourceW = SelectionWidget(self, "Set Source")
        self.blendshapeW = SelectionWidget(self, "Set Blendshape(s)", "multi")
        self.targetW = SelectionWidget(self, "Set Target")
        
        for widget in [self.sourceW, self.targetW, self.blendshapeW]:
            widget.signal.connect(self.validate)
            layout.addWidget(widget)

        # create divider
        d = divider(self)
        layout.addWidget(d) 
        
        # create options
        t = title(self, "Options")
        layout.addWidget(t) 
        
        # scale settings
        self.scaleW = CheckBoxWidget(
            self,
            "Scale Delta",
            (
                "If checked, the vertex delta will be scaled based on the \n"
                "difference of the averaged connected edge length between \n"
                "the source and the target."
            )
        )
        layout.addWidget(self.scaleW) 
        
        # rotate settings
        self.rotateW = CheckBoxWidget(
            self,
            "Rotate Delta",
            (
                "If checked, the vertex delta will be rotated based on the \n"
                "difference of the vertex normal between the source and \n"
                "the target."
            )
        )
        layout.addWidget(self.rotateW) 
        
        # create divider
        d = divider(self)
        layout.addWidget(d) 
        
        # create smoothing
        t = title(self, "Smoothing")
        layout.addWidget(t) 
        
        # smooth settings
        self.smoothW = SpinBoxWidget(
            self,
            QDoubleSpinBox,
            "Factor",
            (
                "The targets will be smoothed based on the difference \n"
                "between source and blendshape and original target and \n"
                "output"
            ),
            value=10,
            minimum=0,
            maximum=1000,
            step=0.5
        )
        layout.addWidget(self.smoothW) 
        
        # smooth iter settings
        self.smoothIterW = SpinBoxWidget(
            self,
            QSpinBox,
            "Iterations",
            (
                "The amount of time the smoothing algorithm is applied to \n"
                "the output."
            ),
            value=2,
            minimum=0,
            maximum=10,
            step=1
        )
        layout.addWidget(self.smoothIterW) 
        
        # create divider
        d = divider(self)
        layout.addWidget(d) 
        
        # create space
        t = title(self, "Space")
        layout.addWidget(t) 

        self.spaceW = QComboBox(self)
        self.spaceW.setFont(FONT)
        self.spaceW.addItems(["kObject", "kWorld"])
        self.spaceW.setToolTip(
            "Determine space in which all the calculations take place."
        )
           
        layout.addWidget(self.spaceW) 
        
        # create divider
        d = divider(self)
        layout.addWidget(d) 
        
        # create retarget button
        self.retargetB = QPushButton(self)
        self.retargetB.setText("Retarget")
        self.retargetB.setFont(FONT)
        self.retargetB.setEnabled(False)
        self.retargetB.released.connect(self.retarget)
        layout.addWidget(self.retargetB) 
                
        # create spacer
        spacer = QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer)
        
        # create progress bar
        self.progressBar = QProgressBar(self)   
        layout.addWidget(self.progressBar)

    # ------------------------------------------------------------------------
    
    def validate(self):
        # variables
        source = self.sourceW.selection
        target = self.targetW.selection
        blendshapes = self.blendshapeW.selection
        
        # set button invisible
        self.retargetB.setEnabled(False)
        
        if source and target and blendshapes:
            self.retargetB.setEnabled(True)
  
    # ------------------------------------------------------------------------
        
    def retarget(self):   
        # get selection
        source = self.sourceW.selection
        target = self.targetW.selection
        blendshapes = self.blendshapeW.selection
        
        # get settings
        scale = self.scaleW.isChecked()
        rotate = self.rotateW.isChecked()
        
        # get smoothing
        smooth = self.smoothW.value()
        smoothIterations = self.smoothIterW.value()
        
        # get space
        space = SPACE[self.spaceW.currentIndex()]
    
        # set progress bar
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(len(blendshapes))
        self.progressBar.setValue(0)  

        # convert
        converted = []
        for i, blendshape in enumerate(blendshapes):
            converted.append(
                convert(
                    source, 
                    blendshape, 
                    target, 
                    scale, 
                    rotate, 
                    smooth,
                    smoothIterations,                
                    space
                )
            )
            
            # update spacebar
            self.progressBar.setValue(i+1)  
           
        # select output
        cmds.select(converted)
            
# ----------------------------------------------------------------------------
        
def show():
    retargetBlendshape = RetargetBlendshapeWidget(mayaWindow())
    retargetBlendshape.show()