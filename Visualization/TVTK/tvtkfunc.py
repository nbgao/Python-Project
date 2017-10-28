
# coding: utf-8

# In[1]:

from tvtk.api import tvtk


# In[2]:

def ivtk_scene(actors):
    from tvtk.tools import ivtk
    # 创建一个带Crust的窗口
    win = ivtk.IVTKWithCrustAndBrowser()
    win.open()
    win.scene.add_actor(actors)
    # 修正窗口错误
    dialog = win.control.centralWidget().widget(0).widget(0)
    from pyface.qt import QtCore
    dialog.setWindowFlags(QtCore.Qt.WindowFlags(0x00000000))
    dialog.show()
    return win


# In[3]:

def event_loop():
    from pyface.api import GUI
    gui = GUI()
    gui.start_event_loop()



