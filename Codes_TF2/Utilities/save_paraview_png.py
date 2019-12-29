#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 16:07:17 2019

@author: hwan
"""
import sys
sys.path.append('/usr/lib/paraview/site-packages')
from paraview.simple import *

###############################################################################
#                                   Save png                                  #
###############################################################################
def save_paraview_png():    
    #read a vtu
    reader = XMLPolyDataReader(FileName="rev_maware_thermalfinvary_full_3D_hl5_tl3_hn500_relu_p10_d50000_b1000_e1000_parameter_pred000000.vtp")
    
    #position camera
    view = GetActiveView()
    if not view:
        # When using the ParaView UI, the View will be present, not otherwise.
        view = CreateRenderView()
    view.CameraViewUp = [0, 0, 1]
    view.CameraFocalPoint = [0, 0, 0]
    view.CameraViewAngle = 45
    view.CameraPosition = [5,0,0]
    
    #draw the object
    Show()
    
    #set the background color
    view.Background = [1,1,1]  #white
    
    #set image size
    view.ViewSize = [200, 300] #[width, height]
    
    dp = GetDisplayProperties()
    
    #set point color
    dp.AmbientColor = [1, 0, 0] #red
    
    #set surface color
    dp.DiffuseColor = [0, 1, 0] #blue
    
    #set point size
    dp.PointSize = 2
    
    #set representation
    dp.Representation = "Surface"
    
    Render()
    
    #save screenshot
    WriteImage("test.png")