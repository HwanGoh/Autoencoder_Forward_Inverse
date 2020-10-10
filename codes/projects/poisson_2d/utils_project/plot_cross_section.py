#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 9 22:03:02 2020

@author: hwan
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.ioff() # Turn interactive plotting off
import matplotlib.tri as tri

from utils_fenics.convert_array_to_dolfin_function import\
        convert_array_to_dolfin_function

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def plot_cross_section(function_space,
                       parameter, mean, cov,
                       x_axis_limits, cross_section_y,
                       filepath):

    #=== Convert array to dolfin function ===#
    parameter_fe = convert_array_to_dolfin_function(function_space, parameter)
    mean_fe = convert_array_to_dolfin_function(function_space, mean)
    cov_fe = convert_array_to_dolfin_function(function_space, cov)

    #=== Extract mesh and triangulate ===#
    mesh = parameter_fe.function_space().mesh()
    coords = mesh.coordinates()
    elements = mesh.cells()
    triangulation = tri.Triangulation(coords[:, 0], coords[:, 1], elements)

    #=== Linear Interpolators ===#
    interp_parameter = tri.LinearTriInterpolator(triangulation, parameter)
    interp_mean = tri.LinearTriInterpolator(triangulation, mean)
    interp_cov = tri.LinearTriInterpolator(triangulation, cov)

    #=== Interpolate values of cross section ===#
    x_axis = np.linspace(x_axis_limits[0], x_axis_limits[1], 100, endpoint=True)
    for i in range(0,len(xaxis)):
        parameter_cross = interp_parameter(x_axis[i], cross_section_y)
        mean_cross = interp_mean(x_axis[i], cross_section_y)
        cov_cross = interp_cov(x_axis[i], cross_section_y)
    pdb.set_trace()
