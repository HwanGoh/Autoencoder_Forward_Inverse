#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 11:47:02 2019

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

def plot_fem_function_fenics_2d(function_space, nodal_values,
                                cross_section_y,
                                title, filepath,
                                fig_size, colorbar_limits):

    #=== Convert array to dolfin function ===#
    nodal_values_fe = convert_array_to_dolfin_function(function_space, nodal_values)

    #=== Extract mesh and triangulate ===#
    mesh = nodal_values_fe.function_space().mesh()
    coords = mesh.coordinates()
    elements = mesh.cells()
    triangulation = tri.Triangulation(coords[:, 0], coords[:, 1], elements)

    #=== Plot figure ===#
    nodal_values = nodal_values_fe.compute_vertex_values(mesh)
    v = np.linspace(colorbar_limits[0], colorbar_limits[1], 40, endpoint=True)
    plt.tricontourf(triangulation, nodal_values, v)
    plt.colorbar()
    plt.axis('equal')
    plt.title(title)
    plt.axhline(cross_section_y, color='r', linestyle='dashed', linewidth=3)

    #=== Save figure ===#
    plt.savefig(filepath, dpi=300, bbox_inches = 'tight', pad_inches = 0)
    plt.close()
