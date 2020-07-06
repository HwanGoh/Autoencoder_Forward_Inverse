# Obtained from:
# https://stackoverflow.com/questions/52202014/how-can-i-plot-2d-fem-results-using-matplotlib
import os

import matplotlib.pyplot as plt
plt.ioff() # Turn interactive plotting off
import matplotlib.tri as tri
import numpy as np

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def plot_FEM_function(file_path, plot_title,
        nodes, elements,
        nodal_values):

    nodes_x = nodes[:,0]
    nodes_y = nodes[:,1]

    #=== Plot Mesh ===#
    # for element in elements:
    #     x = [nodes_x[element[i]] for i in range(len(element))]
    #     y = [nodes_y[element[i]] for i in range(len(element))]
    #     plt.fill(x, y, edgecolor='black', fill=False)

    #=== Triangulate Mesh ===#
    triangulation = tri.Triangulation(nodes_x, nodes_y, elements)

    #=== Plot FEM Function ====#
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    plt.tricontourf(triangulation, nodal_values.flatten())
    plt.colorbar()
    plt.axis('equal')
    plt.savefig(file_path)
    plt.close()
