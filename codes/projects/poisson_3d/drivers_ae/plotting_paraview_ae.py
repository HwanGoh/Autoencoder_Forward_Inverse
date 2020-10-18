#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 21:41:12 2020
@author: hwan
"""
import sys
import os
sys.path.insert(0, os.path.realpath('../../../src'))
sys.path.append('../')
sys.path.append('../../../../')

import yaml
from attrdict import AttrDict
import scipy.sparse as sparse

# Import src code
from utils_io.config_io import command_line_json_string_to_dict
from utils_io.filepaths_ae import FilePathsPredictionAndPlotting

# Import project utilities
from utils_project.filepaths_project import FilePathsProject
from utils_project.plot_paraview import plot_paraview

# Import FEM Code
# from FEniCS_Codes.projects.poisson_3d.utils_project.plot_paraview import plot_paraview

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                 Add Options                                 #
###############################################################################
def add_options(options):

    options.model_aware = False
    options.model_augmented = True

    return options

###############################################################################
#                                  Driver                                     #
###############################################################################
if __name__ == "__main__":

    ##################
    #   Setting Up   #
    ##################
    #=== Hyperparameters ===#
    with open('../config_files/hyperparameters_ae.yaml') as f:
        hyperp = yaml.safe_load(f)
    if len(sys.argv) > 1: # if run from scheduler
        hyperp = command_line_json_string_to_dict(sys.argv[1], hyperp)
    hyperp = AttrDict(hyperp)

    #=== Options ===#
    with open('../config_files/options_ae.yaml') as f:
        options = yaml.safe_load(f)
    options = AttrDict(options)
    options = add_options(options)

    #=== File Names ===#
    project_paths = FilePathsProject(options)
    filepaths = FilePathsPredictionAndPlotting(hyperp, options, project_paths)

    #=== Colourbar Scale ===#
    cbar_RGB_parameter = [0.8596829444126151, 0.231373, 0.298039, 0.752941,
            3.8448816349259007, 0.865003, 0.865003, 0.865003, 6.830080325439186,
            0.705882, 0.0156863, 0.14902]
    cbar_RGB_state = [0.335898315086773, 0.231373, 0.298039, 0.752941,
            0.6068782815115794, 0.865003, 0.865003, 0.865003, 0.8778582479363859,
            0.705882, 0.0156863, 0.14902]

    #=== Plot and Save Paraview Figures ===#
    # Parameter test called twice because first image is different from
    # subsequent images for some unknown reason
    plot_paraview(filepaths.figure_vtk_parameter_test + '.pvd',
                  filepaths.figure_parameter_test + '.png',
                  cbar_RGB_parameter)
    plot_paraview(filepaths.figure_vtk_parameter_test + '.pvd',
                  filepaths.figure_parameter_test + '.png',
                  cbar_RGB_parameter)
    plot_paraview(filepaths.figure_vtk_parameter_pred + '.pvd',
                  filepaths.figure_parameter_pred + '.png',
                  cbar_RGB_parameter)
    if options.obs_type == 'full':
        plot_paraview(filepaths.figure_vtk_state_test + '.pvd',
                      filepaths.figure_state_test + '.png',
                      cbar_RGB_state)
        plot_paraview(filepaths.figure_vtk_state_pred + '.pvd',
                      filepaths.figure_state_pred + '.png',
                      cbar_RGB_state)
