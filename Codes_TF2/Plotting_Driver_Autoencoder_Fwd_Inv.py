#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 12:18:38 2019

@author: hwan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:59:59 2019

@author: hwan
"""
import sys
import os

from Utilities.plot_and_save import plot_and_save

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                       HyperParameters and RunOptions                        #
###############################################################################
class HyperParameters:
    data_type         = 'full'
    num_hidden_layers = 5
    truncation_layer  = 3 # Indexing includes input and output layer with input layer indexed by 0
    num_hidden_nodes  = 500
    penalty           = 50
    batch_size        = 1000
    num_epochs        = 500
    gpu               = '0'
    
class RunOptions:
    def __init__(self, hyper_p): 
        #=== Data Set ===#
        data_thermal_fin_nine = 0
        data_thermal_fin_vary = 1
        
        #=== Data Set Size ===#
        self.num_training_data = 50000
        self.num_testing_data = 200
        
        #=== Data Dimensions ===#
        self.fin_dimensions_2D = 0
        self.fin_dimensions_3D = 1
        
        #=== Random Seed ===#
        self.random_seed = 1234

###############################################################################
#                                 File Name                                   #
###############################################################################                
        #=== Parameter and Observation Dimensions === #
        if self.fin_dimensions_2D == 1:
            self.full_domain_dimensions = 1446 
        if self.fin_dimensions_3D == 1:
            self.full_domain_dimensions = 4090 
        if data_thermal_fin_nine == 1:
            self.parameter_dimensions = 9
        if data_thermal_fin_vary == 1:
            self.parameter_dimensions = self.full_domain_dimensions
        
        #=== File name ===#
        if data_thermal_fin_nine == 1:
            self.dataset = 'thermalfin9'
        if data_thermal_fin_vary == 1:
            self.dataset = 'thermalfinvary'
        if self.fin_dimensions_2D == 1:
            fin_dimension = ''
        if self.fin_dimensions_3D == 1:
            fin_dimension = '_3D'
        if hyper_p.penalty >= 1:
            hyper_p.penalty = int(hyper_p.penalty)
            penalty_string = str(hyper_p.penalty)
        else:
            penalty_string = str(hyper_p.penalty)
            penalty_string = 'pt' + penalty_string[2:]

        self.filename = self.dataset + '_' + hyper_p.data_type + fin_dimension + '_hl%d_tl%d_hn%d_p%s_d%d_b%d_e%d' %(hyper_p.num_hidden_layers, hyper_p.truncation_layer, hyper_p.num_hidden_nodes, penalty_string, self.num_training_data, hyper_p.batch_size, hyper_p.num_epochs)

###############################################################################
#                                 File Paths                                  #
###############################################################################         
        #=== Save File Name ===#
        self.NN_savefile_directory = '../Trained_NNs/' + self.filename
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename
  
        #=== Save File Path for One Instance of Test Data ===#
        self.savefile_name_parameter_test = self.NN_savefile_directory + '/parameter_test' + fin_dimension
        if hyper_p.data_type == 'full':
            self.savefile_name_state_test = self.NN_savefile_directory + '/state_test' + fin_dimension
        if hyper_p.data_type == 'bndonly':
            self.savefile_name_state_test = self.NN_savefile_directory + '/state_test_bnd' + fin_dimension           
            
        #=== Loading Predictions ===#    
        self.savefile_name_parameter_pred = self.NN_savefile_name + '_parameter_pred' + fin_dimension
        self.savefile_name_state_pred = self.NN_savefile_name + '_state_pred' + fin_dimension
            
        #=== Savefile Path for Figures ===#    
        self.figures_savefile_directory = '../Figures/' + self.filename
        self.figures_savefile_name = self.figures_savefile_directory + '/' + self.filename
        self.figures_savefile_name_parameter_test = self.figures_savefile_directory + '/' + 'parameter_test' + fin_dimension
        self.figures_savefile_name_state_test = self.figures_savefile_directory + '/' + 'state_test' + fin_dimension
        self.figures_savefile_name_parameter_pred = self.figures_savefile_name + '_parameter_pred' + fin_dimension
        self.figures_savefile_name_state_pred = self.figures_savefile_name + '_state_pred' + fin_dimension
        
        #=== Creating Directories ===#
        if not os.path.exists(self.figures_savefile_directory):
            os.makedirs(self.figures_savefile_directory)

###############################################################################
#                                  Driver                                     #
###############################################################################
if __name__ == "__main__":
    
    #=== Set hyperparameters ===#
    hyper_p = HyperParameters()
    
    if len(sys.argv) > 1:
        hyper_p.data_type         = str(sys.argv[1])
        hyper_p.num_hidden_layers = int(sys.argv[2])
        hyper_p.truncation_layer  = int(sys.argv[3])
        hyper_p.num_hidden_nodes  = int(sys.argv[4])
        hyper_p.penalty           = float(sys.argv[5])
        hyper_p.batch_size        = int(sys.argv[6])
        hyper_p.num_epochs        = int(sys.argv[7])
        hyper_p.gpu               = str(sys.argv[8])
        
    #=== Set run options ===#        
    run_options = RunOptions(hyper_p)
    
    #=== Predict and Save ===#
    plot_and_save(hyper_p, run_options)
    



    
    
    
    
    
    