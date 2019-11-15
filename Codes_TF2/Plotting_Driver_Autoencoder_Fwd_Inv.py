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
sys.path.append('../')

from Utilities.plot_and_save import plot_and_save

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

import os

###############################################################################
#                       HyperParameters and RunOptions                        #
###############################################################################
class HyperParameters:
    data_type         = 'bndonly'
    num_hidden_layers = 5
    truncation_layer  = 3 # Indexing includes input and output layer with input layer indexed by 0
    num_hidden_nodes  = 500
    penalty           = 1
    num_training_data = 50000
    batch_size        = 1000
    num_epochs        = 500
    gpu               = '0'
    
class RunOptions:
    def __init__(self, hyper_p): 
        #=== Use LBFGS Optimizer ===#
        self.use_LBFGS = 0
        
        #=== Random Seed ===#
        self.random_seed = 1234
        
        #=== Data Set ===#
        data_thermal_fin_nine = 0
        data_thermal_fin_vary = 1
        self.num_testing_data = 200
        
###############################################################################
#                                 File Name                                   #
###############################################################################         
        #=== Data Type Names ===#
        self.use_full_domain_data = 0
        self.use_bnd_data = 0
        self.use_bnd_data_only = 0
        if hyper_p.data_type == 'full':
            self.use_full_domain_data = 1
        if hyper_p.data_type == 'bnd':
            self.use_bnd_data = 1
        if hyper_p.data_type == 'bndonly':
            self.use_bnd_data_only = 1
        
        #=== Parameter and Observation Dimensions === #
        self.full_domain_dimensions = 1446 
        if data_thermal_fin_nine == 1:
            self.parameter_dimensions = 9
        if data_thermal_fin_vary == 1:
            self.parameter_dimensions = self.full_domain_dimensions
        if self.use_full_domain_data == 1:
            self.state_obs_dimensions = self.full_domain_dimensions 
        if self.use_bnd_data == 1 or self.use_bnd_data_only == 1:
            self.state_obs_dimensions = 614
        
        #=== File name ===#
        if data_thermal_fin_nine == 1:
            self.dataset = 'thermalfin9'
        if data_thermal_fin_vary == 1:
            self.dataset = 'thermalfinvary'
        if hyper_p.penalty >= 1:
            hyper_p.penalty = int(hyper_p.penalty)
            penalty_string = str(hyper_p.penalty)
        else:
            penalty_string = str(hyper_p.penalty)
            penalty_string = 'pt' + penalty_string[2:]

        self.filename = self.dataset + '_' + hyper_p.data_type + '_hl%d_tl%d_hn%d_p%s_d%d_b%d_e%d' %(hyper_p.num_hidden_layers, hyper_p.truncation_layer, hyper_p.num_hidden_nodes, penalty_string, hyper_p.num_training_data, hyper_p.batch_size, hyper_p.num_epochs)

###############################################################################
#                                 File Paths                                  #
###############################################################################         
        #=== Save File Name ===#
        self.NN_savefile_directory = '../Trained_NNs/' + self.filename
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename
  
        #=== Loading Test Case ===#       
        self.savefile_name_parameter_test = self.NN_savefile_name + '_parameter_test'
        if hyper_p.data_type == 'full':
            self.savefile_name_state_test = self.NN_savefile_name + '_state_test'
        if hyper_p.data_type == 'bndonly':
            self.savefile_name_state_test = self.NN_savefile_name + '_state_test_bnd'
            
        #=== Loading Predictions ===#    
        self.savefile_name_parameter_pred = self.NN_savefile_name + '_parameter_pred'
        self.savefile_name_state_pred = self.NN_savefile_name + '_state_pred'
            
        #=== Savefile Path for Figures ===#    
        self.figures_savefile_directory = '../Figures/' + self.filename
        self.figures_savefile_name = self.figures_savefile_directory + '/' + self.filename
        self.figures_savefile_name_parameter_test = self.figures_savefile_directory + '/' + 'parameter_test'
        self.figures_savefile_name_state_test = self.figures_savefile_directory + '/' + 'state_test'
        self.figures_savefile_name_parameter_pred = self.figures_savefile_name + '_parameter_pred'
        self.figures_savefile_name_state_pred = self.figures_savefile_name + '_state_pred'
        
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
        hyper_p.num_training_data = int(sys.argv[6])
        hyper_p.batch_size        = int(sys.argv[7])
        hyper_p.num_epochs        = int(sys.argv[8])
        hyper_p.gpu               = str(sys.argv[9])
        
    #=== Set run options ===#        
    run_options = RunOptions(hyper_p)
    
    #=== Predict and Save ===#
    plot_and_save(hyper_p, run_options)
    



    
    
    
    
    
    