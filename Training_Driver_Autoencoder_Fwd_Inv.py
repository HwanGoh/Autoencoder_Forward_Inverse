#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 14:35:58 2019

@author: Hwan Goh
"""
import sys
sys.path.append('../')

from forward_solve import Fin
from thermal_fin import get_space
from parameter_generator import ParameterGeneratorNineValues

import numpy as np
from NN_Autoencoder_Fwd_Inv import AutoencoderFwdInv

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '6'
sys.path.insert(0, '../../Utilities/')

np.random.seed(1234)

###############################################################################
#                               Parameters                                    #
###############################################################################
class RunOptions:
    num_hidden_nodes = 200
    penalty = 1
    num_training_data = 10
    batch_size = 4
    num_batches = int(num_training_data/batch_size)
    num_epochs = 1000
    gpu    = '0'
    
    filename = f'hnodes{num_hidden_nodes}_pen{penalty}_data{num_training_data}_batch{batch_size}_epochs{num_epochs}'
    savefilepath = 'Models/' + filename
    savefilename = savefilepath + '/' + filename
   
###############################################################################
#                                  Driver                                     #
###############################################################################
if __name__ == "__main__":
    
    run_options = RunOptions()
    
    ###################################
    #   Generate Parameters and Data  #
    ###################################  following "test_thermal_fin_gradient.py" code
    V = get_space(40)
    solver = Fin(V) 
    
    parameter_true = np.zeros((run_options.num_training_data,1446))
    state_data = np.zeros((run_options.num_training_data,1446))
    
    for m in range(run_options.num_training_data): 
        print(run_options.filename[:-3])
        # Randomly generate piecewise constant true parameter with 9 values
        parameter_true[m,:], parameter_true_dl = ParameterGeneratorNineValues(V,solver) # True conductivity values       
        # Solve PDE for state variable
        state_data_dl,_, _, _,_ = solver.forward(parameter_true_dl)
        state_data[m,:] = state_data_dl.vector().get_local()
    
    ########################
    #   Train Autoencoder  #
    ########################
    NN = AutoencoderFwdInv(run_options,parameter_true,state_data)
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     