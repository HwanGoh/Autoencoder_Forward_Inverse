#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 14:35:58 2019

@author: Hwan Goh
"""
import sys
sys.path.append('../')

import dolfin as dl
from forward_solve import Fin
from thermal_fin import get_space
from gaussian_field import make_cov_chol

import numpy as np
import matplotlib.pyplot as plt
from Autoencoder_Fwd_Inv_NN import AutoencoderFwdInv

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '6'
sys.path.insert(0, '../../Utilities/')

np.random.seed(1234)

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
#                               Functions                                     #
###############################################################################
def ParameterGeneratorNineValues(length = 0.8):
    chol = make_cov_chol(V, length)
    norm = np.random.randn(len(chol))
    generated_parameter = np.exp(0.5 * chol.T @ norm) 
    parameter_true_dl = ConvertArraytoDolfinFunction(generated_parameter)
    parameter_true_dl = solver.nine_param_to_function(solver.subfin_avg_op(parameter_true_dl))
    generated_parameter = parameter_true_dl.vector().get_local()
    return generated_parameter, parameter_true_dl

def ConvertArraytoDolfinFunction(nodal_values):
    nodal_values_dl = dl.Function(V)
    nodal_values_dl.vector().set_local(np.squeeze(nodal_values))
    return nodal_values_dl
   
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
        parameter_true[m,:], parameter_true_dl = ParameterGeneratorNineValues() # True conductivity values       
        # Solve PDE for state variable
        state_data_dl,_, _, _,_ = solver.forward(parameter_true_dl)
        state_data[m,:] = state_data_dl.vector().get_local()
    
    ########################
    #   Train Autoencoder  #
    ########################
    ae = AutoencoderFwdInv(run_options,parameter_true,state_data)
    
    #######################
    #   Form Predictions  #
    #######################
    #=== Forward Prediction ===#
    parameter_test, parameter_test_dl = ParameterGeneratorNineValues()    
    state_pred = ae.forward_problem_prediction(parameter_test.reshape(1,parameter_test.shape[0]))
      
    #=== Inverse Prediction ===#
    state_test_dl,_, _, _,_ = solver.forward(parameter_test_dl)
    state_test = state_data_dl.vector().get_local()
    parameter_pred = ae.inverse_problem_prediction(state_test.reshape(1,state_test.shape[0]))
    
    ##############
    #  Plotting  #
    ##############
    #=== Plotting test parameter and test state ===#
    p = dl.plot(parameter_test_dl)
    plt.show()
    dl.plot(state_test_dl)
    plt.show()
    
    #=== Plotting predictions of test parameter and test state ===#
    parameter_pred_dl = ConvertArraytoDolfinFunction(parameter_pred)
    state_pred_dl = ConvertArraytoDolfinFunction(state_pred)
    p = dl.plot(parameter_pred_dl)
    plt.show()
    dl.plot(state_pred_dl)
    plt.show()
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     