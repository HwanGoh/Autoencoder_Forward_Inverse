#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 15:34:49 2019

@author: hwan
"""

import sys
sys.path.append('../')

import tensorflow as tf
tf.reset_default_graph()
import numpy as np
import pandas as pd
import dolfin as dl
from forward_solve import Fin
from thermal_fin import get_space

import matplotlib.pyplot as plt
from NN_Autoencoder_Fwd_Inv import AutoencoderFwdInv
from Generate_and_Save_Thermal_Fin_Data import convert_array_to_dolfin_function

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '6'
sys.path.insert(0, '../../Utilities/')

###############################################################################
#                               Parameters                                    #
###############################################################################
class HyperParameters:
    data_type         = 'full'
    num_hidden_layers = 3
    truncation_layer  = 2 # Indexing includes input and output layer with input layer indexed by 0
    num_hidden_nodes  = 1446
    penalty           = 1
    num_training_data = 2000
    batch_size        = 20
    num_epochs        = 50000
    gpu               = '1'
    
class RunOptions:
    def __init__(self, hyper_p):
        # Data type
        self.use_full_domain_data = 0
        self.use_bnd_data = 0
        self.use_bnd_data_only = 0
        if hyper_p.data_type == 'full':
            self.use_full_domain_data = 1
        if hyper_p.data_type == 'bnd':
            self.use_bnd_data = 1
        if hyper_p.data_type == 'bndonly':
            self.use_bnd_data_only = 1
        
        # Observation Dimensions
        self.full_domain_dimensions = 1446 
        if self.use_full_domain_data == 1:
            self.state_obs_dimensions = self.full_domain_dimensions 
        if self.use_bnd_data == 1 or self.use_bnd_data_only == 1:
            self.state_obs_dimensions = 614
        
        # Number of Testing Data
        self.num_testing_data = 200
        
        # File name            
        self.filename = data_type + '_hl%d_tl%d_hn%d_p%d_d%d_b%d_e%d' %(hyper_p.num_hidden_layers, hyper_p.truncation_layer, hyper_p.num_hidden_nodes, hyper_p.penalty, hyper_p.num_training_data, hyper_p.batch_size, hyper_p.num_epochs)

        # Loading and saving data
        if self.use_full_domain_data == 1:
            self.observation_indices_savefilepath = '../Data/' + 'thermal_fin_full_domain'
            self.parameter_test_savefilepath = '../Data/' + 'parameter_test_%d' %(self.num_testing_data) 
            self.state_obs_test_savefilepath = '../Data/' + 'state_test_%d' %(self.num_testing_data)             
        if self.use_bnd_data == 1 or self.use_bnd_data_only == 1:
            self.observation_indices_savefilepath = '../Data/' + 'thermal_fin_bnd_indices'
            self.parameter_test_savefilepath = '../Data/' + 'parameter_test_bnd_%d' %(self.num_testing_data) 
            self.state_obs_test_savefilepath = '../Data/' + 'state_test_bnd_%d' %(self.num_testing_data) 
    
        self.NN_savefile_directory = '../Trained_NNs/' + self.filename
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename
        self.figures_savefile_directory = '../Figures/' + self.filename
        self.figures_savefile_name_parameter_test = self.figures_savefile_directory + '/' + 'parameter_test'
        self.figures_savefile_name_state_test = self.figures_savefile_directory + '/' + 'state_test'
        self.figures_savefile_name_parameter_pred = self.figures_savefile_directory + '/' + 'parameter_pred'
        self.figures_savefile_name_state_pred = self.figures_savefile_directory + '/' + 'state_pred'
        
        # Creating Directories
        if not os.path.exists('../Data'):
            os.makedirs('../Data')
        if not os.path.exists(self.figures_savefile_directory):
            os.makedirs(self.figures_savefile_directory)
       
###############################################################################
#                                  Driver                                     #
###############################################################################
if __name__ == "__main__":
    
    # Set hyperparameters
    hyper_p = HyperParameters()
        
    # Set run options        
    run_options = RunOptions(hyper_p)
           
    #####################################
    #   Form Test Parameters and State  #
    #####################################
    V,_ = get_space(40)
    solver = Fin(V) 
    
    # Load observation indices  
    print('Loading Boundary Indices')
    df_obs_indices = pd.read_csv(run_options.observation_indices_savefilepath + '.csv')    
    obs_indices = df_obs_indices.to_numpy()    
    
    # Load testing data 
    if os.path.isfile(run_options.parameter_test_savefilepath + '.csv'):
        print('Loading Test Data')
        df_parameter_test = pd.read_csv(run_options.parameter_test_savefilepath + '.csv')
        df_state_obs_test = pd.read_csv(run_options.state_obs_test_savefilepath + '.csv')
        parameter_test = df_parameter_test.to_numpy()
        state_obs_test = df_state_obs_test.to_numpy()
        parameter_test = parameter_test.reshape((run_options.num_testing_data, 9))
        state_obs_test = state_obs_test.reshape((run_options.num_testing_data, run_options.state_obs_dimensions))  
        parameter_test = parameter_test[0,:]
        state_obs_test = state_obs_test[0,:]
    else:
        raise ValueError('Test Data of size %d has not yet been generated' %(run_options.num_testing_data)) 
        
    ####################################
    #   Import Trained Neural Network  #
    ####################################        
    with tf.Session() as sess:    
        new_saver = tf.train.import_meta_graph(run_options.NN_savefile_name + '.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(run_options.NN_savefile_directory))        
        
        # Labelling loaded variables as a class
        NN = AutoencoderFwdInv(hyper_p, run_options, len(parameter_test), run_options.full_domain_dimensions, obs_indices, construct_flag = 0) 
                
        #######################
        #   Form Predictions  #
        #######################      
        parameter_pred = sess.run(NN.inverse_pred, feed_dict = {NN.state_obs_inverse_input_tf: state_obs_test.reshape((1, len(state_obs_test)))})    
        if run_options.use_full_domain_data == 1 or run_options.use_bnd_data == 1: # No state prediction if the truncation layer only consists of the observations
            state_pred = sess.run(NN.encoded, feed_dict = {NN.parameter_input_tf: parameter_test.reshape((1, len(parameter_test)))})  
        
        ##############
        #  Plotting  #
        ##############
        #=== Plotting test parameter and test state ===#
        parameter_test_dl = solver.nine_param_to_function(parameter_test.T)
        state_test_dl, _ = solver.forward(parameter_test_dl) # generate true state for comparison
        state_test = state_test_dl.vector().get_local()         
        
        p_test_fig = dl.plot(parameter_test_dl)
        p_test_fig.ax.set_title('True Parameter', fontsize=18)  
        plt.savefig(run_options.figures_savefile_name_parameter_test, dpi=300)
        print('Figure saved to ' + run_options.figures_savefile_name_parameter_test)   
        plt.show()
        
        s_test_fig = dl.plot(state_test_dl)
        s_test_fig.ax.set_title('True State', fontsize=18) 
        plt.savefig(run_options.figures_savefile_name_state_test, dpi=300)
        print('Figure saved to ' + run_options.figures_savefile_name_state_test) 
        plt.show()
        
        #=== Plotting predictions of test parameter and test state ===#
        parameter_pred_dl = solver.nine_param_to_function(parameter_pred.T)
        
        p_pred_fig = dl.plot(parameter_pred_dl)
        p_pred_fig.ax.set_title('Decoder Estimation of True Parameter', fontsize=18)  
        plt.savefig(run_options.figures_savefile_name_parameter_pred, dpi=300)
        print('Figure saved to ' + run_options.figures_savefile_name_parameter_pred) 
        plt.show()
        parameter_pred_error = np.linalg.norm(parameter_pred - parameter_test,2)/np.linalg.norm(parameter_test,2)
        print(parameter_pred_error)
        
        if run_options.use_full_domain_data == 1 or run_options.use_bnd_data == 1: # No state prediction if the truncation layer only consists of the observations
            state_pred_dl = convert_array_to_dolfin_function(V, state_pred)
            s_pred_fig = dl.plot(state_pred_dl)
            s_pred_fig.ax.set_title('Encoder Estimation of True State', fontsize=18)  
            plt.savefig(run_options.figures_savefile_name_state_pred, dpi=300)
            print('Figure saved to ' + run_options.figures_savefile_name_state_pred) 
            plt.show()
            state_pred_error = np.linalg.norm(state_pred - state_test,2)/np.linalg.norm(state_test,2)
            print(state_pred_error)