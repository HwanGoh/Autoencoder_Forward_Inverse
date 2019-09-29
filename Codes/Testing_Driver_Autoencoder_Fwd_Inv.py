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
import pandas as pd
import dolfin as dl
from forward_solve import Fin
from thermal_fin import get_space

import matplotlib.pyplot as plt
from NN_Autoencoder_Fwd_Inv import AutoencoderFwdInv
from Generate_and_Save_Thermal_Fin_Data import generate_thermal_fin_data, convert_array_to_dolfin_function

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
    num_hidden_layers = 1
    truncation_layer  = 1 # Indexing includes input and output layer with input layer indexed by 0
    num_hidden_nodes  = 1446
    penalty           = 10
    num_training_data = 20
    batch_size        = 20
    num_epochs        = 2000
    gpu               = '1'
    
class FileNames:
    def __init__(self, hyper_p, use_bnd_data, num_testing_data):
        # File name
        if use_bnd_data == 1:
            self.filename = f'bnd_hl{hyper_p.num_hidden_layers}_tl{hyper_p.truncation_layer}_hn{hyper_p.num_hidden_nodes}_p{hyper_p.penalty}_d{hyper_p.num_training_data}_b{hyper_p.batch_size}_e{hyper_p.num_epochs}'
        else:
            self.filename = f'hl{hyper_p.num_hidden_layers}_tl{hyper_p.truncation_layer}_hn{hyper_p.num_hidden_nodes}_p{hyper_p.penalty}_d{hyper_p.num_training_data}_b{hyper_p.batch_size}_e{hyper_p.num_epochs}'

        # Loading and saving data
        if use_bnd_data == 1:
            self.observation_indices_savefilepath = '../Data/' + 'thermal_fin_bnd_indices'
            self.parameter_train_savefilepath = '../Data/' + 'parameter_train_bnd_%d' %(hyper_p.num_training_data) 
            self.state_train_savefilepath = '../Data/' + 'state_train_bnd_%d' %(hyper_p.num_training_data) 
            self.parameter_test_savefilepath = '../Data/' + 'parameter_test_bnd_%d' %(num_testing_data) 
            self.state_test_savefilepath = '../Data/' + 'state_test_bnd_%d' %(num_testing_data) 
        else:
            self.observation_indices_savefilepath = '../Data/' + 'thermal_fin_bnd_indices'
            self.parameter_train_savefilepath = '../Data/' + 'parameter_train_%d' %(hyper_p.num_training_data) 
            self.state_train_savefilepath = '../Data/' + 'state_train_%d' %(hyper_p.num_training_data) 
            self.parameter_test_savefilepath = '../Data/' + 'parameter_test_%d' %(num_testing_data) 
            self.state_test_savefilepath = '../Data/' + 'state_test_%d' %(num_testing_data) 
    
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
    
    hyper_p = HyperParameters()
    
    use_bnd_data = 0
    num_testing_data = 20
    full_domain_dimensions = 1446    
    if use_bnd_data == 1:
        state_data_dimensions = 614
    else:
        state_data_dimensions = full_domain_dimensions 
    
    filenames = FileNames(hyper_p, use_bnd_data, num_testing_data)
           
    #####################################
    #   Form Test Parameters and State  #
    #####################################
    V,_ = get_space(40)
    solver = Fin(V) 
    
    # Load observation indices  
    print('Loading Boundary Indices')
    df_obs_indices = pd.read_csv(filenames.observation_indices_savefilepath + '.csv')    
    obs_indices = df_obs_indices.to_numpy()    
    
    # Load testing data 
    if os.path.isfile(filenames.parameter_test_savefilepath + '.csv'):
        print('Loading Test Data')
        df_parameter_test = pd.read_csv(filenames.parameter_test_savefilepath + '.csv')
        df_state_test = pd.read_csv(filenames.state_test_savefilepath + '.csv')
        parameter_test = df_parameter_test.to_numpy()
        state_test = df_state_test.to_numpy()
        parameter_test = parameter_test.reshape((num_testing_data, 9))
        state_test = state_test.reshape((num_testing_data, state_data_dimensions))  
        parameter_test = parameter_test[0,:]
        state_test = state_test[0,:]
    else:
        raise ValueError('Test Data of size %d has not yet been generated' %(num_testing_data)) 
        
    ####################################
    #   Import Trained Neural Network  #
    ####################################        
    with tf.Session() as sess:    
        new_saver = tf.train.import_meta_graph(filenames.NN_savefile_name + '.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(filenames.NN_savefile_directory))        
        
        # Labelling loaded variables as a class
        NN = AutoencoderFwdInv(hyper_p, len(parameter_test), full_domain_dimensions, obs_indices, construct_flag = 0) 
                
        #######################
        #   Form Predictions  #
        #######################        
        state_pred = sess.run(NN.forward_pred, feed_dict = {NN.parameter_input_tf: parameter_test.reshape((1, len(parameter_test)))})  
        parameter_pred = sess.run(NN.inverse_pred, feed_dict = {NN.state_input_tf: state_test.reshape((1, len(state_test)))})    
        
        ##############
        #  Plotting  #
        ##############
        #=== Plotting test parameter and test state ===#
        parameter_test_dl = solver.nine_param_to_function(parameter_test.T)
        state_test_dl = convert_array_to_dolfin_function(V,state_test)
        
        p_test_fig = dl.plot(parameter_test_dl)
        p_test_fig.ax.set_title('True Parameter', fontsize=18)  
        plt.savefig(filenames.figures_savefile_name_parameter_test, dpi=300)
        print('Figure saved to ' + filenames.figures_savefile_name_parameter_test)   
        plt.show()
        
        s_test_fig = dl.plot(state_test_dl)
        s_test_fig.ax.set_title('True State', fontsize=18) 
        plt.savefig(filenames.figures_savefile_name_state_test, dpi=300)
        print('Figure saved to ' + filenames.figures_savefile_name_state_test) 
        plt.show()
        
        #=== Plotting predictions of test parameter and test state ===#
        parameter_pred_dl = solver.nine_param_to_function(parameter_pred.T)
        state_pred_dl = convert_array_to_dolfin_function(V,state_pred)
        
        p_pred_fig = dl.plot(parameter_pred_dl)
        p_pred_fig.ax.set_title('Decoder Estimation of True Parameter', fontsize=18)  
        plt.savefig(filenames.figures_savefile_name_parameter_pred, dpi=300)
        print('Figure saved to ' + filenames.figures_savefile_name_parameter_pred) 
        plt.show()
        parameter_pred_error = tf.norm(parameter_pred - parameter_test,2)/tf.norm(parameter_test,2)
        print(sess.run(parameter_pred_error, feed_dict = {NN.parameter_input_tf: parameter_test.reshape((1, len(parameter_test)))}))
        
        s_pred_fig = dl.plot(state_pred_dl)
        s_pred_fig.ax.set_title('Encoder Estimation of True State', fontsize=18)  
        plt.savefig(filenames.figures_savefile_name_state_pred, dpi=300)
        print('Figure saved to ' + filenames.figures_savefile_name_state_pred) 
        plt.show()
        state_pred_error = tf.norm(state_pred - state_test,2)/tf.norm(state_test,2)
        print(sess.run(state_pred_error, feed_dict = {NN.state_input_tf: state_test.reshape((1, len(state_test)))}))