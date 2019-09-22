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
    truncation_layer = 2 # Indexing includes input and output layer
    num_hidden_nodes = 200
    penalty = 40
    num_training_data = 2000
    batch_size = 20
    num_epochs = 50000
    
class FileNames:
    def __init__(self,hyper_p):
        self.filename = f'hl{hyper_p.num_hidden_layers}_tl{hyper_p.truncation_layer}_hn{hyper_p.num_hidden_nodes}_p{hyper_p.penalty}_d{hyper_p.num_training_data}_b{hyper_p.batch_size}_e{hyper_p.num_epochs}'
        self.test_data_savefilepath = '../Data/' + 'test_data'
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
    filenames = FileNames(hyper_p)
           
    #####################################
    #   Form Test Parameters and State  #
    #####################################
    V = get_space(40)
    solver = Fin(V) 
    
    if os.path.isfile(filenames.test_data_savefilepath + '.csv'):
        print('Loading Test Data')
        df = pd.read_csv(filenames.test_data_savefilepath + '.csv')
        test_data = df.to_numpy()
        parameter_test = test_data[:,0].reshape((1, V.dim()))
        state_test = test_data[:,1].reshape((1, V.dim()))
    else:
        parameter_test, state_test= generate_thermal_fin_data(1)
        # Saving Parameters and State Data
        test_data = {'parameter_test': parameter_test.flatten(), 'state_test': state_test.flatten()}
        df = pd.DataFrame(test_data)   
        df.to_csv(filenames.test_data_savefilepath + '.csv', index=False)  
        
    ####################################
    #   Import Trained Neural Network  #
    ####################################        
    with tf.Session() as sess:    
        new_saver = tf.train.import_meta_graph(filenames.NN_savefile_name + '.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(filenames.NN_savefile_directory))        
        
        # Labelling loaded variables as a class
        NN = AutoencoderFwdInv(hyper_p,parameter_test.shape[1],state_test.shape[1], construct_flag = 0) 
                
        #######################
        #   Form Predictions  #
        #######################        
        state_pred = sess.run(NN.forward_pred, feed_dict = {NN.parameter_input_tf: parameter_test.reshape((1,parameter_test.shape[1]))})  
        parameter_pred = sess.run(NN.inverse_pred, feed_dict = {NN.state_input_tf: state_test.reshape((1,state_test.shape[1]))})    
        
        ##############
        #  Plotting  #
        ##############
        #=== Plotting test parameter and test state ===#
        parameter_test_dl = convert_array_to_dolfin_function(V,parameter_test)
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
        parameter_pred_dl = convert_array_to_dolfin_function(V,parameter_pred)
        state_pred_dl = convert_array_to_dolfin_function(V,state_pred)
        
        p_pred_fig = dl.plot(parameter_pred_dl)
        p_pred_fig.ax.set_title('Decoder Estimation of True Parameter', fontsize=18)  
        plt.savefig(filenames.figures_savefile_name_parameter_pred, dpi=300)
        print('Figure saved to ' + filenames.figures_savefile_name_parameter_pred) 
        plt.show()
        parameter_pred_error = tf.norm(parameter_pred - parameter_test,2)/tf.norm(parameter_test,2)
        print(sess.run(parameter_pred_error, feed_dict = {NN.parameter_input_tf: parameter_test.reshape((1,parameter_test.shape[1]))}))
        
        s_pred_fig = dl.plot(state_pred_dl)
        s_pred_fig.ax.set_title('Encoder Estimation of True State', fontsize=18)  
        plt.savefig(filenames.figures_savefile_name_state_pred, dpi=300)
        print('Figure saved to ' + filenames.figures_savefile_name_state_pred) 
        plt.show()
        state_pred_error = tf.norm(state_pred - state_test,2)/tf.norm(state_test,2)
        print(sess.run(state_pred_error, feed_dict = {NN.state_input_tf: state_test.reshape((1,state_test.shape[1]))}))