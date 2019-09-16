#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 15:34:49 2019

@author: hwan
"""

import sys
sys.path.append('../')

import dolfin as dl
from forward_solve import Fin
from thermal_fin import get_space

import matplotlib.pyplot as plt
from NN_Autoencoder_Fwd_Inv import AutoencoderFwdInv
from parameter_generator import ParameterGeneratorNineValues, ConvertArraytoDolfinFunction
import tensorflow as tf

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '6'
sys.path.insert(0, '../../Utilities/')

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
    
    #####################################
    #   Form Test Parameters and State  #
    #####################################
    V = get_space(40)
    solver = Fin(V) 
    parameter_test, parameter_test_dl = ParameterGeneratorNineValues(V,solver) 
    state_test_dl, _, _, _, _ = solver.forward(parameter_test_dl)
    state_test = state_test_dl.vector().get_local()
    
    ####################################
    #   Import Trained Neural Network  #
    ####################################
    # Neural network
    NN = AutoencoderFwdInv(run_options,parameter_test.shape[0],state_test.shape[0])
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables()) 
        new_saver = tf.train.import_meta_graph(run_options.savefilename + '.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(run_options.savefilepath))
    
        #######################
        #   Form Predictions  #
        #######################        
        state_pred = sess.run(NN.forward_pred, feed_dict = {NN.parameter_input_tf: parameter_test.reshape((1,parameter_test.shape[0]))})  
        parameter_pred = sess.run(NN.inverse_pred, feed_dict = {NN.state_input_tf: state_test.reshape((1,state_test.shape[0]))})    
        
        ##############
        #  Plotting  #
        ##############
        #=== Plotting test parameter and test state ===#
        p = dl.plot(parameter_test_dl)
        plt.show()
        dl.plot(state_test_dl)
        plt.show()
        
        #=== Plotting predictions of test parameter and test state ===#
        parameter_pred_dl = ConvertArraytoDolfinFunction(V,parameter_pred)
        state_pred_dl = ConvertArraytoDolfinFunction(V,state_pred)
        p = dl.plot(parameter_pred_dl)
        plt.show()
        dl.plot(state_pred_dl)
        plt.show()
     