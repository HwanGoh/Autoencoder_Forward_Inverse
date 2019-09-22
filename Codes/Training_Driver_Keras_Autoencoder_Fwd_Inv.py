#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 13:55:15 2019

@author: hwan
"""

import sys
sys.path.append('../')

import tensorflow as tf # for some reason this must be first! Or else I get segmentation fault
tf.reset_default_graph()
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
import pandas as pd

from NN_Autoencoder_Fwd_Inv import AutoencoderFwdInv
from random_mini_batches import random_mini_batches
import time

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '6'
sys.path.insert(0, '../../Utilities/')

np.random.seed(1234)

###############################################################################
#                       Hyperparameters and Filenames                         #
###############################################################################
class HyperParameters:
    num_hidden_layers = 1
    truncation_layer  = 2 # Indexing includes input and output layer
    num_hidden_nodes  = 200
    penalty           = 10
    num_training_data = 20
    batch_size        = 20
    num_epochs        = 2000
    gpu               = '0'
    
class FileNames:
    def __init__(self,hyper_p):        
        self.filename = f'hl{hyper_p.num_hidden_layers}_tl{hyper_p.truncation_layer}_hn{hyper_p.num_hidden_nodes}_p{hyper_p.penalty}_d{hyper_p.num_training_data}_b{hyper_p.batch_size}_e{hyper_p.num_epochs}'
        self.NN_savefile_directory = '../Trained_NNs/' + self.filename # Since we need to save four different types of files to save a neural network model, we need to create a new folder for each model
        self.NN_savefile_name = self.NN_savefile_directory + '/' + self.filename # The file path and name for the four files
        self.data_savefilepath = '../Data/' + 'data_%d' %(hyper_p.num_training_data)
        
        # Creating Directories
        if not os.path.exists(self.NN_savefile_directory):
            os.makedirs(self.NN_savefile_directory)
            
###############################################################################
#                                   Executor                                  #
###############################################################################     
if __name__ == "__main__":     
    hyper_p = HyperParameters()
    if len(sys.argv) > 1:
            hyper_p.num_hidden_layers = int(sys.argv[1])
            hyper_p.truncation_layer  = int(sys.argv[2])
            hyper_p.num_hidden_nodes  = int(sys.argv[3])
            hyper_p.penalty           = int(sys.argv[4])
            hyper_p.num_training_data = int(sys.argv[5])
            hyper_p.batch_size        = int(sys.argv[6])
            hyper_p.num_epochs        = int(sys.argv[7])
            hyper_p.gpu               = str(sys.argv[8])
        
    filenames = FileNames(hyper_p)
    trainer(hyper_p, filenames) 
