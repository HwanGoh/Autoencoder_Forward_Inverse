#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 14:53:47 2019

@author: hwan
"""

import pandas as pd
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def save_weights_and_biases(sess, truncation_layer, layers, savefilepath):
    #=== Save Trained Weights and Biases ===#
    # encoder
    for l in range(1, truncation_layer+1):
        trained_weights = sess.run("autoencoder/encoder/W" + str(l) + ':0')
        trained_biases = sess.run("autoencoder/encoder/b" + str(l) + ':0')
        trained_weights_dict = {"W"+str(l): trained_weights.flatten()}
        trained_biases_dict = {"b"+str(l): trained_biases.flatten()}
        df_trained_weights = pd.DataFrame(trained_weights_dict)
        df_trained_biases = pd.DataFrame(trained_biases_dict)
        df_trained_weights.to_csv(savefilepath + "_W" + str(l) + '.csv', index=False)
        df_trained_biases.to_csv(savefilepath + "_b" + str(l) + '.csv', index=False)
     
    # decoder
    for l in range(truncation_layer+1, len(layers)):
        trained_weights = sess.run("autoencoder/decoder/W" + str(l) + ':0')
        trained_biases = sess.run("autoencoder/decoder/b" + str(l) + ':0')
        trained_weights_dict = {"W"+str(l): trained_weights.flatten()}
        trained_biases_dict = {"b"+str(l): trained_biases.flatten()}
        df_trained_weights = pd.DataFrame(trained_weights_dict)
        df_trained_biases = pd.DataFrame(trained_biases_dict)
        df_trained_weights.to_csv(savefilepath + "_W" + str(l) + '.csv', index=False)
        df_trained_biases.to_csv(savefilepath + "_b" + str(l) + '.csv', index=False)
    
# =============================================================================
#         #=== Testing restore ===#
#         df_trained_weights = pd.read_csv(savefilepath + "_W" + str(l) + '.csv')
#         df_trained_biases = pd.read_csv(savefilepath + "_b" + str(l) + '.csv')
#         restored_W = df_trained_weights.values.reshape([layers[l-1], layers[l]])
#         restored_b = df_trained_biases.values.reshape([1, layers[l]])
#         pdb.set_trace()
# =============================================================================
