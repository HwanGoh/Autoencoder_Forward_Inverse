#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:25:44 2019

@author: hwan
"""

import numpy as np
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"


###############################################################################
#                         Check Gradients Function                            #
###############################################################################
def check_gradients_objects(layers):
    perturb_h = 1e-7  
    rand_v_weights = []
    rand_v_biases = []      
    for l in range(0, len(layers) - 1): 
        W = np.random.random_sample((layers[l], layers[l + 1]))
        b = np.random.random_sample((1, layers[l + 1]))                                 
        rand_v_weights.append(W)
        rand_v_biases.append(b)
    return perturb_h, rand_v_weights, rand_v_biases

def check_gradients(sess, NN, loss, gradients_tf, loss_value, tf_dict):
    #===  Finite Difference Directional Derivative Approximation of Gradient ===#
    perturb_h, rand_v_weights, rand_v_biases = check_gradients_objects(NN.layers)
    weights_current = NN.weights
    biases_current = NN.biases
    for l in range(0, len(NN.weights)): 
       pdb.set_trace() 
       NN.weights[l].assign(weights_current[l] + 100*rand_v_weights[l]) 
       NN.biases[l].assign(biases_current[l] + 100*rand_v_biases[l])
    perturbed_loss = sess.run(loss, feed_dict=tf_dict)
    finite_difference_grad = (perturbed_loss - loss_value)/perturb_h
    for l in range(0, len(NN.weights)): 
       NN.weights[l].assign(weights_current[l]) 
       NN.biases[l].assign(biases_current[l])
    
    #===  Directional Derivative with Tensorflow Gradient ===#  
    grad_tf_norm = 0    
    for l in range(0, len(NN.layers) - 1, 2):
        W = sess.run(NN.weights[l])
        b = sess.run(NN.weights[l+1])
        W_grad_tf = sess.run(gradients_tf[l][0])
        b_grad_tf = sess.run(gradients_tf[l+1][0])
        grad_tf_norm = grad_tf_norm + np.sum(np.multiply(W, W_grad_tf)) + np.sum(np.multiply(b, b_grad_tf))
        
    print(abs(finite_difference_grad - grad_tf_norm)/(abs(finite_difference_grad)))