#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:25:44 2019

@author: hwan
"""

import tensorflow as tf
import numpy as np
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"


###############################################################################
#                         Check Gradients Function                            #
###############################################################################
def check_gradients_objects(layers):
    perturb_h = 1e-6
    rand_v_weights = []
    rand_v_biases = []      
    for l in range(0, len(layers) - 1): 
        W = np.random.random_sample((layers[l], layers[l + 1]))
        b = np.random.random_sample((1, layers[l + 1]))                                 
        rand_v_weights.append(W)
        rand_v_biases.append(b)
    return perturb_h, rand_v_weights, rand_v_biases

def check_gradients(sess, NN, loss, gradients_tf, tf_dict):
    #===  Finite Difference Directional Derivative Approximation of Gradient ===#
    perturb_h, rand_v_weights, rand_v_biases = check_gradients_objects(NN.layers)
    current_loss = sess.run(loss, feed_dict=tf_dict)
    weights_current = sess.run(NN.weights)
    biases_current = sess.run(NN.biases)
    for l in range(0, len(NN.weights)):  
       sess.run(tf.assign(NN.weights[l], weights_current[l] + perturb_h*rand_v_weights[l]))
       sess.run(tf.assign(NN.biases[l], biases_current[l] + perturb_h*rand_v_biases[l]))
    perturbed_loss = sess.run(loss, feed_dict=tf_dict)
    finite_difference_grad = (perturbed_loss - current_loss)/perturb_h
    for l in range(0, len(NN.weights)): 
       sess.run(tf.assign(NN.weights[l], weights_current[l]))
       sess.run(tf.assign(NN.biases[l], biases_current[l]))
    
    #===  Directional Derivative with Tensorflow Gradient ===#  
    grad_tf_norm = 0    
    for l in range(0, len(NN.layers) - 1, 2):
        W = sess.run(NN.weights[l])
        b = sess.run(NN.biases[l])
        W_grad_vals = sess.run(gradients_tf[l][0], feed_dict = tf_dict)
        b_grad_vals = sess.run(gradients_tf[l+1][0], feed_dict = tf_dict)
        grad_tf_norm = grad_tf_norm + np.sum(np.multiply(W, W_grad_vals)) + np.sum(np.multiply(b, b_grad_vals))
        
    print(abs(finite_difference_grad - grad_tf_norm)/(abs(finite_difference_grad)))
    pdb.set_trace()

        
        
        
        
        
        
