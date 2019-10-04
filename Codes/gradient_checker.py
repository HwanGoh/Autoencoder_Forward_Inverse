#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:25:44 2019

@author: hwan
"""

###############################################################################
#                         Check Gradients Function                            #
###############################################################################
def check_gradients_objects(layers):
    perturb_h = 1e-7  
    rand_v_weights = []
    rand_v_biases = []      
    for l in range(0, len(layers) - 1): 
        rand_v_weights = np.random.random_sample(layers[l], layers[l + 1])
        rand_v_biases = np.random.random_sample(1, layers[l + 1])                                 
        rand_v_weights.append(W)
        rand_v_biases.append(b)
    return perturb_h, rand_v_weights, rand_v_biases

def check_gradients(sess, gradients_tf, loss_value, layers, perturb_weights_operation_tf, assign_weights_back_operation_tf, tf_dict):
    #===  Finite Difference Directional Derivative Approximation of Gradient ===#
    sess.run(perturb_weights_operation_tf, feed_dict=tf_dict)
    perturbed_loss = sess.run(loss, feed_dict=tf_dict)
    finite_difference_grad = (perturbed_loss - loss_value)/h
    sess.run(assign_weights_back_operation_tf, feed_dict=tf_dict)
    
    #===  Directional Derivative with Tensorflow Gradient ===#
    stacked_w_and_b_dimensions = 0
    for l in range(0, len(layers) - 1):
        stacked_w_and_b_dimensions = stacked_w_and_b_dimensions + layers[l]*layers[l+1] + layers[l+1]
    
    grad_all_w_and_b = np.array([])    
    for l in range(0, len(layers) - 1):
        W1 = sess.run('autoencoder/encoder/W1:0', feed_dict=tf_dict)
        grad_all_w_and_b = np.concatenate([grad_all_w_and_b, vectorized_weights])

    for l in range(1, len(layers)):
   

###############################################################################
#                 Assigning Weights for Evaluation of Loss                    #
###############################################################################    
def perturb_weights(NN, weights_current, biases_current, perturb_h, rand_v_weights, rand_v_biases):
    for l in range(0, len(NN.weights)): 
       NN.weights[l].assign(weights_current[l] + rand_v_weights[l]) 
       NN.biases[l].assign(biases_current[l] + rand_v_biases[l])
    return 1 # the operations above do absolutely nothing, in the end, all this function does is assign the number 1 to update_weights_flag. Jonathan Wittmer is a very clever boy
   
def assign_weights_back(NN, weights_current, biases_current):
    for l in range(0, len(NN.weights)): 
       NN.weights[l].assign(weights_current[l]) 
       NN.biases[l].assign(biases_current[l])
    return 1 # the operations above do absolutely nothing, in the end, all this function does is assign the number 1 to update_weights_flag. Jonathan Wittmer is a very clever boy
