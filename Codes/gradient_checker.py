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
#                    Check Gradients Direct Comparison                        #
###############################################################################
def check_gradients_direct_comparison(sess, NN, loss, gradients_tf, tf_dict):
    perturb_h = 1e-6
    current_loss = sess.run(loss, feed_dict=tf_dict)
    weights_current = sess.run(NN.weights)
    biases_current = sess.run(NN.biases)
    for l in range(0, len(NN.weights)): 
        for i in range(0, NN.layers[l]*NN.layers[l+1]):
            unit_vec = np.zeros((NN.layers[l]*NN.layers[l+1])) 
            unit_vec[i] = 1
            sess.run(tf.assign(NN.weights[l], weights_current[l] + perturb_h*unit_vec[l]))
            sess.run(tf.assign(NN.biases[l], biases_current[l] + perturb_h*unit_vec[l]))
            perturbed_loss = sess.run(loss, feed_dict=tf_dict)
            
###############################################################################
#                    Check Gradients Directional Derivative                   #
###############################################################################
def check_gradients_directional_derivative(sess, NN, loss, gradients_tf, tf_dict):
    #===  Finite Difference Directional Derivative Approximation of Gradient ===#
    perturb_h = 1e-8
    rand_v_weights, rand_v_biases = check_gradients_objects(NN.layers)
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
    grad_tf_directional_derivative = 0    
    for l in range(0, len(NN.layers) - 1, 2):
        W = sess.run(NN.weights[l])
        b = sess.run(NN.biases[l])
        W_grad_vals = sess.run(gradients_tf[l][0], feed_dict = tf_dict)
        b_grad_vals = sess.run(gradients_tf[l+1][0], feed_dict = tf_dict)
        grad_tf_directional_derivative = grad_tf_directional_derivative + np.sum(np.multiply(W, W_grad_vals)) + np.sum(np.multiply(b, b_grad_vals))
        
    print(abs(finite_difference_grad - grad_tf_directional_derivative)/(abs(finite_difference_grad)))
    pdb.set_trace()

def check_gradients_objects(layers):
    rand_v_weights = []
    rand_v_biases = []      
    for l in range(0, len(layers) - 1): 
        W = np.random.random_sample((layers[l], layers[l + 1]))
        b = np.random.random_sample((1, layers[l + 1]))                                 
        rand_v_weights.append(W)
        rand_v_biases.append(b)
    return rand_v_weights, rand_v_biases        

def define_weight_assignment_operations(NN, perturb_h, rand_v_weights, rand_v_biases):
    weights_current = NN.weights
    biases_current = NN.biases
    perturb_weights_flag = tf.Variable(0) # instead of doing sess.run(tf.assign(NN.weights[l], weights_current[l] + NN.weights[l])), this way circumvents all those session runs with one session run of the below operation
    perturb_weights_operation_tf = perturb_weights_flag.assign(perturb_weights(NN, weights_current, biases_current, perturb_h, rand_v_weights, rand_v_biases)) # assigns to update_weights_flag the value 1
    assign_weights_back_flag = tf.Variable(0)  
    assign_weights_back_operation_tf = assign_weights_back_flag.assign(assign_weights_back(NN, weights_current, biases_current))
    return perturb_weights_operation_tf, assign_weights_back_operation_tf

def check_gradients(sess, loss, gradients_tf, loss_value, layers, perturb_h, perturb_weights_operation_tf, assign_weights_back_operation_tf, rand_v_weights, rand_v_biases, tf_dict):
    #===  Finite Difference Directional Derivative Approximation of Gradient ===#
    sess.run(perturb_weights_operation_tf, feed_dict=tf_dict)
    perturbed_loss = sess.run(loss, feed_dict=tf_dict)
    finite_difference_grad = (perturbed_loss - loss_value)/perturb_h
    sess.run(assign_weights_back_operation_tf, feed_dict=tf_dict)
    
    #===  Directional Derivative with Tensorflow Gradient ===#    
    grad_tf_norm = 0   
    for l in range(0, len(layers) - 1,2):
        grad_weights_values = sess.run(gradients_tf[l][0], feed_dict=tf_dict)
        grad_biases_values = sess.run(gradients_tf[l+1][0], feed_dict=tf_dict)
        grad_weights_sum = np.sum(np.multiply(grad_weights_values, rand_v_weights[l]))
        grad_biases_sum = np.sum(np.multiply(grad_biases_values, rand_v_biases[l]))
        grad_tf_norm = grad_tf_norm + grad_weights_sum + grad_biases_sum
   
    pdb.set_trace()
    print(abs(finite_difference_grad - grad_tf_norm)/abs(finite_difference_grad))
    
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
