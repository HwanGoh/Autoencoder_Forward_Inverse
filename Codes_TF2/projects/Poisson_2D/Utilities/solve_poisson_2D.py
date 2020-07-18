import tensorflow as tf
import numpy as np
import pandas as pd
import time

from Utilities.integrals_pwl_prestiffness import integrals_pwl_prestiffness

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                 Standard                                    #
###############################################################################
def solve_PDE_standard(run_options, file_paths,
                       parameters,
                       obs_indices, nodes, elements,
                       boundary_matrix, load_vector):

    #=== Create Storage ===#
    stiffness_matrix = np.zeros(
            (run_options.parameter_dimensions, run_options.parameter_dimensions))

    #=== Construct Matrices ===#
    for k in range(0, elements.shape[0]):
        ver = elements[k,:]
        vertices_coords = nodes[ver,:]
        p_k = parameters.numpy()[ver]
        stiffness_matrix[np.ix_(ver,ver)] +=\
                integrals_pwl_prestiffness(vertices_coords)*(p_k[0] + p_k[1] + p_k[2])

    return np.transpose(np.linalg.solve(stiffness_matrix + boundary_matrix, load_vector))

###############################################################################
#                                 Sensitivity                                 #
###############################################################################
def construct_sensitivity(run_options, file_paths,
                          parameters, state,
                          obs_indices, nodes, elements,
                          boundary_matrix, load_vector):

    #=== Create Storage ===#
    stiffness_matrix = np.zeros(
            (run_options.parameter_dimensions, run_options.parameter_dimensions))
    partial_derivative_parameter = np.zeros(
            (run_options.parameter_dimensions, run_options.parameter_dimensions))
    partial_derivative_matrix = np.zeros(
            (run_options.parameter_dimensions, run_options.parameter_dimensions))

    #=== Construct Matrices ===#
    for k in range(0, elements.shape[0]):
        ver = elements[k,:]
        vertices_coords = nodes[ver,:]
        p_k = parameters.numpy()[ver]
        stiffness_matrix[np.ix_(ver,ver)] +=\
                integrals_pwl_prestiffness(vertices_coords)*(p_k[0] + p_k[1] + p_k[2])
        partial_derivative_parameter[np.ix_(ver,ver)] +=\
                integrals_pwl_prestiffness(vertices_coords)

    #=== Forming Objects ===#
    partial_derivative_matrix = np.repeat(
            np.matmul(
                partial_derivative_parameter, np.expand_dims(state, axis = 1)),
            repeats=run_options.parameter_dimensions, axis=1)

    return -np.linalg.solve(stiffness_matrix + boundary_matrix, partial_derivative_matrix)

###############################################################################
#                               Using Prematrices                             #
###############################################################################
def solve_PDE_prematrices(run_options, obs_indices,
        parameters,
        prestiffness, boundary_matrix, load_vector):

    #=== Solving PDE ===#
    prestiffness = tf.linalg.matmul(prestiffness, tf.transpose(parameters))
    stiffness_matrix = tf.reshape(prestiffness[:,0],
                (run_options.parameter_dimensions, run_options.parameter_dimensions))
    state = tf.transpose(tf.linalg.solve(stiffness_matrix + boundary_matrix, load_vector))
    for n in range(1, parameters.shape[0]):
        stiffness_matrix = tf.reshape(prestiffness[:,n],
                (run_options.parameter_dimensions, run_options.parameter_dimensions))
        solution = tf.linalg.solve(stiffness_matrix + boundary_matrix, load_vector)
        state = tf.concat([state, tf.transpose(solution)], axis=0)

    #=== Generate Measurement Data ===#
    if run_options.obs_type == 'obs':
        obs_indices = tf.cast(obs_indices, tf.int32)
        state_obs = tf.gather(state, obs_indices, axis=1)
        return tf.squeeze(state_obs)
    else:
        return state

#def solve_PDE_prematrices(run_options, obs_indices,
#        parameters,
#        prestiffness, boundary_matrix, load_vector):

#    state = tf.Variable(tf.zeros((parameters.shape[0], run_options.parameter_dimensions)))

#    #=== Solving PDE ===#
#    prestiffness = tf.linalg.matmul(prestiffness, tf.transpose(parameters))
#    for n in range(0, parameters.shape[0]):
#        stiffness_matrix = tf.reshape(prestiffness[:,n:n+1],
#                (run_options.parameter_dimensions, run_options.parameter_dimensions))
#        state[n:n+1,:].assign(tf.transpose(
#                tf.linalg.solve(stiffness_matrix + boundary_matrix, load_vector)))

#    #=== Generate Measurement Data ===#
#    if run_options.obs_type == 'obs':
#        state = state[:,obs_indices]

#    return state

###############################################################################
#                           Using Sparse Prematrices                          #
###############################################################################
def solve_PDE_prematrices_sparse(run_options, obs_indices,
        parameters,
        prestiffness, boundary_matrix, load_vector):

    #=== Solving PDE ===#
    stiffness_matrix = tf.reshape(
            tf.sparse.sparse_dense_matmul(
                prestiffness, tf.expand_dims(tf.transpose(parameters[0,:]), axis=1)),
                (run_options.parameter_dimensions, run_options.parameter_dimensions))
    state = tf.transpose(tf.linalg.solve(stiffness_matrix + boundary_matrix, load_vector))
    for n in range(1, parameters.shape[0]):
        stiffness_matrix = tf.reshape(
                tf.sparse.sparse_dense_matmul(
                    prestiffness, tf.expand_dims(tf.transpose(parameters[n,:]), axis=1)),
                    (run_options.parameter_dimensions, run_options.parameter_dimensions))
        solution = tf.linalg.solve(stiffness_matrix + boundary_matrix, load_vector)
        state = tf.concat([state, tf.transpose(solution)], axis=0)

    #=== Generate Measurement Data ===#
    if run_options.obs_type == 'obs':
        obs_indices = tf.cast(obs_indices, tf.int32)
        state_obs = tf.gather(state, obs_indices, axis=1)
        return tf.squeeze(state_obs)
    else:
        return state
