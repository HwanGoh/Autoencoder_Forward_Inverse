import tensorflow as tf
import pandas as pd
import time

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def solve_PDE(run_options, obs_indices,
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
        state = state[:,obs_indices]

    return state
