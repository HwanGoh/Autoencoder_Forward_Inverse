import tensorflow as tf
import pandas as pd
import time

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def solve_PDE(run_options,
        parameters,
        prestiffness, boundary_matrix, load_vector):

    state = tf.zeros((run_options.parameter_dimensions,
        run_options.parameter_dimensions))

    #=== Solving PDE ===#
    prestiffness = tf.linalg.matmul(
            prestiffness, tf.transpose(parameters))
    for n in range(0, run_options.num_data_train):
        stiffness_matrix = tf.reshape(prestiffness[:,n],
                (run_options.parameter_dimensions, run_options.parameter_dimensions))

        state[n,:] = tf.transpose(
                tf.linalg.solve(stiffness_matrix + boundary_matrix, load_vector))

    #=== Generate Measurement Data ===#
    if run_options.obs_type == 'obs':
        obs_indices = np.random.choice(boundary_indices_selected,
                run_options.num_obs_points, replace = False)
        state = state[:,obs_indices]

    return state
