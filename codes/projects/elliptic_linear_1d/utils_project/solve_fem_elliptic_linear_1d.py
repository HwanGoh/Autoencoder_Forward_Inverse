import tensorflow as tf
import numpy as np
import pandas as pd
import time

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class SolveFEMEllipticLinear1D:
    def __init__(self, options, filepaths,
                 obs_indices,
                 forward_operator, mass_matrix):

        #=== Defining Attributes ===#
        self.options = options
        self.filepaths = filepaths
        self.obs_indices = tf.cast(obs_indices, tf.int32)
        self.forward_operator = forward_operator
        self.mass_matrix = mass_matrix

###############################################################################
#                                 PDE Solvers                                 #
###############################################################################
    def solve_pde(self, parameters):

        #=== Solving PDE ===#
        prestiffness = tf.linalg.matmul(self.prestiffness, tf.transpose(parameters))
        stiffness_matrix = tf.reshape(prestiffness[:,0],
                    (self.options.parameter_dimensions, self.options.parameter_dimensions))
        state = tf.transpose(
                tf.linalg.solve(stiffness_matrix + self.boundary_matrix, self.load_vector))
        for n in range(1, parameters.shape[0]):
            stiffness_matrix = tf.reshape(self.prestiffness[:,n],
                    (self.options.parameter_dimensions, self.options.parameter_dimensions))
            solution = tf.linalg.solve(stiffness_matrix + self.boundary_matrix, self.load_vector)
            state = tf.concat([state, tf.transpose(solution)], axis=0)

        #=== Generate Measurement Data ===#
        if options.obs_type == 'obs':
            state_obs = tf.gather(state, self.obs_indices, axis=1)
            return tf.squeeze(state_obs)
        else:
            return state

    def solve_pde(self, parameters):

        state = tf.Variable(tf.zeros((parameters.shape[0], self.options.parameter_dimensions)))

        #=== Solving PDE ===#
        prestiffness = tf.linalg.matmul(self.prestiffness, tf.transpose(parameters))
        for n in range(0, parameters.shape[0]):
            stiffness_matrix = tf.reshape(prestiffness[:,n:n+1],
                    (self.options.parameter_dimensions, self.options.parameter_dimensions))
            state[n:n+1,:].assign(tf.transpose(
                    tf.linalg.solve(stiffness_matrix + self.boundary_matrix, self.load_vector)))

        #=== Generate Measurement Data ===#
        if self.options.obs_type == 'obs':
            state = state[:,self.obs_indices]

        return state
