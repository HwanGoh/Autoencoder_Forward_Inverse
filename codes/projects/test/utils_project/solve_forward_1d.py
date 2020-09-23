import numpy as np
import tensorflow as tf

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class SolveForward1D:
    def __init__(self, options, filepaths, obs_indices):

        #=== Defining Attributes ===#
        self.options = options
        self.filepaths = filepaths
        self.obs_indices = tf.cast(obs_indices, tf.int32)
        self.mesh = np.linspace(0, 1, options.mesh_dimensions, endpoint = True)

###############################################################################
#                                   Solvers                                   #
###############################################################################
    def exponential(self, parameters):

        #=== Form State Batch ===#
        state = parameters[0,0]*tf.exp(-parameters[0,1]*self.mesh.flatten())
        for n in range(1, parameters.shape[0]):
            solution = parameters[n,0]*tf.exp(-parameters[n,1]*self.mesh.flatten())
            state = tf.concat([state, tf.transpose(solution)], axis=0)

        #=== Generate Measurement Data ===#
        if self.options.obs_type == 'obs':
            state_obs = tf.gather(state, self.obs_indices, axis=1)
            return tf.squeeze(state_obs)
        else:
            return state
