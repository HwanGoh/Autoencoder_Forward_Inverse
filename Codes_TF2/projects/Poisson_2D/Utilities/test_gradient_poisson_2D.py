import numpy as np
import tensorflow as tf

from get_prior import load_prior
from Utilities.get_FEM_matrices_tf import load_FEM_matrices_tf
from Utilities.solve_poisson_2D import solve_PDE
from NN_Autoencoder_Fwd_Inv import AutoencoderFwdInv
from loss_and_relative_errors import loss_penalized_difference

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def test_gradient(hyperp, run_options, file_paths):
###############################################################################
#                         Construct Tensorflow Gradient                       #
###############################################################################
    ##################
    #   Load Prior   #
    ##################
    prior_mean, _, prior_covariance_cholesky\
    = load_prior(run_options, file_paths,
                 load_mean = 1,
                 load_covariance = 0, load_covariance_cholesky = 1)
    k = 0.5

    #####################
    #   Generate Data   #
    #####################
    #=== Draw True Parameter ===#
    normal_draw = np.random.normal(0, 1, run_options.parameter_dimensions)
    parameter_true = np.matmul(prior_covariance_cholesky, normal_draw) + prior_mean.T
    parameter_true = (1/k)*np.log(np.exp(k*parameter_true)+1);
    parameter_true = tf.cast(parameter_true, tf.float32)
    parameter_true = tf.expand_dims(parameter_true, axis = 0)

    #=== Load Observation Indices ===#
    if run_options.obs_type == 'full':
        obs_dimensions = run_options.parameter_dimensions
        obs_indices = []
    if run_options.obs_type == 'obs':
        obs_dimensions = run_options.num_obs_points
        print('Loading Boundary Indices')
        df_obs_indices = pd.read_csv(file_paths.obs_indices_savefilepath + '.csv')
        obs_indices = df_obs_indices.to_numpy()

    #=== Generate Observation Data ===#
    premass, prestiffness, boundary_matrix, load_vector =\
            load_FEM_matrices_tf(run_options, file_paths)
    state_obs_true = solve_PDE(
            run_options, obs_indices,
            parameter_true,
            prestiffness, boundary_matrix, load_vector)

    ########################
    #   Compute Gradient   #
    ########################
    #=== Draw Input Parameter ===#
    normal_draw = np.random.normal(0, 1, run_options.parameter_dimensions)
    parameter_input = np.matmul(prior_covariance_cholesky, normal_draw) + prior_mean.T
    parameter_input = (1/k)*np.log(np.exp(k*parameter_input)+1);
    parameter_input = tf.cast(parameter_input, tf.float32)
    parameter_input = tf.expand_dims(parameter_input, axis = 0)

    #=== Data and Latent Dimensions of Autoencoder ===#
    if run_options.use_standard_autoencoder == 1:
        input_dimensions = run_options.parameter_dimensions
        latent_dimensions = obs_dimensions
    if run_options.use_reverse_autoencoder == 1:
        input_dimensions = obs_dimensions
        latent_dimensions = run_options.parameter_dimensions

    #=== Form Neural Network ===#
    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
    bias_initializer = 'zeros'
    NN = AutoencoderFwdInv(hyperp,
                           input_dimensions, latent_dimensions,
                           kernel_initializer, bias_initializer)

    #=== Compute Gradient ===#
    with tf.GradientTape() as tape:
        NN_output = NN(parameter_input)
        loss = loss_penalized_difference(state_obs_true, NN_output, 1)
    gradients = tape.gradient(loss, NN.trainable_variables)

    ############################
    #   Direction Derivative   #
    ############################
    #=== Draw Direction ===#
    normal_draw = np.random.normal(0, 1, run_options.parameter_dimensions)
    direction_vector = np.matmul(prior_covariance_cholesky, normal_draw) + prior_mean.T
    direction_vector = direction_vector/np.linalg.norm(direction_vector, 2)
    direction_vector = tf.cast(direction_vector, tf.float32)
