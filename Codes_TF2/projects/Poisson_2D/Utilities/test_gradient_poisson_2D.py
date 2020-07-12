import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from get_prior import load_prior
from positivity_constraints import positivity_constraint_log_exp
from Utilities.get_FEM_matrices_tf import load_FEM_matrices_tf
from Utilities.solve_poisson_2D import solve_PDE
from NN_Autoencoder_Fwd_Inv import AutoencoderFwdInv
from loss_and_relative_errors import loss_penalized_difference

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def test_gradient(hyperp, run_options, file_paths):
###############################################################################
#                        Construct Directional Derivative                     #
###############################################################################
    ##################
    #   Load Prior   #
    ##################
    prior_mean, _, prior_covariance_cholesky, _\
    = load_prior(run_options, file_paths,
                 load_mean = 1,
                 load_covariance = 0,
                 load_covariance_cholesky = 1, load_covariance_cholesky_inverse = 0)
    k = 0.5

    #####################
    #   Generate Data   #
    #####################
    #=== Draw True Parameter ===#
    normal_draw = np.random.normal(0, 1, run_options.parameter_dimensions)
    parameter_true = np.matmul(prior_covariance_cholesky, normal_draw) + prior_mean.T
    parameter_true = positivity_constraint_log_exp(parameter_true)
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

    #=== Display Neural Network Architecture ===#
    NN.build((hyperp.batch_size, input_dimensions))
    NN.summary()

    #=== Draw and Set Weights ===#
    weights_list = []
    for n in range(0, len(NN.weights)):
        weights_list.append(tf.random.normal(NN.weights[n].shape, 0, 0.05, tf.float32))
    NN.set_weights(weights_list)

    #=== Compute Gradient ===#
    with tf.GradientTape() as tape:
        NN_output = NN(parameter_true)
        test = positivity_constraint_log_exp(NN_output)
        forward_model_pred = solve_PDE(
                run_options, obs_indices,
                positivity_constraint_log_exp(NN_output),
                prestiffness, boundary_matrix, load_vector)
        loss_0 = loss_penalized_difference(state_obs_true, forward_model_pred, 1)
    gradients = tape.gradient(loss_0, NN.trainable_variables)

    ############################
    #   Direction Derivative   #
    ############################
    #=== Draw Direction ===#
    directions_list = []
    for n in range(0, len(gradients)):
        directions_list.append(tf.random.normal(gradients[n].shape, 0, 0.05, tf.float32))
        directions_list[n] = directions_list[n]/tf.linalg.norm(directions_list[n],2)

    #=== Directional Derivative ===#
    directional_derivative = 0.0
    for n in range(0, len(gradients)):
        directional_derivative += tf.reduce_sum(tf.multiply(gradients[n], directions_list[n]))
    directional_derivative = directional_derivative.numpy()

###############################################################################
#                     Construct Finite Difference Derivative                  #
###############################################################################
    loss_h_list = []
    gradients_fd_list = []
    errors_list = []
    h_collection = np.power(2., -np.arange(32))

    for h in h_collection:
        #=== Perturbed Loss ===#
        weights_perturbed_list = []
        for n in range(0, len(NN.weights)):
            weights_perturbed_list.append(weights_list[n] + h*directions_list[n])
        NN.set_weights(weights_perturbed_list)
        NN_perturbed_output = NN(parameter_true)
        forward_model_perturbed_pred = solve_PDE(
                run_options, obs_indices,
                positivity_constraint_log_exp(NN_perturbed_output),
                prestiffness, boundary_matrix, load_vector)

        loss_h = loss_penalized_difference(state_obs_true, forward_model_perturbed_pred, 1)
        gradient_fd = (loss_h - loss_0)/h
        error = abs(gradient_fd - directional_derivative)/abs(directional_derivative)

        loss_h = loss_h.numpy()
        gradient_fd = gradient_fd.numpy()
        error = error.numpy()

        loss_h_list.append(loss_h)
        gradients_fd_list.append(gradient_fd)
        errors_list.append(error)

###############################################################################
#                                   Plotting                                  #
###############################################################################
    #=== Plot Functional ===#
    plt.loglog(h_collection, loss_h_list, "-ob", label="Functional")
    plt.savefig('functional.png', dpi=200)
    plt.close()

    #=== Plot Error ===#
    plt.loglog(h_collection, errors_list, "-ob", label="Error")
    plt.loglog(h_collection,
            (.5*errors_list[0]/h_collection[0])*h_collection, "-.k", label="First Order")
    plt.savefig('grad_test.png', dpi=200)
    plt.cla()
    plt.clf()

    print(f"FD gradients: {gradients_fd_list}")
    print(f"Errors: {errors_list}")
