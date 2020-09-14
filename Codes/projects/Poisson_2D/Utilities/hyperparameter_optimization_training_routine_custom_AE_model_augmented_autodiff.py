import os
import sys
sys.path.insert(0, os.path.realpath('../../src'))

import tensorflow as tf
import numpy as np
import pandas as pd

# Load FilePaths class and data retrieval function
from Utilities.file_paths_AE import FilePathsHyperparameterOptimization

# Import src code
from get_train_and_test_data import load_train_and_test_data
from add_noise import add_noise
from form_train_val_test import form_train_val_test_tf_batches
from get_prior import load_prior
from Utilities.get_FEM_matrices_tf import load_FEM_matrices_tf
from Utilities.FEM_prematrices_poisson_2D import FEMPrematricesPoisson2D
from NN_AE_Fwd_Inv import AutoencoderFwdInv
from loss_and_relative_errors import loss_penalized_difference, loss_weighted_penalized_difference,\
        relative_error, reg_prior
from optimize_custom_AE_model_augmented_autodiff import optimize
from optimize_distributed_custom_AE_model_aware import optimize_distributed
from positivity_constraints import positivity_constraint_log_exp

# Import skopt code
from skopt.utils import use_named_args
from skopt import gp_minimize

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                 Training                                    #
###############################################################################
def trainer_custom(hyperp, run_options, file_paths,
        n_calls, space,
        autoencoder_loss, project_name,
        data_options, dataset_directory):

    #=== GPU Settings ===#
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if run_options.distributed_training == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = run_options.which_gpu
    if run_options.distributed_training == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = run_options.dist_which_gpus
        gpus = tf.config.experimental.list_physical_devices('GPU')

    #=== Load Observation Indices ===#
    if run_options.obs_type == 'full':
        obs_dimensions = run_options.parameter_dimensions
        obs_indices = []
    if run_options.obs_type == 'obs':
        obs_dimensions = run_options.num_obs_points
        print('Loading Boundary Indices')
        df_obs_indices = pd.read_csv(file_paths.obs_indices_savefilepath + '.csv')
        obs_indices = df_obs_indices.to_numpy()

    #=== Load Data ===#
    parameter_train, state_obs_train,\
    parameter_test, state_obs_test\
    = load_train_and_test_data(file_paths,
            hyperp.num_data_train, run_options.num_data_test,
            run_options.parameter_dimensions, obs_dimensions,
            load_data_train_flag = 1,
            normalize_input_flag = 0, normalize_output_flag = 0)

    #=== Add Noise to Data ===#
    if run_options.add_noise == 1:
        state_obs_train, state_obs_test, noise_regularization_matrix\
        = add_noise(run_options, state_obs_train, state_obs_test, load_data_train_flag = 1)
    else:
        noise_regularization_matrix = tf.eye(obs_dimensions)

    #=== Prior ===#
    if hyperp.penalty_prior != 0:
        load_flag = 1
    else:
        load_flag = 0
    prior_mean, _, _,\
    prior_covariance_cholesky_inverse\
    = load_prior(run_options, file_paths,
                 load_mean = 1,
                 load_covariance = 0,
                 load_covariance_cholesky = 0,
                 load_covariance_cholesky_inverse = load_flag)

    ############################
    #   Objective Functional   #
    ############################
    @use_named_args(space)
    def objective_functional(**hyperp_of_interest_dict):
        #=== Assign Hyperparameters of Interest ===#
        for key, val in hyperp_of_interest_dict.items():
            setattr(hyperp, key, val)

        #=== Construct Validation Set and Batches ===#
        if run_options.standard_autoencoder == 1:
            input_and_latent_train, input_and_latent_val, input_and_latent_test,\
            hyperp.num_data_train, num_data_val, run_options.num_data_test,\
            num_batches_train, num_batches_val, num_batches_test,\
            input_dimensions\
            = form_train_val_test_tf_batches(parameter_train, state_obs_train,
                    parameter_test, state_obs_test,
                    hyperp.batch_size, run_options.random_seed)
        if run_options.reverse_autoencoder == 1:
            input_and_latent_train, input_and_latent_val, input_and_latent_test,\
            hyperp.num_data_train, num_data_val, run_options.num_data_test,\
            num_batches_train, num_batches_val, num_batches_test,\
            input_dimensions\
            = form_train_val_test_tf_batches(state_obs_train, parameter_train,
                    state_obs_test, parameter_test,
                    hyperp.batch_size, run_options.random_seed)

        #=== Update File Paths with New Hyperparameters ===#
        file_paths = FilePathsHyperparameterOptimization(hyperp, run_options,
                                                     autoencoder_loss, project_name,
                                                     data_options, dataset_directory)

        #=== Data and Latent Dimensions of Autoencoder ===#
        if run_options.standard_autoencoder == 1:
            input_dimensions = run_options.parameter_dimensions
            latent_dimensions = obs_dimensions
        if run_options.reverse_autoencoder == 1:
            input_dimensions = obs_dimensions
            latent_dimensions = run_options.parameter_dimensions

        #=== Load FEM Matrices ===#
        _, prestiffness, boundary_matrix, load_vector =\
                load_FEM_matrices_tf(run_options, file_paths,
                                    load_premass = 0,
                                    load_prestiffness = 1)

        #=== Construct Forward Model ===#
        forward_model = FEMPrematricesPoisson2D(run_options, file_paths,
                                                obs_indices,
                                                prestiffness,
                                                boundary_matrix, load_vector)

        #=== Neural Network Regularizers ===#
        kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
        bias_initializer = 'zeros'

        #=== Non-distributed Training ===#
        if run_options.distributed_training == 0:
            #=== Neural Network ===#
            NN = AutoencoderFwdInv(hyperp, run_options,
                                   input_dimensions, latent_dimensions,
                                   kernel_initializer, bias_initializer,
                                   positivity_constraint_log_exp)

            #=== Optimizer ===#
            optimizer = tf.keras.optimizers.Adam()

            #=== Training ===#
            optimize(hyperp, run_options, file_paths,
                    NN, optimizer,
                    loss_penalized_difference, relative_error,
                    input_and_latent_train, input_and_latent_val, input_and_latent_test,
                    input_dimensions,
                    num_batches_train,
                    loss_weighted_penalized_difference, noise_regularization_matrix,
                    reg_prior, prior_mean, prior_covariance_cholesky_inverse,
                    positivity_constraint_log_exp,
                    forward_model.solve_PDE_prematrices_sparse)

        #=== Distributed Training ===#
        if run_options.distributed_training == 1:
            dist_strategy = tf.distribute.MirroredStrategy()
            with dist_strategy.scope():
                #=== Neural Network ===#
                NN = AutoencoderFwdInv(hyperp, run_options,
                                       input_dimensions, latent_dimensions,
                                       kernel_initializer, bias_initializer,
                                       positivity_constraint_log_exp)

                #=== Optimizer ===#
                optimizer = tf.keras.optimizers.Adam()

            #=== Training ===#
            optimize_distributed(dist_strategy,
                    hyperp, run_options, file_paths,
                    NN, optimizer,
                    loss_penalized_difference, relative_error,
                    input_and_latent_train, input_and_latent_val, input_and_latent_test,
                    input_dimensions,
                    num_batches_train,
                    loss_weighted_penalized_difference, noise_regularization_matrix,
                    reg_prior, prior_mean, prior_covariance_cholesky_inverse,
                    positivity_constraint_log_exp,
                    forward_model.solve_PDE_prematrices_sparse)

        #=== Loading Metrics For Output ===#
        print('Loading Metrics')
        df_metrics = pd.read_csv(file_paths.NN_savefile_name + "_metrics" + '.csv')
        array_metrics = df_metrics.to_numpy()
        storage_array_loss_val = array_metrics[:,5]

        return storage_array_loss_val[-1]

    ################################
    #   Optimize Hyperparameters   #
    ################################
    hyperp_opt_result = gp_minimize(objective_functional, space, n_calls=n_calls, random_state=None)

    return hyperp_opt_result
