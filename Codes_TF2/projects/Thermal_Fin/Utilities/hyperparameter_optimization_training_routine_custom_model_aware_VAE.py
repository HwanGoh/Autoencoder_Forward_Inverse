import os
import sys
sys.path.insert(0, os.path.realpath('../../src'))

import tensorflow as tf
import numpy as np
import pandas as pd

# Load FilePaths class and data retrieval function
from Utilities.file_paths_VAE import FilePathsHyperparameterOptimization

# Import src code
from get_train_and_test_data import load_train_and_test_data
from form_train_val_test import form_train_val_test_tf_batches
from NN_VAE_Fwd_Inv import VAEFwdInv
from loss_and_relative_errors import loss_penalized_difference,\
        KLD_diagonal_post_cov, KLD_full_post_cov, relative_error
from optimize_custom_model_aware_VAE import optimize
from optimize_distributed_custom_model_aware_VAE import optimize_distributed

# Import skopt code
from skopt.utils import use_named_args
from skopt import gp_minimize

###############################################################################
#                                 Training                                    #
###############################################################################
def trainer_custom(hyperp, run_options, file_paths,
        n_calls, space,
        autoencoder_loss, dataset_directory):

    #=== GPU Settings ===#
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if run_options.use_distributed_training == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = run_options.which_gpu
    if run_options.use_distributed_training == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = run_options.dist_which_gpus
        gpus = tf.config.experimental.list_physical_devices('GPU')

    #=== Load observation indices ===#
    print('Loading Boundary Indices')
    df_obs_indices = pd.read_csv(file_paths.observation_indices_savefilepath + '.csv')
    obs_indices = df_obs_indices.to_numpy()
    run_options.state_dimensions = len(obs_indices)

    #=== Load Data ===#
    parameter_train, state_obs_train,\
    parameter_test, state_obs_test,\
    = load_train_and_test_data(file_paths,
            run_options.num_data_train, run_options.num_data_test,
            run_options.parameter_dimensions, run_options.state_dimensions,
            load_data_train_flag = 1)
    output_dimensions = run_options.state_dimensions

    ############################
    #   Objective Functional   #
    ############################
    @use_named_args(space)
    def objective_functional(**hyperp_of_interest_objective_args_tuple):
        #=== Assign Hyperparameters of Interest ===#
        for key, val in hyperp_of_interest_objective_args_tuple.items():
            setattr(hyperp, key, val)
        hyperp.truncation_layer = int(np.ceil(hyperp.num_hidden_layers/2))

        #=== Update File Paths with New Hyperparameters ===#
        file_paths = FilePathsHyperparameterOptimization(hyperp, run_options,
                autoencoder_loss, dataset_directory)

        #=== Construct Validation Set and Batches ===#
        if run_options.use_distributed_training == 0:
            GLOBAL_BATCH_SIZE = hyperp.batch_size
        if run_options.use_distributed_training == 1:
            GLOBAL_BATCH_SIZE = hyperp.batch_size * len(gpus)
        data_and_latent_train, data_and_latent_val, data_and_latent_test,\
        run_options.num_data_train, num_data_val, run_options.num_data_test,\
        num_batches_train, num_batches_val, num_batches_test,\
        data_input_shape\
        = form_train_val_test_tf_batches(state_obs_train, parameter_train,
                state_obs_test, parameter_test,
                GLOBAL_BATCH_SIZE, run_options.random_seed)

        #=== Data and Latent Dimensions of Autoencoder ===#
        if hyperp.data_type == 'full':
            data_dimension = run_options.full_domain_dimensions
        if hyperp.data_type == 'bnd':
            data_dimension = len(obs_indices)
        latent_dimension = parameter_dimension

        #=== Posterior Covariance Loss Functional ===#
        if run_options.diagonal_posterior_covariance == 1:
            KLD_loss = KLD_diagonal_post_cov
        if run_options.full_posterior_covariance == 1:
            KLD_loss = KLD_full_post_cov

        #=== Prior Regularization ===#
        print('Loading Prior Matrix')
        df_cov = pd.read_csv(file_paths.prior_savefilepath + '.csv')
        prior_cov = df_cov.to_numpy()
        prior_cov = prior_cov.reshape((run_options.full_domain_dimensions,
            run_options.full_domain_dimensions))
        prior_cov = prior_cov.astype(np.float32)

        #=== Neural Network Regularizers ===#
        kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
        bias_initializer = 'zeros'

        #=== Non-distributed Training ===#
        if run_options.use_distributed_training == 0:
            #=== Neural Network ===#
            NN = VAEFwdInv(hyperp, data_dimension, latent_dimension,
                           kernel_initializer, bias_initializer)

            #=== Optimizer ===#
            optimizer = tf.keras.optimizers.Adam()

            #=== Training ===#
            optimize(hyperp, run_options, file_paths,
                    NN, optimizer,
                    loss_penalized_difference, KLD_loss, relative_error, prior_cov,
                    data_and_latent_train, data_and_latent_val, data_and_latent_test,
                    data_input_shape, latent_dimension,
                    num_batches_train)

        #=== Distributed Training ===#
        if run_options.use_distributed_training == 1:
            dist_strategy = tf.distribute.MirroredStrategy()
            with dist_strategy.scope():
                #=== Neural Network ===#
                NN = VAEFwdInv(hyperp, data_dimension, latent_dimension,
                               kernel_initializer, bias_initializer )

                #=== Optimizer ===#
                optimizer = tf.keras.optimizers.Adam()

            #=== Training ===#
            optimize_distributed(dist_strategy, GLOBAL_BATCH_SIZE,
                    hyperp, run_options, file_paths,
                    NN, optimizer,
                    loss_penalized_difference, KLD_loss, relative_error, prior_cov,
                    data_and_latent_train, data_and_latent_val, data_and_latent_test,
                    data_input_dimension, latent_dimension,
                    num_batches_train)

        #=== Loading Metrics For Output ===#
        print('Loading Metrics')
        df_metrics = pd.read_csv(file_paths.NN_savefile_name + "_metrics" + '.csv')
        array_metrics = df_metrics.to_numpy()
        storage_array_loss_val = array_metrics[:,4]

        return storage_array_loss_val[-1]

    ################################
    #   Optimize Hyperparameters   #
    ################################
    hyperp_opt_result = gp_minimize(objective_functional, space, n_calls=n_calls, random_state=None)

    return hyperp_opt_result
