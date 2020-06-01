import os
import sys
sys.path.insert(0, os.path.realpath('../../src'))

import tensorflow as tf
import numpy as np
import pandas as pd

# Load FilePaths class and data retrieval function
from Utilities.file_paths_VAE import FilePathsHyperparameterOptimization
from Utilities.get_thermal_fin_data import load_thermal_fin_data

# Import src code
from form_train_val_test_batches import form_train_val_test_batches
from NN_VAE_Fwd_Inv import VAEFwdInv
from loss_and_relative_errors import loss_penalized_difference, KLD_diagonal_post_cov, KLD_full_post_cov, relative_error
from optimize_model_aware_VAE import optimize
from optimize_distributed_model_aware_VAE import optimize_distributed

# Import skopt code
from skopt.utils import use_named_args
from skopt import gp_minimize

###############################################################################
#                                 Training                                    #
###############################################################################
def trainer(hyperp, run_options, file_paths, n_calls, space,
        autoencoder_loss, dataset_directory):

    #=== GPU Settings ===#
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if run_options.use_distributed_training == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = run_options.which_gpu
    if run_options.use_distributed_training == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = run_options.dist_which_gpus
        gpus = tf.config.experimental.list_physical_devices('GPU')

    #=== Load Data ===#
    obs_indices, parameter_train, state_obs_train,\
    parameter_test, state_obs_test,\
    data_input_shape_temp, parameter_dimension\
    = load_thermal_fin_data(file_paths, run_options.num_data_train,
            run_options.num_data_test, run_options.parameter_dimensions)

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
        num_batches_train, num_batches_val, num_batches_test\
        = form_train_val_test_batches(state_obs_train, parameter_train,
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

        #=== Non-distributed Training ===#
        if run_options.use_distributed_training == 0:
            #=== Neural Network ===#
            NN = VAEFwdInv(hyperp, data_dimension, latent_dimension)

            #=== Training ===#
            optimize(hyperp, run_options, file_paths, NN,
                    loss_penalized_difference, KLD_loss, relative_error, prior_cov,
                    data_and_latent_train, data_and_latent_val, data_and_latent_test,
                    data_dimension, latent_dimension, num_batches_train)

        #=== Distributed Training ===#
        if run_options.use_distributed_training == 1:
            dist_strategy = tf.distribute.MirroredStrategy()
            with dist_strategy.scope():
                #=== Neural Network ===#
                NN = VAEFwdInv(hyperp, data_dimension, latent_dimension)

            #=== Training ===#
            optimize_distributed(dist_strategy, GLOBAL_BATCH_SIZE,
                    hyperp, run_options, file_paths, NN,
                    loss_penalized_difference, KLD_loss, relative_error, prior_cov,
                    data_and_latent_train, data_and_latent_val, data_and_latent_test,
                    data_dimension, latent_dimension, num_batches_train)

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
