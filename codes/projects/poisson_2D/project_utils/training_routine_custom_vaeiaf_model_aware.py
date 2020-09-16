import os
import sys

import tensorflow as tf
import numpy as np
import pandas as pd

# Import src code
from get_train_and_test_data import load_train_and_test_data
from add_noise import add_noise
from form_train_val_test import form_train_val_test_tf_batches
from get_prior import load_prior
from NN_VAEIAF_Fwd_Inv import VAEIAFFwdInv
from loss_and_relative_errors import\
        loss_penalized_difference, loss_weighted_penalized_difference, relative_error
from optimize_custom_VAEIAF_model_aware import optimize
from optimize_distributed_custom_VAEIAF_model_aware import optimize_distributed
from positivity_constraints import positivity_constraint_log_exp

import pdb

###############################################################################
#                                  Training                                   #
###############################################################################
def trainer_custom(hyperp, options, file_paths):
    #=== GPU Settings ===#
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if options.distributed_training == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = options.which_gpu
    if options.distributed_training == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = options.dist_which_gpus
        gpus = tf.config.experimental.list_physical_devices('GPU')

    #=== Load Observation Indices ===#
    if options.obs_type == 'full':
        obs_dimensions = options.parameter_dimensions
        obs_indices = []
    if options.obs_type == 'obs':
        obs_dimensions = options.num_obs_points
        print('Loading Boundary Indices')
        df_obs_indices = pd.read_csv(file_paths.obs_indices_savefilepath + '.csv')
        obs_indices = df_obs_indices.to_numpy()

    #=== Load Data ===#
    parameter_train, state_obs_train,\
    parameter_test, state_obs_test\
    = load_train_and_test_data(file_paths,
            hyperp.num_data_train, options.num_data_test,
            options.parameter_dimensions, obs_dimensions,
            load_data_train_flag = 1,
            normalize_input_flag = 0, normalize_output_flag = 0)

    #=== Add Noise to Data ===#
    if options.add_noise == 1:
        state_obs_train, state_obs_test, noise_regularization_matrix\
        = add_noise(options, state_obs_train, state_obs_test, load_data_train_flag = 1)
    else:
        noise_regularization_matrix = tf.eye(obs_dimensions)

    #=== Construct Validation Set and Batches ===#
    input_and_latent_train, input_and_latent_val, input_and_latent_test,\
    num_batches_train, num_batches_val, num_batches_test\
    = form_train_val_test_tf_batches(state_obs_train, parameter_train,
            state_obs_test, parameter_test,
            hyperp.batch_size, options.random_seed)

    #=== Data and Latent Dimensions of Autoencoder ===#
    input_dimensions = obs_dimensions
    latent_dimensions = options.parameter_dimensions

    #=== Prior ===#
    prior_mean,\
    _ , _, prior_covariance_cholesky_inverse\
    = load_prior(options, file_paths,
                 load_mean = 1,
                 load_covariance = 0,
                 load_covariance_cholesky = 0, load_covariance_cholesky_inverse = 1)

    #=== Neural Network Regularizers ===#
    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
    bias_initializer = 'zeros'
    kernel_initializer_IAF = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
    bias_initializer_IAF = 'zeros'

    #=== Non-distributed Training ===#
    if options.distributed_training == 0:
        #=== Neural Network ===#
        NN = VAEIAFFwdInv(hyperp, options,
                          input_dimensions, latent_dimensions,
                          kernel_initializer, bias_initializer,
                          kernel_initializer_IAF, bias_initializer_IAF,
                          positivity_constraint_log_exp)

        #=== Optimizer ===#
        optimizer = tf.keras.optimizers.Adam()

        #=== Training ===#
        optimize(hyperp, options, file_paths,
                 NN, optimizer,
                 loss_penalized_difference, relative_error,
                 prior_mean, prior_covariance_cholesky_inverse,
                 input_and_latent_train, input_and_latent_val, input_and_latent_test,
                 input_dimensions, latent_dimensions,
                 num_batches_train,
                 loss_weighted_penalized_difference, noise_regularization_matrix,
                 positivity_constraint_log_exp)

    #=== Distributed Training ===#
    if options.distributed_training == 1:
        dist_strategy = tf.distribute.MirroredStrategy()
        with dist_strategy.scope():
            #=== Neural Network ===#
            NN = VAEIAFFwdInv(hyperp, options,
                              input_dimensions, latent_dimensions,
                              kernel_initializer, bias_initializer)

            #=== Optimizer ===#
            optimizer = tf.keras.optimizers.Adam()

        #=== Training ===#
        optimize_distributed(dist_strategy,
                hyperp, options, file_paths,
                NN, optimizer,
                loss_penalized_difference, loss_posterior_IAF, relative_error,
                prior_mean, prior_covariance,
                input_and_latent_train, input_and_latent_val, input_and_latent_test,
                input_dimensions, latent_dimensions,
                num_batches_train,
                loss_weighted_penalized_difference, noise_regularization_matrix,
                positivity_constraint_log_exp)
