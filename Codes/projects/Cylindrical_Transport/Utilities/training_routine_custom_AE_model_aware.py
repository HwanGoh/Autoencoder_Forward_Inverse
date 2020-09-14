import os
import sys

import tensorflow as tf
import numpy as np
import pandas as pd

# Import src code
from get_train_and_test_data import load_train_and_test_data
from get_prior import load_prior
from form_train_val_test import form_train_val_test_tf_batches
from NN_AE_Fwd_Inv import AutoencoderFwdInv
from loss_and_relative_errors import loss_penalized_difference, relative_error,\
        reg_prior
from optimize_custom_AE_model_aware import optimize
from optimize_distributed_custom_AE_model_aware import optimize_distributed

###############################################################################
#                                 Training                                    #
###############################################################################
def trainer_custom(hyperp, run_options, file_paths):
    #=== GPU Settings ===#
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if run_options.use_distributed_training == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = run_options.which_gpu
    if run_options.use_distributed_training == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = run_options.dist_which_gpus
        gpus = tf.config.experimental.list_physical_devices('GPU')

    #=== Load Data ===#
    parameter_train, state_obs_train,\
    parameter_test, state_obs_test,\
    = load_train_and_test_data(file_paths,
            run_options.num_data_train, run_options.num_data_test,
            run_options.parameter_dimensions, run_options.state_dimensions,
            load_data_train_flag = 1,
            normalize_input_flag = 0, normalize_output_flag = 0)

    #=== Construct Validation Set and Batches ===#
    if run_options.use_standard_autoencoder == 1:
        input_and_latent_train, input_and_latent_val, input_and_latent_test,\
        run_options.num_data_train, num_data_val, run_options.num_data_test,\
        num_batches_train, num_batches_val, num_batches_test,\
        input_dimensions\
        = form_train_val_test_tf_batches(parameter_train, state_obs_train,
                parameter_test, state_obs_test,
                hyperp.batch_size, run_options.random_seed)
    if run_options.use_reverse_autoencoder == 1:
        input_and_latent_train, input_and_latent_val, input_and_latent_test,\
        run_options.num_data_train, num_data_val, run_options.num_data_test,\
        num_batches_train, num_batches_val, num_batches_test,\
        input_dimensions\
        = form_train_val_test_tf_batches(state_obs_train, parameter_train,
                state_obs_test, parameter_test,
                hyperp.batch_size, run_options.random_seed)

    #=== Data and Latent Dimensions of Autoencoder ===#
    if run_options.use_standard_autoencoder == 1:
        latent_dimensions = run_options.state_dimensions
    if run_options.use_reverse_autoencoder == 1:
        latent_dimensions = run_options.parameter_dimensions

    #=== Prior ===#
    if hyperp.penalty_prior != 0:
        load_flag = 1
    else:
        load_flag = 0
    prior_mean,\
    prior_covariance, prior_covariance_cholesky\
    = load_prior(run_options, file_paths,
                 load_mean = 0,
                 load_covariance = 0, load_covariance_cholesky = load_flag)

    #=== Neural Network Regularizers ===#
    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
    bias_initializer = 'zeros'

    #=== Non-distributed Training ===#
    if run_options.use_distributed_training == 0:
        #=== Neural Network ===#
        NN = AutoencoderFwdInv(hyperp, input_dimensions, latent_dimensions,
                               kernel_initializer, bias_initializer)

        #=== Optimizer ===#
        optimizer = tf.keras.optimizers.Adam()

        #=== Training ===#
        optimize(hyperp, run_options, file_paths,
                NN, optimizer,
                loss_penalized_difference, relative_error,
                reg_prior, prior_mean, prior_covariance_cholesky,
                input_and_latent_train, input_and_latent_val, input_and_latent_test,
                input_dimensions, num_batches_train)

    #=== Distributed Training ===#
    if run_options.use_distributed_training == 1:
        dist_strategy = tf.distribute.MirroredStrategy()
        with dist_strategy.scope():
            #=== Neural Network ===#
            NN = AutoencoderFwdInv(hyperp, input_dimensions, latent_dimensions,
                                   kernel_initializer, bias_initializer)

            #=== Optimizer ===#
            optimizer = tf.keras.optimizers.Adam()

        #=== Training ===#
        optimize_distributed(dist_strategy,
                hyperp, run_options, file_paths,
                NN, optimizer,
                loss_penalized_difference, relative_error,
                reg_prior, prior_mean, prior_covariance_cholesky,
                input_and_latent_train, input_and_latent_val, input_and_latent_test,
                input_dimensions, num_batches_train)
