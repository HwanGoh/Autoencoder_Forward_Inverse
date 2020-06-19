import os
import sys

import tensorflow as tf
import numpy as np
import pandas as pd

# Import src code
from get_train_and_test_data import load_train_and_test_data
from form_train_val_test import form_train_val_test_tf_batches
from NN_Autoencoder_Fwd_Inv import AutoencoderFwdInv
from loss_and_relative_errors import loss_penalized_difference,\
        loss_forward_model, relative_error, reg_prior
from optimize_custom_model_induced_autoencoder import optimize
from optimize_distributed_custom_model_aware_autoencoder import optimize_distributed

###############################################################################
#                                  Training                                   #
###############################################################################
def trainer_custom(hyperp, run_options, file_paths):
    #=== GPU Settings ===#
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if run_options.use_distributed_training == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = run_options.which_gpu
        GLOBAL_BATCH_SIZE = hyperp.batch_size
    if run_options.use_distributed_training == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = run_options.dist_which_gpus
        gpus = tf.config.experimental.list_physical_devices('GPU')
        GLOBAL_BATCH_SIZE = hyperp.batch_size * len(gpus)

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
                GLOBAL_BATCH_SIZE, run_options.random_seed)
    if run_options.use_reverse_autoencoder == 1:
        input_and_latent_train, input_and_latent_val, input_and_latent_test,\
        run_options.num_data_train, num_data_val, run_options.num_data_test,\
        num_batches_train, num_batches_val, num_batches_test,\
        input_dimensions\
        = form_train_val_test_tf_batches(state_obs_train, parameter_train,
                state_obs_test, parameter_test,
                GLOBAL_BATCH_SIZE, run_options.random_seed)

    #=== Data and Latent Dimensions of Autoencoder ===#
    if run_options.use_standard_autoencoder == 1:
        data_dimension = run_options.parameter_dimensions
        if hyperp.data_type == 'full':
            latent_dimension = run_options.full_domain_dimensions
        if hyperp.data_type == 'bnd':
            latent_dimension = len(obs_indices)
    if run_options.use_reverse_autoencoder == 1:
        if hyperp.data_type == 'full':
            data_dimension = run_options.full_domain_dimensions
        if hyperp.data_type == 'bnd':
            data_dimension = len(obs_indices)
        latent_dimension = run_options.parameter_dimensions

    #=== Prior Regularization ===#
    if hyperp.penalty_prior != 0:
        print('Loading Prior Matrix')
        df_L_pr = pd.read_csv(file_paths.prior_chol_savefilepath + '.csv')
        L_pr = df_L_pr.to_numpy()
        L_pr = L_pr.reshape((run_options.full_domain_dimensions, run_options.full_domain_dimensions))
        L_pr = L_pr.astype(np.float32)
    else:
        L_pr = 0.0

    #=== Neural Network Regularizers ===#
    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
    bias_initializer = 'zeros'

    #=== Non-distributed Training ===#
    if run_options.use_distributed_training == 0:
        #=== Neural Network ===#
        NN = AutoencoderFwdInv(hyperp, data_dimension, latent_dimension,
                               kernel_initializer, bias_initializer)

        #=== Optimizer ===#
        optimizer = tf.keras.optimizers.Adam()

        #=== Training ===#
        optimize(hyperp, run_options, file_paths,
                NN, optimizer,
                obs_indices,
                loss_penalized_difference, loss_forward_model,
                relative_error, reg_prior, L_pr,
                input_and_latent_train, input_and_latent_val, input_and_latent_test,
                input_dimensions,
                num_batches_train)

    #=== Distributed Training ===#
    if run_options.use_distributed_training == 1:
        dist_strategy = tf.distribute.MirroredStrategy()
        with dist_strategy.scope():
            #=== Neural Network ===#
            NN = AutoencoderFwdInv(hyperp, data_dimension, latent_dimension,
                                   kernel_initializer, bias_initializer)

            #=== Optimizer ===#
            optimizer = tf.keras.optimizers.Adam()

        #=== Training ===#
        optimize_distributed(dist_strategy, GLOBAL_BATCH_SIZE,
                hyperp, run_options, file_paths,
                NN, optimizer,
                obs_indices,
                loss_penalized_difference, loss_forward_model,
                relative_error, reg_prior, L_pr,
                input_and_latent_train, input_and_latent_val, input_and_latent_test,
                input_dimensions,
                num_batches_train)
