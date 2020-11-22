import os
import sys

import tensorflow as tf

# Import src code
from utils_training.form_train_val_test import form_train_val_test_tf_batches
from neural_networks.nn_ae import AE
from utils_training.functionals import\
        loss_penalized_difference, loss_weighted_penalized_difference,\
        relative_error
from optimize.optimize_custom_ae_model_aware import optimize
from optimize.optimize_distributed_custom_ae_model_aware import optimize_distributed
from utils_misc.positivity_constraints import positivity_constraint_exp,\
                                              positivity_constraint_log_exp

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                  Training                                   #
###############################################################################
def trainer_custom(hyperp, options, filepaths,
                   data_dict, prior_dict):

    #=== GPU Settings ===#
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if options.distributed_training == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = options.which_gpu
    if options.distributed_training == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = options.dist_which_gpus
        gpus = tf.config.experimental.list_physical_devices('GPU')

    #=== Construct Validation Set and Batches ===#
    if options.standard_autoencoder == 1:
        input_and_latent_train, input_and_latent_val, input_and_latent_test,\
        num_batches_train, num_batches_val, num_batches_test\
        = form_train_val_test_tf_batches(
                data_dict["parameter_train"], data_dict["state_obs_train"],
                data_dict["parameter_test"], data_dict["state_obs_test"],
                hyperp.batch_size, options.random_seed)
    if options.reverse_autoencoder == 1:
        input_and_latent_train, input_and_latent_val, input_and_latent_test,\
        num_batches_train, num_batches_val, num_batches_test\
        = form_train_val_test_tf_batches(
                data_dict["state_obs_train"], data_dict["parameter_train"],
                data_dict["state_obs_test"], data_dict["parameter_test"],
                hyperp.batch_size, options.random_seed)

    #=== Data and Latent Dimensions of Autoencoder ===#
    if options.standard_autoencoder == 1:
        input_dimensions = options.parameter_dimensions
        latent_dimensions = data_dict["obs_dimensions"]
    if options.reverse_autoencoder == 1:
        input_dimensions = data_dict["obs_dimensions"]
        latent_dimensions = options.parameter_dimensions

    #=== Neural Network Regularizers ===#
    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
    bias_initializer = 'zeros'

    #=== Non-distributed Training ===#
    if options.distributed_training == 0:
        #=== Neural Network ===#
        NN = AE(hyperp, options,
                input_dimensions, latent_dimensions,
                kernel_initializer, bias_initializer,
                tf.identity)

        #=== Optimizer ===#
        optimizer = tf.keras.optimizers.Adam()

        #=== Training ===#
        optimize(hyperp, options, filepaths,
                 NN, optimizer,
                 input_and_latent_train, input_and_latent_val, input_and_latent_test,
                 input_dimensions, num_batches_train,
                 loss_penalized_difference, loss_weighted_penalized_difference, relative_error,
                 data_dict["noise_regularization_matrix"],
                 prior_dict["prior_mean"], prior_dict["prior_covariance_cholesky_inverse"],
                 tf.identity)

    #=== Distributed Training ===#
    if options.distributed_training == 1:
        dist_strategy = tf.distribute.MirroredStrategy()
        with dist_strategy.scope():
            #=== Neural Network ===#
            NN = AE(hyperp, options,
                    input_dimensions, latent_dimensions,
                    kernel_initializer, bias_initializer,
                    tf.identity)

            #=== Optimizer ===#
            optimizer = tf.keras.optimizers.Adam()

        #=== Training ===#
        optimize_distributed(dist_strategy,
                hyperp, options, filepaths,
                NN, optimizer,
                input_and_latent_train, input_and_latent_val, input_and_latent_test,
                input_dimensions, num_batches_train,
                loss_penalized_difference, loss_weighted_penalized_difference, relative_error,
                data_dict["noise_regularization_matrix"],
                prior_dict["prior_mean"], prior_dict["prior_covariance_cholesky_inverse"],
                tf.identity)
