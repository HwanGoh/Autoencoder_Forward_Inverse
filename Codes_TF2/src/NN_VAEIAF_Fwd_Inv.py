#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:45:14 2020

@author: hwan
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.initializers import RandomNormal
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                           Variational Autoencoder                           #
###############################################################################
class VAEIAFFwdInv(tf.keras.Model):
    def __init__(self, hyperp, run_options,
                 input_dimensions, latent_dimensions,
                 kernel_initializer, bias_initializer,
                 kernel_initializer_IAF, bias_initializer_IAF,
                 positivity_constraint):
        super(VAEIAFFwdInv, self).__init__()

        #=== Define Architecture and Create Layer Storage ===#
        self.architecture = [input_dimensions] +\
                [hyperp.num_hidden_nodes]*hyperp.num_hidden_layers + [input_dimensions]
        self.architecture[hyperp.truncation_layer] = latent_dimensions + latent_dimensions

        #=== Define Other Attributes ===#
        self.run_options = run_options
        self.activations = ['not required'] +\
                [hyperp.activation]*hyperp.num_hidden_layers + ['linear']
        self.activations[hyperp.truncation_layer] = 'linear' # This is the identity activation
        self.positivity_constraint = positivity_constraint

        #=== Encoder, IAF Chain and Decoder ===#
        self.encoder = Encoder(hyperp.truncation_layer,
                               self.architecture, self.activations,
                               kernel_initializer, bias_initializer)
        self.IAF_chain_posterior = IAFChainPosterior(run_options.IAF_LSTM_update,
                                                     hyperp.num_IAF_transforms,
                                                     hyperp.num_hidden_nodes_IAF,
                                                     hyperp.activation_IAF,
                                                     kernel_initializer_IAF, bias_initializer_IAF)
        if self.run_options.model_aware == 1:
            self.decoder = Decoder(hyperp.truncation_layer,
                                   self.architecture, self.activations,
                                   kernel_initializer, bias_initializer,
                                   len(self.architecture))

    #=== Variational Autoencoder Propagation ===#
    def reparameterize(self, mean, log_var):
        return self.IAF_chain_posterior((mean, log_var),
                                        sample_flag = True, infer_flag = False)

    def call(self, X):
        post_mean, log_post_var = self.encoder(X)
        if self.run_options.model_augmented == 1:
            return reparameterize(post_mean, log_post_var)
        if self.run_options.model_aware == 1:
            z = self.reparameterize(post_mean, log_post_var)
            likelihood_mean = self.decoder(self.positivity_constraint(z))
            return likelihood_mean

###############################################################################
#                                  Encoder                                    #
###############################################################################
class Encoder(tf.keras.layers.Layer):
    def __init__(self, truncation_layer,
                 architecture,
                 activations,
                 kernel_initializer, bias_initializer):
        super(Encoder, self).__init__()

        self.hidden_layers_encoder = []
        for l in range(1, truncation_layer+1):
            hidden_layer_encoder = tf.keras.layers.Dense(units = architecture[l],
                                                         activation = activations[l],
                                                         use_bias = True,
                                                         kernel_initializer = kernel_initializer,
                                                         bias_initializer = bias_initializer,
                                                         name = "W" + str(l))
            self.hidden_layers_encoder.append(hidden_layer_encoder)

    def call(self, X):
        for hidden_layer in self.hidden_layers_encoder:
            X = hidden_layer(X)
        post_mean, log_post_var = tf.split(X, num_or_size_splits=2, axis=1)
        return post_mean, log_post_var

###############################################################################
#                                  Decoder                                    #
###############################################################################
class Decoder(tf.keras.layers.Layer):
    def __init__(self, truncation_layer,
                 architecture,
                 activations,
                 kernel_initializer, bias_initializer,
                 num_layers):
        super(Decoder, self).__init__()

        self.hidden_layers_decoder = []
        for l in range(truncation_layer+1, num_layers):
            hidden_layer_decoder = tf.keras.layers.Dense(units = architecture[l],
                                                         activation = activations[l],
                                                         use_bias = True,
                                                         kernel_initializer = kernel_initializer,
                                                         bias_initializer = bias_initializer,
                                                         name = "W" + str(l))
            self.hidden_layers_decoder.append(hidden_layer_decoder)

    def call(self, X):
        for hidden_layer in self.hidden_layers_decoder:
            X = hidden_layer(X)
        return X

###############################################################################
#                    Chain of Inverse Autoregressive Flow                     #
###############################################################################
class IAFChainPosterior(tf.keras.layers.Layer):
    def __init__(self, IAF_LSTM_update_flag,
                 num_IAF_transforms,
                 hidden_units,
                 activation,
                 kernel_initializer, bias_initializer):
        super(IAFChainPosterior, self).__init__()

        #=== Attributes ===#
        self.IAF_LSTM_update_flag = IAF_LSTM_update_flag
        self.num_IAF_transforms = num_IAF_transforms
        self.hidden_units = hidden_units
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        #=== IAF Chain ===#
        latent_dimensions = input_shape[0][1]
        self.event_shape = [latent_dimensions]
        bijectors_list = []
        if self.IAF_LSTM_update_flag == 0:
            for i in range(0, self.num_IAF_transforms):
                bijectors_list.append(tfb.Invert(
                    tfb.MaskedAutoregressiveFlow(
                        shift_and_log_scale_fn = Made(params=2,
                                                      event_shape = self.event_shape,
                                                      hidden_units = self.hidden_units,
                                                      activation = self.activation,
                                                      kernel_initializer = self.kernel_initializer,
                                                      bias_initializer = self.bias_initializer))))
                bijectors_list.append(tfb.Permute(list(reversed(range(latent_dimensions)))))
        self.IAF_chain = tfb.Chain(bijectors_list[:-1])

    def call(self, inputs, sample_flag = True, infer_flag = False):
        mean = inputs[0]
        log_var = inputs[1]

        #=== Base Distribution ===#
        base_distribution = tfd.MultivariateNormalDiag(loc = mean,
                                                       scale_diag = tf.exp(0.5*log_var))
        #=== Transformed Distribution ===#
        self.distribution = tfd.TransformedDistribution(distribution = base_distribution,
                                                        bijector = self.IAF_chain)
        #=== Inference and Sampling ===#
        sample_draw = self.distribution.sample()
        if sample_flag == 1:
            return sample_draw
        if infer_flag == 1:
            return self.distribution.log_prob(sample_draw)

###############################################################################
#                          Masked Autoregressive Flow                         #
###############################################################################
class Made(tf.keras.layers.Layer):
    def __init__(self, params,
                 event_shape,
                 hidden_units,
                 activation,
                 kernel_initializer, bias_initializer):
        super(Made, self).__init__()

        self.network = tfb.AutoregressiveNetwork(params = params,
                                                 event_shape = event_shape,
                                                 hidden_units = [hidden_units, hidden_units],
                                                 activation = activation,
                                                 kernel_initializer = kernel_initializer,
                                                 bias_initializer = bias_initializer)

    def call(self, X):
        mean, log_var = tf.unstack(self.network(X), num=2, axis=-1)
        return mean, log_var
