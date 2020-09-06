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
        print(self.architecture)

        #=== Define Other Attributes ===#
        self.run_options = run_options
        self.hidden_layers_decoder = [] # This will be a list of layers
        activation = hyperp.activation
        self.activations = ['not required'] + [activation]*hyperp.num_hidden_layers + ['linear']
        self.activations[hyperp.truncation_layer] = 'linear' # This is the identity activation
        self.positivity_constraint = positivity_constraint

        #=== Encoder, IAF Chain and Decoder ===#
        self.encoder = Encoder(hyperp.truncation_layer,
                               self.architecture, self.activations,
                               kernel_initializer, bias_initializer)
        self.IAF_chain = IAFChain(hyperp.num_IAF_transforms,
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
        eps = tf.random.normal(shape=mean.shape)
        return mean + eps*tf.exp(log_var*0.5)

    def call(self, X):
        post_mean, log_post_var = self.encoder(X)
        if self.run_options.model_augmented == 1:
            return post_mean, log_post_var
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
        self.hidden_layers_encoder = [] # This will be a list of layers
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
        self.hidden_layers_decoder = [] # This will be a list of layers
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
#                          Masked Autoregressive Flow                         #
###############################################################################
class Made(tf.keras.layers.Layer):
    def __init__(self, params,
                 event_shape,
                 hidden_units,
                 activation,
                 kernel_initializer, bias_initializer):

        super(Made, self).__init__(name=name)

        self.params = params
        self.event_shape = event_shape
        self.hidden_units = hidden_units
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.network = tfb.AutoregressiveNetwork(params=params,
                                                 event_shape=event_shape,
                                                 hidden_units=hidden_units,
                                                 activation=activation,
                                                 kernel_initializer=kernel_initializer,
                                                 bias_initializer=bias_initializer)

    def call(self, x):
        shift, log_scale = tf.unstack(self.network(x), num=2, axis=-1)

        return shift, tf.math.tanh(log_scale)

###############################################################################
#                    Chain of Inverse Autoregressive Flow                     #
###############################################################################
class IAFChain:
    def __init__(self, num_IAF_tranforms,
                 hidden_units,
                 activation,
                 kernel_initializer, bias_initializer):

        base_dist = tfd.Normal(loc=0.0, scale=1.0)  # specify base distribution
        bijectors = []

        for i in range(0, num_IAF_transforms):
            bijectors.append(tfb.Invert(
                tfb.MaskedAutoregressiveFlow(
                shift_and_log_scale_fn = Made(params=2,
                                              hidden_units=hidden_units,
                                              activation=activation,
                                              kernel_initializer=kernel_initializer,
                                              bias_initializer=bias_initializer))))
            bijectors.append(tfb.Permute(permutation=[1, 0]))

        bijector = tfb.Chain(bijectors=list(reversed(bijectors)))

        maf = tfd.TransformedDistribution(distribution=base_dist,
                                          bijector=bijector,
                                          event_shape=[2])
