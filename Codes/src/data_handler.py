import time

import numpy as np
import pandas as pd
import tensorflow as tf

class DataHandler:
    def __init__(self,hyperp, options, file_paths,
                 input_dimensions, output_dimensions):

        self.file_path_input_train = file_paths.input_train_savefilepath
        self.file_path_output_train = file_paths.output_train_savefilepath
        self.file_path_input_test = file_paths.input_test_savefilepath
        self.file_path_output_test = file_paths.output_test_savefilepath

        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions

        self.num_data_train = hyperp.num_data_train
        self.num_data_test = options.num_data_test

        self.noise_level = options.noise_level

        self.num_obs_points = options.num_obs_points
        self.num_noisy_obs = options.num_noisy_obs
        self.num_noisy_obs_unregularized = options.num_noisy_obs_unregularized

        self.dampening_scalar = 0.001

        self.random_seed = options.random_seed

###############################################################################
#                                 Load Data                                   #
###############################################################################
    def load_data_train(self):
        print('Loading Training Data')
        self.input_train, self.output_train = self.load_data(
                self.file_path_input_train,
                self.file_path_output_train,
                self.num_data_train)
    def load_data_test(self):
        print('Loading Training Data')
        self.input_test, self.output_test = self.load_data(
                self.file_path_input_test,
                self.file_path_output_test,
                self.num_data_test)

    def load_data(self, file_path_input_data, file_path_output_data,
                  num_data):

        start_time_load_data = time.time()

        df_input_data = pd.read_csv(file_path_input_data + '.csv')
        df_output_data = pd.read_csv(file_path_output_data + '.csv')
        input_data = df_input_data.to_numpy()
        output_data = df_output_data.to_numpy()
        input_data = input_data.reshape((-1,self.input_dimensions))
        output_data = output_data.reshape((-1,self.output_dimensions))
        input_data = input_data[0:num_data,:]
        output_data = output_data[0:num_data,:]

        elapsed_time_load_data = time.time() - start_time_load_data
        print('Time taken to load data: %.4f' %(elapsed_time_load_data))

        return input_data, output_data

###############################################################################
#                                 Add Noise                                   #
###############################################################################
    def add_noise_output_train(self):
        self.output_train_max = np.max(self.output_train)
        self.output_train = self.add_noise(self.output_train, self.output_train_max)
    def add_noise_output_test(self):
        self.output_test_max = np.max(self.output_test)
        self.output_test = self.add_noise(self.output_test, self.output_test_max)

    def add_noise(self, data, data_max):
        np.random.seed(self.random_seed)
        #=== Add Noise ===#
        noisy_obs = np.random.choice(
                range(0, data.shape[1]), self.num_noisy_obs , replace=False)
        non_noisy_obs = np.setdiff1d(range(0, self.num_obs_points), noisy_obs)

        noise = np.random.normal(0, 1, data.shape)
        noise[:, non_noisy_obs] = self.dampening_scalar*noise[:, non_noisy_obs]
        data += self.noise_level*data_max*noise

        return data

###############################################################################
#                         Noise Regularization Matrix                         #
###############################################################################
    def construct_noise_regularization_matrix_train(self):
        return self.construct_noise_regularization_matrix(self.output_train, self.output_train_max)
    def construct_noise_regularization_matrix_test(self):
        return self.construct_noise_regularization_matrix(self.output_test, self.output_test_max)

    def construct_noise_regularization_matrix(self, data, data_max):
        #=== Noise Regularization Matrix ===#
        noisy_obs = np.random.choice(
                range(0, data.shape[1]), self.num_noisy_obs , replace=False)
        non_noisy_obs = np.setdiff1d(range(0, self.num_obs_points), noisy_obs)
        diagonal = 1/(self.noise_level*data_max)*np.ones(data.shape[1])

        if self.num_noisy_obs_unregularized != 0:
            diagonal[non_noisy_obs[0:self.num_noisy_obs_unregularized]] =\
                    (1/self.dampening_scalar)*\
                    diagonal[non_noisy_obs[0:self.num_noisy_obs_unregularized]]
        noise_regularization_matrix = tf.linalg.diag(diagonal)
        noise_regularization_matrix = tf.cast(noise_regularization_matrix, dtype = tf.float32)

        return noise_regularization_matrix

###############################################################################
#                               Normalize Data                                #
###############################################################################
    def normalize_data_input_train(self):
        self.input_train = normalize_data(self.input_train)
    def normalize_data_output_train(self):
        self.output_train = normalize_data(self.output_train)
    def normalize_data_input_test(self):
        self.input_test = normalize_data(self.input_test)
    def normalize_data_output_test(self):
        self.output_test = normalize_data(self.output_test)

    def normalize_data(self, data):
        data = data/np.expand_dims(np.linalg.norm(data, ord = 2, axis = 1), 1)

        return data
