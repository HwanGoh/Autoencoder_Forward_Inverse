import numpy as np
import pandas as pd

from get_train_and_test_data import load_train_and_test_data
from add_noise import add_noise

from get_prior import load_prior

def load_data_dict(hyperp, options, file_paths):
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
            normalize_input_flag = options.normalize_input,
            normalize_output_flag = options.normalize_output)

    #=== Add Noise to Data ===#
    if options.add_noise == 1:
        state_obs_train, state_obs_test, noise_regularization_matrix\
        = add_noise(options, state_obs_train, state_obs_test, load_data_train_flag = 1)
    else:
        noise_regularization_matrix = tf.eye(obs_dimensions)

    #=== Construct Dictionary ===#
    data_dict = {}
    data_dict["obs_dimensions"] = obs_dimensions
    data_dict["obs_indices"] = obs_indices
    data_dict["parameter_train"] = parameter_train
    data_dict["state_obs_train"] = state_obs_train
    data_dict["parameter_test"] = parameter_test
    data_dict["state_obs_test"] = state_obs_test
    data_dict["noise_regularization_matrix"] = noise_regularization_matrix

    return data_dict
