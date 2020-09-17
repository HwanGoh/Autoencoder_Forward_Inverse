import numpy as np
import pandas as pd

from utils_data.data_handler import DataHandler

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def construct_data_dict(hyperp, options, file_paths):

    #=== Load Observation Indices ===#
    if options.obs_type == 'full':
        obs_dimensions = options.parameter_dimensions
        obs_indices = []
    if options.obs_type == 'obs':
        obs_dimensions = options.num_obs_points
        print('Loading Boundary Indices')
        df_obs_indices = pd.read_csv(file_paths.project.obs_indices_file_path + '.csv')
        obs_indices = df_obs_indices.to_numpy()

    #=== Prepare Data ===#
    data = DataHandler(hyperp, options, file_paths,
                       options.parameter_dimensions, obs_dimensions)
    data.load_data_train()
    data.load_data_test()
    if options.add_noise == 1:
        data.add_noise_output_train()
        data.add_noise_output_test()
        noise_regularization_matrix = data.construct_noise_regularization_matrix_train()
    else:
        noise_regularization_matrix = np.eye(obs_dimensions).astype(np.float32)

    #=== Construct Dictionary ===#
    data_dict = {}
    data_dict["obs_dimensions"] = obs_dimensions
    data_dict["obs_indices"] = obs_indices
    data_dict["parameter_train"] = data.input_train
    data_dict["state_obs_train"] = data.output_train
    data_dict["parameter_test"] = data.input_test
    data_dict["state_obs_test"] = data.output_test
    data_dict["noise_regularization_matrix"] = noise_regularization_matrix

    return data_dict
