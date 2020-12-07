import tensorflow as tf
import pandas as pd

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def load_fem_matrices_tf(options, filepaths):

    #=== Load Spatial Operator ===#
    df_forward_operator = pd.read_csv(filepaths.project.forward_operator + '.csv')
    forward_operator = df_forward_operator.to_numpy()
    df_mass_matrix = pd.read_csv(filepaths.project.mass_matrix + '.csv')
    mass_matrix = df_mass_matrix.to_numpy()

    return forward_operator.reshape((options.parameter_dimensions, options.parameter_dimensions)),\
           mass_matrix.reshape((options.parameter_dimensions, options.parameter_dimensions))
