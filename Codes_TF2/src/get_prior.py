#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:16:28 2019

@author: hwan
"""
import numpy as np
import pandas as pd

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                 All Data                                    #
###############################################################################
def load_prior(run_options, file_paths,
        load_mean = 0,
        load_covariance = 0,
        load_covariance_cholesky = 0, load_covariance_cholesky_inverse = 0):

    print('Loading Prior')

    #=== Prior Mean ===#
    if load_mean == 1:
        df_mean = pd.read_csv(file_paths.prior_mean_savefilepath + '.csv')
        prior_mean = df_mean.to_numpy()
        prior_mean = prior_mean.astype(np.float32).flatten()
    else:
        prior_mean = np.zeros(run_options.parameter_dimensions)

    #=== Prior Covariance ===#
    if load_covariance == 1:
        df_covariance = pd.read_csv(file_paths.prior_covariance_savefilepath + '.csv')
        prior_covariance = df_covariance.to_numpy()
        prior_covariance = prior_covariance.reshape((run_options.parameter_dimensions,
            run_options.parameter_dimensions))
        prior_covariance = prior_covariance.astype(np.float32)
    else:
        prior_covariance = np.identity(run_options.parameter_dimensions)

    #=== Prior Covariance Cholesky ===#
    if load_covariance_cholesky == 1:
        df_covariance_cholesky =\
                pd.read_csv(file_paths.prior_covariance_cholesky_savefilepath + '.csv')
        prior_covariance_cholesky = df_covariance_cholesky.to_numpy()
        prior_covariance_cholesky =\
                prior_covariance_cholesky.reshape((run_options.parameter_dimensions,
            run_options.parameter_dimensions))
        prior_covariance_cholesky = prior_covariance_cholesky.astype(np.float32)
    else:
        prior_covariance_cholesky = np.identity(run_options.parameter_dimensions)

    #=== Prior Covariance Cholesky Inverse ===#
    if load_covariance_cholesky_inverse == 1:
        df_covariance_cholesky_inverse =\
                pd.read_csv(file_paths.prior_covariance_cholesky_inverse_savefilepath + '.csv')
        prior_covariance_cholesky_inverse = df_covariance_cholesky_inverse.to_numpy()
        prior_covariance_cholesky_inverse =\
                prior_covariance_cholesky_inverse.reshape((run_options.parameter_dimensions,
            run_options.parameter_dimensions))
        prior_covariance_cholesky_inverse = prior_covariance_cholesky_inverse.astype(np.float32)
    else:
        prior_covariance_cholesky_inverse = np.identity(run_options.parameter_dimensions)

    return prior_mean, prior_covariance,\
            prior_covariance_cholesky, prior_covariance_cholesky_inverse
