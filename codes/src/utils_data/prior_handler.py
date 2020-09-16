#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:16:28 2019

@author: hwan
"""
import numpy as np
import pandas as pd

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class PriorHandler:
    def __init__(self, hyperp, options, file_paths, input_dimensions):

        self.prior_mean_savefilepath = file_paths.prior_mean_savefilepath
        self.prior_covariance_savefilepath = file_paths.prior_covariance_savefilepath
        self.prior_covariance_cholesky_savefilepath =\
            file_paths.prior_covariance_cholesky_savefilepath
        self.prior_covariance_cholesky_inverse_savefilepath =\
            file_paths.prior_covariance_cholesky_inverse_savefilepath

        self.input_dimensions = input_dimensions

    def load_prior_mean(self):
        return self.load_vector(self.prior_mean_savefilepath)

    def load_prior_covariance(self):
        return self.load_matrix(self.prior_covariance_savefilepath)

    def load_prior_covariance_cholesky(self):
        return self.load_matrix(self.prior_covariance_cholesky_savefilepath)

    def load_prior_covariance_cholesky_inverse(self):
        return self.load_matrix(self.prior_covariance_cholesky_inverse_savefilepath)

    def load_vector(self, file_path):
        df_vector = pd.read_csv(file_path + '.csv')
        vector = df_vector.to_numpy()
        return vector.astype(np.float32).flatten()

    def load_matrix(self, file_path):
        df_matrix = pd.read_csv(file_path + '.csv')
        matrix = df_matrix.to_numpy()
        matrix = matrix.reshape((self.input_dimensions, self.input_dimensions))
        return matrix.astype(np.float32)
