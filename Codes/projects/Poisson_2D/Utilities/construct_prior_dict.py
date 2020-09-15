import numpy as np
import pandas as pd

from prior_handler import PriorHandler

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def construct_prior_dict(hyperp, options, file_paths,
                         load_mean = 1,
                         load_covariance = 0,
                         load_covariance_cholesky = 0,
                         load_covariance_cholesky_inverse = 0):

    prior_dict = {}
    prior = PriorHandler(hyperp, options, file_paths,
                         options.parameter_dimensions)

    #=== Prior Mean ===#
    if load_mean == 1:
        prior_mean = prior.load_prior_mean()
        prior_dict["prior_mean"] = prior_mean

    #=== Prior Covariance ===#
    if load_covariance == 1:
        prior_covariance = prior.load_prior_covariance()
        prior_dict["prior_covariance"] = prior_covariance

    #=== Prior Covariance Cholesky ===#
    if load_covariance_cholesky == 1:
        prior_covariance_cholesky = prior.load_prior_covariance_cholesky()
        prior_dict["prior_covariance_cholesky"] = prior_covariance_cholesky

    #=== Prior Covariance Cholesky Inverse ===#
    if load_covariance_cholesky_inverse == 1:
        prior_covariance_cholesky_inverse = prior.load_prior_covariance_cholesky_inverse()
        prior_dict["prior_covariance_cholesky_inverse"] = prior_covariance_cholesky_inverse

    return prior_dict
