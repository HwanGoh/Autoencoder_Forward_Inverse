import numpy as np
import pandas as pd

from get_prior import load_prior

def load_prior_dict(hyperp, options, file_paths):

    #=== Load Objects ===#
    prior_mean, _, _,\
    prior_covariance_cholesky_inverse\
    = load_prior(options, file_paths,
                 load_mean = 1,
                 load_covariance = 0,
                 load_covariance_cholesky = 0,
                 load_covariance_cholesky_inverse = 1)

    #=== Construct Dictionary ===#
    prior_dict = {}
    prior_dict["prior_mean"] = prior_mean
    prior_dict["prior_covariance_cholesky_inverse"] = prior_covariance_cholesky_inverse

    return prior_dict
