import numpy as np
import pandas as pd


def save_prior(filepaths, prior_mean, prior_covariance,
        prior_covariance_cholesky, prior_covariance_cholesky_inverse):

    #=== Save Prior ===#
    df_prior_mean = pd.DataFrame({'prior_mean': prior_mean.flatten()})
    df_prior_mean.to_csv(filepaths.prior_mean + '.csv', index=False)

    df_prior_covariance = pd.DataFrame({'prior_covariance': prior_covariance.flatten()})
    df_prior_covariance.to_csv(filepaths.prior_covariance + '.csv', index=False)

    df_prior_covariance_cholesky = pd.DataFrame(
            {'prior_covariance_cholesky': prior_covariance_cholesky.flatten()})
    df_prior_covariance_cholesky.to_csv(
            filepaths.prior_covariance_cholesky + '.csv', index=False)

    df_prior_covariance_cholesky_inverse = pd.DataFrame(
            {'prior_covariance_cholesky_inverse': prior_covariance_cholesky_inverse.flatten()})
    df_prior_covariance_cholesky_inverse.to_csv(
            filepaths.prior_covariance_cholesky_inverse + '.csv', index=False)
