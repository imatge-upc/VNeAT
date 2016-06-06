"""
Score functions to be used in GridSearch.
These functions must have the following prototype:

    Parameters
    ----------
    y_true : numpy.array(N x M)
        Observations, where N is the number of observations and M the number of explained variables
    y_predicted: numpy.array(N x M)
        Predicted observations
    df : numpy.array(M,)
        degrees of freedom for each explained variable

    Returns
    -------
    numpy.array(M,)
        Error for each of the M explained variables
"""

import numpy as np
import scipy.stats as stats


def mse(y_true, y_predicted, df):
    """
    Calculates the Mean Squared Error for the N data points
    """
    N = y_true.shape[0]
    sum_SE = np.sum(np.square(y_predicted - y_true), axis=0)
    return sum_SE / N


def statisticC_p(y_true, y_predicted, df):
    """
    Calculates the statistic Cp
    """
    N = y_true.shape[0]

    # Caculate the training error (err) assuming a L2 loss
    err = mse(y_true, y_predicted, df)

    # Estimate the error variance
    eps = y_true - y_predicted
    eps_mean = np.mean(eps, axis=0)
    eps_var = (1.0 / (N - 1)) * np.sum(np.square(eps - eps_mean))

    # Compute Cp statistic
    return err + (2.0 / N)*df*eps_var


def anova_error(y_true, y_predicted, df):
    """
    Analysis of Variance (ANOVA) based error measure (that is, this score has to be minimized)
    http://reliawiki.org/index.php/Simple_Linear_Regression_Analysis
    """
    N = y_true.shape[0]
    residual = y_true - y_predicted
    SST = (np.square(y_true - y_true.mean(axis=0))).sum(axis=0)         # Total Sum of Squares
    SSE = (np.square(residual - residual.mean(axis=0))).sum(axis=0)     # Error Sum of Squares
    SSR = SST - SSE                                                     # Regression Sum of Squares
    df_SST = N - 1                                                      # Total degrees of freedom
    df_SSR = df                                                         # Regression degrees of freedom
    df_SSE = df_SST - df_SSR                                            # Error degrees of freedom
    num = (SSR/(df_SSR + 1e-12))
    den = (SSE/df_SSE) + 1e-12
    F_score = num / den
    return stats.f.sf(F_score, df_SSR, df_SSE)