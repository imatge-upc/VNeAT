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
    eps_var = (1 / (N-1)) * (eps - np.mean(eps)).dot(eps - np.mean(eps))

    # Compute Cp statistic
    return err + (2.0 / N)*df*eps_var

