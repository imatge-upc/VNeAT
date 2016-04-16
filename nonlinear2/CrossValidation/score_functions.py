"""
Score functions to be used in crossvalidation
"""

import numpy as np

def mse(y_true, y_predicted, N):
    """
    Calculates the Mean Squared Error for the N data points

    Parameters
    ----------
    y_true : numpy.array(N x M)
        Observations, where N is the number of observations and M the number of explained variables
    y_predicted:
        Predicted observations
    N:
        number of data points for each explained variable

    Returns
    -------
    numpy.array(1xM)
        MSE for each of the M explained variables
    """
    sum_SE = np.sum(np.square(y_predicted - y_true), axis=0)
    return sum_SE / N

def leaveOneOut(y_true, y_predicted, N):
    # TODO
    pass