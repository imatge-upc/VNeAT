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
    y_predicted: numpy.array(N x M)
        Predicted observations
    N: int
        number of data points for each explained variable

    Returns
    -------
    numpy.array(1xM)
        MSE for each of the M explained variables
    """
    sum_SE = np.sum(np.square(y_predicted - y_true), axis=0)
    return sum_SE / N

# def statisticC_p(y_true, y_predicted, N, fitter):
#     """
#     Calculates the statistic Cp
#
#     Parameters
#     ----------
#     y_true : numpy.array(N x M)
#         Observations, where N is the number of observations and M the number of explained variables
#     y_predicted: numpy.array(N x M)
#         Predicted observations
#     N: int
#         number of data points for each explained variable
#     fitter : CurveFitter
#         fitter used to calculate the effective degrees of freedom corresponding to
#         to the model used to fit the data
#     Returns
#     -------
#     numpy.array(1xM)
#         MSE for each of the M explained variables
#     """
#     # Caculate the training error (err) assuming a L2 loss
#     err = mse(y_true, y_predicted, N)
#
#     # Get the effective degrees of freedom
#     df = fitter.degrees_of_freedom()
#
#     # Compute Cp statistic
#     return err + (2.0 / N)*df

