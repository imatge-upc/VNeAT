"""
Functions to compute the (effective) degrees of freedom for different fitters

All functions must have the following prototype:

    Parameters
    ----------
    y : numpy.array (NxM)
        M explained variable vector with N observations
    fitter : CurveFitter
        Fitted CurveFitter or one of its subclasses

    Returns
    -------
    numpy.array(M,)
        The degrees of freedom for this specific fit of X with respect to each
        explained variable

"""

import numpy as np

def df_SVR(y, fitter):
    # TODO
    return np.zeros(y.shape[1])

def df_GAM(y, fitter):
    # TODO
    return np.zeros(y.shape[1])


