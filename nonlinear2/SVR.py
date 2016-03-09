""" Support Vector Regression fitters
        * Linear SVR
        * Polynomial SVR
        * Gaussian SVR
"""
from CurveFitting import AdditiveCurveFitter
from sklearn.svm import SVR
import sklearn.preprocessing as preprocessing
import numpy as np
from numpy import array, ravel
from Transforms import polynomial
from joblib import Parallel, delayed

class LinearSVR(AdditiveCurveFitter):
    """
    LINEAR SVR
    Class that implements linear Support Vector Regression
    """

    @staticmethod
    def __fit__(correctors, regressors, observations, *args, **kwargs):
        """

        Parameters
        ----------
        correctors
        regressors
        observations
        args
        kwargs

        Returns
        -------

        """
        # Parameters for linear SVR
        C = kwargs['C'] if 'C' in kwargs else 1000.0
        epsilon = kwargs['epsilon'] if 'epsilon' in kwargs else 0.01
        shrinking = kwargs['shrinking'] if 'shrinking' in kwargs else True
        max_iter = kwargs['max_iter'] if 'max_iter' in kwargs else -1
        tol = kwargs['tol'] if 'tol' in kwargs else 1e-3
        sample_weight = kwargs['sample_weight'] if 'sample_weight' in kwargs else None
        n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else 4

        # Initialize linear SVR from scikit-learn
        svr_fitter = SVR(kernel='linear', C=C, epsilon=epsilon, shrinking=shrinking, max_iter=max_iter, tol=tol)

        # Create features matrix and standardize data
        X = np.concatenate((correctors, regressors), axis=1)
        num_variables = observations.shape[1]
        X_std = preprocessing.scale(X)

        # Fit data per voxel
        params = Parallel(n_jobs=n_jobs)(delayed(__fit_features__) \
                                        (svr_fitter, X_std, observations[:, i], sample_weight)
                                         for i in range(num_variables))
        params = array(params)

        # Get correction and regression coefficients
        end_correctors = int(correctors.shape[1])
        c_params = params[:end_correctors, :]
        r_params = params[end_correctors:, :]
        return c_params, r_params


    @staticmethod
    def __predict__(regressors, regression_parameters, *args, **kwargs):
        """

        Parameters
        ----------
        regressors
        regression_parameters
        args
        kwargs

        Returns
        -------

        """
        return regressors.dot(regression_parameters)


class PolySVR(LinearSVR):
    """ POLYNOMIAL SVR """

    def __init__(self, features, regressors = None, degrees = None, homogeneous = True):
        """

        Parameters
        ----------
        features NxF (2-dimensional) matrix
        regressors int / iterable object (default None)
        degrees iterable of F elements (default None)
        homogeneous bool (default True)

        Returns
        -------

        """

        # Check features matrix
        self._svr_features = array(features)
        if len(self._svr_features.shape) != 2:
            raise ValueError('Argument \'features\' must be a 2-dimensional matrix')
        self._svr_features = self._svr_features.T

        # Check regressors indexes
        if regressors is None:
            self._svr_is_regressor = [True]*len(self._svr_features)
            regressors = []
        else:
            self._svr_is_regressor = [False]*len(self._svr_features)
            if isinstance(regressors, int):
                regressors = [regressors]
        try:
            for r in regressors:
                try:
                    self._svr_is_regressor[r] = True
                except TypeError:
                    raise ValueError('All elements in argument \'regressors\' must be valid indices')
                except IndexError:
                    raise IndexError('Index out of range in argument \'regressors\'')
        except TypeError:
            raise TypeError('Argument \'regressors\' must be iterable or int')

        # Check degrees indexes
        if degrees is None:
            self._svr_degrees = [1]*len(self._svr_features)
        else:
            degrees = list(degrees)
            if len(degrees) != len(self._svr_features):
                raise ValueError('Argument \'degrees\' must have a length equal to the number of features')
            for deg in degrees:
                if not isinstance(deg, int):
                    raise ValueError('Expected integer in \'degrees\' list, got ' + str(type(deg)) + ' instead')
                if deg < 1:
                    raise ValueError('All degrees must be >= 1')
            self._svr_degrees = degrees

        # Check homogeneous term
        self._svr_homogeneous = homogeneous

        # Call function to expand the feature space with polynomial terms
        self.__svr_polynomial_expansion()

    def __svr_polynomial_expansion(self):
        """
        Expands the input space to a feature space with the corresponding polynomial terms,
        and then uses this expanded space to initialize the correctors and regressors for a linear SVR
        Returns
        -------

        """
        correctors = []
        regressors = []
        for index in xrange(len(self._svr_is_regressor)):
            for p in polynomial(self._svr_degrees[index], [self._svr_features[index]]):
                if self._svr_is_regressor[index]:
                    regressors.append(p)
                else:
                    correctors.append(p)

        if len(correctors) == 0:
            correctors = None
        else:
            correctors = array(correctors).T

        if len(regressors) == 0:
            regressors = None
        else:
            regressors = array(regressors).T

        # Instance a LinearSVR (parent) with the expanded polynomial features
        super(PolySVR, self).__init__(regressors, correctors, self._svr_homogeneous)

class GaussianSVR(object):
    """ GAUSSIAN SVR """
    pass


""" HELPER FUNCTIONS """

def __fit_features__(fitter, X, y, sample_weight=None):
        """
        Fits the features from X to the observation y given the linear fitter and the optional sample_weights
        Parameters
        ----------
        fitter sklearn linear fitter, must have the fit method and the coef_ attribute
        X NxF matrix, where N is the number of observations and F the number of features
        y Nx1 the variable that we want to explain with the features
        [sample_weight]

        Returns
        -------
        Fx1 vector with the fitting coefficients
        """
        return ravel(fitter.fit(X, y, sample_weight=sample_weight).coef_).T