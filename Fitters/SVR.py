""" Support Vector Regression fitters
        * Linear SVR
        * Polynomial SVR
        * Gaussian SVR
"""
import numpy as np
from joblib import Parallel, delayed

from sklearn.svm import LinearSVR, SVR

from Utils.Transforms import polynomial
from Fitters.CurveFitting import CurveFitter
from Fitters.CurveFitting import AdditiveCurveFitter

class LinSVR(AdditiveCurveFitter):
    """
    LINEAR SVR
    Class that implements linear Support Vector Regression
    """

    def __init__(self, predictors = None, correctors = None, intercept = CurveFitter.NoIntercept):
        self._svr_intercept = intercept
        # Don't allow a intercept feature to be created, use instead the intercept term from the fitter
        super(LinSVR, self).__init__(predictors, correctors, intercept)

    def __fit__(self, correctors, predictors, observations, *args, **kwargs):

        # Parameters for linear SVR
        C = kwargs['C'] if 'C' in kwargs else 100.0
        epsilon = kwargs['epsilon'] if 'epsilon' in kwargs else 0.1
        max_iter = kwargs['max_iter'] if 'max_iter' in kwargs else 2000
        tol = kwargs['tol'] if 'tol' in kwargs else 1e-4
        n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else 4

        # Initialize linear SVR from scikit-learn
        svr_fitter = LinearSVR(epsilon=epsilon, tol=tol, C=C, fit_intercept=self._svr_intercept,
                               intercept_scaling=C, max_iter=max_iter)

        num_variables = observations.shape[1]

        # Intercept handling
        predictor_intercept = self._svr_intercept == AdditiveCurveFitter.PredictionIntercept
        corrector_intercept = self._svr_intercept == AdditiveCurveFitter.CorrectionIntercept

        # Predictors preprocessing
        fit_predictors = True
        if predictor_intercept:
            predictors = predictors[:, 1:]
        p_size = predictors.size

        if p_size == 0 and predictor_intercept:
            predictors = np.array([[]])
        elif p_size == 0:
            fit_predictors = False
            pparams = np.array([[]])

        # Correctors preprocessing
        fit_correctors = True
        if corrector_intercept:
            correctors = correctors[:, 1:]
        c_size = correctors.size

        if c_size == 0 and corrector_intercept:
            correctors = np.array([[]])
        elif c_size == 0:
            fit_correctors = False
            cparams = np.array([[]])

        # Fit correctors
        if fit_correctors:
            params = Parallel(n_jobs=n_jobs)(delayed(__fit_features__) \
                                        (svr_fitter, correctors, observations[:, i], corrector_intercept)
                                         for i in range(num_variables))

            cparams = np.array(params).T

            # Correct observations
            observations = self.__correct__(observations, correctors, cparams)

        # Fit predictors
        if fit_predictors:
            params = Parallel(n_jobs=n_jobs)(delayed(__fit_features__) \
                                            (svr_fitter, predictors, observations[:, i], predictor_intercept)
                                             for i in range(num_variables))
            pparams = np.array(params).T

        # Get correction and regression coefficients
        return cparams, pparams

    @staticmethod
    def __predict__(predictors, prediction_parameters, *args, **kwargs):
        return predictors.dot(prediction_parameters)


class PolySVR(LinSVR):
    """ POLYNOMIAL SVR """

    def __init__(self, features, predictors = None, degrees = None, intercept = CurveFitter.NoIntercept):
        """

        Parameters
        ----------
        features NxF (2-dimensional) matrix
        predictors int / iterable object (default None)
        degrees iterable of F elements (default None)
        intercept bool (default True)

        Returns
        -------

        """

        # Check features matrix
        self._svr_features = np.array(features)
        if len(self._svr_features.shape) != 2:
            raise ValueError('Argument \'features\' must be a 2-dimensional matrix')
        self._svr_features = self._svr_features.T

        # Check predictors indexes
        if predictors is None:
            self._svr_is_predictor = [True]*len(self._svr_features)
            predictors = []
        else:
            self._svr_is_predictor = [False]*len(self._svr_features)
            if isinstance(predictors, int):
                predictors = [predictors]
        try:
            for r in predictors:
                try:
                    self._svr_is_predictor[r] = True
                except TypeError:
                    raise ValueError('All elements in argument \'predictors\' must be valid indices')
                except IndexError:
                    raise IndexError('Index out of range in argument \'predictors\'')
        except TypeError:
            raise TypeError('Argument \'predictors\' must be iterable or int')

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

        # Check intercept term
        self._svr_intercept = intercept

        # Call function to expand the feature space with polynomial terms
        self.__svr_polynomial_expansion()

    def __svr_polynomial_expansion(self):
        """
        Expands the input space to a feature space with the corresponding polynomial terms,
        and then uses this expanded space to initialize the correctors and predictors for a linear SVR
        Returns
        -------

        """
        correctors = []
        predictors = []
        for index in xrange(len(self._svr_is_predictor)):
            for p in polynomial(self._svr_degrees[index], [self._svr_features[index]]):
                if self._svr_is_predictor[index]:
                    predictors.append(p)
                else:
                    correctors.append(p)

        if len(correctors) == 0:
            correctors = None
        else:
            correctors = np.array(correctors).T

        if len(predictors) == 0:
            predictors = None
        else:
            predictors = np.array(predictors).T

        # Instance a LinSVR (parent) with the expanded polynomial features
        super(PolySVR, self).__init__(predictors, correctors, self._svr_intercept)



class GaussianSVR(CurveFitter):
    """ Support Vector Regression fitter with Gaussian Kernel """

    def __init__(self, predictors=None, correctors=None, intercept=CurveFitter.NoIntercept):
        self._svr_intercept = intercept
        # Don't allow a intercept feature to be created, use instead the intercept term from the fitter
        super(GaussianSVR, self).__init__(predictors, correctors, intercept)


    def __fit__(self, correctors, predictors, observations, *args, **kwargs):
        # Parameters for SVR training
        C = kwargs['C'] if 'C' in kwargs else 100.0
        epsilon = kwargs['epsilon'] if 'epsilon' in kwargs else 0.1
        gamma = kwargs['gamma'] if 'gamma' in kwargs else 0.25
        max_iter = kwargs['max_iter'] if 'max_iter' in kwargs else -1
        tol = kwargs['tol'] if 'tol' in kwargs else 1e-4
        cache_size = kwargs['cache_size'] if 'cache_size' in kwargs else 1000
        n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else 4

        # Initialize linear SVR from scikit-learn
        svr_fitter = SVR(kernel='rbf', gamma=gamma, tol=tol, C=C, epsilon=epsilon,
                         cache_size=cache_size, max_iter=max_iter)
        num_variables = observations.shape[1]

        # Save gamma for prediction
        self._svr_gamma = svr_fitter.__getattribute__('gamma')

        # Intercept handling
        predictor_intercept = self._svr_intercept == AdditiveCurveFitter.PredictionIntercept
        corrector_intercept = self._svr_intercept == AdditiveCurveFitter.CorrectionIntercept

        # Predictors preprocessing
        fit_predictors = True
        if predictor_intercept:
            predictors = predictors[:, 1:]
        p_size = predictors.size

        if p_size == 0 and predictor_intercept:
            predictors = np.array([[]])
        elif p_size == 0:
            fit_predictors = False
            pparams = np.array([[]])

        # Correctors preprocessing
        fit_correctors = True
        if corrector_intercept:
            correctors = correctors[:, 1:]
        c_size = correctors.size

        if c_size == 0 and corrector_intercept:
            correctors = np.array([[]])
        elif c_size == 0:
            fit_correctors = False
            cparams = np.array([[]])

        # Fit correctors and correct (if necessary)
        if fit_correctors:
            params = Parallel(n_jobs=n_jobs)(delayed(__fit_SVR_features__) \
                                        (svr_fitter, correctors, observations[:, i], corrector_intercept)
                                         for i in range(num_variables))

            cparams = np.array(params).T

            # Correct observations
            observations = self.__correct__(observations, correctors, cparams)

        # Fit predictors
        if fit_predictors:
            params = Parallel(n_jobs=n_jobs)(delayed(__fit_SVR_features__) \
                                            (svr_fitter, predictors, observations[:, i], predictor_intercept)
                                             for i in range(num_variables))
            pparams = np.array(params).T

        # Return correction and regression coefficients
        return cparams, pparams

    def __predict__(self, predictors, prediction_parameters, *args, **kwargs):
        # Intercept term
        if self._svr_intercept == self.PredictionIntercept:
            # Get intercept term as the last coefficient in pparams
            intercept = prediction_parameters[-1, :]
            prediction_parameters = prediction_parameters[:-1,:]
            # Get rid of the ones column for the intercept term
            predictors = predictors[:, 1:]
            training_examples = self.predictors[:, 1:]
        else:
            intercept = 0
            training_examples = self.predictors

        # Training data
        N = training_examples.shape[0]

        # Prediction function with gaussian kernel
        num_variables = prediction_parameters.shape[1]
        num_predictors = predictors.shape[0]
        prediction = np.zeros((num_predictors, num_variables))
        for i in range(num_predictors):
            x = np.atleast_2d(predictors[i])
            X_p = np.ones( (N, 1) ).dot(x)
            x_norm = np.sum((X_p - training_examples)**2, axis=1)
            exponential = np.exp(-self._svr_gamma*x_norm)
            prediction[i, :] = prediction_parameters.T.dot(exponential)

        return prediction + intercept

    def __correct__(self, observations, correctors, correction_parameters, *args, **kwargs):
        # TODO
        return np.zeros((correctors.shape[0], 1))





""" HELPER FUNCTIONS """

def __fit_features__(fitter, X, y, intercept):
        """
        Fits the features from X to the observation y given the linear fitter
        Parameters
        ----------
        fitter : sklearn.svm.LinearSVR
            Linear fitter that must have the fit method and the coef_ and intercept_ attributes
        X : numpy.array(NxF)
            Features array where N is the number of observations and F the number of features
        y : numpy.array(Nx1)
            The variable that we want to explain with the features
        intercept : Boolean
            Whether the intercept term must be computed or not

        Returns
        -------
        numpy.array(F,) or numpy.array(F+1,)
            Array with the fitting coefficients plus the intercept term if intercept=True
        """
        num_features = X.shape[1]
        if num_features <= 0:
            if intercept:
                # If the features array is empty and we need to compute the intercept term,
                # create dummy features to fit and then get only the intercept term
                X = np.ones((y.shape[0], 1))
            else:
                raise Exception("Features array X is not a NxF array")
        fitter.fit(X, y)

        if intercept:
            if num_features > 0:
                params = np.zeros((num_features + 1, 1))
                coefficients = np.atleast_2d(fitter.coef_)
                params[1:, :] = coefficients.T
            else:
                params = np.zeros((1, 1)) # Only the intercept term
            params[0, :] = float(fitter.intercept_)
        else:
            params = fitter.coef_.T
        return np.ravel(params)


def __fit_SVR_features__(fitter, X, y, intercept):
        """
        Fits the features from X to the observation y given the Support Vector Regression
        fitter
        Parameters
        ----------
        fitter : sklearn.svm.SVR
            SVR fitter that must have the fit method and the support_, support_vectors_,
            dual_coef_ and intercept_ attributes
        X : numpy.array(NxF)
            Features array where N is the number of observations and F the number of features
        y : numpy.array(Nx1)
            The variable that we want to explain with the features
        intercept : Boolean
            Whether the intercept term must be computed or not

        Returns
        -------
        numpy.array(N,) or numpy.array(N+1,)
            Array with the dual coefficients for all feature vectors (the ones that are
            not support vectors have a zero dual coefficient) plus the intercept term
            if intercept=True
        """
        N, num_features = X.shape
        if num_features <= 0:
            if intercept:
                # If the features array is empty and we need to compute the intercept term,
                # create dummy features to fit and then get only the intercept term
                X = np.ones((N,1))
            else:
                raise Exception("Features array X is not a NxF array")
        fitter.fit(X, y)

        if intercept:
            if num_features > 0:
                params = np.zeros(N+1)
                params[fitter.support_] = np.ravel(fitter.dual_coef_)
            else:
                params = np.zeros(1) # Only the intercept term
            params[-1] = fitter.intercept_
        else:
            params = np.zeros(N)
            params[fitter.support_] = np.ravel(fitter.dual_coef_)
        return params