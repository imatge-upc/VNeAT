""" Support Vector Regression fitters
        * Linear SVR
        * Polynomial SVR
        * Gaussian SVR
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.Fitters.CurveFitting import AdditiveCurveFitter
from src.Fitters.CurveFitting import CurveFitter
from src.Utils.Transforms import polynomial


class LinSVR(AdditiveCurveFitter):
    """
    LINEAR SVR
    Class that implements linear Support Vector Regression
    """

    def __init__(self, predictors=None, correctors=None, intercept=CurveFitter.NoIntercept,
                 C=1, epsilon=0.1):
        self._svr_C = C
        self._svr_epsilon = epsilon
        self._svr_intercept = intercept
        # Don't allow a intercept feature to be created, use instead the intercept term from the fitter
        super(LinSVR, self).__init__(predictors, correctors, CurveFitter.NoIntercept)

    def __fit__(self, correctors, predictors, observations, *args, **kwargs):

        # Parameters for linear SVR
        self._svr_C = kwargs['C'] if 'C' in kwargs else self._svr_C
        self._svr_epsilon = kwargs['epsilon'] if 'epsilon' in kwargs else self._svr_epsilon
        max_iter = kwargs['max_iter'] if 'max_iter' in kwargs else -1
        cache_size = kwargs['cache_size'] if 'cache_size' in kwargs else 1000
        tol = kwargs['tol'] if 'tol' in kwargs else 1e-4
        n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else 4

        # Initialize linear SVR from scikit-learn
        svr_fitter = SVR(kernel="linear", epsilon=self._svr_epsilon, C=self._svr_C,
                         cache_size=cache_size, tol=tol, max_iter=max_iter)

        # Initialize standard scaler
        self._correctors_scaler = StandardScaler()
        self._predictors_scaler = StandardScaler()

        num_variables = observations.shape[1]

        # Intercept handling
        predictor_intercept = self._svr_intercept == AdditiveCurveFitter.PredictionIntercept
        corrector_intercept = self._svr_intercept == AdditiveCurveFitter.CorrectionIntercept

        # Predictors preprocessing
        fit_predictors = True
        p_size = predictors.size

        if p_size == 0 and predictor_intercept:
            scaled_predictors = np.array([[]])
        elif p_size == 0:
            fit_predictors = False
            pparams = np.array([[]])
        else:
            scaled_predictors = self._predictors_scaler.fit_transform(predictors)

        # Correctors preprocessing
        fit_correctors = True
        c_size = correctors.size

        if c_size == 0 and corrector_intercept:
            scaled_correctors = np.array([[]])
        elif c_size == 0:
            fit_correctors = False
            cparams = np.array([[]])
        else:
            scaled_correctors = self._correctors_scaler.fit_transform(correctors)

        # Fit correctors
        if fit_correctors:
            params = Parallel(n_jobs=n_jobs)(
                delayed(__fit_features__)(
                    svr_fitter,
                    scaled_correctors,
                    observations[:, i],
                    corrector_intercept
                ) for i in range(num_variables)
            )

            cparams = np.array(params).T

            # Correct observations
            observations = self.__correct__(observations, correctors, cparams)

        # Fit predictors
        if fit_predictors:
            params = Parallel(n_jobs=n_jobs)(
                delayed(__fit_features__)(
                    svr_fitter,
                    scaled_predictors,
                    observations[:, i],
                    predictor_intercept
                ) for i in range(num_variables)
            )

            pparams = np.array(params).T

        # Get correction and regression coefficients
        return cparams, pparams

    def __predict__(self, predictors, prediction_parameters, *args, **kwargs):
        # Compute prediction (first remove df from the end of the params vector)
        pred_params = prediction_parameters[:-1, :]
        intercept = 0
        if self._svr_intercept == self.PredictionIntercept:
            intercept = pred_params[0, :]
            pred_params = pred_params[1:, :]

        # Scale predictors to match the scaling used in fitting
        try:
            scaled_predictors = self._predictors_scaler.transform(predictors)
        except AttributeError:
            # Assume that the data used to predict has the similar statistics than the used
            # in learning and therefore the scaling can be learned from this data to be
            # predicted
            scaled_predictors = StandardScaler().fit_transform(predictors)

        # Return prediction
        return scaled_predictors.dot(pred_params) + intercept

    def __correct__(self, observations, correctors, correction_parameters, *args, **kwargs):
        # Compute correction (first remove df from the end of the params vector)
        corr_params = correction_parameters[:-1, :]
        intercept = 0
        if self._svr_intercept == self.CorrectionIntercept:
            intercept = corr_params[0, :]
            corr_params = corr_params[1:, :]

        # Scale correctors to match the scaling used in fitting
        try:
            scaled_correctors = self._correctors_scaler.transform(correctors)
        except AttributeError:
            # Assume that the data used to predict has the similar statistics than the used
            # in learning and therefore the scaling can be learned from this data to be
            # predicted
            scaled_correctors = StandardScaler().fit_transform(correctors)
        correction = scaled_correctors.dot(corr_params) + intercept

        # Return observations corrected by correction
        return observations - correction

    def __df_correction__(self, observations, correctors, correction_parameters):
        # Get the df from the correction parameters
        df = correction_parameters[-1, :]
        return df

    def __df_prediction__(self, observations, predictors, prediction_parameters):
        # Get the df from the prediction parameters
        df = prediction_parameters[-1, :]
        return df


class PolySVR(LinSVR):
    """ POLYNOMIAL SVR """

    def __init__(self, features, predictors=None, degrees=None, intercept=CurveFitter.NoIntercept):
        # Check features matrix
        self._svr_features = np.array(features)
        if len(self._svr_features.shape) != 2:
            raise ValueError('Argument \'features\' must be a 2-dimensional matrix')
        self._svr_features = self._svr_features.T

        # Check predictors indexes
        if predictors is None:
            self._svr_is_predictor = [True] * len(self._svr_features)
            predictors = []
        else:
            self._svr_is_predictor = [False] * len(self._svr_features)
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
            self._svr_degrees = [1] * len(self._svr_features)
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
    """ GAUSSIAN SVR """

    def __init__(self, predictors=None, correctors=None, intercept=CurveFitter.NoIntercept,
                 C=1, epsilon=0.1, gamma=0.1):
        self._svr_intercept = intercept
        self._svr_C = C
        self._svr_epsilon = epsilon
        self._svr_gamma = gamma
        super(GaussianSVR, self).__init__(predictors, correctors, self.NoIntercept)

    def __fit__(self, correctors, predictors, observations, *args, **kwargs):
        # Parameters for SVR training
        self._svr_C = kwargs['C'] if 'C' in kwargs else self._svr_C
        self._svr_epsilon = kwargs['epsilon'] if 'epsilon' in kwargs else self._svr_epsilon
        self._svr_gamma = kwargs['gamma'] if 'gamma' in kwargs else self._svr_gamma
        max_iter = kwargs['max_iter'] if 'max_iter' in kwargs else -1
        tol = kwargs['tol'] if 'tol' in kwargs else 1e-4
        cache_size = kwargs['cache_size'] if 'cache_size' in kwargs else 1000
        n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else 4

        # Initialize linear SVR from scikit-learn
        svr_fitter = SVR(kernel='rbf', C=self._svr_C, epsilon=self._svr_epsilon, gamma=self._svr_gamma,
                         tol=tol, cache_size=cache_size, max_iter=max_iter)
        num_variables = observations.shape[1]

        # Intercept handling
        predictor_intercept = self._svr_intercept == AdditiveCurveFitter.PredictionIntercept
        corrector_intercept = self._svr_intercept == AdditiveCurveFitter.CorrectionIntercept

        # Predictors preprocessing
        fit_predictors = True
        p_size = predictors.size

        if p_size == 0 and predictor_intercept:
            predictors = np.array([[]])
        elif p_size == 0:
            fit_predictors = False
            pparams = np.array([[]])

        # Correctors preprocessing
        fit_correctors = True
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
            prediction_parameters = prediction_parameters[:-1, :]
        else:
            intercept = 0

        training_examples = self.predictors

        return self.__predict_from_params__(predictors, prediction_parameters,
                                            intercept, training_examples)

    def __correct__(self, observations, correctors, correction_parameters, *args, **kwargs):
        # Intercept term
        if self._svr_intercept == self.CorrectionIntercept:
            # Get intercept term as the last coefficient in pparams
            intercept = correction_parameters[-1, :]
            correction_parameters = correction_parameters[:-1, :]
        else:
            intercept = 0

        training_examples = self.correctors[:, 1:]

        # Correction
        correction = self.__predict_from_params__(correctors, correction_parameters,
                                                  intercept, training_examples)

        return observations - correction

    def __df_correction__(self, observations, correctors, correction_parameters):
        # Manually compute correction (as in __correct__() )
        if self._svr_intercept == self.CorrectionIntercept:
            # Get intercept term as the last coefficient in pparams
            intercept = correction_parameters[-1, :]
            correction_parameters = correction_parameters[:-1, :]
        else:
            intercept = 0

        training_examples = correctors[:, 1:]
        correction = self.__predict_from_params__(correctors, correction_parameters,
                                                  intercept, training_examples)
        # Create kernel diagonal for each variable to explain (ones, because the kernel is gaussian,
        # K(x_i, x_i) = 1
        kernel_diag = np.ones(correction_parameters.shape)

        # Compute pseudoresiduals (refer to F. Dinuzzo et al.
        # On the Representer Theorem and Equivalent Degrees of Freedom of SVR
        # [http://www.jmlr.org/papers/volume8/dinuzzo07a/dinuzzo07a.pdf]
        pseudoresiduals = observations - correction + correction_parameters * kernel_diag

        # Compute effective degrees of freedom from pseudoresiduals
        _C = self._svr_C
        _epsilon = self._svr_epsilon

        # Logical operations
        min_value = _epsilon * np.ones(pseudoresiduals.shape)
        max_value = min_value + _C * kernel_diag
        comp_min = min_value <= np.abs(pseudoresiduals)
        comp_max = np.abs(pseudoresiduals) <= max_value
        return np.sum(np.logical_and(comp_min, comp_max), axis=0)

    def __df_prediction__(self, observations, predictors, prediction_parameters):
        # Compute prediction
        prediction = self.__predict__(predictors, prediction_parameters)
        # Delete intercept term, if any
        if self._svr_intercept == self.PredictionIntercept:
            pred_params = prediction_parameters[:-1, :]
        else:
            pred_params = prediction_parameters
        # Create kernel diagonal for each variable to explain (ones, because the kernel is gaussian,
        # K(x_i, x_i) = 1
        kernel_diag = np.ones(pred_params.shape)

        # Compute pseudoresiduals (refer to F. Dinuzzo et al.
        # On the Representer Theorem and Equivalent Degrees of Freedom of SVR
        # [http://www.jmlr.org/papers/volume8/dinuzzo07a/dinuzzo07a.pdf]
        pseudoresiduals = observations - prediction + pred_params * kernel_diag

        # Compute effective degrees of freedom from pseudoresiduals
        _C = self._svr_C
        _epsilon = self._svr_epsilon

        # Logical operations
        min_value = _epsilon * np.ones(pseudoresiduals.shape)
        max_value = min_value + _C * kernel_diag
        comp_min = min_value <= np.abs(pseudoresiduals)
        comp_max = np.abs(pseudoresiduals) <= max_value
        df = np.sum(np.logical_and(comp_min, comp_max), axis=0)
        return df

    def __predict_from_params__(self, test_data, params, intercept, training_data):
        """
        Using the parameters and the intercept (that can be 0) learned from learning,
        and the training data used in the fitting process, predicts the values of test data
        Parameters
        ----------
        test_data : numpy.array
            Data to predict
        params : numpy.array
            Learned parameters
        intercept : numpy.array or int
            Intercept term (can be 0 if no intercept term was fitted)
        training_data : numpy.array
            Features used to train

        Returns
        -------
        numpy.array
            Predictions of test_data
        """
        # Training data
        N = training_data.shape[0]

        # Prediction function with gaussian kernel
        num_variables = params.shape[1]
        num_predictors = test_data.shape[0]
        prediction = np.zeros((num_predictors, num_variables))
        for i in range(num_predictors):
            x = np.atleast_2d(test_data[i])
            X_p = np.ones((N, 1)).dot(x)
            x_norm = np.sum((X_p - training_data) ** 2, axis=1)
            exponential = np.exp(-self._svr_gamma * x_norm)
            prediction[i, :] = params.T.dot(exponential)

        return prediction + intercept


""" HELPER FUNCTIONS """


def __fit_features__(fitter, X, y, intercept):
    """
    Fits the features from X to the observation y given the linear fitter, and computes
    the degrees of freedom for this fit
    Parameters
    ----------
    fitter : sklearn.svm.SVR
        Linear fitter that must have the fit method and the coef_ and intercept_ attributes
    X : numpy.array(NxF)
        Features array where N is the number of observations and F the number of features
    y : numpy.array(Nx1)
        The variable that we want to explain with the features
    intercept : Boolean
        Whether the intercept term must be computed or not

    Returns
    -------
    numpy.array(F+1,) or numpy.array(F+2,)
        Array with the fitting coefficients, the degrees of freedom
         and, if intercept=True, the intercept term. The order is the following:
         [(intercept) param1 param2 ... paramF degrees_of_freedom]
    """
    num_features = X.shape[1]
    if num_features <= 0:
        if intercept:
            # If the features array is empty and we need to compute the intercept term,
            # create dummy features to fit and then get only the intercept term
            X = np.ones((y.shape[0] + 1, 1))
        else:
            raise Exception("Features array X is not a NxF array")
    fitter.fit(X, y)

    if intercept:
        if num_features > 0:
            params = np.zeros((num_features + 2, 1))
            coefficients = np.atleast_2d(fitter.coef_)
            params[1:-1, :] = coefficients.T
        else:
            params = np.zeros((2, 1))  # Only the intercept term and df=1
            params[-1, :] = 1  # Hardcoded degrees_of_freedom, as we can't compute them
        params[0, :] = float(fitter.intercept_)
    else:
        params = np.zeros((num_features + 1, 1))
        params[:-1, :] = fitter.coef_.T

    # Get the variables needed to compute the df (only if needed, that is, df is 0)
    if params[-1] == 0:
        prediction = fitter.predict(X)
        dual_coeff = np.zeros((X.shape[0],))
        dual_coeff[fitter.support_] = np.ravel(fitter.dual_coef_)

        # Compute degrees of freedom for this fitting
        df = __compute_df_linSVR__(np.ravel(y), np.ravel(prediction), X, dual_coeff,
                                   fitter.C, fitter.epsilon)

        # Append df into params
        params[-1, :] = df

    # Return params
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
            X = np.ones((N, 1))
        else:
            raise Exception("Features array X is not a NxF array")
    fitter.fit(X, y)

    if intercept:
        if num_features > 0:
            params = np.zeros(N + 1)
            params[fitter.support_] = np.ravel(fitter.dual_coef_)
        else:
            params = np.zeros(1)  # Only the intercept term
        params[-1] = fitter.intercept_
    else:
        params = np.zeros(N)
        params[fitter.support_] = np.ravel(fitter.dual_coef_)
    return params


def __compute_df_linSVR__(observations, predicted_observations, features,
                          dual_coefficients, C, epsilon):
    """
    Computes the degrees of freedom for a linear SVR ( i.e. K(x_i, x_j) = <x_i, x_j> )

    Parameters
    ----------
    observations : numpy.array (N)
        Observations used to fit the data
    predicted_observations : numpy.array (N)
        Predicted observations from the features used to fit the real observations
    features : numpy.array (N x F)
        Features used to fit the observations
    dual_coefficients : numpy.array (N)
        Dual coefficients found after solving the SVR optimization problem
    C : float
        C regularization parameter used in SVR
    epsilon : float
        Amplitude of the epsilon-insensitive tube loss function used in SVR

    Returns
    -------
    int
        The degrees of freedom for this explained variable given the predicted
    """

    # Create kernel diagonal: K(x_i, x_i) = <x_i, x_i>
    kernel_diag = np.diag(features.dot(features.T))  # Column vector

    # Compute pseudoresiduals (refer to F. Dinuzzo et al.
    # On the Representer Theorem and Equivalent Degrees of Freedom of SVR
    # [http://www.jmlr.org/papers/volume8/dinuzzo07a/dinuzzo07a.pdf]
    pseudoresiduals = observations - predicted_observations + dual_coefficients * kernel_diag

    # Compute effective degrees of freedom from pseudoresiduals
    min_value = epsilon * np.ones(pseudoresiduals.shape)
    max_value = min_value + C * kernel_diag
    comp_min = min_value <= np.abs(pseudoresiduals)
    comp_max = np.abs(pseudoresiduals) <= max_value
    return np.sum(np.logical_and(comp_min, comp_max), axis=0)
