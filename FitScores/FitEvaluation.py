'''
#    import numpy as np
#    from scipy.stats import f as f_stat
#    
#    class evaluation_function:
#        def __init__(self, f):
#            self._eval_function = f
#            self._requirements = []
#            self._implicit = {}
#            self._default = {}
#            self._bound = {}
#    
#        def require(self, method_name):
#            self._requirements.append(method_name)
#    
#        def set_implicit(self, method_name, method):
#            self._implicit[method_name] = method
#    
#        def set_default(self, method_name, method):
#            self._default[method_name] = method
#    
#        def bind(self, fittertype, *args, **kwargs):
#            binding = 
#            for name in self._requirements.iteritems():
#                # method = kwargs (TODO: check for KeyErrors)
#                # call self.setattr(name, method) (or should it be setattribute?)
#    
#    #TODO: finish this
#    #TODO: add 'self' parameter to all the evaluation functions
#    
#        def __call__(self, correction_fitter, prediction_fitter, observations, correctors = None, correction_parameters = None, predictors = None, prediction_parameters = None, *args, **kwargs):
#            """Evaluates the degree to which the correctors and predictors get to explain the observational
#                data passed in the 'observations' argument.
#    
#                Parameters:
#    
#                    - observations: array-like structure of shape (N, X1, ..., Xn), representing the observational data,
#                        i.e., values obtained by measuring the variables of interest, whose behaviour is wanted to be
#                        explained by the correctors and predictors in the system, where M = X1*...*Xn is the number of
#                        variables and N the number of observations for each variable.
#    
#                    - correctors: NxC (2-dimensional) matrix (default None), representing the covariates, i.e., features
#                        that (may) explain a part of the observational data in which we are not interested, where C is
#                        the number of correctors and N the number of elements for each corrector. If set to None, the
#                        internal correctors will be used.
#    
#                    - correction_parameters: array-like structure of shape (Kc, X1, ..., Xn) (default None), representing
#                        the parameters to fit the correctors to the observations for each variable, where M = X1*...*Xn
#                        is the number of variables and Kc the number of correction parameters for each variable. If set
#                        to None, the correction parameters obtained in the last call to 'fit' will be used.
#    
#                    - predictors: NxR (2-dimensional) matrix (default None), representing the predictors, i.e., features
#                        to be used to try to explain/predict the observations (experimental data), where R is the number
#                        of predictors and N the number of elements for each predictor. If set to None, the internal re-
#                        gressors will be used.
#    
#                    - prediction_parameters: array-like structure of shape (Kr, X1, ..., Xn) (default None), representing
#                        the parameters to fit the predictors to the corrected observations for each variable, where M =
#                        X1*...*Xn is the number of variables and Kr is the number of prediction parameters for each
#                        variable. If set to None, the prediction parameters obtained in the last call to 'fit' will be
#                        used.
#    
#                    - any other arguments will be passed to the 'func' method.
#    
#                Returns:
#    
#                    - Fitting scores: array-like structure of shape (X1, ..., Xn), containing floats that indicate the
#                        goodness of the fit, that is, how well the predicted curves represent the corrected observational
#                        data, or equivalently, how well the model applied to the predictors explains the observed data.
#            """
#    
#            obs = np.array(observations, dtype = np.float64)
#            dims = obs.shape
#            obs = obs.reshape(dims[0], -1)
#    
#            
#            if 0 in dims:
#                raise ValueError('There are no elements in argument \'observations\'')
#    
#            if correctors is None:
#                cors = correction_fitter.correctors
#                if 0 in cors.shape:
#                    correctors_present = False
#                else:
#                    correctors_present = True
#            else:
#                cors = np.array(correctors, dtype = np.float64)
#                if len(cors.shape) != 2:
#                    raise TypeError('Argument \'correctors\' must be a 2-dimensional matrix')
#                
#                if 0 in cors.shape:
#                    raise ValueError('There are no elements in argument \'correctors\'')
#    
#                correctors_present = True
#    
#            if correctors_present:
#                if obs.shape[0] != cors.shape[0]:
#                    raise ValueError('The dimensions of the observations and the correctors are incompatible')
#    
#                if correction_parameters is None:
#                    cparams = correction_fitter.correction_parameters
#                    if 0 in cparams.shape:
#                        raise AttributeError('There are no correction parameters in this instance')
#                else:
#                    cparams = np.array(correction_parameters, dtype = np.float64)
#                    cparams = cparams.reshape(cparams.shape[0], -1)
#    
#                    if 0 in cparams.shape:
#                        raise ValueError('There are no elements in argument \'correction_parameters\'')
#    
#                if obs.shape[1] != cparams.shape[1]:
#                    raise ValueError('The dimensions of the observations and the correction parameters are incompatible')
#    
#            else:
#                cparams = np.zeros((0, 0))
#    
#    
#            if predictors is None:
#                preds = prediction_fitter.predictors
#                if 0 in preds.shape:
#                    raise AttributeError('There are no predictors in this instance')
#            else:
#                preds = np.array(predictors, dtype = np.float64)
#    
#                if len(preds.shape) != 2:
#                    raise TypeError('Argument \'predictors\' must be a 2-dimensional matrix')
#    
#                if 0 in preds.shape:
#                    raise ValueError('There are no elements in argument \'predictors\'')
#    
#            if obs.shape[0] != preds.shape[0]:
#                    raise ValueError('The dimensions of the observations and the predictors are incompatible')
#    
#            if prediction_parameters is None:
#                pparams = prediction_fitter.prediction_parameters
#                if 0 in pparams.shape:
#                    raise AttributeError('There are no prediction parameters in this instance')
#            else:
#                pparams = np.array(prediction_parameters, dtype = np.float64)
#                # Make matrix 2-dimensional
#                pparams = pparams.reshape(pparams.shape[0], -1)
#    
#                if 0 in pparams.shape:
#                    raise ValueError('There are no elements in argument \'prediction_parameters\'')
#    
#            if obs.shape[1] != pparams.shape[1]:
#                raise ValueError('The dimensions of the observations and the prediction parameters are incompatible')
#    
#            fitting_scores = self._eval_function(correction_fitter, prediction_fitter, obs, cors, cparams, preds, pparams, *args, **kwargs)
#    
#            return fitting_scores.reshape(dims[1:])
#    
#    
#    @evaluation_function
#    def mse(correction_fitter, prediction_fitter, observations, correctors, correction_parameters, predictors, prediction_parameters):
#        """Evaluates the significance of the predictors as regards the behaviour of the observations by computing
#            the Mean Squared Error of the prediction with respect to the corrected data. The smaller the result,
#            the better the fit.
#        """
#        if 0 in correctors.shape:
#            # There is no correction -> Corrected data is same as observations
#            corrected_data = observations
#        else:
#            # Compute corrected data
#            corrected_data = correction_fitter.correct(observations, correctors, correction_parameters)
#    
#        # prediction_error = corrected_data - prediction
#        prediction_error = corrected_data - prediction_fitter.predict(predictors, prediction_parameters)
#    
#        return (prediction_error**2).sum(axis = 0)/np.float64(len(prediction_error))
#    
#    @evaluation_function
#    def relvar(correction_fitter, prediction_fitter, observations, correctors, correction_parameters, predictors, prediction_parameters):
#        """Evaluates the significance of the predictors as regards the behaviour of the observations by computing
#            the relative variance of the prediction error with respect to the corrected data. In particular, this
#            method computes the ratio between the variance of the error and the variance of the corrected data.
#            Thus, the smaller the result, the better the fit.
#        """
#        if 0 in correctors.shape:
#            # There is no correction -> Corrected data is same as observations
#            corrected_data = observations
#        else:
#            # Compute corrected data
#            corrected_data = correction_fitter.correct(observations, correctors, correction_parameters)
#    
#        correction_variance = ((corrected_data - corrected_data.mean(axis = 0))**2).sum(axis = 0)
#        # We don't divide it by N-1 because the final ratio will eliminate this factor
#    
#        # prediction_error = corrected_data - prediction
#        prediction_error = corrected_data - prediction_fitter.predict(predictors, prediction_parameters)
#    
#        error_variance = ((prediction_error - prediction_error.mean(axis = 0))**2).sum(axis = 0)
#        # We don't divide it by N-1 because the final ratio will eliminate this factor
#    
#        return error_variance / correction_variance
#    
#    @evaluation_function
#    def fstat(correction_fitter, prediction_fitter, observations, correctors, correction_parameters, predictors, prediction_parameters):
#        """Evaluates the significance of the predictors as regards the behaviour of the observations by computing
#            the value of the F-statistic for a test in which the null hypothesis states that the predictors do not
#            explain the variation of the observations at all. The calculated F-statistic value compares the variance
#            of the prediction error with the variance of the corrected data, WITHOUT then mapping the result to its
#            corresponding p-value (which takes into account the degrees of freedom of both, the corrected data and
#            the prediction error). Please, refer to the "ftest" method if what you wish is a p-value related measure
#            rather than the F-statistic itself.
#        """
#        if 0 in correctors.shape:
#            # There is no correction -> Correction error is same as observations
#            correction_error = observations
#        else:
#            # Compute correction error
#            correction_error = correction_fitter.correct(observations, correctors, correction_parameters)
#    
#        ## Get the error obtained when using the full model (correctors + predictors)
#        # prediction = self.__predict__(predictors, prediction_parameters)
#    
#        # prediction_error = correction_error - prediction
#        prediction_error = correction_error - prediction_fitter.predict(predictors, prediction_parameters)
#    
#        ## Now compare the variances of the errors
#    
#        # Residual Sum of Squares for restricted model
#        rss1 = ((correction_error - correction_error.mean(axis = 0))**2).sum(axis = 0)
#        p1 = correctors.shape[1]
#    
#        # Residual Sum of Squares for full model
#        rss2 = (prediction_error**2).sum(axis = 0) # TODO: Check if this is correct or the following line should replace it
#        # rss2 = ((prediction_error - prediction_error.mean(axis = 0))**2).sum(axis = 0)
#        p2 = p1 + predictors.shape[1]
#    
#        # Degrees of freedom
#        n = observations.shape[0]
#        df1 = p2 - p1
#        df2 = n - p2
#    
#        # Compute f-scores and p-values
#        var1 = (rss1 - rss2)/df1
#        var2 = rss2/df2
#        f_score = var1/var2
#    
#        # print rss1, rss2
#        # print 'Df Residuals:', df2
#        # print 'Df Model:', df1
#        # print 'F-statistic:', f_score
#        # print 'R-squared:', 1 - rss2/rss1
#    
#        return f_score
#    
#    
#    @evaluation_function
#    def ftest(correction_fitter, prediction_fitter, observations, correctors, correction_parameters, predictors, prediction_parameters):
#        """Evaluates the significance of the predictors as regards the behaviour of the observations by performing
#            an F-test. In particular, the null hypothesis states that the predictors do not explain the variation
#            of the observations at all. The inverse of the p-value of such experiment (1 - p_value) is returned.
#            Refer to the "fstats" method if what you are looking for is the value of the f-statistic rather than
#            the p-value.
#        """
#    
#        f_score = fstats(correction_fitter, prediction_fitter, observations, correctors, correction_parameters, predictors, prediction_parameters)
#    
#        p1 = correctors.shape[1]
#        p2 = p1 + predictors.shape[1]
#        n = observations.shape[0]
#    
#        # Degrees of freedom
#        df1 = p2 - p1
#        df2 = n - p2
#    
#        return f_stat.cdf(f_score, df1, df2)
#    
#    
#    @evaluation_function
#    def aic(correction_fitter, prediction_fitter, observations, correctors, correction_parameters, predictors, prediction_parameters):
#        """Evaluates the significance of the predictors as regards the behaviour of the observations by computing
#            the Akaike Information Criterion (AIC).
#        """
#        k = correction_parameters.shape[0] + prediction_parameters.shape[0] + 1 # the residual error counts as an estimated variable
#        n = observations.shape[0]
#    
#        if 0 in correctors.shape:
#            # There is no correction -> Corrected data is same as observations
#            corrected_data = observations
#        else:
#            # Compute corrected data
#            corrected_data = correction_fitter.correct(observations, correctors, correction_parameters)
#    
#        err = corrected_data - prediction_fitter.predict(predictors, prediction_parameters)
#    
#        rss = (err**2).sum(axis = 0)
#    
#        return 2*k + n*np.log(rss)
#    
#    
#    @evaluation_function
#    def aicc(correction_fitter, prediction_fitter, observations, correctors, correction_parameters, predictors, prediction_parameters):
#        """Evaluates the significance of the predictors as regards the behaviour of the observations by computing
#            the corrected Akaike Information Criterion (AICc).
#        """
#        k = correction_parameters.shape[0] + prediction_parameters.shape[0] + 1 # the residual error counts as an estimated variable
#        n = observations.shape[0]
#    
#        return aic(correction_fitter, prediction_fitter, observations, correctors, correction_parameters, predictors, prediction_parameters) + 2*k*(k+1)/np.float64(n - k - 1)
#    
#    
#    
#    
#    
#    # Curve based measures
#    
#    #TODO: create some methods to evaluate the fit based on the curve, the predictions and the corrected data
#    # One method: Penalized Residual Sum of Squares (PSRR) = MSE / sum(d2(curve)/d(x2))
#    # Another method: Modified Penalized Residual Sum of Squares (MPSRR) = MSE / max(d2(curve)/d(x2))
#    
'''

import numpy as np
from scipy.stats import f as f_stat


def evaluation_function(func):
    def evaluation_wrapper(correction_fitter, prediction_fitter, observations, correctors = None, correction_parameters = None, predictors = None, prediction_parameters = None, *args, **kwargs):
        """Evaluates the degree to which the correctors and predictors get to explain the observational
            data passed in the 'observations' argument.
    
            Parameters:
    
                - observations: array-like structure of shape (N, X1, ..., Xn), representing the observational data,
                    i.e., values obtained by measuring the variables of interest, whose behaviour is wanted to be
                    explained by the correctors and predictors in the system, where M = X1*...*Xn is the number of
                    variables and N the number of observations for each variable.
    
                - correctors: NxC (2-dimensional) matrix (default None), representing the covariates, i.e., features
                    that (may) explain a part of the observational data in which we are not interested, where C is
                    the number of correctors and N the number of elements for each corrector. If set to None, the
                    internal correctors will be used.
    
                - correction_parameters: array-like structure of shape (Kc, X1, ..., Xn) (default None), representing
                    the parameters to fit the correctors to the observations for each variable, where M = X1*...*Xn
                    is the number of variables and Kc the number of correction parameters for each variable. If set
                    to None, the correction parameters obtained in the last call to 'fit' will be used.
    
                - predictors: NxR (2-dimensional) matrix (default None), representing the predictors, i.e., features
                    to be used to try to explain/predict the observations (experimental data), where R is the number
                    of predictors and N the number of elements for each predictor. If set to None, the internal re-
                    gressors will be used.
    
                - prediction_parameters: array-like structure of shape (Kr, X1, ..., Xn) (default None), representing
                    the parameters to fit the predictors to the corrected observations for each variable, where M =
                    X1*...*Xn is the number of variables and Kr is the number of prediction parameters for each
                    variable. If set to None, the prediction parameters obtained in the last call to 'fit' will be
                    used.
    
                - any other arguments will be passed to the 'func' method.
    
            Returns:
    
                - Fitting scores: array-like structure of shape (X1, ..., Xn), containing floats that indicate the
                    goodness of the fit, that is, how well the predicted curves represent the corrected observational
                    data, or equivalently, how well the model applied to the predictors explains the observed data.
        """
    
        obs = np.array(observations, dtype = np.float64)
        dims = obs.shape
        obs = obs.reshape(dims[0], -1)

        if 0 in dims:
            raise ValueError('There are no elements in argument \'observations\'')
    
        if correctors is None:
            cors = correction_fitter.correctors
            if 0 in cors.shape:
                correctors_present = False
            else:
                correctors_present = True
        else:
            cors = np.array(correctors, dtype = np.float64)
            if len(cors.shape) != 2:
                raise TypeError('Argument \'correctors\' must be a 2-dimensional matrix')
            
            if 0 in cors.shape:
                raise ValueError('There are no elements in argument \'correctors\'')
    
            correctors_present = True
    
        if correctors_present:
            if obs.shape[0] != cors.shape[0]:
                raise ValueError('The dimensions of the observations and the correctors are incompatible')
    
            if correction_parameters is None:
                cparams = correction_fitter.correction_parameters
                if 0 in cparams.shape:
                    raise AttributeError('There are no correction parameters in this instance')
            else:
                cparams = np.array(correction_parameters, dtype = np.float64)
                cparams = cparams.reshape(cparams.shape[0], -1)
    
                if 0 in cparams.shape:
                    raise ValueError('There are no elements in argument \'correction_parameters\'')
    
            if obs.shape[1] != cparams.shape[1]:
                raise ValueError('The dimensions of the observations and the correction parameters are incompatible')
    
        else:
            cparams = np.zeros((0, 0))

        if predictors is None:
            preds = prediction_fitter.predictors
            if 0 in preds.shape:
                raise AttributeError('There are no predictors in this instance')
        else:
            preds = np.array(predictors, dtype = np.float64)
    
            if len(preds.shape) != 2:
                raise TypeError('Argument \'predictors\' must be a 2-dimensional matrix')
    
            if 0 in preds.shape:
                raise ValueError('There are no elements in argument \'predictors\'')
    
        if obs.shape[0] != preds.shape[0]:
                raise ValueError('The dimensions of the observations and the predictors are incompatible')
    
        if prediction_parameters is None:
            pparams = prediction_fitter.prediction_parameters
            if 0 in pparams.shape:
                raise AttributeError('There are no prediction parameters in this instance')
        else:
            pparams = np.array(prediction_parameters, dtype = np.float64)
            # Make matrix 2-dimensional
            pparams = pparams.reshape(pparams.shape[0], -1)
    
            if 0 in pparams.shape:
                raise ValueError('There are no elements in argument \'prediction_parameters\'')
    
        if obs.shape[1] != pparams.shape[1]:
            raise ValueError('The dimensions of the observations and the prediction parameters are incompatible')
    
        fitting_scores = func(correction_fitter, prediction_fitter, obs, cors, cparams, preds, pparams, *args, **kwargs)
    
        return fitting_scores.reshape(dims[1:])
    
    return evaluation_wrapper


@evaluation_function
def mse(correction_fitter, prediction_fitter, observations, correctors, correction_parameters, predictors, prediction_parameters):
    """Evaluates the significance of the predictors as regards the behaviour of the observations by computing
        the Mean Squared Error of the prediction with respect to the corrected data. The smaller the result,
        the better the fit.
    """
    if 0 in correctors.shape:
        # There is no correction -> Corrected data is same as observations
        corrected_data = observations
    else:
        # Compute corrected data
        corrected_data = correction_fitter.correct(observations, correctors, correction_parameters)

    # prediction_error = corrected_data - prediction
    prediction_error = corrected_data - prediction_fitter.predict(predictors, prediction_parameters)

    return (prediction_error**2).sum(axis = 0)/np.float64(len(prediction_error))

@evaluation_function
def r2(correction_fitter, prediction_fitter, observations, correctors, correction_parameters, predictors, prediction_parameters):
    '''Evaluates the significance of the predictors as regards the behaviour of the observations by computing
        the value of the R-squared measurement, which is basically a range adjusted version of the MSE.
        In this case, however, the larger the result, the better the fit.
    '''
    if 0 in correctors.shape:
        # There is no correction -> Corrected data is same as observations
        corrected_data = observations
    else:
        # Compute corrected data
        corrected_data = correction_fitter.correct(observations, correctors, correction_parameters)

    correction_variance = ((corrected_data - corrected_data.mean(axis = 0))**2).sum(axis = 0)
    # We don't divide it by N-1 because the final ratio will eliminate this factor

    # prediction_error = corrected_data - prediction
    prediction_error = corrected_data - prediction_fitter.predict(predictors, prediction_parameters)

    error_variance = ((prediction_error - prediction_error.mean(axis = 0))**2).sum(axis = 0)
    # We don't divide it by N-1 because the final ratio will eliminate this factor

    return 1 - error_variance / correction_variance

@evaluation_function
def fstat(correction_fitter, prediction_fitter, observations, correctors, correction_parameters, predictors, prediction_parameters):
    """Evaluates the significance of the predictors as regards the behaviour of the observations by computing
        the value of the F-statistic for a test in which the null hypothesis states that the predictors do not
        explain the variation of the observations at all. The calculated F-statistic value compares the variance
        of the prediction error with the variance of the corrected data, WITHOUT then mapping the result to its
        corresponding p-value (which takes into account the degrees of freedom of both, the corrected data and
        the prediction error). Please, refer to the "ftest" method if what you wish is a p-value related measure
        rather than the F-statistic itself.
    """
    if 0 in correctors.shape:
        # There is no correction -> Correction error is same as observations
        correction_error = observations
    else:
        # Compute correction error
        correction_error = correction_fitter.correct(observations, correctors, correction_parameters)
    
    ## Get the error obtained when using the full model (correctors + predictors)
    # prediction = self.__predict__(predictors, prediction_parameters)
    
    # prediction_error = correction_error - prediction
    prediction_error = correction_error - prediction_fitter.predict(predictors, prediction_parameters)
    
    ## Now compare the variances of the errors
    
    # Residual Sum of Squares for restricted model
    rss1 = ((correction_error - correction_error.mean(axis = 0))**2).sum(axis = 0)
    p1 = correction_fitter.df_correction(observations, correctors, correction_parameters)
    
    # Residual Sum of Squares for full model
    rss2 = (prediction_error**2).sum(axis = 0) # TODO: Check if this is correct or the following line should replace it
    # rss2 = ((prediction_error - prediction_error.mean(axis = 0))**2).sum(axis = 0)
    p2 = p1 + prediction_fitter.df_prediction(observations, predictors, prediction_parameters)
    
    # Degrees of freedom
    n = observations.shape[0]
    df1 = p2 - p1
    df2 = n - p2
    
    # Compute f-scores and p-values
    var1 = (rss1 - rss2)/df1
    var2 = rss2/df2
    f_score = var1/var2
    
    # print rss1, rss2
    # print 'Df Residuals:', df2
    # print 'Df Model:', df1
    # print 'F-statistic:', f_score
    # print 'R-squared:', 1 - rss2/rss1
    
    return f_score

@evaluation_function
def ftest(correction_fitter, prediction_fitter, observations, correctors, correction_parameters, predictors, prediction_parameters):
    """Evaluates the significance of the predictors as regards the behaviour of the observations by performing
        an F-test. In particular, the null hypothesis states that the predictors do not explain the variation
        of the observations at all. The inverse of the p-value of such experiment (1 - p_value) is returned.
        Refer to the "fstats" method if what you are looking for is the value of the f-statistic rather than
        the p-value.
    """

    f_score = fstat(correction_fitter, prediction_fitter, observations, correctors, correction_parameters, predictors, prediction_parameters)

    p1 = correctors.shape[1]
    p2 = p1 + predictors.shape[1]
    n = observations.shape[0]

    # Degrees of freedom
    df1 = p2 - p1
    df2 = n - p2

    return f_stat.cdf(f_score, df1, df2)


@evaluation_function
def aic(correction_fitter, prediction_fitter, observations, correctors, correction_parameters, predictors, prediction_parameters):
    """Evaluates the significance of the predictors as regards the behaviour of the observations by computing
        the Akaike Information Criterion (AIC).
    """
    k = correction_parameters.shape[0] + prediction_parameters.shape[0] + 1 # the residual error counts as an estimated variable
    n = observations.shape[0]

    if 0 in correctors.shape:
        # There is no correction -> Corrected data is same as observations
        corrected_data = observations
    else:
        # Compute corrected data
        corrected_data = correction_fitter.correct(observations, correctors, correction_parameters)

    err = corrected_data - prediction_fitter.predict(predictors, prediction_parameters)

    rss = (err**2).sum(axis = 0)

    return 2*k + n*np.log(rss)


@evaluation_function
def aicc(correction_fitter, prediction_fitter, observations, correctors, correction_parameters, predictors, prediction_parameters):
    """Evaluates the significance of the predictors as regards the behaviour of the observations by computing
        the corrected Akaike Information Criterion (AICc).
    """
    k = correction_parameters.shape[0] + prediction_parameters.shape[0] + 1 # the residual error counts as an estimated variable
    n = observations.shape[0]

    return aic(correction_fitter, prediction_fitter, observations, correctors, correction_parameters, predictors, prediction_parameters) + 2*k*(k+1)/np.float64(n - k - 1)


# Curve based measures

#TODO: create some methods to evaluate the fit based on the curve, the predictions and the corrected data
# One method: Penalized Residual Sum of Squares (PSRR) = MSE / sum(d2(curve)/d(x2))
# Another method: Modified Penalized Residual Sum of Squares (MPSRR) = MSE / max(d2(curve)/d(x2))