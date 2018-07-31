from abc import abstractmethod
from warnings import warn

import numpy as np
from numpy import float64
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splev
from sklearn.linear_model import LinearRegression as LR
import statsmodels.api as sm
from patsy import dmatrix
import pandas as pd
from vneat.Fitters.CurveFitting import AdditiveCurveFitter



class GAM(AdditiveCurveFitter):
    """
    Generalized Additive Model with non-parametric, smoothed components
    """

    def __init__(self, corrector_smoothers=None, predictor_smoothers=None, intercept=AdditiveCurveFitter.PredictionIntercept):


        if corrector_smoothers is None or not corrector_smoothers:
            correctors = None
        else:
            correctors = corrector_smoothers.get_covariates()

        if predictor_smoothers is None or not predictor_smoothers:
            predictors = None
        else:
            predictors = predictor_smoothers.get_covariates()
        self._gam_intercept = intercept
        self.intercept_smoother = InterceptSmoother(1)
        self.predictor_smoothers = SmootherSet(predictor_smoothers)
        self.corrector_smoothers = SmootherSet(corrector_smoothers)

        super(GAM, self).__init__(predictors=predictors, correctors=correctors, intercept=AdditiveCurveFitter.NoIntercept)

    def __fit__(self, correctors, predictors, observations, rtol=1.0e-08, maxiter=500, *args, **kwargs):

        dims = observations.shape
        for smoother, corr in zip(self.corrector_smoothers, correctors.T[1:]):
            smoother.set_covariate(corr.reshape(dims[0], -1))

        for smoother, reg in zip(self.predictor_smoothers, predictors.T):
            smoother.set_covariate(reg.reshape(dims[0], -1))

        smoother_functions = SmootherSet(self.corrector_smoothers + self.predictor_smoothers)
        crv_reg = []
        crv_corr = []
        for obs in observations.T:
            alpha, smoothers = self.__backfitting_algorithm(obs, smoother_functions, rtol=rtol, maxiter=maxiter, *args,
                                                            **kwargs)

            self.intercept_smoother.set_parameters(alpha)
            self.corrector_smoothers = SmootherSet(smoothers[:self.corrector_smoothers.n_smoothers])
            self.predictor_smoothers = SmootherSet(smoothers[self.corrector_smoothers.n_smoothers:])

            if self._gam_intercept == AdditiveCurveFitter.PredictionIntercept:
                corr = self.__code_parameters(self.corrector_smoothers)
                pred = np.concatenate((np.array([TYPE_SMOOTHER.index(InterceptSmoother), 1, self.alpha]),
                                       self.__code_parameters(self.predictor_smoothers)))
            else:
                corr = np.concatenate((np.array([TYPE_SMOOTHER.index(InterceptSmoother), 1, self.alpha]),
                                       self.__code_parameters(self.corrector_smoothers)))
                pred = self.__code_parameters(self.predictor_smoothers)

            crv_corr.append(corr)
            crv_reg.append(pred)

        return (np.array(crv_corr).T, np.array(crv_reg).T)

    def __predict__(self, predictors, prediction_parameters, *args, **kwargs):

        y_predict = []
        for pred_param in prediction_parameters.T:
            indx_pred = 0
            indx_smthr = 0
            while indx_smthr < len(pred_param):
                y_pred = np.zeros((predictors.shape[0],))
                if pred_param[indx_smthr] == 0:
                    y_pred += pred_param[indx_smthr + 2]
                    indx_smthr += pred_param[1] + 2
                else:
                    pred = predictors[:, indx_pred]
                    smoother = TYPE_SMOOTHER[int(pred_param[indx_smthr])](pred)

                    ###### VALID!!!!! ####### for adContinuum
                    n_params = int(pred_param[indx_smthr + 1])
                    smoother.set_parameters(pred_param[indx_smthr + 2:indx_smthr + 2 + n_params])
                    indx_smthr += n_params + 2
                    #########################

                    ###### TEMPORAL!!! ####### for agingAPOE
                    # n_params = 6
                    # smoother.set_parameters(pred_param[indx_smthr + 1:indx_smthr + 1 + n_params])
                    # indx_smthr += n_params + 1
                    #########################
                    indx_pred += 1
                    y_pred += smoother.predict()
            y_predict.append(y_pred)

        return np.asarray(y_predict).T

    def __init_iter(self, observations, n_smoothers):
        self.iter = 0
        self.alpha = np.mean(observations, axis=0)
        mu = np.zeros((observations.shape[0],), np.float64)
        offset = np.zeros((n_smoothers,), np.float64)
        return self.alpha, mu, offset

    def __cont(self, convergence_num, convergence_den, maxiter, rtol):
        if self.iter == 0:
            self.iter += 1
            return True

        if self.iter > maxiter:
            print(self.iter)
            return False
        if (convergence_num / (1 + convergence_den)) < rtol:
            return False

        return True

    def __code_parameters(self, smoother_set):
        parameters = np.array([])
        for smoother in smoother_set:
            params = smoother.get_parameters()
            if not isinstance(params, (list,np.ndarray)):
                params = [params]
            parameters = np.concatenate((parameters, [TYPE_SMOOTHER.index(smoother.__class__)], params))
        return parameters

    def __backfitting_algorithm(self, observation, smoother_functions, rtol=1e-8, maxiter=500, *args, **kwargs):

        N = observation.shape[0]
        alpha, mu, offset = self.__init_iter(observation, smoother_functions.n_smoothers)

        for smoother in smoother_functions:
            r = observation - alpha - mu
            smoother.fit(r, *args, **kwargs)
            f_i_pred = smoother.predict()
            offset = f_i_pred.sum() / N
            f_i_pred -= offset
            mu += f_i_pred
        self.iter += 1

        mu_old = 0
        convergence_num = sum(mu ** 2)
        while self.__cont(convergence_num, mu_old, maxiter, rtol):
            mu_old = sum(mu ** 2)
            convergence_num = 0
            for smoother in smoother_functions:
                f_i_prev = smoother.predict() - smoother.predict().sum() / N
                mu = mu - f_i_prev
                r = observation - alpha - mu
                smoother.fit(r, *args, **kwargs)
                f_i_pred = smoother.predict()
                offset = f_i_pred.sum() / N
                f_i_pred -= offset
                mu += f_i_pred
                convergence_num = convergence_num + sum((f_i_prev - f_i_pred) ** 2)
            self.iter += 1

        return (alpha, smoother_functions)

    def __df_correction__(self, observations, correctors, correction_parameters):

        df, df_partial = [], []
        for reg_param in correction_parameters.T:
            y_pred = np.zeros((correctors.shape[0],))
            if reg_param[0] == 0:
                y_pred += reg_param[2]
                indx_smthr = 3
            else:
                indx_smthr = 0
            for corr in correctors.T:
                smoother = TYPE_SMOOTHER[int(reg_param[indx_smthr])](corr)
                df_partial.append(smoother.df_mo)
        df.append(df_partial)
        return np.asarray(df)

    def __df_prediction__(self, observations, predictors, prediction_parameters):
        df = []
        for obs, pred_param in zip(observations.T, prediction_parameters.T):
            indx_smthr = 0
            indx_pred = 0
            df_partial = 0
            y_pred = np.zeros((predictors.shape[0],))
            while indx_smthr < len(pred_param):
                if pred_param[indx_smthr] == 0:
                    y_pred += pred_param[2]
                    indx_smthr += pred_param[1] + 2
                else:
                    indx_smthr = 0
                for pred in predictors.T:
                    pred = predictors[:, indx_pred]
                    smoother = TYPE_SMOOTHER[int(pred_param[indx_smthr])](pred)

                    ###### VALID!!!!! ####### for adContinuum
                    n_params = int(pred_param[indx_smthr + 1])
                    df_partial += smoother.df_model(obs, pred_param[indx_smthr + 2:indx_smthr + 2 + n_params])
                    indx_smthr += n_params + 2
                    #########################

                    ###### TEMPORAL!!! ####### for agingApoe
                    # n_params = 6
                    # df_partial += smoother.df_model(obs, pred_param[indx_smthr + 1:indx_smthr + 1 + n_params])
                    # indx_smthr += n_params + 1
                    #########################

                    indx_pred += 1
            df.append(df_partial)
        return np.asarray(df)


class SmootherSet(list):
    def __init__(self, smoothers=None):
        self.n_smoothers = 0
        if smoothers is not None:
            self.extend(smoothers)

    def extend(self, smoothers, name=None):
        if isinstance(smoothers, list):
            self.n_smoothers += len(smoothers)
            super(SmootherSet, self).extend(smoothers)
        else:
            self.n_smoothers += 1
            super(SmootherSet, self).append(smoothers)

    def get_covariates(self):
        return np.array([smoother.get_covariate() for smoother in self]).T

    def get_parameters(self):
        return np.array([smoother.get_parameters() for smoother in self])


class Smoother():
    @abstractmethod
    def fit(self, ydata, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_parameters(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_covariate(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def set_covariate(self, covariate, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def set_parameters(self, parameters, *args, **kwargs):
        raise NotImplementedError()


class SplinesSmoother(Smoother):

    _name = 'SplinesSmoother'
    def __init__(self, xdata, order=3, smoothing_factor=None, df=None, spline_parameters=None):
        # Implements smoothing splines regression
        # xdata:
        # order:
        # smoothing_factor:
        #
        # Spline parameters coding: TYPE_SMOOTHER, ParameterLenght, SmoothingFactor,
        #                           n_Knots, Knots, n_Coeff, Coeff, 1, Order

        if smoothing_factor is None:
            smoothing_factor = len(xdata)
        self.smoothing_factor = smoothing_factor
        self.df = df
        self.order = order

        xdata = np.squeeze(xdata)
        for x in xdata:
            index_xdata = np.where(xdata == x)[0]
            xdata[index_xdata] = x * 0.00001 * np.random.randn(len(index_xdata))

        self.xdata = np.sort(xdata)
        self.index_xdata = np.argsort(xdata)
        self.spline_parameters = spline_parameters

    def df_model(self, ydata, parameters=None):
        """
        Degrees of freedom used in the fit.
        """

        if parameters is not None:
            self.set_parameters(parameters)

        std_estimation = np.std(ydata)
        if std_estimation == 0:
            weights = None
        else:
            weights = 1 / std_estimation * np.ones(len(ydata))
        spline = UnivariateSpline(self.xdata, ydata[self.index_xdata], k=self.order, s=self.smoothing_factor,
                                  w=weights)
        return len(spline.get_coeffs())

    def df_resid(self, ydata, parameters=None):
        """
        Residual degrees of freedom from last fit.
        """
        return self.N - self.df_model(ydata, parameters=parameters)

    def fit(self, ydata, *args, **kwargs):

        self.df = kwargs['df'] if 'df' in kwargs else self.df

        if self.df is not None:
            self.smoothing_factor = self.compute_smoothing_factor(ydata, self.df)
        elif 's' in kwargs:
            self.smoothing_factor = kwargs['smoothing_factor']

        if ydata.ndim == 1:
            ydata = ydata[:, None]

        std_estimation = np.std(ydata)
        if std_estimation == 0:
            weights = None
        else:
            weights = 1 / std_estimation * np.ones(len(ydata))



        spline = UnivariateSpline(self.xdata, ydata[self.index_xdata], k=self.order, s=self.smoothing_factor,
                                  w=weights)
        # / (max(ydata) * np.sqrt(len(ydata))) * np.ones(len(ydata))
        self.spline_parameters = spline._eval_args  # spline.get_knots(),spline.get_coeffs(),self.order

    def predict(self, xdata=None, spline_parameters=None, *args, **kwargs):

        if xdata is None:
            xdata = self.xdata
        else:
            xdata = np.sort(xdata)
        if xdata.ndim > 1:
            raise ValueError("Each smoother must have a single covariate.")

        if spline_parameters is None:
            if self.spline_parameters is None:
                warn(
                    "Spline parameters are not specified, you should either fit a model or specify them. "
                    "Output is set to 0")
                return np.zeros((xdata.shape[0], 1))
            else:
                spline_parameters = self.spline_parameters

        y_pred = splev(xdata, spline_parameters)
        if np.any(np.isnan(y_pred)):
            warn("Spline parameters are too restrictive that it cannot predict. Output is set to 0")
            return np.zeros((xdata.shape[0],))

        # if len(y_pred.shape) == 1:
        #     y_pred=y_pred[...,np.newaxis]
        return y_pred

    def get_parameters(self):
        shape_parameters = 2 * (self.xdata.shape[0] + self.order + 1) + 7
        parameters = np.append(1, self.smoothing_factor)
        for param in self.spline_parameters:
            try:
                parameters = np.append(parameters, len(param))
                parameters = np.append(parameters, [p for p in param])
            except:
                parameters = np.append(parameters, 1)
                parameters = np.append(parameters, param)

        parameters = np.append(shape_parameters, parameters)

        parameters_reshaped = np.zeros((shape_parameters,))
        parameters_reshaped[:len(parameters)] = parameters
        return parameters_reshaped

    def get_covariate(self):
        return np.array(self.xdata)

    def set_covariate(self, covariate, *args, **kwargs):
        self.xdata = np.squeeze(covariate)

    def set_parameters(self, parameters, *args, **kwargs):

        parameters = np.asarray(parameters).reshape((-1,))
        if parameters[0] == 0:
            self.df = parameters[1]
        else:
            self.smoothing_factor = parameters[1]

        try:
            n_knots = int(parameters[2])
            n_coeff = int(parameters[3 + n_knots])
            self.spline_parameters = tuple([parameters[3:3 + n_knots], parameters[4 + n_knots:4 + n_knots + n_coeff],
                                            int(parameters[5 + n_knots + n_coeff])])
            self.order = int(parameters[5 + n_knots + n_coeff])
        except:
            self.order = int(parameters[-1])

    def compute_smoothing_factor(self, ydata, df_target, xdata=None):

        # Check if std of data is 0 to avoid unnecessary computations
        if np.std(ydata) == 0:
            return len(ydata)

        found = False
        change = True
        tol = 1e-6
        s = 100
        step = 20.0
        max_iter = 50
        n_iter = 0
        while not found:
            n_iter += 1
            df = self.df_model(ydata, parameters=[1, s, self.order])
            if df == df_target:
                found = True
            elif df < df_target:
                if change is True:
                    change = False
                    step -= step / 2
                s -= step
            else:
                if change is False:
                    change = True
                    step -= step / 2
                s += step
            if step < tol:
                warning_msg = "WARNING: Couldn't find a curve with the desired number of degrees of freedom. " \
                              "(Df - 1) has been chosen"
                warn(warning_msg)
                s = self.compute_smoothing_factor(ydata, df_target - 1, xdata=xdata)
                found = True
            if n_iter >= max_iter:
                warning_msg = "WARNING: Reached maximum number of iterations. df={} has been chosen".format(df)
                warn(warning_msg)
                found = True

        return s

    @staticmethod
    def name():
        return SplinesSmoother._name


class RegressionSplinesSmoother(Smoother):
    # http://patsy.readthedocs.io/en/latest/spline-regression.html


    _spline_type_list = ['bs','cr'] #B-splines and Natural splines respectively. Degree of the polynomial is set to 3.
    _name = 'RegressionSplinesSmoother'

    def __init__(self, xdata, df=None, spline_type=None, spline_parameters=None):

        self.xdata = xdata
        self.df = df
        self.spline_type = spline_type
        self.spline_parameters = spline_parameters

    def df_model(self, ydata, parameters=None):
        if parameters is not None:
            self.set_parameters(parameters)

        return self.df

    def df_resid(self, ydata, parameters=None):
        if parameters is not None:
            self.set_parameters(parameters)

        return ydata.shape[0] - self.df_model(ydata,parameters=parameters)

    def fit(self, ydata, *args, **kwargs):
        if self.spline_type == RegressionSplinesSmoother._spline_type_list.index('bs'):
            transformed_matrix = dmatrix("bs(predictor, df="+str(self.df)+",include_intercept=False)",
                                     {"predictor": pd.Series(self.xdata)})
        elif self.spline_type == RegressionSplinesSmoother._spline_type_list.index('cr'):
            transformed_matrix =dmatrix('cr(predictor, df='+str(self.df)+')',
                                        {'predictor': pd.Series(self.xdata)}, return_type='dataframe')
        else:
            raise ValueError('Please, specify a valid splines_type')

        fit_wrapper = LR(fit_intercept=False).fit(transformed_matrix,ydata)

        self.spline_parameters = list(fit_wrapper.coef_)

    def predict(self, xdata=None, spline_parameters=None, df=None, spline_type=None, *args, **kwargs):
        if xdata is None:
            xdata = self.xdata

        if spline_parameters is None:
            if self.spline_parameters is None:
                warn("Spline parameters are not specified, you should either fit a model or specify them. "
                     "Output is set to 0")
                return np.zeros((xdata.shape[0], 1))
            else:
                spline_parameters = self.spline_parameters

        if df is None:
            if self.df is None:
                warn("Splines df is not specified, you should either fit a model or specify them. "
                     "Output is set to 0")
                return np.zeros((xdata.shape[0], 1))
            else:
                df = self.df

        if spline_type is None:
            if self.spline_type is None:
                warn("Spline spline_type are not specified, you should either fit a model or specify them. "
                     "Output is set to 0")
                return np.zeros((xdata.shape[0], 1))
            else:
                spline_type = self.spline_type

        fit_wrapper = LR(fit_intercept=False)
        fit_wrapper.coef=spline_parameters

        if spline_type == RegressionSplinesSmoother._spline_type_list.index('bs'):
            y_pred = np.dot(dmatrix("bs(predictor, df="+str(self.df)+", include_intercept=False)",
                                         {"predictor": xdata}, return_type='dataframe'),spline_parameters)

        elif spline_type == RegressionSplinesSmoother._spline_type_list.index('cr'):
            y_pred = np.dot(dmatrix("cr(predictor, df="+str(self.df)+")",
                                    {"predictor": xdata}, return_type='dataframe'),spline_parameters)

        else:
            raise ValueError('Please, specify a valid splines_type')


        return y_pred

    def get_covariate(self):
        return np.array(self.xdata)

    def set_covariate(self, covariate, *args, **kwargs):
        self.xdata = np.squeeze(covariate)

    def get_parameters(self):

        parameters = np.asarray([2+len(self.spline_parameters),self.spline_type, self.df] + self.spline_parameters)

        return parameters

    def set_parameters(self, parameters, *args, **kwargs):
        self.spline_type = parameters[0]
        self.df = parameters[1]
        self.spline_parameters = parameters[2:]

    @staticmethod
    def name():
        return RegressionSplinesSmoother._name

class PolynomialSmoother(Smoother):
    """
    Polynomial smoother up to a given order.
    """
    _name = 'PolynomialSmoother'

    def __init__(self, xdata, order=3, coefficients=None):

        self.order = order
        xdata = np.squeeze(xdata)
        if xdata.ndim > 1:
            raise ValueError("Error, each smoother a single covariate associated.")

        self.xdata = xdata

        if coefficients is None:
            coefficients = np.zeros((order + 1,), np.float64)
        self.coefficients = coefficients

        self._N = len(xdata)

    def fit(self, ydata, sample_weight=None, *args, **kwargs):
        try:
            n_jobs = kwargs['n_jobs']
        except KeyError:
            n_jobs = -1
        curve = LR(fit_intercept=False, normalize=False, copy_X=False, n_jobs=n_jobs)

        xdata = np.array([np.squeeze(self.xdata) ** i for i in range(self.order + 1)]).T
        curve.fit(xdata, ydata, sample_weight)
        self.coefficients = curve.coef_.T

    def predict(self, xdata=None, coefficients=None):
        if xdata is None:
            xdata = self.xdata
        elif xdata.ndim > 1:
            raise ValueError("Each smoother must have a single covariate.")

        if coefficients is None:
            if self.coefficients is None:
                warn(
                    "Polynomial coefficients are not specified, you should either fit a model or specify them. "
                    "Output is set to 0"
                )
                return np.zeros((xdata.shape[0], 1))
            else:
                coefficients = self.coefficients

        xdata = np.array([np.squeeze(xdata) ** i for i in range(self.order + 1)]).T
        y_pred = xdata.dot(coefficients)

        return y_pred

    def get_parameters(self, prediction_parameters = None, name=None):
        params = np.append((len(self.coefficients) + 1, self.order), self.coefficients)
        return params

    def set_parameters(self, parameters, *args, **kwargs):
        self.order = int(parameters[0])
        self.coefficients = np.asarray(parameters[1:])

    def get_covariate(self):
        return np.array(self.xdata)

    def set_covariate(self, covariate, *args, **kwargs):
        self.xdata = np.squeeze(covariate)

    @staticmethod
    def name():
        return PolynomialSmoother._name

    def df_model(self, ydata, parameters=None):
        """
        Degrees of freedom used in the fit.
        """
        if parameters is not None:
            self.set_parameters(parameters)

        return self.order + 1

    def df_resid(self, ydata, parameters=None):
        """
        Residual degrees of freedom from last fit.
        """
        return self._N - self.df_model(parameters=parameters)

class InterceptSmoother(Smoother):
    _name = 'InterceptSmoother'


    def __init__(self, xdata, alpha=None):
        self.xdata = xdata
        self.alpha = alpha

    def fit(self, ydata, *args, **kwargs):
        self.alpha = np.mean(ydata)

    def predict(self):
        # try:
        #     dims = self.xdata.shape
        # except:
        #     dims=1

        return self.alpha

    def get_parameters(self):
        return self.alpha

    def get_covariate(self):
        return 1

    def set_covariate(self, covariate, *args, **kwargs):
        pass

    def set_parameters(self, alpha, *args, **kwargs):
        self.alpha = alpha

    @staticmethod
    def name():
        return InterceptSmoother._name

    def df_model(self, ydata, parameters=None):

        return 1


class KernelSmoother(Smoother):

    _name = 'KernelSmoother'
    def __init__(self, xdata, std_kernel=1, name=None):

        xdata = np.squeeze(xdata)
        if xdata.ndim > 1:
            raise ValueError("Error, each smoother a single covariate associated.")

        self.xdata = xdata
        self.std_kernel = std_kernel
        self.Kernel = GaussianKernel(std_kernel)

        if name is None:
            name = 'KernelSmoother'

        self._name = name
        self._N = len(xdata)

    def fit(self, ydata, *args, **kwargs):
        self.ydata = ydata

    def predict(self, x = None, ydata=None):
        """
        Returns the kernel smoothed prediction at x

        If x is a real number then a single value is returned.

        Otherwise an attempt is made to cast x to numpy.ndarray and an array of
        corresponding y-points is returned.
        """

        if x is None:
            x = self.xdata
        elif x.ndim > 1:
            raise ValueError("Each smoother must have a single covariate.")

        if ydata is None:
            if not hasattr(self, 'ydata'):
                raise ValueError("You should fit your data before predicting")
            else:
                ydata = self.ydata


        if np.shape(x) == 1:
            return self.Kernel.smooth(self.xdata, ydata, x)
        else:
            return np.array([self.Kernel.smooth(self.xdata, self.ydata, xx) for xx
                             in np.array(x)])

    def get_parameters(self):
        params = self.std_kernel
        return params

    def set_parameters(self, parameters, *args, **kwargs):
        self.std_kernel = parameters[0] if isinstance(parameters,(list,np.ndarray)) else parameters
        self.Kernel = GaussianKernel(self.std_kernel)

    def get_covariate(self):
        return np.array(self.xdata)

    def set_covariate(self, covariate, *args, **kwargs):
        self.xdata = np.squeeze(covariate)

    @staticmethod
    def name():
        return KernelSmoother._name

    def df_model(self):
        """
        Degrees of freedom used in the fit.
        """
        return self.Kernel.df(self.xdata)

    def df_resid(self, ydata, parameters=None):
        """
        Residual degrees of freedom from last fit.
        """
        return self._N - self.df_model()

    # def conf(self, x):
    #     """
    #     Returns the fitted curve and 1-sigma upper and lower point-wise
    #     confidence.
    #     These bounds are based on variance only, and do not include the bias.
    #     If the bandwidth is much larger than the curvature of the underlying
    #     funtion then the bias could be large.
    #
    #     x is the points on which you want to evaluate the fit and the errors.
    #
    #     Alternatively if x is specified as a positive integer, then the fit and
    #     confidence bands points will be returned after every
    #     xth sample point - so they are closer together where the data
    #     is denser.
    #     """
    #     if isinstance(x, int):
    #         sorted_x = np.array(self.x)
    #         sorted_x.sort()
    #         confx = sorted_x[::x]
    #         conffit = self.conf(confx)
    #         return (confx, conffit)
    #     else:
    #         return np.array([self.Kernel.smoothconf(self.x, self.y, xx)
    #                          for xx in x])
    #
    # def var(self, x):
    #     return np.array([self.Kernel.smoothvar(self.x, self.y, xx) for xx in x])
    #
    # def std(self, x):
    #     return np.sqrt(self.var(x))
    #


class GaussianKernel:
    """
    Gaussian (Normal) Kernel

    K(u) = 1 / (sqrt(2*pi)) exp(-0.5 u**2)
    """

    def __init__(self, sigma=1.0):
        print(sigma)
        self.sigma = sigma

    def fit(self, predictors, observations):
        pass

    def predict(self):
        pass

    def smooth(self, xs, ys, x):
        """Returns the kernel smoothing estimate for point x based on x-values
            xs and y-values ys.
            Not expected to be called by the user.

            Special implementation optimised for Gaussian.
            """
        w = np.sum(np.exp(np.multiply(np.square(np.divide(np.subtract(xs, x), self.sigma)), -0.5)))
        v = np.sum(np.multiply(ys, np.exp(np.multiply(np.square(np.divide(np.subtract(xs, x), self.sigma)), -0.5))))

        return v / w

    def df(self, x):
        M = x.shape[0]
        S = np.zeros((M,M))
        for i in range(M):
            S[i,:] = np.exp(np.multiply(np.square(np.divide(np.subtract(x, x[i]), self.sigma)), -0.5))
            S[i,:] = S[i,:] / np.sum(S[i,:])

        return np.trace(np.dot(S,S.T))




TYPE_SMOOTHER = [InterceptSmoother, PolynomialSmoother, SplinesSmoother, RegressionSplinesSmoother, KernelSmoother]
