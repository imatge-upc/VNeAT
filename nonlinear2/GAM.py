from abc import abstractmethod
from CurveFitting import AdditiveCurveFitter
from numpy import zeros, float64, array as nparray
import numpy as np
from statsmodels.genmod import families
from statsmodels.sandbox.nonparametric import kernels
from sklearn.linear_model import LinearRegression as LR
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splev
# import matplotlib.pyplot as plt
from warnings import warn
import copy


class GAM(AdditiveCurveFitter):
    '''
    Additive model with non-parametric, smoothed components
    '''

    def __init__(self, corrector_smoothers=None, predictor_smoothers=None):

        self.TYPE_SMOOTHER = [InterceptSmoother, PolynomialSmoother, SplinesSmoother]

        if corrector_smoothers is None or not corrector_smoothers:
            correctors = None
        else:
            correctors = corrector_smoothers.get_covariates()

        if predictor_smoothers is None and not predictor_smoothers:
            predictors = None
        else:
            predictors = predictor_smoothers.get_covariates()

        self.intercept_smoother = InterceptSmoother(1)
        self.predictor_smoothers = SmootherSet(predictor_smoothers)
        self.corrector_smoothers = SmootherSet(corrector_smoothers)

        super(GAM, self).__init__(predictors, correctors, True)

    def __fit__(self, correctors, predictors, observations, rtol=1.0e-08, maxiter=500):

        dims = observations.shape

        [smoother.set_covariate(corr.reshape(dims[0], -1)) for smoother, corr in
         zip(self.corrector_smoothers, correctors.T[1:])]
        [smoother.set_covariate(reg.reshape(dims[0], -1)) for smoother, reg in
         zip(self.predictor_smoothers, predictors.T)]

        smoother_functions = SmootherSet(self.corrector_smoothers + self.predictor_smoothers)
        crv_reg = []
        crv_corr = []
        for obs in observations.T:
            corr, reg = self.__backfitting_algorithm(obs, smoother_functions, rtol=rtol, maxiter=maxiter)
            crv_corr.append(corr)
            crv_reg.append(reg)

        return (np.array(crv_corr).T, np.array(crv_reg).T)

    def __predict__(self, predictors, prediction_parameters):

        y_predict = []
        for reg_param in prediction_parameters.T:
            y_pred = np.zeros((predictors.shape[0],))
            indx_smthr = 0
            for reg in predictors.T:
                smoother = self.TYPE_SMOOTHER[int(reg_param[indx_smthr])](reg)
                n_params = int(reg_param[indx_smthr + 1])
                smoother.set_parameters(reg_param[indx_smthr + 2:indx_smthr + 2 + n_params])
                indx_smthr += n_params + 2
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
            parameters = np.concatenate((parameters, [self.TYPE_SMOOTHER.index(smoother.__class__)], params))
        return parameters

    def __backfitting_algorithm(self, observation, smoother_functions, rtol=1e-8, maxiter=500):

        N = observation.shape[0]
        alpha, mu, offset = self.__init_iter(observation, smoother_functions.n_smoothers)

        for smoother in smoother_functions:
            r = observation - alpha - mu
            smoother.fit(r)
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
                smoother.fit(r)
                f_i_pred = smoother.predict()
                offset = f_i_pred.sum() / N
                f_i_pred -= offset
                mu += f_i_pred
                convergence_num = convergence_num + sum((f_i_prev - f_i_pred) ** 2)
            self.iter += 1

        self.intercept_smoother.set_parameters(self.alpha)
        self.corrector_smoothers = SmootherSet(smoother_functions[:self.corrector_smoothers.n_smoothers])
        self.predictor_smoothers = SmootherSet(smoother_functions[self.corrector_smoothers.n_smoothers:])
        return (np.concatenate((np.array([self.TYPE_SMOOTHER.index(InterceptSmoother), 1, self.alpha]),
                                self.__code_parameters(self.corrector_smoothers))),
                self.__code_parameters(self.predictor_smoothers))


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
    def fit(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, ydata, *args, **kwargs):
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
    def __init__(self, xdata, order=3, smoothing_factor=None, spline_parameters=None, name=None):

        if smoothing_factor is None:
            smoothing_factor = len(xdata)
        self.smoothing_factor = smoothing_factor
        self.order = order
        self.xdata = np.sort(xdata)
        self.spline_parameters = spline_parameters
        if name is None:
            name = 'SplinesSmoother'
        self._name = name

    def df_model(self, parameters=None):
        pass

    #     """
    #     Degrees of freedom used in the fit.
    #     """
    #     return (parameters[2]+1)*(parameters[0]+1)-parameters[2]*parameters[0]

    def df_resid(self, parameters=None):
        pass

    #     """
    #     Residual degrees of freedom from last fit.
    #     """
    #     return self.N - self.df_model(parameters=parameters)


    def fit(self, ydata):
        if ydata.ndim == 1:
            ydata = ydata[:, None]
        spline = UnivariateSpline(self.xdata, ydata, k=self.order, s=self.smoothing_factor,
                                  w=1 / max(ydata) * np.ones(len(ydata)))
        self.spline_parameters = spline._eval_args  # spline.get_knots(),spline.get_coeffs(),self.order

    def predict(self, xdata=None, spline_parameters=None):

        if xdata is None:
            xdata = self.xdata
        elif xdata.ndim > 1:
            raise ValueError("Each smoother must have a single covariate.")

        if spline_parameters is None:
            if self.spline_parameters is None:
                warn(
                    "Spline parameters are not specified, you should either fit a model or specify them. Output is set to 0")
                return np.zeros((xdata.shape[0], 1))
            else:
                spline_parameters = self.spline_parameters

        y_pred = splev(xdata, spline_parameters)
        if np.any(np.isnan(y_pred)):
            warn("Spline parameters are too restrictive that it cannot predict. Output is set to 0")
            shape_parameters = 2 * (self.xdata.shape[0] + self.order + 1) + 6
            return np.zeros((shape_parameters,))

        # if len(y_pred.shape) == 1:
        #     y_pred=y_pred[...,np.newaxis]
        return y_pred

    def get_parameters(self):
        shape_parameters = 2 * (self.xdata.shape[0] + self.order + 1) + 6
        parameters = np.array([self.smoothing_factor], dtype=float64)
        for param in self.spline_parameters:
            try:
                parameters = np.append(parameters, len(param))
                parameters = np.append(parameters, [p for p in param])
            except:
                parameters = np.append(parameters, 1)
                parameters = np.append(parameters, param)

        parameters = np.append(len(parameters), parameters)

        parameters_reshaped = np.zeros((shape_parameters,))
        parameters_reshaped[:len(parameters)] = parameters
        return parameters_reshaped

    def get_covariate(self):
        return np.array(self.xdata)

    def set_covariate(self, covariate):
        self.xdata = np.squeeze(covariate)

    def set_parameters(self, parameters):

        parameters = np.squeeze(parameters)
        self.smoothing_factor = parameters[0]
        n_knots = int(parameters[1])
        self.spline_parameters = tuple([parameters[2:2 + n_knots], parameters[3 + n_knots:-2], int(parameters[-1])])
        self.order = int(parameters[-1])

    @property
    def name(self):
        return self._name


class PolynomialSmoother(Smoother):
    """
    Polynomial smoother up to a given order.
    """

    def __init__(self, xdata, order=3, coefficients=None, name=None):

        self.order = order

        if xdata.ndim > 1:
            raise ValueError("Error, each smoother a single covariate associated.")

        self.xdata = xdata

        if coefficients is None:
            coefficients = np.zeros((order + 1,), np.float64)
        self.coefficients = coefficients

        if name is None:
            name = 'PolynomialSmoother'

        self._name = name
        self._N = len(xdata)

    def fit(self, ydata, sample_weight=None, num_threads=-1):

        curve = LR(fit_intercept=False, normalize=False, copy_X=False, n_jobs=num_threads)

        xdata = np.array([np.squeeze(self.xdata) ** i for i in np.arange(1,self.order + 1)]).T
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
                    "Polynomial coefficients are not specified, you should either fit a model or specify them. Output is set to 0")
                return np.zeros((xdata.shape[0], 1))
            else:
                coefficients = self.coefficients

        xdata = np.array([np.squeeze(xdata) ** i for i in np.arange(1,self.order + 1)]).T
        y_pred = xdata.dot(coefficients)

        return y_pred

    def get_parameters(self):
        return np.append((len(self.coefficients) + 1, self.order), self.coefficients)

    def set_parameters(self, parameters):
        self.order = int(parameters[0])
        self.coefficients = parameters[1:]

    def get_covariate(self):
        return np.array(self.xdata)

    def set_covariate(self, covariate):
        self.xdata = np.squeeze(covariate)

    @property
    def name(self):
        return self._name

    def df_model(self):
        """
        Degrees of freedom used in the fit.
        """
        return self.order + 1

    def df_resid(self):
        """
        Residual degrees of freedom from last fit.
        """
        return self._N - self.order - 1


class InterceptSmoother(Smoother):
    def __init__(self, xdata, alpha=None):
        self.xdata = xdata
        self.alpha = alpha

    def fit(self, ydata):
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

    def set_covariate(self, covariate):
        pass

    def set_parameters(self, alpha):
        self.alpha = alpha


class KernelSmoother(Smoother):
    def __init__(self, x, y, Kernel=None):
        if Kernel is None:
            Kernel = kernels.Gaussian()
        self.Kernel = Kernel
        self.x = np.array(x)
        self.y = np.array(y)

    def fit(self):
        pass

    def __call__(self, x):
        return np.array([self.predict(xx) for xx in x])

    def predict(self, x):
        """
        Returns the kernel smoothed prediction at x

        If x is a real number then a single value is returned.

        Otherwise an attempt is made to cast x to numpy.ndarray and an array of
        corresponding y-points is returned.
        """
        if np.size(x) == 1:  # if isinstance(x, numbers.Real):
            return self.Kernel.smooth(self.x, self.y, x)
        else:
            return np.array([self.Kernel.smooth(self.x, self.y, xx) for xx
                             in np.array(x)])

    def conf(self, x):
        """
        Returns the fitted curve and 1-sigma upper and lower point-wise
        confidence.
        These bounds are based on variance only, and do not include the bias.
        If the bandwidth is much larger than the curvature of the underlying
        funtion then the bias could be large.

        x is the points on which you want to evaluate the fit and the errors.

        Alternatively if x is specified as a positive integer, then the fit and
        confidence bands points will be returned after every
        xth sample point - so they are closer together where the data
        is denser.
        """
        if isinstance(x, int):
            sorted_x = np.array(self.x)
            sorted_x.sort()
            confx = sorted_x[::x]
            conffit = self.conf(confx)
            return (confx, conffit)
        else:
            return np.array([self.Kernel.smoothconf(self.x, self.y, xx)
                             for xx in x])

    def var(self, x):
        return np.array([self.Kernel.smoothvar(self.x, self.y, xx) for xx in x])

    def std(self, x):
        return np.sqrt(self.var(x))


class GaussianKernel:
    """
    Gaussian (Normal) Kernel

    K(u) = 1 / (sqrt(2*pi)) exp(-0.5 u**2)
    """

    def __init__(self, sigma=1.0):
        pass

    def fit(self, predictors, observations):
        pass

    def predict(self):
        pass
