from CurveFitting import AdditiveCurveFitter
from numpy import zeros, array as nparray
import numpy as np
from statsmodels.genmod import families
from statsmodels.sandbox.nonparametric import kernels
from sklearn.linear_model import LinearRegression as LR
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splev
import warnings
from collections import OrderedDict

class GAM(AdditiveCurveFitter):
    '''Additive model with non-parametric, smoothed components
    '''


    @staticmethod
    def __predict__(regressors):

        for smoother in regressors:
            ypred=ypred + smoother.predict()

        return ypred

    def __fit__(self,correctors,regressors, observations, rtol=1.0e-06, maxiter=30):

        self._init_iter()
        smoothers=correctors+regressors
        n_correctors=len(correctors)
        n_regressors=len(regressors)
        n_features=len(smoothers)


        mu = 0
        alpha = np.mean(observations)

        offset = np.zeros(n_features, np.float64)

        while self.cont():
            for i in range(n_features):
                r = observations - alpha - mu
                smoothers[i].fit(r)
                f_i_pred = smoothers[i].predict()
                offset[i] = f_i_pred.sum() / n_features
                f_i_pred -= offset[i]
                mu += f_i_pred

        results_correctors=OrderedDict()
        results_regressors=OrderedDict()
        results_correctors['Mean']=alpha
        for smoother in smoothers[:n_correctors]:
            results_correctors[smoother.name]=smoother.get_parameters()
        for smoother in smoothers[n_correctors:]:
            results_regressors[smoother.name]=smoother.get_parameters()

        return (results_correctors,results_regressors)

    def _init_iter(self):
        self.iter = 0
        self.dev = np.inf
        return self

    def _cont(self):
        if self.iter == 0:
            return True

        self.iter += 1
        curdev = (((self.results.observations - self.results.predict())**2)).sum()

        if self.iter > self.maxiter:
            return False
        if ((self.dev - curdev) / curdev) < self.rtol:
            self.dev = curdev
            return False

        self.dev = curdev

        return True


# class Smoothers(list):
#
#     def __init__(self):
#         self.name =[]
#         self.num_smoothers = 0
#
#     def set_smoother(self,smoother, name = None):
#         self.num_smoothers = self.num_smoothers +1
#         self.append(smoother)

class Smoother():

    def fit(self):
        raise NotImplementedError()
    def predict(self):
        raise NotImplementedError()


class SplinesSmoother(Smoother):

    def __init__(self,xdata,degree=3,smoothing_factor=None,parameters=None,name=None):

        self.smoothing_factor=smoothing_factor
        self.degree=degree
        self.xdata=xdata
        self.parameters=parameters
        if name is None:
            name='SplinesSmoother'
        self._name=name

    # def df_model(self,parameters=None):
    #     """
    #     Degrees of freedom used in the fit.
    #     """
    #     return (parameters[2]+1)*(parameters[0]+1)-parameters[2]*parameters[0]

    # def df_resid(self,parameters=None):
    #     """
    #     Residual degrees of freedom from last fit.
    #     """
    #     return self.N - self.df_model(parameters=parameters)


    def fit(self,ydata):
        if ydata.ndim == 1:
            ydata = ydata[:,None]

        spline=UnivariateSpline(self.xdata , ydata, k=self.degree,s=self.smoothing_factor)
        self.parameters=spline.get_knots(),spline.get_coeffs(),self.degree


    def predict(self,xdata=None,parameters=None):

        if xdata is None:
            xdata = self.xdata
        elif xdata.ndim > 1:
            raise ValueError("Each smoother must have a single covariate.")

        if parameters is None:
            if self.parameters is None:
                raise ValueError("You should either fit first the model to the data or specify the parameters")
            else:
                parameters = self.parameters

        ydata_pred=splev(xdata,parameters)
        return np.squeeze(ydata_pred)

    def get_parameters(self):
        return self.parameters


    @property
    def name(self):
        return self.name


class PolynomialSmoother(Smoother):
    """
    Polynomial smoother up to a given order.
    """

    def __init__(self, x, order=3,parameters=None,name=None):

        self.order = order
        if parameters is None:
            parameters = np.zeros((order+1,), np.float64)
        self.parameters=parameters
        self.xdata = x
        if name is None:
            name='SplinesSmoother'
        self._name=name
        if x.ndim > 1:
            raise ValueError("Error, each smoother a single covariate associated.")



    # def df_model(self):
    #     """
    #     Degrees of freedom used in the fit.
    #     """
    #     return self.order + 1
    #
    # def df_resid(self):
    #     """
    #     Residual degrees of freedom from last fit.
    #     """
    #     return self.N - self.order - 1


    def fit(self, ydata,sample_weight=None,num_threads = -1):

        if ydata.ndim == 1:
            ydata = ydata[:,None]

        curve = LR(fit_intercept = False, normalize = False, copy_X = False,n_jobs=num_threads)
        curve.fit(self.xdata, ydata, sample_weight)
        self.parameters = curve.coef_.T

    def predict(self, xdata=None,parameters=None):
        if xdata is None:
            xdata=self.xdata
        elif xdata.ndim > 1:
            raise ValueError("Each smoother must have a single covariate.")

        if parameters is None:
            if self.parameters is None:
                raise ValueError("You should either fit first the model to the data or specify the parameters")
            else:
                parameters = self.parameters
        return xdata.dot(self.parameters)

    def get_parameters(self):
        return self.parameters

    @property
    def name(self):
        return 'SplinesSmoother'


class KernelSmoother(Smoother):
    def __init__(self, x, y, Kernel = None):
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
        if np.size(x) == 1: # if isinstance(x, numbers.Real):
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

    def fit(self, regressors,observations):
        pass
    def predict(self):
        pass

