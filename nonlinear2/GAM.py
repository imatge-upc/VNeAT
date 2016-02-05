from abc import  abstractmethod
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
    '''
    Additive model with non-parametric, smoothed components
    '''

    def __init__(self,corrector_smoothers=None, regressor_smoothers=None,maxiter=20):

        self.corrector_smoothers=corrector_smoothers
        self.regressor_smoothers=regressor_smoothers

        if corrector_smoothers is not None:
            correctors=corrector_smoothers.get_covariates()
        if regressor_smoothers is not None:
            regressors=regressor_smoothers.get_covariates()

        super(GAM, self).__init__(regressors, correctors, False)


    def __predict__(self,regressors,regression_parameters):

        y_pred=0
        for reg in self.regressor_smoothers:
            y_pred = y_pred + reg.predict()

        return y_pred

    def __fit__(self,correctors,regressors,observations, rtol=1.0e-06, maxiter=1):

        self.maxiter=maxiter
        self.rtol=rtol

        dims=observations.shape
        smoothers = SmootherSet()
        for smoother,corr in  zip(self.corrector_smoothers,correctors.T):
            smoother.set_covariate(corr.reshape(dims[0],-1))
            smoothers.set_smoother(smoother)
        for smoother,reg in zip(self.regressor_smoothers,regressors.T):
            smoother.set_covariate(reg.reshape(dims[0],-1))
            smoothers.set_smoother(smoother)

        n_correctors=self.corrector_smoothers.num_smoothers
        n_smoothers=len(smoothers)

        self.__init_iter()
        alpha = np.mean(observations)
        mu = 0.0
        offset = np.zeros(n_smoothers, np.float64)

        while self.__cont(observations):
            for i in range(n_smoothers):
                r = observations - alpha - mu
                smoothers[i].fit(r)
                f_i_pred = smoothers[i].predict()
                offset[i] = f_i_pred.sum() / n_smoothers
                f_i_pred -= offset[i]
                mu += f_i_pred
                self.iter += 1

        # results_correctors=[]
        # results_regressors=[]
        # results_correctors.append('Mean',alpha)
        # for smoother in smoothers[:n_correctors]:
        #     results_correctors.append((smoother.name,smoother.get_parameters()))
        # for smoother in smoothers[n_correctors:]:
        #     results_regressors.append((smoother.name,smoother.get_parameters()))

        self.corrector_smoothers=smoothers[:n_correctors+1]
        self.regressor_smoothers=smoothers[n_correctors+1:]
        print([smooth.parameters for smooth in self.corrector_smoothers])
        print([smooth.parameters for smooth in self.regressor_smoothers])

        return (self.corrector_smoothers,self.regressor_smoothers)

    def __init_iter(self):
        self.iter = 0
        self.dev = np.inf
        return self

    def __cont(self,observations):
        if self.iter == 0:
            return True


        curdev = (((observations - self.predict())**2)).sum()

        if self.iter > self.maxiter:
            return False
        if ((self.dev - curdev) / curdev) < self.rtol:
            self.dev = curdev
            return False

        self.dev = curdev

        return True


class SmootherSet(list):

    def __init__(self):
        self.num_smoothers = 0

    def set_smoother(self,smoother, name = None):
        self.num_smoothers = self.num_smoothers +1
        self.append(smoother)

    def get_covariates(self):
        return np.array([smoother.get_covariate() for smoother in self]).T



class Smoother():

    @abstractmethod
    def fit(self):
        raise NotImplementedError()

    @abstractmethod
    def predict(self):
        raise NotImplementedError()

    @abstractmethod
    def get_parameters(self):
        raise NotImplementedError()

    @abstractmethod
    def get_covariate(self):
        raise NotImplementedError()

    @abstractmethod
    def set_covariate(self):
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

    def get_covariate(self):
        return np.array(self.xdata)

    def set_covariate(self,xdata):
        self.xdata=xdata

    @property
    def name(self):
        return self.name

class PolynomialSmoother(Smoother):
    """
    Polynomial smoother up to a given order.
    """

    def __init__(self, x, order=3,parameters=None,name=None):

        self.order = order

        if x.ndim > 1:
            raise ValueError("Error, each smoother a single covariate associated.")

        self.xdata = x #np.array([x**i for i in range(order+1)]).T

        if parameters is None:
            parameters = np.zeros((order+1,), np.float64)
        self.parameters=parameters

        if name is None:
            name='SplinesSmoother'

        self._name=name


    def fit(self,ydata,sample_weight=None,num_threads = -1):

        curve = LR(fit_intercept = False, normalize = False, copy_X = False,n_jobs=num_threads)

        xdata = np.array([np.squeeze(self.xdata)**i for i in range(self.order+1)]).T
        curve.fit(xdata, ydata, sample_weight)
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

        xdata=np.array([np.squeeze(xdata)**i for i in range(self.order+1)]).T
        return xdata.dot(self.parameters)

    def get_parameters(self):
        return self.parameters

    def get_covariate(self):
        return np.array(self.xdata)

    def set_covariate(self,xdata):
        self.xdata=xdata

    @property
    def name(self):
        return self.name


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

