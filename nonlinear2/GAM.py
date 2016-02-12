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

    def __init__(self,corrector_smoothers=None, regressor_smoothers=None):

        self.TYPE_SMOOTHER=[PolynomialSmoother,SplinesSmoother]

        if corrector_smoothers is not None:
            correctors=corrector_smoothers.get_covariates()
        else:
            correctors=None

        if regressor_smoothers is not None:
            regressors=regressor_smoothers.get_covariates()
        else:
            regressors=None

        self.regressor_smoothers=regressor_smoothers
        self.corrector_smoothers=SmootherSet(corrector_smoothers)

        super(GAM, self).__init__(regressors, correctors, False)

    def __fit__(self,correctors,regressors,observations, rtol=1.0e-06, maxiter=20):

        dims=observations.shape

        [smoother.set_covariate(corr.reshape(dims[0],-1)) for smoother,corr in  zip(self.corrector_smoothers,correctors.T)]
        [smoother.set_covariate(corr.reshape(dims[0],-1)) for smoother,corr in  zip(self.corrector_smoothers,correctors.T)]


        smoother_functions = SmootherSet(self.corrector_smoothers+self.regressor_smoothers)
        alpha,mu,offset=self.__init_iter(observations,smoother_functions.n_smoothers)

        while self.__cont(observations,alpha+mu,maxiter,rtol):
            for smoother in smoother_functions:
                r = observations - alpha - mu
                smoother.fit(r)
                f_i_pred = smoother.predict()
                offset = f_i_pred.sum() / smoother_functions.n_smoothers
                f_i_pred -= offset
                mu += f_i_pred
                self.iter += 1

        self.corrector_smoothers=SmootherSet(smoother_functions[:self.corrector_smoothers.n_smoothers])
        self.regressor_smoothers=SmootherSet(smoother_functions[self.corrector_smoothers.n_smoothers:])
        return (self.__code_parameters(self.corrector_smoothers),self.__code_parameters(self.regressor_smoothers))


    def __predict__(self,regressors,regression_parameters):

        y_pred=np.zeros((regressors.shape[0],))
        for reg, parameters in zip(regressors.T,regression_parameters):


            smoother=self.TYPE_SMOOTHER[int(parameters[0])](reg)
            print(smoother)
            print(parameters)
            smoother.set_parameters(parameters[1:])

            y_pred = smoother.predict()

        return y_pred


    def __init_iter(self,observations,n_smoothers):
        self.iter = 0
        self.dev = np.inf
        self.alpha=np.mean(observations)
        mu = 0.0
        offset = np.zeros(n_smoothers, np.float64)
        return self.alpha,mu,offset

    def __cont(self,observations,observations_pred,maxiter,rtol):
        if self.iter == 0:
            self.iter += 1
            return True

        curdev = (((observations - observations_pred)**2)).sum()

        if self.iter > maxiter:
            return False
        if ((self.dev - curdev) / curdev) < rtol:
            self.dev = curdev
            return False

        self.dev = curdev

        return True

    def __code_parameters(self,smoother_set):
        parameters=[]
        for smoother in smoother_set:
            params=smoother.get_parameters()
            parameters.append(np.append(self.TYPE_SMOOTHER.index(smoother.__class__),params))
        return np.array(parameters)


class SmootherSet(list):

    def __init__(self, smoothers = None):
        self.n_smoothers=0
        if smoothers is not None:
            self.extend(smoothers)

    def extend(self,smoothers, name = None):
        if isinstance(smoothers,list):
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
    def fit(self,*args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def predict(self,ydata,*args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_parameters(self,*args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_covariate(self,*args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def set_covariate(self,covariate,*args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def set_parameters(self,parameters,*args, **kwargs):
        raise NotImplementedError()

class SplinesSmoother(Smoother):

    def __init__(self,xdata,degree=3,smoothing_factor=None,parameters=None,name=None):

        if smoothing_factor is None:
            smoothing_factor = len(xdata)
        self.smoothing_factor=smoothing_factor
        self.degree=degree
        self.xdata=xdata
        self.parameters=parameters
        if name is None:
            name='SplinesSmoother'
        self._name=name

    def df_model(self,parameters=None):
        pass
    #     """
    #     Degrees of freedom used in the fit.
    #     """
    #     return (parameters[2]+1)*(parameters[0]+1)-parameters[2]*parameters[0]

    def df_resid(self,parameters=None):
        pass
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
        return np.append(self.smoothing_factor,self.parameters)

    def get_covariate(self):
        return np.array(self.xdata)

    def set_covariate(self,covariate):
        self.xdata=np.squeeze(covariate)

    def set_parameters(self,parameters):
        self.parameters = parameters

    @property
    def name(self):
        return self._name

class PolynomialSmoother(Smoother):
    """
    Polynomial smoother up to a given order.
    """

    def __init__(self, xdata, order=3,coeficients=None,name=None):

        self.order = order

        if xdata.ndim > 1:
            raise ValueError("Error, each smoother a single covariate associated.")

        self.xdata = xdata

        if coeficients is None:
            coeficients = np.zeros((order+1,), np.float64)
        self.coeficients=coeficients

        if name is None:
            name='PolynomialSmoother'

        self._name=name
        self._N=len(xdata)

    def fit(self,ydata,sample_weight=None,num_threads = -1):

        curve = LR(fit_intercept = False, normalize = False, copy_X = False,n_jobs=num_threads)

        xdata = np.array([np.squeeze(self.xdata)**i for i in range(self.order+1)]).T
        curve.fit(xdata, ydata, sample_weight)
        self.coefficients = curve.coef_.T

    def predict(self, xdata=None,coeficients=None):
        if xdata is None:
            xdata=self.xdata
        elif xdata.ndim > 1:
            raise ValueError("Each smoother must have a single covariate.")

        if coeficients is None:
            if self.coefficients is None:
                raise ValueError("You should either fit first the model to the data or specify the parameters")
            else:
                coefficients = self.coefficients
        xdata=np.array([np.squeeze(xdata)**i for i in range(self.order+1)]).T
        return xdata.dot(self.coefficients)

    def get_parameters(self):
        return np.append(self.order,self.coeficients)

    def set_parameters(self,parameters):
        self.order = int(parameters[1])
        self.coefficients = parameters[2:]

    def get_covariate(self):
        return np.array(self.xdata)

    def set_covariate(self,covariate):
        self.xdata=np.squeeze(covariate)

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

    def __init__(self):
        self._name = 'Intercept'

    @abstractmethod
    def fit(self,ydata):
        self.alpha = np.mean(ydata)
        self._N = ydata.shape[0]

    @abstractmethod
    def predict(self):
        return self.alpha * np.ones((self._N,))

    @abstractmethod
    def get_parameters(self):
        return self.alpha

    @abstractmethod
    def get_covariate(self):
        pass

    @abstractmethod
    def set_covariate(self,covariate):
        pass

    @abstractmethod
    def set_parameters(self,parameters):
        pass

    @property
    def name(self):
        return self._name

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

