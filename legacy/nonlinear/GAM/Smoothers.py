from numpy import array as nparray
import numpy as np
from statsmodels.sandbox.nonparametric import kernels
from scipy.interpolate import UnivariateSpline
import patsy as pt

class Smoother:
    # smoothers should have the method 'smooth'.

    def __init__(self, xdata=None, smoother=None):
        if xdata is None:
            self.xdata = np.array([])
            self.num_reg = 0
            self.smoother = []
        elif smoother is None:
            self.xdata = nparray(xdata, dtype=float)
            self.num_reg = self.xdata.shape[1]
            self.smoother = self.set_polySmoother(self.xdata,1)  # Set the identity as smoother
        else:
            self.xdata=nparray(xdata, dtype=float)
            self.smoother=smoother
            self.num_reg = len(smoother)

    def set_polySmoother(self, xdata, d=None):
        xdata=nparray(xdata, dtype=float)
        if d is None:
            d = 1

        if self.num_reg == 0:
            self.xdata = xdata.T
        else:
            self.xdata=np.c_[self.xdata, xdata]

        if xdata.ndim == 1:
            nreg=1
            self.smoother.append(PolySmoother(d, xdata.copy()))
        else:
            nreg=xdata.shape[1]
            self.smoother.append([PolySmoother(d, xdata[:, i].copy()) for i in range(nreg)])

        self.num_reg =self.num_reg + nreg


    def set_splinesSmoother(self,xdata,d=None,s=None):

        if d is None:
            d=3
        xdata=nparray(xdata, dtype=float)

        if self.num_reg == 0:
            self.xdata = xdata.T
        else:
            self.xdata=np.c_[self.xdata, xdata]

        if xdata.ndim == 1:
            nreg=1
            self.smoother.append(SplinesSmoother(xdata.copy(),d,s))
        else:
            nreg=xdata.shape[1]
            self.smoother.append([SplinesSmoother(xdata[:, i].copy(),d,s) for i in range(nreg)])

        self.num_reg =self.num_reg + nreg

#    def set_kernelSmoother(self,xdata):

    def set_smoother(self, smoother):
        self.smoother = smoother
        self.num_reg = len(smoother)


class KernelSmoother(object):
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


class SplinesSmoother(object):

    def __init__(self,x,d,s=None):
        self.d=d
        self.X=x[:,None]
        self.spline=[]
        if s is None:
            s = x.shape[0]
        self.s=s

    def __call__(self, x=None):
        return self.predict(x=x)

    def smooth(self,*args, **kwds):
        # For compatibility
        return self.fit(*args, **kwds)


    def fit(self,y,weights=None):
        if y.ndim == 1:
            y = y[:,None]
        if weights is None or np.isnan(weights).all():
            _w = 1
        else:
            _w = np.sqrt(weights)[:,None]

        self.spline=UnivariateSpline(self.X * _w, y * _w, s=self.s)

    def predict(self,x=None):
        if x is None:
            x = self.X

        y_pred=self.spline(x)
        return np.squeeze(y_pred)


class PolySmoother(object):
    """
    Polynomial smoother up to a given order.
    """


    def __init__(self, order, x=None):
        #order = 4 # set this because we get knots instead of order
        self.order = order

        #print order, x.shape
        self.coef = np.zeros((order+1,), np.float64)
        if x is not None:
            if x.ndim > 1:
                print('Warning: 2d x detected in PolySmoother init, shape:', x.shape)
                x=x[0,:] #check orientation
            self.X = np.array([x**i for i in range(order+1)]).T

    def df_fit(self):
        '''alias of df_model for backwards compatibility
        '''
        return self.df_model()

    def df_model(self):
        """
        Degrees of freedom used in the fit.
        """
        return self.order + 1

    def gram(self, d=None):
        #fake for spline imitation
        pass

    def smooth(self,*args, **kwds):
        '''alias for fit,  for backwards compatibility,

        do we need it with different behavior than fit?

        '''
        return self.fit(*args, **kwds)

    def df_resid(self):
        """
        Residual degrees of freedom from last fit.
        """
        return self.N - self.order - 1

    def __call__(self, x=None):
        return self.predict(x=x)


    def predict(self, x=None):

        if x is not None:
            #if x.ndim > 1: x=x[0,:]  #why this this should select column not row
            if x.ndim > 1:
                print('Warning: 2d x detected in PolySmoother predict, shape:', x.shape)
                x=x[:,0]  #TODO: check and clean this up
            X = np.array([(x**i) for i in range(self.order+1)])
        else: X = self.X
        #return np.squeeze(np.dot(X.T, self.coef))
        #need to check what dimension this is supposed to be
        if X.shape[1] == self.coef.shape[0]:
            return np.squeeze(np.dot(X, self.coef))#[0]
        else:
            return np.squeeze(np.dot(X.T, self.coef))#[0]

    def fit(self, y, x=None, weights=None):
        self.N = y.shape[0]
        if y.ndim == 1:
            y = y[:,None]
        if weights is None or np.isnan(weights).all():
            weights = 1
            _w = 1
        else:
            _w = np.sqrt(weights)[:,None]
        if x is None:
            if not hasattr(self, "X"):
                raise ValueError("x needed to fit PolySmoother")
        else:
            if x.ndim > 1:
                print('Warning: 2d x detected in PolySmoother predict, shape:', x.shape)
                #x=x[0,:] #TODO: check orientation, row or col
            self.X = np.array([(x**i) for i in range(self.order+1)]).T
        #print _w.shape

        X = self.X * _w

        _y = y * _w#[:,None]
        #self.coef = np.dot(L.pinv(X).T, _y[:,None])
        #self.coef = np.dot(L.pinv(X), _y)
        self.coef = np.linalg.lstsq(X, _y)[0]
        self.params = np.squeeze(self.coef)



class Gaussian:
    """
    Gaussian (Normal) Kernel

    K(u) = 1 / (sqrt(2*pi)) exp(-0.5 u**2)
    """
    def __init__(self, h=1.0):
        self._L2Norm = 1.0/(2.0*np.sqrt(np.pi))
        self._kernel_var = 1.0
        self._order = 2

    def smooth(self, xs, ys, x):
        """Returns the kernel smoothing estimate for point x based on x-values
        xs and y-values ys.
        Not expected to be called by the user.

        Special implementation optimised for Gaussian.
        """
        w = np.sum(exp(multiply(square(divide(subtract(xs, x),
                                              self.h)),-0.5)))
        v = np.sum(multiply(ys, exp(multiply(square(divide(subtract(xs, x),
                                                          self.h)), -0.5))))
        return v/w

