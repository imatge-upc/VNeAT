from CurveFitting import AdditiveCurveFitter
from numpy import zeros, array as nparray
import numpy as np

class GAM(AdditiveCurveFitter):
    '''Additive model with non-parametric, smoothed components

    Parameters
    ----------
    exog : ndarray
    smoothers : None or list of smoother instances
        smoother instances not yet checked
    weights : None or ndarray
    family : None or family instance
        I think only used because of shared results with GAM and subclassing.
        If None, then Gaussian is used.
    '''

    def __init__(self, exog, smoothers=None, weights=None, family=None):
        self.exog = exog
        if not weights is None:
            self.weights = weights
        else:
            self.weights = np.ones(self.exog.shape[0])

        self.smoothers = smoothers

        if family is None:
            self.family = families.Gaussian()
        else:
            self.family = family
        #self.family = families.Gaussian()

    def _iter__(self):
        self.iter = 0
        self.dev = np.inf
        return self

    def cont(self):
        self.iter += 1 #moved here to always count, not necessary

        curdev = (((self.results.Y - self.results.predict(self.exog))**2) * self.weights).sum()

        if self.iter > self.maxiter: #kill it, no max iterationoption
            return False
        if ((self.dev - curdev) / curdev) < self.rtol:
            self.dev = curdev
            return False

        self.dev = curdev
        return True

    # def df_resid(self):
    #     return self.results.Y.shape[0] - np.array([self.smoothers[i].df_fit() for i in range(self.exog.shape[1])]).sum()
    #
    # def estimate_scale(self):
    #     return ((self.results.Y - self.results(self.exog))**2).sum() / self.df_resid()

    def fit(self, Y, rtol=1.0e-06, maxiter=30):
        self.rtol = rtol
        self.maxiter = maxiter
        self._iter__()
        mu = 0
        alpha = (Y * self.weights).sum() / self.weights.sum()

        offset = np.zeros(self.smoothers.__len__(), np.float64)

        for i in range(self.smoothers.__len__()):
            r=Y - alpha - mu
            self.smoothers[i].smooth(Y - alpha - mu,
                                     weights=self.weights)
            f_i_pred = self.smoothers[i].predict()
            offset[i] = (f_i_pred * self.weights).sum() / self.weights.sum()
            f_i_pred -= f_i_pred.sum()/Y.shape[0]
            mu += f_i_pred

        self.results = Results(Y, alpha, self.exog, self.smoothers, self.family, offset)

        while self.cont():
            self.results = self.next()

        if self.iter >= self.maxiter:
            warnings.warn(iteration_limit_doc, IterationLimitWarning)

        return self.results

    def next(self):
        _results = self.results
        Y = self.results.Y
        mu = _results.predict(self.exog)
        offset = np.zeros(self.smoothers.__len__(), np.float64)
        alpha = (Y * self.weights).sum() / self.weights.sum()
        for i in range(self.smoothers.__len__()):
            tmp = self.smoothers[i].predict()
            bad = np.isnan(Y - alpha - mu + tmp).any()
            if bad: #temporary assert while debugging
                print(Y, alpha, mu, tmp)
                raise ValueError("nan encountered")
            #self.smoothers[i].smooth(Y - alpha - mu + tmp,
            self.smoothers[i].smooth(Y - mu + tmp,
                                     weights=self.weights)
            f_i_pred = self.smoothers[i].predict() #fit the values of previous smooth/fit
            self.results.offset[i] = (f_i_pred*self.weights).sum() / self.weights.sum()
            f_i_pred -= f_i_pred.sum()/Y.shape[0]
            if DEBUG:
                print(self.smoothers[i].params)
            mu += f_i_pred - tmp
        offset = self.results.offset
        return Results(Y, alpha, self.exog, self.smoothers, self.family, offset)


class Smoother:
    # smoothers should have the method 'smooth'.

    def __init__(self, xdata=None, smoother=None):
        if xdata is None and smoother is None:
            self.xdata = nparray([])
            self.num_reg = 0
            self.smoother = []
        elif xdata is not None and smoother is not None:
            self.xdata=nparray(xdata, dtype=float)
            self.smoother=smoother
            self.num_reg = len(smoother)
        else:
            raise TypeError('You should specifify a smoother for each regressor and viceversa')

    def set_polySmoother(self, xdata, d=None):

        xdata=nparray(xdata, dtype=float)
        if d is None:
            d = 1

        if self.num_reg == 0:
            self.xdata = xdata.T
        else:
            self.xdata=np.c_[self.xdata, xdata]

        for i in range(xdata.shape[1]):
            self.smoother.append(PolySmoother(d, xdata[:, i].copy()))

        self.num_reg =self.num_reg + xdata.shape[1]


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


class Results(object):

    def __init__(self, Y, alpha, exog, smoothers, family, offset):

        self.nobs = Y.shape[0]
        self.k_vars = smoothers.__len__()
        self.Y = Y
        self.alpha = alpha
        self.smoothers = smoothers
        self.offset = offset
        self.family = family
        self.exog = exog
        self.offset = offset
        self.mu = self.linkinversepredict(exog)

    def linkinversepredict(self, exog):  #For GAM (link function)
        '''expected value ? check new GLM, same as mu for given exog
        '''
        return self.family.link.inverse(self.predict(exog))

    def predict(self, exog):
        '''predict response, sum of smoothed components
        TODO: What's this in the case of GLM, corresponds to X*beta ?
        '''
        #note: sum is here over axis=0,
        #TODO: transpose in smoothed and sum over axis=1

        #BUG: there is some inconsistent orientation somewhere
        #temporary hack, won't work for 1d
        #print dir(self)
        #print 'self.nobs, self.k_vars', self.nobs, self.k_vars
        exog_smoothed = self.smoothed(exog)
        #print 'exog_smoothed.shape', exog_smoothed.shape
        if exog_smoothed.shape[0] == self.k_vars:
            import warnings
            warnings.warn("old orientation, colvars, will go away",
                          FutureWarning)
            return np.sum(self.smoothed(exog), axis=0) + self.alpha
        if exog_smoothed.shape[1] == self.k_vars:
            return np.sum(exog_smoothed, axis=1) + self.alpha
        else:
            raise ValueError('shape mismatch in predict')

    def smoothed(self, exog=None): #ADRIA changed this from exog -- exog==None
        if exog is None:
            exog=self.exog
        '''get smoothed prediction for each component

        '''
        #bug: with exog in predict I get a shape error
        #print 'smoothed', exog.shape, self.smoothers[0].predict(exog).shape
        #there was a mistake exog didn't have column index i


        if exog.ndim == 1:
            return np.array([self.smoothers[0].predict(exog) + self.offset])
        else:
            return np.array([self.smoothers[i].predict(exog[:,i]) + self.offset[i]
                            for i in range(self.smoothers.__len__())]).T
        #ADRIA changed this: BEFORE: return np.array([self.smoothers[i].predict(exog[:,i]) + self.offset[i]
            #shouldn't be a mistake because exog[:,i] is attached to smoother, but
            #it is for different exog
            #return np.array([self.smoothers[i].predict() + self.offset[i]
                         # for i in range(self.smoothers.__len__())]).T

