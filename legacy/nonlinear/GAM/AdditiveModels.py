import warnings

import numpy as np
from statsmodels.compat.python import next, range
from statsmodels.sandbox.nonparametric.smoothers import PolySmoother
from statsmodels.tools.sm_exceptions import iteration_limit_doc

DEBUG = False


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

    def linkinversepredict(self, exog):  # For GAM (link function)
        '''expected value ? check new GLM, same as mu for given exog
        '''
        return self.family.link.inverse(self.predict(exog))

    def predict(self, exog):
        '''predict response, sum of smoothed components
        TODO: What's this in the case of GLM, corresponds to X*beta ?
        '''
        # note: sum is here over axis=0,
        # TODO: transpose in smoothed and sum over axis=1

        # BUG: there is some inconsistent orientation somewhere
        # temporary hack, won't work for 1d
        # print dir(self)
        # print 'self.nobs, self.k_vars', self.nobs, self.k_vars
        exog_smoothed = self.smoothed(exog)
        # print 'exog_smoothed.shape', exog_smoothed.shape
        if exog_smoothed.shape[0] == self.k_vars:
            import warnings
            warnings.warn("old orientation, colvars, will go away",
                          FutureWarning)
            return np.sum(self.smoothed(exog), axis=0) + self.alpha
        if exog_smoothed.shape[1] == self.k_vars:
            return np.sum(exog_smoothed, axis=1) + self.alpha
        else:
            raise ValueError('shape mismatch in predict')

    def smoothed(self, exog=None):  # ADRIA changed this from exog -- exog==None
        if exog is None:
            exog = self.exog
        '''get smoothed prediction for each component

        '''
        # bug: with exog in predict I get a shape error
        # print 'smoothed', exog.shape, self.smoothers[0].predict(exog).shape
        # there was a mistake exog didn't have column index i


        if exog.ndim == 1:
            return np.array([self.smoothers[0].predict(exog) + self.offset])
        else:
            return np.array([self.smoothers[i].predict(exog[:, i]) + self.offset[i]
                             for i in range(self.smoothers.__len__())]).T
            # ADRIA changed this: BEFORE: return np.array([self.smoothers[i].predict(exog[:,i]) + self.offset[i]
            # shouldn't be a mistake because exog[:,i] is attached to smoother, but
            # it is for different exog
            # return np.array([self.smoothers[i].predict() + self.offset[i]
            # for i in range(self.smoothers.__len__())]).T


class AdditiveModel(object):
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
            # self.family = families.Gaussian()

    def _iter__(self):
        self.iter = 0
        self.dev = np.inf
        return self

    def cont(self):
        self.iter += 1  # moved here to always count, not necessary
        if DEBUG:
            print(self.iter, self.results.Y.shape)
            print(self.results.predict(self.exog).shape, self.weights.shape)
        curdev = (((self.results.Y - self.results.predict(self.exog)) ** 2) * self.weights).sum()

        if self.iter > self.maxiter:  # kill it, no max iterationoption
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
            r = Y - alpha - mu
            self.smoothers[i].smooth(Y - alpha - mu,
                                     weights=self.weights)
            f_i_pred = self.smoothers[i].predict()
            offset[i] = (f_i_pred * self.weights).sum() / self.weights.sum()
            f_i_pred -= f_i_pred.sum() / Y.shape[0]
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
            if bad:  # temporary assert while debugging
                print(Y, alpha, mu, tmp)
                raise ValueError("nan encountered")
            # self.smoothers[i].smooth(Y - alpha - mu + tmp,
            self.smoothers[i].smooth(Y - mu + tmp,
                                     weights=self.weights)
            f_i_pred = self.smoothers[i].predict()  # fit the values of previous smooth/fit
            self.results.offset[i] = (f_i_pred * self.weights).sum() / self.weights.sum()
            f_i_pred -= f_i_pred.sum() / Y.shape[0]
            if DEBUG:
                print(self.smoothers[i].params)
            mu += f_i_pred - tmp
        offset = self.results.offset
        return Results(Y, alpha, self.exog, self.smoothers, self.family, offset)
