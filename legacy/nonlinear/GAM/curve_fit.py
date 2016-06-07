from numpy import array as nparray
from scipy.optimize import curve_fit

from Smoothers import Smoother


class FunctionWrapper:
    def __init__(self, function, xdata):
        self.function = function
        self.xdata = xdata

    @staticmethod
    def wrap(num_params):
        func = 'lambda fm, '
        for i in range(num_params - 1):
            func += 'x' + str(i) + ', '
        func += 'x' + str(num_params - 1) + ': fm.function(fm.xdata, '
        for i in range(num_params - 1):
            func += 'x' + str(i) + ', '
        func += 'x' + str(num_params - 1) + ')'
        return eval(func)


class GLM:
    def __init__(self, xdata, ydata):
        self.xdata = nparray(xdata, dtype=float)
        self.ydata = nparray(ydata, dtype=float)

        assert len(self.xdata.shape) == 2 and self.xdata.shape[1] != 0
        assert len(self.ydata.shape) == 1
        assert self.xdata.shape[1] == self.ydata.shape[0]

        orig_num_regressors = self.xdata.shape[0]
        self.num_regressors = orig_num_regressors
        i = 0
        while i < self.num_regressors:
            self.__filter(i)
            if self.num_regressors == orig_num_regressors:
                i += 1
            else:
                orig_num_regressors -= 1

    @staticmethod
    def pred_function(xdata, *args):
        return sum(xdata[i] * args[i] for i in range(min(xdata.shape[0], len(args))))

    def orthogonalize(self):
        orig_ydata = self.ydata
        orig_num_regressors = self.num_regressors
        try:
            i = 1
            while i < orig_num_regressors:
                self.num_regressors = i
                self.ydata = self.xdata[i, :]
                self.optimize()
                self.xdata[i, :] -= self.pred_function(self.xdata, *self.opt_params)
                del self.opt_params, self.opt_params_cov
                self.__filter(i)
                if self.num_regressors < i:
                    orig_num_regressors -= 1
                else:
                    i += 1
        except Exception as e:
            raise e
        finally:
            self.num_regressors = orig_num_regressors
            self.ydata = orig_ydata

    def __filter(self, i):
        if max(abs(self.xdata[i, :])) < 1e-5:
            xdata = list(self.xdata)
            del xdata[i]
            self.xdata = nparray(xdata)
            self.num_regressors -= 1

    def optimize(self, p0=None, sigma=None, absolute_sigma=False, check_finite=True, **kw):
        num_params = self.num_regressors
        fw = FunctionWrapper(GLM.pred_function, self.xdata[:num_params, :])
        self.opt_params, self.opt_params_cov = curve_fit(fw.wrap(num_params), fw, self.ydata, p0, sigma, absolute_sigma,
                                                         check_finite, **kw)


import AdditiveModels as am


class GAM:
    def __init__(self, ydata, basisFunctions=None):

        self.ydata = nparray(ydata, dtype=float)
        assert len(self.ydata.shape) == 1

        # self.xdata = nparray(xdata, dtype=float)
        # assert len(self.xdata.shape) == 2 and self.xdata.shape[1] != 0
        # assert self.xdata.shape[0] == self.ydata.shape[0]
        # orig_num_regressors = self.xdata.shape[0]
        # self.num_regressors = orig_num_regressors

        if basisFunctions is None:
            self.basisFunctions = Smoother()
        else:
            self.basisFunctions = basisFunctions
            self.AM = am.AdditiveModel(basisFunctions.xdata)

    def set_smoother(self, smoother):
        self.basisFunctions = smoother

    def backfitting_algorithm(self):
        self.AM = am.AdditiveModel(self.basisFunctions.xdata, smoothers=self.basisFunctions.smoother)
        self.results = self.AM.fit(self.ydata)
        self.basisFunctions.set_smoother = self.results.smoothers

    def local_scoring(self):
        # TODO
        self.AM = am.AdditiveModel(self.basisFunctions.xdata)
        self.results = self.AM.fit(self.ydata)
        self.basisFunctions.set_smoother = self.results.smoothers

    def pred_function(self, xdata=None):
        # results.smoothed should be changed to just predict the function associated with xdata. Xdata can be all the covariates
        # or just some of them.
        return self.AM.results.smoothed(self.basisFunctions.xdata)

    def prediction(self):
        return self.AM.results.predict(self.basisFunctions.xdata)

    def __filter(self, i):
        if max(abs(self.xdata[i, :])) < 1e-5:
            xdata = list(self.xdata)
            del xdata[i]
            self.xdata = nparray(xdata)
            self.num_regressors -= 1
