from numpy import array as nparray
import numpy as np
from scipy.optimize import curve_fit


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


from statsmodels.sandbox.gam import AdditiveModel, Results


class GAM:
    def __init__(self, ydata, basisFunctions = None):

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
            self.AM = AdditiveModel(basisFunctions.xdata)

    def set_smoother(self, smoother):
        self.baseFunctions = smoother

    # possibility of different smoothers for each covariate

    def backfitting_algorithm(self):
        self.AM = AdditiveModel(self.basisFunctions.xdata)
        self.results = self.AM.fit(self.ydata)
        self.basisFunctions.set_smoother=self.results.smoothers

    def local_scoring(self):
        #TODO
        self.AM = AdditiveModel(self.basisFunctions.xdata)
        self.results = self.AM.fit(self.ydata)
        self.basisFunctions.set_smoother=self.results.smoothers

    def pred_function(self):
        return self.AM.results.smoothed(self.basisFunctions.xdata)

    def prediction(self):
        return self.AM.results.predict(self.basisFunctions.xdata)

    def __filter(self, i):
        if max(abs(self.xdata[i, :])) < 1e-5:
            xdata = list(self.xdata)
            del xdata[i]
            self.xdata = nparray(xdata)
            self.num_regressors -= 1


from statsmodels.sandbox.nonparametric.smoothers import PolySmoother


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
            self.smoother.append(PolySmoother(d, xdata))
        else:
            nreg=xdata.shape[1]
            self.smoother.append([PolySmoother(d, xdata[:, i].copy()) for i in range(nreg)])

        self.num_reg =self.num_reg + nreg


    def set_splines(self,xdata):
        self.xdata= np.append(self.xdata,nparray(xdata, dtype=float))
        if self.xdata.ndim == 1:
            self.num_reg +=self.num_reg
        else:
            self.num_reg =self.num_reg + xdata.shape[1]

        x_sort = np.sort(self.xdata)
        n = self.xdata.shape[0]
        if n < 500:
            nknots = n
        else:
            a1 = np.log(50) / np.log(2)
        a2 = np.log(100) / np.log(2)
        a3 = np.log(140) / np.log(2)
        a4 = np.log(200) / np.log(2)
        if n < 200:
            nknots = 2 ** (a1 + (a2 - a1) * (n - 50) / 150.)
        elif n < 800:
            nknots = 2 ** (a2 + (a3 - a2) * (n - 200) / 600.)
        elif n < 3200:
            nknots = 2 ** (a3 + (a4 - a3) * (n - 800) / 2400.)
        else:
            nknots = 200 + (n - 3200.) ** 0.2
        knots = x_sort[np.linspace(0, n - 1, nknots).astype(np.int32)]
        self.smoother = SmoothingSpline(knots, x=self.xdata.copy())


    def set_smoother(self, smoother):
        self.smoother = smoother
        self.num_reg = len(smoother)






# --------  Dead code goes here (ignore)  --------

#	def add_params(func, num_added_params):
#		num_orig_params = func.func_code.co_argcount
#		new_nparams = num_orig_params + num_added_params
#		function = 'lambda '
#		for i in range(new_nparams - 1):
#			function += 'x' + str(i) + ', '
#		function += 'x' + str(new_nparams - 1) + ': ' + func.func_code.co_name + '('
#		for i in range(new_nparams - 1):
#			function += 'x' + str(i) + ', '
#		function += 'x' + str(new_nparams - 1) + ')'
#		return eval(function)

#	def glm(xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=True, **kw):
#		# xdata dimensions = (k, M) or just (M) in case it is 1-dimensional
#		# ydata dimensions = (M)
#		# watch documention of scipy.optimize.curve_fit for information about other options
#		if len(xdata.shape) == 1:
#			xdata = nparray([xdata])
#			# convert dimension (M) to (1, M) so that we can treat this case equally
#
#
#		def pred_function(xdata, *args):
#			return sum(xdata[i]*args[i] for i in range(xdata.shape[0]))
#
#		num_params = xdata.shape[0]
#
#		fw = FunctionWrapper(pred_function, xdata)
#
#		return curve_fit(fw.wrap(num_params), fw, ydata, p0, sigma, absolute_sigma, check_finite, **kw)



#	INSIDE GLM CLASS:
#
#	def orthogonalize(self):
#		if self.num_regressors > 1:
#			orig_ydata = self.ydata
#			orig_num_regressors = self.num_regressors
#			self.ydata = self.xdata[self.num_regressors - 1, :]
#			try:
#				self.num_regressors -= 1
#				self.orthogonalize()
#				self.optimize()
#				self.xdata[self.num_regressors, :] -= self.pred_function(self.xdata, *self.opt_params)
#				del self.opt_params, self.opt_params_cov
#			except Exception as e:
#				raise e
#			finally:
#				self.num_regressors = orig_num_regressors
#				self.ydata = orig_ydata
