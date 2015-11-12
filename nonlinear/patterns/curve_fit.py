from numpy import array as nparray
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


class GLM:
	def __init__(self, xdata, ydata):
		if len(xdata.shape) == 1:
			self.xdata = nparray([xdata.copy()])
		else:
			self.xdata = xdata.copy()
		self.num_regressors = self.xdata.shape[0]
		self.ydata = ydata.copy()

	@staticmethod
	def pred_function(xdata, *args):
		return sum(xdata[i]*args[i] for i in range(min(xdata.shape[0], len(args))))

		'''
	def orthogonalize(self):
		if self.num_regressors > 1:
			orig_ydata = self.ydata
			orig_num_regressors = self.num_regressors
			self.ydata = self.xdata([self.num_regressors - 1, :])
			try:
				self.num_regressors -= 1
				self.optimize()
				self.xdata[self.num_regressors, :] -= self.pred_function(self.xdata, *self.opt_params)
			except Exception as e:
				raise e
			finally:
				self.num_regressors = orig_num_regressors
				self.ydata = orig_ydata
		'''

	def optimize(self, p0=None, sigma=None, absolute_sigma=False, check_finite=True, **kw):
		num_params = self.num_regressors
		fw = FunctionWrapper(GLM.pred_function, self.xdata[:num_params, :])
		self.opt_params, self.opt_params_cov = curve_fit(fw.wrap(num_params), fw, self.ydata, p0, sigma, absolute_sigma, check_finite, **kw)

#	def glm(xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=True, **kw):
#		# xdata dimensions = (k, M) or just (M) in case it is 1-dimensional
#		# ydata dimensions = (M)
#		# watch documention of scipy.optimize.curve_fit for information about other options
#		if len(xdata.shape) == 1:
#			xdata = nparray([xdata])
#			# convert dimension (M) to (1, M) so that we can treat this case equally
#	
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

	