from abc import ABCMeta, abstractmethod
from Documentation import docstring_inheritor
from numpy import array as nparray, zeros, ones, float64
from scipy.stats import f as f_stat


class abstractstatic(staticmethod):
	__slots__ = ()
	__isabstractmethod__ = True

	def __init__(self, function):
		super(abstractstatic, self).__init__(function)
		function.__isabstractmethod__ = True


class CurveFitter:
	'''Abstract class that implements the framework to develop curve fitting algorithms.
	'''
	__metaclass__ = docstring_inheritor(ABCMeta)
	__threshold = (1e-14 ** 2)

	def __init__(self, regressors = None, correctors = None, homogeneous = True):
		'''[Constructor]

			Parameters:
			    
			    - regressors: NxR (2-dimensional) matrix (default None), representing the predictors,
			        i.e., features to be used to try to explain/predict some observations (experimental
			        data), where R is the number of regressors and N the number of elements for each
			        regressor.

			    - correctors: NxC (2-dimensional) matrix (default None), representing the covariates,
			        i.e., features that (may) explain a part of the observational data in which we are
			        not interested, where C is the number of correctors and N the number of elements
			        for each corrector (this number must be the same as that in the 'regressors' argu-
			        ment).

			    - homogeneous: bool (default True), indicating whether the homogeneous term (a.k.a.
			        intercept) must be incorporated to the model as a corrector or not. If so, a column
			        of ones will be added as the first column (feature) of the internal correctors matrix.
		'''

		if regressors is None:
			self._crvfitter_regressors = zeros((0, 0))
		else:
			self._crvfitter_regressors = nparray(regressors, dtype = float64)

			if len(self._crvfitter_regressors.shape) != 2:
				raise TypeError('Argument \'regressors\' must be a 2-dimensional matrix')

		if correctors is None:
			if homogeneous:
				self._crvfitter_correctors = ones((self._crvfitter_regressors.shape[0], 1))
			else:
				self._crvfitter_correctors = zeros((0, 0))

		else:
			correctors = nparray(correctors, dtype = float64)

			if len(correctors.shape) != 2:
				raise TypeError('Argument \'correctors\' must be a 2-dimensional matrix (or None)')

			if homogeneous:
				dims = correctors.shape
				self._crvfitter_correctors = ones((dims[0], dims[1]+1))
				self._crvfitter_correctors[:, 1:] = correctors
			else:
				self._crvfitter_correctors = correctors
		
			if self._crvfitter_correctors.shape[0] != self._crvfitter_regressors.shape[0]:
				raise ValueError('Correctors and regressors must have the same number of samples (same length in the first dimension)')

		self._crvfitter_correction_parameters = zeros((0, 0))
		self._crvfitter_regression_parameters = zeros((0, 0))

	@property
	def correctors(self):
		'''Matrix of shape (N, C), representing the correctors of the model.
			Note: do not modify the result (use copy method present in it if you need to treat this data).
		'''
		return self._crvfitter_correctors
	
	@property
	def regressors(self):
		'''Matrix of shape (N, R), representing the regressors of the model.
			Note: do not modify the result (use copy method present in it if you need to treat this data).
		'''
		return self._crvfitter_regressors

	@property
	def correction_parameters(self):
		'''Array-like structure of shape (Kc, X1, ..., Xn), representing the correction parameters for which
			the correctors best explain the observational data passed as argument in the last call to 'fit',
			where Kc is the number of parameters for each variable in such observations, and X1, ..., Xn are
			the dimensions of the 'observations' argument in the last call to 'fit' (there are X1*...*Xn
			variables).
			Note: do not modify the result (use copy method present in it if you need to treat this data).
		'''
		return self._crvfitter_correction_parameters.reshape(-1, *self._crvfitter_dims)

	@property
	def regression_parameters(self):
		'''Array-like structure of shape (Kr, X1, ..., Xn), representing the regression parameters for which
			the regressors best explain the observational data passed as argument in the last call to 'fit',
			where Kr is the number of parameters for each variable in such observations, and X1, ..., Xn are
			the dimensions of the 'observations' argument in the last call to 'fit' (there are X1*...*Xn
			variables).
			Note: do not modify the result (use copy method present in it if you need to treat this data).
		'''
		return self._crvfitter_regression_parameters.reshape(-1, *self._crvfitter_dims)

	def orthogonalize_correctors(self):
		'''Orthogonalizes each corrector in the structure w.r.t. all the previous ones. That is, for each
			column in the correctors matrix, its projection over the previous columns is computed and sub-
			tracted from it.

			Modifies:

			    - Correctors: each column has been orthogonalized with respect to the previous ones.

			Returns:

			    - None
		'''
		# Gram-Schmidt
		threshold = self._crvfitter_correctors.shape[0]*CurveFitter.__threshold
		for i in xrange(self._crvfitter_correctors.shape[1] - 1):
			u = self._crvfitter_correctors[:, i]
			norm_sq = u.dot(u)
			if norm_sq < threshold:
				u[:] = 0.0
				continue
			u2 = u/norm_sq
			for j in xrange(i+1, self._crvfitter_correctors.shape[1]):
				v = self._crvfitter_correctors[:, j]
				v -= v.dot(u)*u2

	def normalize_correctors(self):
		'''Normalizes the energy of each corrector (the magnitude of each feature interpreted as a vector,
			that is, the magnitude of each column of the internal correctors matrix).

			Modifies:

			    - Correctors: each column has been normalized to have unit magnitude.

			Returns:

			    - None
		'''
		threshold = self._crvfitter_correctors.shape[0]*CurveFitter.__threshold
		for i in xrange(self._crvfitter_correctors.shape[1]):
			u = self._crvfitter_correctors[:, i]
			norm_sq = u.dot(u)
			if norm_sq >= threshold:
				u /= norm_sq**0.5
			elif norm_sq != 0.0:
				u[:] = 0.0

	def orthonormalize_correctors(self):
		'''Orthogonalizes each corrector with respect to all the previous ones, and normalizes the results.
			This is equivalent to applying orthogonalize_correctors and normalize_correctors consecutively
			(in that same order), but slightly faster.
			
			Modifies:

			    - Correctors: each column has been orthogonalized w.r.t. the previous ones, and normal-
			        ized afterwards.

			Returns:

			    - None
		'''
		# Gram-Schmidt
		threshold = self._crvfitter_correctors.shape[0]*CurveFitter.__threshold
		for i in xrange(self._crvfitter_correctors.shape[1]):
			u = self._crvfitter_correctors[:, i]
			norm_sq = u.dot(u)
			if norm_sq < threshold:
				u[:] = 0.0
				continue
			u /= norm_sq**0.5 # Normalize u
			for j in xrange(i+1, self._crvfitter_correctors.shape[1]):
				v = self._crvfitter_correctors[:, j]
				v -= v.dot(u)*u # Orthogonalize v with respect to u

	def orthogonalize_regressors(self):
		'''Orthogonalizes each regressor in the structure w.r.t. all the previous ones. That is, for each
			column in the regressors matrix, its projection over the previous columns is computed and sub-
			tracted from it.

			Modifies:

			    - Regressors: each column has been orthogonalized with respect to the previous ones.

			Returns:

			    - None
		'''
		# Gram-Schmidt
		threshold = self._crvfitter_regressors.shape[0]*CurveFitter.__threshold
		for i in xrange(self._crvfitter_regressors.shape[1] - 1):
			u = self._crvfitter_regressors[:, i]
			norm_sq = u.dot(u)
			if norm_sq < threshold:
				u[:] = 0.0
				continue
			u2 = u/norm_sq
			for j in xrange(i+1, self._crvfitter_regressors.shape[1]):
				v = self._crvfitter_regressors[:, j]
				v -= v.dot(u)*u2

	def normalize_regressors(self):
		'''Normalizes the energy of each regressor (the magnitude of each feature interpreted as a vector,
			that is, the magnitude of each column of the internal regressors matrix).

			Modifies:

			    - Regressors: each column has been normalized to have unit magnitude.

			Returns:

			    - None
		'''
		threshold = self._crvfitter_regressors.shape[0]*CurveFitter.__threshold
		for i in xrange(self._crvfitter_regressors.shape[1]):
			u = self._crvfitter_regressors[:, i]
			norm_sq = u.dot(u)
			if norm_sq >= threshold:
				u /= norm_sq**0.5
			elif norm_sq != 0.0:
				u[:] = 0.0

	def orthonormalize_regressors(self):
		'''Orthogonalizes each regressors with respect to all the previous ones, and normalizes the results.
			This is equivalent to applying orthonormalize_regressors and normalize_regressors consecutively
			(in that same order), but slightly faster.
			
			Modifies:

			    - Regressors: each column has been orthogonalized w.r.t. the previous ones, and normal-
			        ized afterwards.

			Returns:

			    - None
		'''
		# Gram-Schmidt
		threshold = self._crvfitter_regressors.shape[0]*CurveFitter.__threshold
		for i in xrange(self._crvfitter_regressors.shape[1]):
			u = self._crvfitter_regressors[:, i]
			norm_sq = u.dot(u)
			if norm_sq < threshold:
				u[:] = 0.0
				continue
			u /= norm_sq**0.5 # Normalize u
			for j in xrange(i+1, self._crvfitter_regressors.shape[1]):
				v = self._crvfitter_regressors[:, j]
				v -= v.dot(u)*u # Orthogonalize v with respect to u

	def orthogonalize_all(self):
		'''Orthogonalizes each regressor w.r.t the others, all correctors w.r.t. the others, and all the
			regressors w.r.t. all the correctors.

			Modifies:

			    - Correctors: each column has been orthogonalized with respect to the previous ones.
			    - Regressors: each column has been orthogonalized with respect to all the columns in
			        the correctors matrix and all the previous columns in the regressors matrix.

			Returns:

			    - None
		'''
		# Gram-Schmidt
		threshold = self._crvfitter_correctors.shape[0]*CurveFitter.__threshold
		for i in xrange(self._crvfitter_correctors.shape[1]):
			u = self._crvfitter_correctors[:, i]
			norm_sq = u.dot(u)
			if norm_sq < threshold:
				u[:] = 0.0
				continue
			u2 = u/norm_sq
			for j in xrange(i+1, self._crvfitter_correctors.shape[1]):
				v = self._crvfitter_correctors[:, j]
				v -= v.dot(u)*u2
			for j in xrange(self._crvfitter_regressors.shape[1]):
				v = self._crvfitter_regressors[:, j]
				v -= v.dot(u)*u2
		self.orthogonalize_regressors()

	def normalize_all(self):
		'''Normalizes the energy of each corrector and each regressor (the magnitude of each feature
			interpreted as a vector, that is, the magnitude of each column of the internal correctors and
			regressors matrices).

			Modifies:

			    - Correctors: each column has been normalized to have unit magnitude.
			    - Regressors: each column has been normalized to have unit magnitude.

			Returns:

			    - None
		'''
		self.normalize_correctors()
		self.normalize_regressors()

	def orthonormalize_all(self):
		'''Orthogonalizes each regressor w.r.t the others, all correctors w.r.t. the others, and all the
			regressors w.r.t. all the correctors, and normalizes the results. This is equivalent to applying
			orthogonalize_all and normalize_all consecutively (in that same order), but slightly faster.

			Modifies:

			    - Correctors: each column has been orthogonalized with respect to the previous ones and nor-
			        malized afterwards.
			    - Regressors: each column has been orthogonalized with respect to all the columns in the
			        correctors matrix and all the previous columns in the regressors matrix, and normalized
			        afterwards.

			Returns:

			    - None
		'''
		# Gram-Schmidt
		threshold = self._crvfitter_correctors.shape[0]*CurveFitter.__threshold
		for i in xrange(self._crvfitter_correctors.shape[1]):
			u = self._crvfitter_correctors[:, i]
			norm_sq = u.dot(u)
			if norm_sq < threshold:
				u[:] = 0.0
				continue
			u /= norm_sq**0.5 # Normalize u
			for j in xrange(i+1, self._crvfitter_correctors.shape[1]):
				v = self._crvfitter_correctors[:, j]
				v -= v.dot(u)*u # Orthogonalize v with respect to u
			for j in xrange(self._crvfitter_regressors.shape[1]):
				v = self._crvfitter_regressors[:, j]
				v -= v.dot(u)*u # Orthogonalize v with respect to u
		self.orthonormalize_regressors()

	@abstractstatic
	def __fit__(correctors, regressors, observations, *args, **kwargs):
		'''[Abstract method] Computes the correction and regression parameters that best fit the observations.
			This method is not intended to be called outside the CurveFitter class.

			Parameters:

			    - correctors: NxC (2-dimensional) matrix, representing the covariates, i.e., features that
			        (may) explain a part of the observational data in which we are not interested, where C
			        is the number of correctors and N the number of elements for each corrector.

			    - regressors: NxR (2-dimensional) matrix, representing the predictors, i.e., features to be used
			        to try to explain/predict the observations (experimental data), where R is the number of
			        regressors and N the number of elements for each regressor (the latter is ensured to be the
			        same as that in the 'correctors' argument).

			    - observations: NxM (2-dimensional) matrix, representing the observational data, i.e., values
			        obtained by measuring the variables of interest, whose behaviour is wanted to be explained
			        by the correctors and regressors, where M is the number of variables and N the number of
			        observations for each variable (the latter is ensured to be the same as those in the
			        'correctors' and the 'regressors' arguments).

			    - any other arguments will also be passed to the method in the subclass.

			Returns:

			    - Correction parameters: (Kc)xM (2-dimensional) matrix, representing the parameters that best
			        fit the correctors to the observations for each variable, where M is the number of variables
			        (same as that in the 'observations' argument) and Kc is the number of correction parameters
			        for each variable.

			    - Regression parameters: (Kr)xM (2-dimensional) matrix, representing the parameters that best
			        fit the regressors to the corrected observations for each variable, where M is the number of
			        variables (same as that in the 'observations' argument) and Kr is the number of regression
			        parameters for each variable.


			[Developer notes]
			    - Assertions regarding the size and type of the arguments have already been performed before the
			        call to this method to ensure that the sizes of the arguments are coherent and the observations
			        matrix has at least one element.

			    - The 'correctors' and 'regressors' matrices may have zero elements, in which case the behaviour
			        of the method is left to be decided by the subclass.

			    - You may modify the 'observations' matrix if needed, but both the 'correctors' and the 'regressors'
			        arguments should be left unchanged.

			    - The result should be returned as a tuple of 2 elements, containing the correction parameters in
			        the first position and the regression parameters in the second position.

			    - Although it is defined as a static method here, this method supports a non-static implementation.
		'''
		raise NotImplementedError

	def fit(self, observations, *args, **kwargs):
		'''Computes the correction and regression parameters that best fit the observations.

			Parameters:

			    - observations: array-like structure of shape (N, X1, ..., Xn), representing the observational data,
			        i.e., values obtained by measuring the variables of interest, whose behaviour is wanted to be
			        explained by the correctors and regressors in the system, where M = X1*...*Xn is the number of
			        variables and N the number of observations for each variable.

			    - any other arguments will be passed to the __fit__ method.

			Modifies:

			    - [created] Correction parameters: array-like structure of shape (Kc, X1, ..., Xn), representing the
			        parameters that best fit the correctors to the observations, where X1, ..., Xn are the original
			        dimensions of the 'observations' argument and Kc is the number of correction parameters for each
			        variable.

			    - [created] Regression parameters: array-like structure of shape (Kr, X1, ..., Xn), representing the
			        parameters that best fit the regressors to the observations, where X1, ..., Xn are the original
			        dimensions of the 'observations' argument and Kr is the number of regression parameters for each
			        variable.
		'''
		obs = nparray(observations, dtype = float64)
		dims = obs.shape
		self._crvfitter_dims = dims[1:]
		if dims[0] != self._crvfitter_regressors.shape[0]:
			raise ValueError('Observations and features (correctors and/or regressors) have incompatible sizes')
		
		if 0 in dims:
			raise ValueError('There are no elements in argument \'observations\'')

		obs = obs.reshape(dims[0], -1)
		self._crvfitter_correction_parameters, self._crvfitter_regression_parameters = self.__fit__(self._crvfitter_correctors, self._crvfitter_regressors, obs, *args, **kwargs)

	@abstractstatic
	def __evaluate_fit__(correctors, correction_parameters, regressors, regression_parameters, observations, *args, **kwargs):
		'''[Abstract method] Evaluates the degree to which the correctors and regressors get to explain the
			observational data passed for the last time to the 'fit' method.
			This method is not intended to be called outside the CurveFitter class.

			Parameters:

			    - correctors: NxC (2-dimensional) matrix, representing the covariates, i.e., features that (may)
			        explain a part of the observational data in which we are not interested, where C is the number
			        of correctors and N the number of elements for each corrector.

			    - correction_parameters: (Kc)xM (2-dimensional) matrix, representing the parameters that best fit
			        the correctors to the observations for each variable, where M is the number of variables and Kc
			        is the number of correction parameters for each variable.

			    - regressors: NxR (2-dimensional) matrix, representing the predictors, i.e., features to be used
			        to try to explain/predict the observations (experimental data), where R is the number of
			        regressors and N the number of elements for each regressor (the latter is ensured to be the
			        same as that in the 'correctors' argument).

			    - regression_parameters: (Kr)xM (2-dimensional) matrix, representing the parameters that best fit
			        the regressors to the corrected observations for each variable, where M is the number of variables
			        and Kr is the number of regression parameters for each variable.

			    - observations: NxM (2-dimensional) matrix, representing the observational data, i.e., values
			        obtained by measuring the variables of interest, whose behaviour is wanted to be explained
			        by the correctors and regressors, where M is the number of variables and N the number of
			        observations for each variable (the latter is ensured to be the same as those in the
			        'correctors' and the 'regressors' arguments).

			    - any other arguments will also be passed to the method in the subclass.

			Returns:

			    - fitting scores: array of length M, containing floats between 0 and 1 that indicate the degree
			        to which the correctors and regressors fit the observations for each variable; the significance
			        of the regressors in the behaviour of the observations for each variable; the greater the score,
			        the greater the significance, and thus, the better the fit.

			[Developer notes]
			    - Assertions regarding the size and type of the arguments have already been performed before the
			        call to this method to ensure that the sizes of the arguments are coherent and all, the
			        'observations', 'regressors', and 'regression_parameters' matrices have at least one element
			        each.

			   	- You may modify the 'observations' matrix if needed, but all the 'correctors', the 'regressors',
			        the 'correction_parameters' and the 'regression_parameters' arguments should be left unchanged.

			    - The 'correctors' and 'correction_parameters' matrices may have zero elements.

			    - Although it is defined as a static method here, this method supports a non-static implementation.
		'''
		raise NotImplementedError

	def evaluate_fit(self, observations, correctors = None, correction_parameters = None, regressors = None, regression_parameters = None, *args, **kwargs):
		'''Evaluates the degree to which the correctors and regressors get to explain the observational
			data passed in the 'observations' argument.

			Parameters:

			    - observations: array-like structure of shape (N, X1, ..., Xn), representing the observational data,
			        i.e., values obtained by measuring the variables of interest, whose behaviour is wanted to be
			        explained by the correctors and regressors in the system, where M = X1*...*Xn is the number of
			        variables and N the number of observations for each variable.

			    - correctors: NxC (2-dimensional) matrix (default None), representing the covariates, i.e., features
			        that (may) explain a part of the observational data in which we are not interested, where C is
			        the number of correctors and N the number of elements for each corrector. If set to None, the
			        internal correctors will be used.

			    - correction_parameters: array-like structure of shape (Kc, X1, ..., Xn) (default None), representing
			        the parameters to fit the correctors to the observations for each variable, where M = X1*...*Xn
			        is the number of variables and Kc the number of correction parameters for each variable. If set
			        to None, the correction parameters obtained in the last call to 'fit' will be used.

			    - regressors: NxR (2-dimensional) matrix (default None), representing the predictors, i.e., features
			        to be used to try to explain/predict the observations (experimental data), where R is the number
			        of regressors and N the number of elements for each regressor. If set to None, the internal re-
			        gressors will be used.

			    - regression_parameters: array-like structure of shape (Kr, X1, ..., Xn) (default None), representing
			        the parameters to fit the regressors to the corrected observations for each variable, where M =
			        X1*...*Xn is the number of variables and Kr is the number of regression parameters for each
			        variable. If set to None, the regression parameters obtained in the last call to 'fit' will be
			        used.

			    - any other arguments will be passed to the __evaluate_fit__ method.

			Returns:

			    - fitting scores: array-like structure of shape (X1, ..., Xn), containing floats between 0 and 1
			        that indicate the significance of the regressors in the behaviour of the observations for each
			        variable; the greater the score, the greater the significance, and thus, the better the fit.
		'''

		obs = nparray(observations, dtype = float64)
		dims = obs.shape
		obs = obs.reshape(dims[0], -1)

		
		if 0 in dims:
			raise ValueError('There are no elements in argument \'observations\'')

		if correctors is None:
			cors = self._crvfitter_correctors
			if 0 in cors.shape:
				correctors_present = False
			else:
				correctors_present = True
		else:
			cors = nparray(correctors, dtype = float64)
			if len(cors.shape) != 2:
				raise TypeError('Argument \'correctors\' must be a 2-dimensional matrix')
			
			if 0 in cors.shape:
				raise ValueError('There are no elements in argument \'correctors\'')

			correctors_present = True

		if correctors_present:
			if obs.shape[0] != cors.shape[0]:
				raise ValueError('The dimensions of the observations and the correctors are incompatible')

			if correction_parameters is None:
				cparams = self._crvfitter_correction_parameters
				if 0 in cparams.shape:
					raise AttributeError('There are no correction parameters in this instance')
			else:
				cparams = nparray(correction_parameters, dtype = float64)
				cparams = cparams.reshape(cparams.shape[0], -1)

				if 0 in cparams.shape:
					raise ValueError('There are no elements in argument \'correction_parameters\'')

			if obs.shape[1] != cparams.shape[1]:
				raise ValueError('The dimensions of the observations and the correction parameters are incompatible')

		else:
			cparams = zeros((0, 0))


		if regressors is None:
			regs = self._crvfitter_regressors
			if 0 in regs.shape:
				raise AttributeError('There are no regressors in this instance')
		else:
			regs = nparray(regressors, dtype = float64)

			if len(regs.shape) != 2:
				raise TypeError('Argument \'regressors\' must be a 2-dimensional matrix')

			if 0 in regs.shape:
				raise ValueError('There are no elements in argument \'regressors\'')

		if obs.shape[0] != regs.shape[0]:
				raise ValueError('The dimensions of the observations and the regressors are incompatible')

		if regression_parameters is None:
			rparams = self._crvfitter_regression_parameters
			if 0 in rparams.shape:
				raise AttributeError('There are no regression parameters in this instance')
		else:
			rparams = nparray(regression_parameters, dtype = float64)
			# Make matrix 2-dimensional
			rparams = rparams.reshape(rparams.shape[0], -1)

			if 0 in rparams.shape:
				raise ValueError('There are no elements in argument \'regression_parameters\'')

		if obs.shape[1] != rparams.shape[1]:
			raise ValueError('The dimensions of the observations and the regression parameters are incompatible')

		fitting_scores = self.__evaluate_fit__(cors, cparams, regs, rparams, obs, *args, **kwargs)

		return fitting_scores.reshape(dims[1:])


	@abstractstatic
	def __predict__(regressors, regression_parameters, *args, **kwargs):
		'''[Abstract method] Computes a prediction using the regressors together with the regression parameters.
			This method is not intended to be called outside the CurveFitter class.

			Parameters:

			    - regressors: NxR (2-dimensional) matrix, representing the predictors, i.e., features to be used
			        to try to explain/predict the observations (experimental data), where R is the number of
			        regressors and N the number of elements for each regressor.

			    - regression_parameters: (Kr)xM (2-dimensional) matrix, representing the parameters that best fit
			        the regressors to the corrected observations for each variable, where M is the number of
			        variables and Kr is the number of regression parameters for each variable.

			    - any other arguments will also be passed to the method in the subclass.

			Returns:

			    - Prediction: NxM (2-dimensional) matrix, containing N predicted values for each of the M variables.

			[Developer notes]
			    - Assertions regarding the size and type of the arguments have already been performed before the
			        call to this method to ensure that the sizes of the arguments are coherent and both, the
			        'regressors' and the 'regression_parameters' matrices have at least one element each.

			    - Both the 'regressors' and the 'regression_parameters' arguments should be left unchanged.

			    - Although it is defined as a static method here, this method supports a non-static implementation.
		'''
		raise NotImplementedError

	def predict(self, regressors = None, regression_parameters = None, *args, **kwargs):
		'''Computes a prediction using the regressors together with the regression parameters.

			Parameters:

			    - regressors: NxR (2-dimensional) matrix (default None), representing the predictors, i.e., features
			        to be used to try to explain/predict the observations (experimental data), where R is the number
			        of regressors and N the number of elements for each regressor. If set to None, the regressors of
			        the instance will be used.

			    - regression_parameters: array-like structure of shape (Kr, X1, ..., Xn) (default None), representing
			        the parameters to fit the regressors to the corrected observations for each variable, where M =
			        X1*...*Xn is the number of variables and Kr is the number of regression parameters for each
			        variable. If set to None, the regression parameters obtained in the last call to 'fit' will be
			        used.

			    - any other arguments will be passed to the __predict__ method.

			Returns:

			    - Prediction: array-like structure of shape (N, X1, ..., Xn), containing N predicted values for each of
			        the M = X1*...*Xn variables.
		'''
		if regressors is None:
			regs = self._crvfitter_regressors
			if 0 in regs.shape:
				raise AttributeError('There are no regressors in this instance')
		else:
			regs = nparray(regressors, dtype = float64)
			if len(regs.shape) != 2:
				raise TypeError('Argument \'regressors\' must be a 2-dimensional matrix')
			if 0 in regs.shape:
				raise ValueError('There are no elements in argument \'regressors\'')

		if regression_parameters is None:
			params = self._crvfitter_regression_parameters
			dims = (1,) + self._crvfitter_dims
		else:
			params = nparray(regression_parameters, dtype=float64)
			# Keep original dimensions (to reset dimensions of prediction)
			dims = params.shape
			# Make matrix 2-dimensional
			params = params.reshape(dims[0], -1)

		if 0 in dims:
			raise ValueError('There are no elements in argument \'regression_parameters\'')

		prediction = self.__predict__(regs, params, *args, **kwargs)

		# Restore original dimensions (except for the first axis)
		return prediction.reshape(-1, *dims[1:])

	@abstractstatic
	def __correct__(observations, correctors, correction_parameters, *args, **kwargs):
		'''[Abstract method] Computes the values of the data after accounting for the correctors by using the
			correction parameters. 
			This method is not intended to be called outside the CurveFitter class.

			Parameters:

			    - observations: NxM (2-dimensional) matrix, representing the observational data to be corrected,
			        i.e., the values obtained by measuring the variables of interest from which we want to subtract
			        the contribution of the correctors, where M is the number of variables and N the number of
			        observations for each variable.

			    - correctors: NxC (2-dimensional) matrix, representing the covariates, i.e., features that (may)
			        explain a part of the observational data in which we are not interested, where C is the number
			        of correctors and N the number of elements for each corrector.

			    - correction_parameters: (Kc)xM (2-dimensional) matrix, representing the parameters that best fit
			        the correctors to the observations for each variable, where M is the number of variables and Kc
			        is the number of correction parameters for each variable.

			    - any other arguments will also be passed to the method in the subclass.

			Returns:

			    - corrected data: NxM (2-dimensional) matrix, containing the observational data after having sub-
			        tracted the contribution of the correctors by using the correction parameters.

			[Developer notes]
			    - Assertions regarding the size and type of the arguments have already been performed before the
			        call to this method to ensure that the sizes of the arguments are coherent and all, the
			        'observations', 'correctors', and 'correction_parameters' matrices have at least one element
			        each.

			    - The 'correction_parameters' matrix may have zero elements.

			    - You may modify the 'observations' matrix if needed. However, both the 'correctors' and the
			        'correction_parameters' arguments should be left unchanged.

			    - Although it is defined as a static method here, this method supports a non-static implementation.
		'''
		raise NotImplementedError

	def correct(self, observations, correctors = None, correction_parameters = None, *args, **kwargs):
		'''Computes the values of the data after accounting for the correctors by using the correction parameters.

			Parameters:

			    - observations: array-like matrix of shape (N, X1, ..., Xn), representing the observational data,
			        i.e., values obtained by measuring the variables of interest, whose behaviour is wanted to be
			        explained by the correctors and regressors in the system, where M = X1*...*Xn is the number of
			        variables and N the number of observations for each variable.

			    - correctors: NxC (2-dimensional) matrix (default None), representing the covariates, i.e., features
			        that (may) explain a part of the observational data in which we are not interested, where C is
			        the number of correctors and N the number of elements for each corrector. If set to None, the
			        internal correctors will be used.

			    - correction_parameters: array-like structure of shape (Kc, X1, ..., Xn) (default None), representing
			        the parameters to fit the correctors to the observations for each variable, where M = X1*...*Xn
			        is the number of variables and Kc the number of correction parameters for each variable. If set
			        to None, the correction parameters obtained in the last call to 'fit' will be used.

			    - any other arguments will be passed to the __correct__ method.

			Returns:

			    - Corrected data: array-like matrix of shape (N, X1, ..., Xn), containing the observational data
			        after having subtracted the contribution of the correctors by using the correction parameters.
		'''

		## Treat observations
		obs = nparray(observations, dtype = float64)
		# Keep original dimensions (to reset dimensions of corrected data)
		dims = obs.shape
		# Make matrix 2-dimensional
		obs = obs.reshape(dims[0], -1)

		# Check correctness of matrix
		if 0 in dims:
			return zeros(dims)

		## Treat correctors
		if correctors is None:
			cors = self._crvfitter_correctors
			if 0 in cors.shape:
				return observations
		else:
			cors = nparray(correctors, dtype = float64)
			if len(cors.shape) != 2:
				raise TypeError('Argument \'correctors\' must be a 2-dimensional matrix')
			
			if 0 in cors.shape:
				raise ValueError('There are no elements in argument \'correctors\'')

		if obs.shape[0] != cors.shape[0]:
			raise ValueError('The dimensions of the observations and the correctors are incompatible')

		## Treat correction parameters
		if correction_parameters is None:
			params = self._crvfitter_correction_parameters
		else:
			params = nparray(correction_parameters, dtype = float64)
			params = params.reshape(params.shape[0], -1)

			if 0 in params.shape:
				raise ValueError('There are no elements in argument \'correction_parameters\'')

		if obs.shape[1] != params.shape[1]:
			raise ValueError('The dimensions of the observations and the correction parameters are incompatible')

		## Compute corrected data
		corrected_data = self.__correct__(obs, cors, params, *args, **kwargs)

		# Restore original dimensions
		return corrected_data.reshape(dims)


class AdditiveCurveFitter(CurveFitter):
	'''CurveFitter subclass for the cases in which the following model is assumed:
		    Y = fp(C, CP) + fp(R, RP) + err
		where 'Y' is the set of observations, 'C' is the set of correctors, 'CP' is the set of correction parameters,
		'R' is the set of regressors, 'RP' the set of regression parameters, 'fp' is an arbitrary function (more
		specifically, it is the prediction function), and 'err' is the residual error of the model.
	'''
	
	def __correct__(self, observations, correctors, correction_parameters, *args, **kwargs):
		'''The correction is performed by applying the following formula:
			    Y - fp(C, CP)
			where 'Y' is the 'observations' argument, 'C' is the 'correctors' argument, 'CP' is the 'correction_parameters'
			argument and 'fp' is the '__predict__' method of the subclass. Any additional arguments are also passed
			to the 'fp' function.
			(See the '__predict__' and '__correct__' abstract methods in CurveFitter)
		'''
		return observations - self.__predict__(correctors, correction_parameters, *args, **kwargs)

	def __evaluate_fit__(self, correctors, correction_parameters, regressors, regression_parameters, observations, *args, **kwargs):
		'''Evaluates the significance of the regressors as regards the behaviour of the observations performing an
			F-test. In particular, the null hypothesis states that the regressors do not explain the variation of
			the observations at all. The inverse of the p-value of such experiment (1 - p_value) is returned.
		'''
		if 0 in correctors.shape:
			# There is no correction -> Correction error is same as observations
			correction_error = observations
		else:
			# Compute correction error
			correction_error = self.__correct__(observations, correctors, correction_parameters)

		## Get the error obtained when using the full model (correctors + regressors)
		# prediction = self.__predict__(regressors, regression_parameters)

		# regression_error = correction_error - prediction
		regression_error = self.__correct__(correction_error, regressors, regression_parameters)

		## Now compare the variances of the errors

		# Residual Sum of Squares for restricted model
		rss1 = sum((correction_error - correction_error.mean())**2)
		p1 = correctors.shape[1]

		# Residual Sum of Squares for full model
		rss2 = sum(regression_error**2)
		p2 = p1 + regressors.shape[1]

		# Degrees of freedom
		n = observations.shape[0]
		df1 = p2 - p1
		df2 = n - p2

		# Compute f-scores and p-values
		var1 = (rss1 - rss2)/df1
		var2 = rss2/df2
		f_score = var1/var2

		# print rss1, rss2
		# print 'Df Residuals:', df2
		# print 'Df Model:', df1
		# print 'F-statistic:', f_score
		# print 'R-squared:', 1 - rss2/rss1

		return f_stat.cdf(f_score, df1, df2)




