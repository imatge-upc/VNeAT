from CurveFitting import AdditiveCurveFitter
from numpy import zeros, array as nparray
from sklearn.linear_model import LinearRegression as LR
from Transforms import polynomial

class GLM(AdditiveCurveFitter):

	''' Class that implements the General Linear Method.

		This method assumes the following situation:

		    - There are M (random) variables whose behaviour we want to explain.

		    - Each of the M variables has been measured N times, obtaining thus
		      an NxM matrix of observations (i-th column contains the N observa-
		      tions for i-th variable).

		    - There are K regressors (in this class both, the correctors and the
		      regressors are called regressors and treated equally) that might
		      explain the behaviour of the M variables in an additive manner, i.e.,
		      a ponderated sum of the K regressors might fit each of the variables.

		    - Each of the K regressors has been measured at the same moments in
		      which the M variables were measured, giving thus a NxK matrix where
		      the i-th column represents the N observations of the i-th regressor.
		
		In this situation, the relationship of the different elements can be ex-
		pressed as follows:

		    OBS(NxM) = MODEL(NxK) * PARAMS(KxM) + eps(NxM),

		where OBS denotes the NxM matrix containing the N observations of each of
		the M variables, MODEL denotes the NxK matrix containing the N observations
		of each of the K regressors, PARAMS denotes the KxM matrix of ponderation
		coefficients (one for each variable and regressor, that is, the amplitude
		each regressor has in each variable), and eps denotes the error commited
		when making the aforementioned assumptions, i.e., a NxM matrix that contains
		the data that is left unexplained after accounting for all the regressors
		in the model.

		This class provides the tools to orthogonalize each of the regressors in
		the matrix with respect to the ones in the previous columns, and to esti-
		mate the ponderation coefficients (the PARAMS matrix) so that the energy
		of the error (the MSE) is minimized.
	'''

	@staticmethod
	def __predict__(regressors, regression_parameters):
		'''Computes a prediction applying the prediction function used in GLM.

			Parameters:

			    - regressors: NxR (2-dimensional) matrix, representing the predictors, i.e., features to be used
			        to try to explain/predict the observations (experimental data), where R is the number of
			        regressors and N the number of elements for each regressor.

			    - regression_parameters: KxM (2-dimensional) matrix, representing the parameters that best fit
			        the regressors to the corrected observations for each variable, where M is the number of
			        variables and K is the number of regression parameters for each variable.

			    - any other arguments will also be passed to the method in the subclass.

			Returns:

			    - Prediction: NxM (2-dimensional) matrix, containing N predicted values for each of the M variables,
			        result of computing the expression 'regressors * regression_parameters' (matrix multiplication).
		'''
		return regressors.dot(regression_parameters)

	@staticmethod
	def __fit__(correctors, regressors, observations, sample_weight = None, num_threads = -1):
		'''Computes the correction and regression parameters that best fit the observations according to the
			General Linear Model.

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

			    - sample_weight: array of length N (default None), indicating the weight of each sample for
			        the fitting algorithm, where N is the number of observations for each variable. If set
			        to None, each sample will have the same weight.

			    - num_threads: integer (default -1), indicating the number of threads to be used by the algo-
			        rithm. If set to -1, all CPUs are used. This will only provide speed-up for M > 1 and
			        sufficiently large problems.

			Returns:

			    - Correction parameters: CxM (2-dimensional) matrix, representing the parameters that best fit
			        the correctors to the observations for each variable, where M is the number of variables
			        (same as that in the 'observations' argument) and C is the number of correction parameters
			        for each variable (same as the number of correctors).

			    - Regression parameters: RxM (2-dimensional) matrix, representing the parameters that best fit
			        the regressors to the corrected observations for each variable, where M is the number of
			        variables (same as that in the 'observations' argument) and R is the number of regression
			        parameters for each variable (same as the number of regressors).
		'''
		curve = LR(fit_intercept = False, normalize = False, copy_X = False, n_jobs = num_threads)
		
		ncols = correctors.shape[1]
		dims = (correctors.shape[0], ncols + regressors.shape[1])
		xdata = zeros(dims)
		xdata[:, :ncols] = correctors.view()
		xdata[:, ncols:] = regressors.view()
		
		curve.fit(xdata, observations, sample_weight)
		params = curve.coef_.T
		return (params[:ncols], params[ncols:])


class PolyGLM(GLM):
	def __init__(self, features, regressors = None, degrees = None, homogeneous = True):
		'''[Constructor]

			Parameters:

			    - features: NxF (2-dimensional) matrix representing the features to be used to try to
			        explain some observations (experimental data), either by using them as correctors/
			        covariates or regressors/predictors in the model, where F is the number of features
			        and N the number of samples for each feature.

			    - regressors: int / iterable object (default None), containing the index/indices of the
			        column(s) in the 'features' matrix that must be used as regressors. If set to None,
			        all the columns of such matrix will be interpreted as regressors.

			    - degrees: iterable of F elements (default None), containing the degree of each feature
			        in the 'features' argument, where F is the number of features. If set to None, only
			        the linear term of each feature will be taken into account (same as setting all the
			        degrees to 1).

			    - homogeneous: bool (default True), indicating whether the homogeneous term (a.k.a.
			        intercept) must be incorporated to the model as a corrector or not. If so, a column
			        of ones will be added as the first column (feature) of the internal correctors matrix.
		'''
		self._pglm_features = nparray(features)
		if len(self._pglm_features.shape) != 2:
			raise ValueError('Argument \'features\' must be a 2-dimensional matrix')
		self._pglm_features = self._pglm_features.T

		if regressors is None:
			self._pglm_is_regressor = [True]*len(self._pglm_features)
			regressors = []
		else:
			self._pglm_is_regressor = [False]*len(self._pglm_features)
			if isinstance(regressors, int):
				regressors = [regressors]

		try:
			for r in regressors:
				try:
					self._pglm_is_regressor[r] = True
				except TypeError:
					raise ValueError('All elements in argument \'regressors\' must be valid indices')
				except IndexError:
					raise IndexError('Index out of range in argument \'regressors\'')
		except TypeError:
			raise TypeError('Argument \'regressors\' must be iterable or int')
		
		if degrees is None:
			self._pglm_degrees = [1]*len(self._pglm_features)
		else:
			degrees = list(degrees)
			if len(degrees) != len(self._pglm_features):
				raise ValueError('Argument \'degrees\' must have a length equal to the number of features')
			for deg in degrees:
				if not isinstance(deg, int):
					raise ValueError('Expected integer in \'degrees\' list, got ' + str(type(deg)) + ' instead')
				if deg < 1:
					raise ValueError('All degrees must be >= 1')
			self._pglm_degrees = degrees

		self._pglm_homogeneous = homogeneous

		self.__pglm_update_GLM()

	def __pglm_update_GLM(self):
		'''Private function. Not meant to be used by anyone outside the PolyGLM class.
		'''
		correctors = []
		regressors = []
		for index in range(len(self._pglm_is_regressor)):
			for p in polynomial(self._pglm_degrees[index], [self._pglm_features[index]]):
				if self._pglm_is_regressor[index]:
					regressors.append(p)
				else:
					correctors.append(p)

		if len(correctors) == 0:
			correctors = None
		else:
			correctors = nparray(correctors).T

		if len(regressors) == 0:
			regressors = None
		else:
			regressors = nparray(regressors).T

		super(PolyGLM, self).__init__(regressors, correctors, self._pglm_homogeneous)

	@property
	def lin_correctors(self):
		'''Matrix containing the linear terms of the features that are interpreted as correctors in the model.
		'''
		r = [self._pglm_features[i] for i in range(len(self._pglm_is_regressor)) if not self._pglm_is_regressor[i]]
		return nparray(r).T

	@property
	def lin_regressors(self):
		'''Matrix containing the linear terms of the features that are interpreted as regressors in the model.
		'''
		r = [self._pglm_features[i] for i in range(len(self._pglm_is_regressor)) if self._pglm_is_regressor[i]]
		return nparray(r).T

	@property
	def features(self):
		'''Original features with which the instance was initialized (not including the homogeneous term).
			Note: do not modify the result (use copy method present in it if you need to treat this data).
		'''
		return self._pglm_features.T

	def set_regressors(self, regressors):
		'''Reselects the regressors of the model.

			Parameters:

			    - regressors: int / iterable object (default None), containing the index/indices of the
			        column(s) in the 'features' matrix that must be used as regressors. If set to None,
			        all the columns of such matrix will be interpreted as regressors.

			Modifies:

			    - Correctors: the new correctors are set, deorthogonalized and denormalized.

			    - Regressors: the new regressors are set, deorthogonalized and denormalized.

			    - [deleted] Correction parameters

			    - [deleted] Regression parameters
		'''
		if regressors is None:
			pglm_is_regressor = [True]*len(self._pglm_features)
			regressors = []
		else:
			pglm_is_regressor = [False]*len(self._pglm_features)
			if isinstance(regressors, int):
				regressors = [regressors]

		try:
			for r in regressors:
				try:
					pglm_is_regressor[r] = True
				except TypeError:
					raise ValueError('All elements in argument \'regressors\' must be valid indices')
				except IndexError:
					raise IndexError('Index out of range in argument \'regressors\'')
		except TypeError:
			raise TypeError('Argument \'regressors\' must be iterable or int')
		
		self._pglm_is_regressor = pglm_is_regressor

		self.__pglm_update_GLM()

