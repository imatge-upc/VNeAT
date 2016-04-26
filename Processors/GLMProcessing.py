import numpy as np
from Processors.Processing import Processor

from Fitters.GLM import GLM, PolyGLM as PGLM


class GLMProcessor(Processor):
	_glmprocessor_perp_norm_options_names = [
		'Orthonormalize all',
		'Orthogonalize all',
		'Normalize all',
		'Orthonormalize predictors',
		'Orthogonalize predictors',
		'Normalize predictors',
		'Orthonormalize correctors',
		'Orthogonalize correctors',
		'Normalize correctors',
		'Use correctors and predictors as they are'
	]

	_glmprocessor_perp_norm_options_list = [
		GLM.orthonormalize_all,
		GLM.orthogonalize_all,
		GLM.normalize_all,
		GLM.orthonormalize_predictors,
		GLM.orthogonalize_predictors,
		GLM.normalize_predictors,
		GLM.orthonormalize_correctors,
		GLM.orthogonalize_correctors,
		GLM.normalize_correctors,
		lambda *args, **kwargs: np.zeros((0, 0))
	]

	_glmprocessor_intercept_options_names = [
		'Do not include the intercept term',
		'As a corrector',
		'As a predictor'
	]

	_glmprocessor_intercept_options_list = [
		GLM.NoIntercept,
		GLM.CorrectionIntercept,
		GLM.PredictionIntercept
	]

	def _glmprocessor_compute_original_parameters(self, Gamma, Beta2):
		'''Given an upper triangular matrix Gamma, and an arbitrary matrix Beta2, computes Beta such
			that Beta2 = Gamma * Beta.
			Notice that if Beta2 is the np.identity matrix, then Beta is the right-pseudoinverse of Gamma.
		'''

		# Gamma is the deorthogonalization (upper triangular) matrix
		# Beta2 is the matrix with the optimal parameters of the orthonormalized design matrix
		# Beta is the matrix with the optimal parameters of the original design matrix

		# The relationships between the different matrices of the system are described below:
		# (1) Y = X * Beta
		# (2) X = Z * Gamma
		# (3) Y = Z * Beta2

		# dim(Y) = NxM
		# dim(X) = dim(Z) = NxK
		# dim(Beta) = dim(Beta2) = KxM
		# dim(Gamma) = KxK

		# Combining the three expressions, we get:
		# Z * Beta2 = Z * Gamma * Beta

		# Thus, we can get the elements of Beta by solving the following equation:
		# Beta2 = Gamma * Beta

		# where Gamma is an upper triangular matrix (look out, it could be singular, which is why
		# we do not apply Beta = inv(Gamma)*Beta2).


		# However, the 'Gamma' argument of this method is actually only a part of the 'Gamma' matrix described
		# above, so we only have to adjust the corresponding part of the 'Beta2' matrix (the rest will be left
		# equal in the 'Beta' matrix).

		Beta = Beta2.copy()
		K = Gamma.shape[0]

		# Get the part of Beta2 that must be 'deorthonormalized'
		if self._glmprocessor_perp_norm_option < 3:
			# All features were orthonormalized/orthogonalized/normalized
			# Process the whole matrix
			dnBeta2 = Beta2.view()
			dnBeta = Beta.view()
		elif self._glmprocessor_perp_norm_option < 6:
			# Only the predictors were orthonormalized/orthogonalized/normalized
			# Only process the last K parameters (the ones belonging to the predictors)
			dnBeta2 = Beta2[-K:].view()
			dnBeta = Beta[-K:].view()
			# Leave the rest as is
			Beta[:-K] = Beta2[:-K].view()
		elif self._glmprocessor_perp_norm_option < 9:
			# Only the correctors were orthonormalized/orthogonalized/normalized
			# Only process the first K parameters (the ones belonging to the correctors)
			dnBeta2 = Beta2[:K].view()
			dnBeta = Beta[:K].view()
			# Leave the rest as is
			Beta[K:] = Beta2[K:].view()
		else:
			# Nothing changed; Beta2 already contains the non-orthogonalized parameters, as does Beta (copied)
			return Beta

		# Work with dnBeta, dnBeta2, and Gamma
		for index in xrange(K):
			j = K - index - 1
			if Gamma[j, j] == 0:
				continue
			dnBeta[j] /= Gamma[j, j]
			for i in xrange(j):
				dnBeta[i] -= dnBeta[j]*Gamma[i, j]

		return Beta


	def __fitter__(self, user_defined_parameters):
		'''Initializes the GLM fitter to be used to process the data.


		'''

		preds = self.predictors.T
		cors = self.correctors.T
		num_features = preds.shape[0] + cors.shape[0] # R + C

		self._glmprocessor_intercept = user_defined_parameters[0]
		self._glmprocessor_perp_norm_option = user_defined_parameters[1]
		self._glmprocessor_degrees = user_defined_parameters[2:(2 + num_features)]
		self._glmprocessor_submodels = user_defined_parameters[(2+num_features):]

		treat_data = GLMProcessor._glmprocessor_perp_norm_options_list[self._glmprocessor_perp_norm_option]
		intercept = GLMProcessor._glmprocessor_intercept_options_list[self._glmprocessor_intercept]

		predictors = []
		correctors = []
		for i in xrange(len(cors)):
			cor = 1
			for _ in xrange(self._glmprocessor_degrees[len(preds) + i]):
				cor *= cors[i]
				correctors.append(cor.copy())
		j = 0
		for i in xrange(len(preds)):
			reg = 1
			for _ in xrange(self._glmprocessor_degrees[i]):
				reg *= preds[i]
				if bool(self._glmprocessor_submodels[j]):
					predictors.append(reg.copy())
				else:
					correctors.append(reg.copy())
				j += 1

		correctors = np.array(correctors).T
		if 0 in correctors.shape:
			correctors = None

		self._glmprocessor_glm = GLM(predictors = np.array(predictors).T, correctors = correctors, intercept = intercept)
		self._glmprocessor_deorthonormalization_matrix = treat_data(self._glmprocessor_glm)
		return self._glmprocessor_glm

	def process(self, x1 = 0, x2 = None, y1 = 0, y2 = None, z1 = 0, z2 = None, mem_usage = None, *args, **kwargs):
		results = super(GLMProcessor, self).process(x1, x2, y1, y2, z1, z2, mem_usage, *args, **kwargs)

		if self._glmprocessor_perp_norm_option >= 6:
			return results

		glm = self._glmprocessor_glm

		ZC = glm.correctors
		ZR = glm.predictors
		Z = np.concatenate((ZC, ZR), axis = 1)

		Beta2R = results.prediction_parameters.reshape(ZR.shape[1], -1)

		GammaR = self._glmprocessor_deorthonormalization_matrix[:, -(ZR.shape[1]):]
		ZGR = Z.dot(GammaR)

		glmInv = GLM(predictors = ZGR.T, intercept = GLM.NoIntercept)
		glmInv.fit(np.identity(ZGR.shape[1]))

		ZGRInv = glmInv.prediction_parameters.T

		BetaR_denorm = ZGRInv.dot(ZR).dot(Beta2R)

		BetaR_denorm = BetaR_denorm.reshape(results.prediction_parameters.shape)
		pparams = np.concatenate((results.prediction_parameters, BetaR_denorm), axis = 0)

#		allparams = np.concatenate((results.correction_parameters, results.prediction_parameters), axis = 0)
#		allparams = allparams.reshape(allparams.shape[0], -1)
#
#		allparams_denorm = self._glmprocessor_compute_original_parameters(self._glmprocessor_deorthonormalization_matrix, allparams)
#		pparams = allparams_denorm[-(results.prediction_parameters.shape[0]):]
#		pparams = pparams.reshape(results.prediction_parameters.shape)
#		pparams = np.concatenate((results.prediction_parameters, pparams), axis = 0)

#		R = results.prediction_parameters.shape[0]
#		pparams = results.prediction_parameters.reshape(R, -1)
#
#		pparams = self._glmprocessor_compute_original_parameters(self._glmprocessor_deorthonormalization_matrix[-R:, -R:], pparams)
#
#		pparams = pparams.reshape(results.prediction_parameters.shape)
#		pparams = np.concatenate((results.prediction_parameters, pparams), axis = 0)

		return self.Results(pparams, results.correction_parameters)

	def __user_defined_parameters__(self, fitter):
		return (self._glmprocessor_intercept, self._glmprocessor_perp_norm_option) + tuple(self._glmprocessor_degrees) + tuple(self._glmprocessor_submodels)

	def __read_user_defined_parameters__(self, predictor_names, corrector_names):
		intercept = GLMProcessor._glmprocessor_intercept_options[super(GLMProcessor, self).__getoneof__(
			GLMProcessor._glmprocessor_intercept_options_names,
			default_value = GLMProcessor._glmprocessor_intercept_options_names[2],
			show_text = 'GLM Processor: How do you want to include the intercept term? (default: ' + GLMProcessor._glmprocessor_intercept_options_names[2] + ')'
		)]

		perp_norm_option = GLMProcessor._glmprocessor_perp_norm_options[super(GLMProcessor, self).__getoneof__(
			GLMProcessor._glmprocessor_perp_norm_options_names,
			default_value = GLMProcessor._glmprocessor_perp_norm_options_names[0],
			show_text = 'GLM Processor: How do you want to treat the features? (default: ' + GLMProcessor._glmprocessor_perp_norm_options_names[0] + ')'
		)]

		degrees = []
		for reg in predictor_names:
			degrees.append(super(GLMProcessor, self).__getint__(
				default_value = 1,
				lower_limit = 1,
				try_ntimes = 3,
				show_text = 'GLM Processor: Please, enter the degree of the feature (predictor) \'' + str(reg) + '\' (or leave blank to set to 1): '
			))
		for cor in corrector_names:
			degrees.append(super(GLMProcessor, self).__getint__(
				default_value = 1,
				try_ntimes = 3,
				show_text = 'GLM Processor: Please, enter the degree of the feature (corrector) \'' + str(cor) + '\' (or leave blank to set to 1): '
			))

		submodels = []
		for i in xrange(len(predictor_names)):
			reg = predictor_names[i]
			if super(GLMProcessor, self).__getyesorno__(default_value = False, show_text = 'GLM Processor: Would you like to analyze a submodel of ' + str(reg) + ' instead of the full model? (Y/N, default N): '):
				# TODO: create a __getmultipleyesorno__ method that allows to check that at least 1 has been selected
				for j in xrange(degrees[i]):
					submodels.append(int(super(GLMProcessor, self).__getyesorno__(default_value = False, show_text = '    Should the power ' + str(j+1) + ' term be considered a predictor? (Y/N, default N): ')))
			else:
				submodels += [int(True)]*degrees[i]

		return (intercept, perp_norm_option) + tuple(degrees) + tuple(submodels)

	def __curve__(self, fitter, predictor, prediction_parameters):

		# Generate all the necessary terms of the predictor

		preds = predictor.T

		predictors = []
		j = 0
		for i in xrange(len(preds)):
			reg = 1
			for _ in xrange(self._glmprocessor_degrees[i]):
				reg *= preds[i]
				if bool(self._glmprocessor_submodels[j]):
					predictors.append(reg.copy())
				j += 1

		# Initialize the glm with such predictors
		glm = GLM(predictors = np.array(predictors).T, intercept = GLMProcessor._glmprocessor_intercept_options_list[self._glmprocessor_intercept])

		# Get the prediction parameters for the original features matrix
		if self._glmprocessor_perp_norm_option < 6:
			Kx2 = prediction_parameters.shape[0]
			pparams = prediction_parameters[(Kx2/2):]
		else:
			pparams = prediction_parameters

		# Call the normal function with such parameters
		return glm.predict(prediction_parameters = pparams)

	@staticmethod
	def evaluate_fit(evaluation_function, correction_processor, correction_parameters, prediction_processor, prediction_parameters, x1 = 0, x2 = None, y1 = 0, y2 = None, z1 = 0, z2 = None, origx = 0, origy = 0, origz = 0, gm_threshold = None, filter_nans = True, default_value = 0.0, mem_usage = None, *args, **kwargs):

		if prediction_processor.predictors.shape[1] != prediction_parameters.shape[0]:
			# Both, original and orthonormalized versions are present in the matrix
			# Use only the orthonormalized elements
			Kx2 = prediction_parameters.shape[0]
			prediction_parameters = prediction_parameters[:(Kx2/2)]

		return super(GLMProcessor, GLMProcessor).evaluate_fit (
			evaluation_function = evaluation_function,
			correction_processor = correction_processor,
			correction_parameters = correction_parameters,
			prediction_processor = prediction_processor,
			prediction_parameters = prediction_parameters,
			x1 = x1, x2 = x2, y1 = y1, y2 = y2, z1 = z1, z2 = z2,
			origx = origx, origy = origy, origz = origz,
			gm_threshold = gm_threshold,
			filter_nans = filter_nans,
			default_value = default_value,
			mem_usage = mem_usage,
			*args, **kwargs
		)


GLMProcessor._glmprocessor_perp_norm_options = {GLMProcessor._glmprocessor_perp_norm_options_names[i] : i for i in xrange(len(GLMProcessor._glmprocessor_perp_norm_options_names))}
GLMProcessor._glmprocessor_intercept_options = {GLMProcessor._glmprocessor_intercept_options_names[i] : i for i in xrange(len(GLMProcessor._glmprocessor_intercept_options_names))}


class PolyGLMProcessor(Processor):
	_pglmprocessor_perp_norm_options_names = [
		'Orthonormalize all',
		'Orthogonalize all',
		'Normalize all',
		'Orthonormalize predictors',
		'Orthogonalize predictors',
		'Normalize predictors',
		'Orthonormalize correctors',
		'Orthogonalize correctors',
		'Normalize correctors',
		'Use correctors and predictors as they are'
	]

	_pglmprocessor_perp_norm_options_list = [
		PGLM.orthonormalize_all,
		PGLM.orthogonalize_all,
		PGLM.normalize_all,
		PGLM.orthonormalize_predictors,
		PGLM.orthogonalize_predictors,
		PGLM.normalize_predictors,
		PGLM.orthonormalize_correctors,
		PGLM.orthogonalize_correctors,
		PGLM.normalize_correctors,
		lambda *args, **kwargs: np.zeros((0, 0))
	]

	_pglmprocessor_intercept_options_names = [
		'Do not include the intercept term',
		'As a corrector',
		'As a predictor'
	]

	_pglmprocessor_intercept_options_list = [
		PGLM.NoIntercept,
		PGLM.CorrectionIntercept,
		PGLM.PredictionIntercept
	]

	def _pglmprocessor_compute_original_parameters(self, Gamma, Beta2):
		'''Given an upper triangular matrix Gamma, and an arbitrary matrix Beta2, computes Beta such
			that Beta2 = Gamma * Beta.
			Notice that if Beta2 is the np.identity matrix, then Beta is the right-pseudoinverse of Gamma.
		'''

		# Gamma is the deorthogonalization (upper triangular) matrix
		# Beta2 is the matrix with the optimal parameters of the orthonormalized design matrix
		# Beta is the matrix with the optimal parameters of the original design matrix

		# The relationships between the different matrices of the system are described below:
		# (1) Y = X * Beta
		# (2) X = Z * Gamma
		# (3) Y = Z * Beta2

		# dim(Y) = NxM
		# dim(X) = dim(Z) = NxK
		# dim(Beta) = dim(Beta2) = KxM
		# dim(Gamma) = KxK

		# Combining the three expressions, we get:
		# Z * Beta2 = Z * Gamma * Beta

		# Thus, we can get the elements of Beta by solving the following equation:
		# Beta2 = Gamma * Beta

		# where Gamma is an upper triangular matrix (look out, it could be singular, which is why
		# we do not apply Beta = inv(Gamma)*Beta2).


		# However, the 'Gamma' argument of this method is actually only a part of the 'Gamma' matrix described
		# above, so we only have to adjust the corresponding part of the 'Beta2' matrix (the rest will be left
		# equal in the 'Beta' matrix).

		Beta = Beta2.copy()
		K = Gamma.shape[0]

		# Get the part of Beta2 that must be 'deorthonormalized'
		if self._pglmprocessor_perp_norm_option < 3:
			# All features were orthonormalized/orthogonalized/normalized
			# Process the whole matrix
			dnBeta2 = Beta2.view()
			dnBeta = Beta.view()
		elif self._pglmprocessor_perp_norm_option < 6:
			# Only the predictors were orthonormalized/orthogonalized/normalized
			# Only process the last K parameters (the ones belonging to the predictors)
			dnBeta2 = Beta2[-K:].view()
			dnBeta = Beta[-K:].view()
			# Leave the rest as is
			Beta[:-K] = Beta2[:-K].view()
		elif self._pglmprocessor_perp_norm_option < 9:
			# Only the correctors were orthonormalized/orthogonalized/normalized
			# Only process the first K parameters (the ones belonging to the correctors)
			dnBeta2 = Beta2[:K].view()
			dnBeta = Beta[:K].view()
			# Leave the rest as is
			Beta[K:] = Beta2[K:].view()
		else:
			# Nothing changed; Beta2 already contains the non-orthogonalized parameters, as does Beta (copied)
			return Beta

		# Work with dnBeta, dnBeta2, and Gamma
		for index in xrange(K):
			j = K - index - 1
			if Gamma[j, j] == 0:
				continue
			dnBeta[j] /= Gamma[j, j]
			for i in xrange(j):
				dnBeta[i] -= dnBeta[j]*Gamma[i, j]

		return Beta

	def __fitter__(self, user_defined_parameters):
		'''Initializes the PolyGLM fitter to be used to process the data.


		'''
		self._pglmprocessor_intercept = user_defined_parameters[0]
		self._pglmprocessor_perp_norm_option = user_defined_parameters[1]
		self._pglmprocessor_degrees = user_defined_parameters[2:]

		treat_data = PolyGLMProcessor._pglmprocessor_perp_norm_options_list[self._pglmprocessor_perp_norm_option]
		intercept = PolyGLMProcessor._pglmprocessor_intercept_options_list[self._pglmprocessor_intercept]

		num_preds = self.predictors.shape[1]
		features = np.zeros((self.predictors.shape[0], num_preds + self.correctors.shape[1]))
		features[:, :num_preds] = self.predictors
		features[:, num_preds:] = self.correctors

		self._pglmprocessor_pglm = PGLM(features = features, predictors = xrange(num_preds), degrees = self._pglmprocessor_degrees, intercept = intercept)
		self._pglmprocessor_deorthonormalization_matrix = treat_data(self._pglmprocessor_pglm)
		return self._pglmprocessor_pglm

	def process(self, x1 = 0, x2 = None, y1 = 0, y2 = None, z1 = 0, z2 = None, mem_usage = None, *args, **kwargs):
		results = super(PolyGLMProcessor, self).process(x1, x2, y1, y2, z1, z2, mem_usage, *args, **kwargs)

		if self._pglmprocessor_perp_norm_option >= 6:
			return results

		pglm = self._pglmprocessor_pglm

		ZC = pglm.correctors
		ZR = pglm.predictors
		Z = np.concatenate((ZC, ZR), axis = 1)

		Beta2R = results.prediction_parameters.reshape(ZR.shape[1], -1)

		GammaR = self._pglmprocessor_deorthonormalization_matrix[:, -(ZR.shape[1]):]
		ZGR = Z.dot(GammaR)

		glmInv = GLM(predictors = ZGR.T, intercept = GLM.NoIntercept)
		glmInv.fit(np.identity(ZGR.shape[1]))

		ZGRInv = glmInv.prediction_parameters.T

		BetaR_denorm = ZGRInv.dot(ZR).dot(Beta2R)

		BetaR_denorm = BetaR_denorm.reshape(results.prediction_parameters.shape)
		pparams = np.concatenate((results.prediction_parameters, BetaR_denorm), axis = 0)

#		allparams = np.concatenate((results.correction_parameters, results.prediction_parameters), axis = 0)
#		allparams = allparams.reshape(allparams.shape[0], -1)
#
#		allparams_denorm = self._pglmprocessor_compute_original_parameters(self._pglmprocessor_deorthonormalization_matrix, allparams)#
#		pparams = allparams_denorm[-(results.prediction_parameters.shape[0]):]
#		pparams = pparams.reshape(results.prediction_parameters.shape)
#		pparams = np.concatenate((results.prediction_parameters, pparams), axis = 0)

#		R = results.prediction_parameters.shape[0]
#		pparams = results.prediction_parameters.reshape(R, -1)
#
#		pparams = self._pglmprocessor_compute_original_parameters(self._pglmprocessor_deorthonormalization_matrix[-R:, -R:], pparams)
#
#		pparams = pparams.reshape(results.prediction_parameters.shape)
#		pparams = np.concatenate((results.prediction_parameters, pparams), axis = 0)

		return self.Results(pparams, results.correction_parameters)

	def __user_defined_parameters__(self, fitter):
		return (self._pglmprocessor_intercept, self._pglmprocessor_perp_norm_option) + tuple(self._pglmprocessor_degrees)

	def __read_user_defined_parameters__(self, predictor_names, corrector_names):
		intercept = GLMProcessor._pglmprocessor_intercept_options[super(PolyGLMProcessor, self).__getoneof__(
			PolyGLMProcessor._pglmprocessor_intercept_options_names,
			default_value = PolyGLMProcessor._pglmprocessor_intercept_options_names[2],
			show_text = 'PolyGLM Processor: How do you want to include the intercept term? (default: ' + PolyGLMProcessor._pglmprocessor_intercept_options_names[2] + ')'
		)]

		perp_norm_option = PolyGLMProcessor._pglmprocessor_perp_norm_options[super(PolyGLMProcessor, self).__getoneof__(
			PolyGLMProcessor._pglmprocessor_perp_norm_options_names,
			default_value = PolyGLMProcessor._pglmprocessor_perp_norm_options_names[0],
			show_text = 'PolyGLM Processor: How do you want to treat the features? (default: ' + PolyGLMProcessor._pglmprocessor_perp_norm_options_names[0] + ')'
		)]

		degrees = []
		for reg in predictor_names:
			degrees.append(super(PolyGLMProcessor, self).__getint__(
				default_value = 1,
				lower_limit = 1,
				try_ntimes = 3,
				show_text = 'PolyGLM Processor: Please, enter the degree of the feature (predictor) \'' + str(reg) + '\' (or leave blank to set to 1): '
			))
		for cor in corrector_names:
			degrees.append(super(PolyGLMProcessor, self).__getint__(
				default_value = 1,
				try_ntimes = 3,
				show_text = 'PolyGLM Processor: Please, enter the degree of the feature (corrector) \'' + str(cor) + '\' (or leave blank to set to 1): '
			))

		return (intercept, perp_norm_option) + tuple(degrees)

	def __curve__(self, fitter, predictor, prediction_parameters):
		pglm = PGLM(predictor, degrees = self._pglmprocessor_degrees[:1], intercept = PolyGLMProcessor._pglmprocessor_intercept_options_list[self._pglmprocessor_intercept])
		# PolyGLMProcessor._pglmprocessor_perp_norm_options_list[self._pglmprocessor_perp_norm_option](pglm)

		# Get the prediction parameters for the original features matrix
		if self._pglmprocessor_perp_norm_option < 6:
			Kx2 = prediction_parameters.shape[0]
			pparams = prediction_parameters[(Kx2/2):]
		else:
			pparams = prediction_parameters

		# Call the normal function with such parameters
		return pglm.predict(prediction_parameters = pparams)

	@staticmethod
	def evaluate_fit(evaluation_function, correction_processor, correction_parameters, prediction_processor, prediction_parameters, x1 = 0, x2 = None, y1 = 0, y2 = None, z1 = 0, z2 = None, origx = 0, origy = 0, origz = 0, gm_threshold = None, filter_nans = True, default_value = 0.0, mem_usage = None, *args, **kwargs):

		if prediction_processor.predictors.shape[1] != prediction_parameters.shape[0]:
			# Both, original and orthonormalized versions are present in the matrix
			# Use only the orthonormalized elements
			Kx2 = prediction_parameters.shape[0]
			prediction_parameters = prediction_parameters[:(Kx2/2)]

		return super(PolyGLMProcessor, PolyGLMProcessor).evaluate_fit (
			evaluation_function = evaluation_function,
			correction_processor = correction_processor,
			correction_parameters = correction_parameters,
			prediction_processor = prediction_processor,
			prediction_parameters = prediction_parameters,
			x1 = x1, x2 = x2, y1 = y1, y2 = y2, z1 = z1, z2 = z2,
			origx = origx, origy = origy, origz = origz,
			gm_threshold = gm_threshold,
			filter_nans = filter_nans,
			default_value = default_value,
			mem_usage = mem_usage,
			*args, **kwargs
		)

PolyGLMProcessor._pglmprocessor_perp_norm_options = {PolyGLMProcessor._pglmprocessor_perp_norm_options_names[i] : i for i in xrange(len(PolyGLMProcessor._pglmprocessor_perp_norm_options_names))}
PolyGLMProcessor._pglmprocessor_intercept_options = {PolyGLMProcessor._pglmprocessor_intercept_options_names[i] : i for i in xrange(len(PolyGLMProcessor._pglmprocessor_intercept_options_names))}
