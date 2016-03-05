from Processing import Processor
from GLM import GLM, PolyGLM as PGLM
from numpy import zeros, array as nparray

class GLMProcessor(Processor):
	_glmprocessor_perp_norm_options_names = [
		'Orthonormalize all',
		'Orthogonalize all',
		'Normalize all',
		'Orthonormalize regressors',
		'Orthogonalize regressors',
		'Normalize regressors',
		'Orthonormalize correctors',
		'Orthogonalize correctors',
		'Normalize correctors',
		'Use correctors and regressors as they are'
	]

	_glmprocessor_perp_norm_options_list = [
		GLM.orthonormalize_all,
		GLM.orthogonalize_all,
		GLM.normalize_all,
		GLM.orthonormalize_regressors,
		GLM.orthogonalize_regressors,
		GLM.normalize_regressors,
		GLM.orthonormalize_correctors,
		GLM.orthogonalize_correctors,
		GLM.normalize_correctors,
		lambda *args, **kwargs: zeros((0, 0))
	]

	def __fitter__(self, user_defined_parameters):
		'''Initializes the GLM fitter to be used to process the data.


		'''
		regs = self.regressors.T
		cors = self.correctors.T
		num_features = regs.shape[0] + cors.shape[0] # R + C

		self._glmprocessor_homogeneous = user_defined_parameters[0]
		self._glmprocessor_perp_norm_option = user_defined_parameters[1]
		self._glmprocessor_degrees = user_defined_parameters[2:(2 + num_features)]
		self._glmprocessor_submodels = user_defined_parameters[(2+num_features):]

		treat_data = GLMProcessor._glmprocessor_perp_norm_options_list[self._glmprocessor_perp_norm_option]

		regressors = []
		correctors = []
		for i in xrange(len(cors)):
			cor = 1
			for _ in xrange(self._glmprocessor_degrees[len(regs) + i]):
				cor *= cors[i]
				correctors.append(cor.copy())
		j = 0
		for i in xrange(len(regs)):
			reg = 1
			for _ in xrange(self._glmprocessor_degrees[i]):
				reg *= regs[i]
				if bool(self._glmprocessor_submodels[j]):
					regressors.append(reg.copy())
				else:
					correctors.append(reg.copy())
				j += 1

		correctors = nparray(correctors).T
		if 0 in correctors.shape:
			correctors = None

		glm = GLM(regressors = nparray(regressors).T, correctors = correctors, homogeneous = bool(self._glmprocessor_homogeneous))
		self._glmprocessor_orthonormalization_matrix = treat_data(glm)
		return glm

	def __user_defined_parameters__(self, fitter):
		return (self._glmprocessor_homogeneous, self._glmprocessor_perp_norm_option) + tuple(self._glmprocessor_degrees) + tuple(self._glmprocessor_submodels)

	def __read_user_defined_parameters__(self, regressor_names, corrector_names):
		homogeneous = int(super(GLMProcessor, self).__getyesorno__(default_value = True, show_text = 'GLM Processor: Do you want to include the homogeneous term? (Y/N, default Y): '))

		perp_norm_option = GLMProcessor._glmprocessor_perp_norm_options[super(GLMProcessor, self).__getoneof__(
			GLMProcessor._glmprocessor_perp_norm_options_names,
			default_value = 'Orthonormalize all',
			show_text = 'GLM Processor: How do you want to treat the features? (default: Orthonormalize all)'
		)]

		degrees = []
		for reg in regressor_names:
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
		for i in xrange(len(regressor_names)):
			reg = regressor_names[i]
			if super(GLMProcessor, self).__getyesorno__(default_value = False, show_text = 'GLM Processor: Would you like to analyze a submodel of ' + str(reg) + ' instead of the full model? (Y/N, default N): '):
				# TODO: create a __getmultipleyesorno__ method that allows to check that at least 1 has been selected
				for j in xrange(degrees[i]):
					submodels.append(int(super(GLMProcessor, self).__getyesorno__(default_value = False, show_text = '    Should the power ' + str(j+1) + ' term be considered a predictor? (Y/N, default N): ')))
			else:
				submodels += [int(True)]*degrees[i]

		return (homogeneous, perp_norm_option) + tuple(degrees) + tuple(submodels)

	def __curve__(self, fitter, regressor, regression_parameters):
		#TODO
		super(GLMProcessor, self).__curve__(fitter, regressor, regression_parameters)

GLMProcessor._glmprocessor_perp_norm_options = {GLMProcessor._glmprocessor_perp_norm_options_names[i] : i for i in xrange(len(GLMProcessor._glmprocessor_perp_norm_options_names))}



class PolyGLMProcessor(Processor):
	_pglmprocessor_perp_norm_options_names = [
		'Orthonormalize all',
		'Orthogonalize all',
		'Normalize all',
		'Orthonormalize regressors',
		'Orthogonalize regressors',
		'Normalize regressors',
		'Orthonormalize correctors',
		'Orthogonalize correctors',
		'Normalize correctors',
		'Use correctors and regressors as they are'
	]

	_pglmprocessor_perp_norm_options_list = [
		PGLM.orthonormalize_all,
		PGLM.orthogonalize_all,
		PGLM.normalize_all,
		PGLM.orthonormalize_regressors,
		PGLM.orthogonalize_regressors,
		PGLM.normalize_regressors,
		PGLM.orthonormalize_correctors,
		PGLM.orthogonalize_correctors,
		PGLM.normalize_correctors,
		lambda *args, **kwargs: zeros((0, 0))
	]


	def __fitter__(self, user_defined_parameters):
		'''Initializes the PolyGLM fitter to be used to process the data.


		'''
		self._pglmprocessor_homogeneous = user_defined_parameters[0]
		self._pglmprocessor_perp_norm_option = user_defined_parameters[1]
		self._pglmprocessor_degrees = user_defined_parameters[2:]

		treat_data = PolyGLMProcessor._pglmprocessor_perp_norm_options_list[self._pglmprocessor_perp_norm_option]

		num_regs = self.regressors.shape[1]
		features = zeros((self.regressors.shape[0], num_regs + self.correctors.shape[1]))
		features[:, :num_regs] = self.regressors
		features[:, num_regs:] = self.correctors

		pglm = PGLM(features = features, regressors = xrange(num_regs), degrees = self._pglmprocessor_degrees, homogeneous = self._pglmprocessor_homogeneous)
		treat_data(pglm)
		return pglm

	def __user_defined_parameters__(self, fitter):
		return (self._pglmprocessor_homogeneous, self._pglmprocessor_perp_norm_option) + tuple(self._pglmprocessor_degrees)

	def __read_user_defined_parameters__(self, regressor_names, corrector_names):
		if super(PolyGLMProcessor, self).__getyesorno__(default_value = True, show_text = 'PolyGLM Processor: Do you want to include the homogeneous term? (Y/N, default Y): '):
			homogeneous = 1
		else:
			homogeneous = 0
		
		perp_norm_option = PolyGLMProcessor._pglmprocessor_perp_norm_options[super(PolyGLMProcessor, self).__getoneof__(
			PolyGLMProcessor._pglmprocessor_perp_norm_options_names,
			default_value = 'Orthonormalize all',
			show_text = 'PolyGLM Processor: How do you want to treat the features? (default: Orthonormalize all)'
		)]

		degrees = []
		for reg in regressor_names:
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

		return (homogeneous, perp_norm_option) + tuple(degrees)

	def __curve__(self, fitter, regressor, regression_parameters):
		pglm = PGLM(regressor, degrees = self._pglmprocessor_degrees[:1], homogeneous = False)
		PolyGLMProcessor._pglmprocessor_perp_norm_options_list[self._pglmprocessor_perp_norm_option](pglm)
		return pglm.predict(regression_parameters = regression_parameters)

PolyGLMProcessor._pglmprocessor_perp_norm_options = {PolyGLMProcessor._pglmprocessor_perp_norm_options_names[i] : i for i in xrange(len(PolyGLMProcessor._pglmprocessor_perp_norm_options_names))}

