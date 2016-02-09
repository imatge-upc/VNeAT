from Processing import Processor
from GLM import GLM, PolyGLM as PGLM
from numpy import zeros

class GLMProcessor(Processor):
	_glmprocessor_perp_norm_options = {
		'Orthonormalize all': 0,
		'Orthogonalize all': 1,
		'Normalize all': 2,
		'Orthonormalize regressors': 3,
		'Orthogonalize regressors': 4,
		'Normalize regressors': 5,
		'Orthonormalize correctors': 6,
		'Orthogonalize correctors': 7,
		'Normalize correctors': 8,
		'Use correctors and regressors as they are': 9
	}

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
		lambda *args, **kwargs: None
	]

	def __fitter__(self, user_defined_parameters):
		'''Initializes the GLM fitter to be used to process the data.


		'''
		self._glmprocessor_homogeneous = user_defined_parameters[0]
		self._glmprocessor_perp_norm_option = user_defined_parameters[1]

		treat_data = GLMProcessor._glmprocessor_perp_norm_options_list[self._glmprocessor_perp_norm_option]

		glm = GLM(regressors = self.regressors, correctors = self.correctors, homogeneous = self._glmprocessor_homogeneous)
		treat_data(glm)
		return glm

	def __user_defined_parameters__(self, fitter):
		return (self._glmprocessor_homogeneous, self._glmprocessor_perp_norm_option)

	def __read_user_defined_parameters__(self, regressor_names, corrector_names):
		if super(GLMProcessor, self).__getyesorno__(default_value = True, show_text = 'GLM Processor: Do you want to include the homogeneous term? (Y/N, default Y): '):
			homogeneous = 1
		else:
			homogeneous = 0
		
		perp_norm_option = GLMProcessor._glmprocessor_perp_norm_options[super(GLMProcessor, self).__getoneof__(
			GLMProcessor._glmprocessor_perp_norm_options,
			default_value = 0,
			show_text = 'GLM Processor: How do you want to treat the features? (default: Orthonormalize all)'
		)]

		return (homogeneous, perp_norm_option)



class PolyGLMProcessor(Processor):
	_pglmprocessor_perp_norm_options = {
		'Orthonormalize all': 0,
		'Orthogonalize all': 1,
		'Normalize all': 2,
		'Orthonormalize regressors': 3,
		'Orthogonalize regressors': 4,
		'Normalize regressors': 5,
		'Orthonormalize correctors': 6,
		'Orthogonalize correctors': 7,
		'Normalize correctors': 8,
		'Use correctors and regressors as they are': 9
	}

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
		lambda *args, **kwargs: None
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

		pglm = PGLM(features = features, regressors = range(num_regs), degrees = self._pglmprocessor_degrees, homogeneous = self._pglmprocessor_homogeneous)
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
			PolyGLMProcessor._pglmprocessor_perp_norm_options,
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






