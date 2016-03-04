from Processing import Processor
from GAM import GAM, PolynomialSmoother, SplinesSmoother, SmootherSet
from numpy import zeros


class GAMProcessor(Processor):
	_gamprocessor_perp_norm_options_names = [
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

	_gamprocessor_perp_norm_options_list = [
		GAM.orthonormalize_all,
		GAM.orthogonalize_all,
		GAM.normalize_all,
		GAM.orthonormalize_regressors,
		GAM.orthogonalize_regressors,
		GAM.normalize_regressors,
		GAM.orthonormalize_correctors,
		GAM.orthogonalize_correctors,
		GAM.normalize_correctors,
		lambda *args, **kwargs: None
	]


	def __fitter__(self, user_defined_parameters):
		'''Initializes the PolyGLM fitter to be used to process the data.


		'''
		TYPE_SMOOTHER=[PolynomialSmoother,SplinesSmoother]
		self._gamprocessor_perp_norm_option = user_defined_parameters[0]
		self._gamprocessor_smoother_parameters = user_defined_parameters[1]

		sm_index = 0
		corrector_smoothers=SmootherSet()
		regressor_smoothers=SmootherSet()
		for corr in self.correctors.T:
			smoother_function=TYPE_SMOOTHER[int(self._gamprocessor_smoother_parameters[sm_index])](corr)
			sm_index += 1
			n_param = self._gamprocessor_smoother_parameters[sm_index]
			sm_index += 1
			smoother_function.set_parameters(self._gamprocessor_smoother_parameters[sm_index:sm_index+n_param])
			sm_index += n_param
			corrector_smoothers.extend(smoother_function)
		for reg in self.regressors.T:
			smoother_function=TYPE_SMOOTHER[int(self._gamprocessor_smoother_parameters[sm_index])](reg)
			sm_index += 1
			n_param = self._gamprocessor_smoother_parameters[sm_index]
			sm_index += 1
			smoother_function.set_parameters(self._gamprocessor_smoother_parameters[sm_index:sm_index+n_param])
			sm_index += n_param
			regressor_smoothers.extend(smoother_function)

		treat_data = GAMProcessor._gamprocessor_perp_norm_options_list[self._gamprocessor_perp_norm_option]

		gam = GAM(corrector_smoothers=corrector_smoothers, regressor_smoothers=regressor_smoothers)

		treat_data(gam)
		return gam

	def __user_defined_parameters__(self, fitter):
		return (self._gamprocessor_perp_norm_option,self._gamprocessor_smoother_parameters)

	def __read_user_defined_parameters__(self, regressor_names, corrector_names):

		perp_norm_option = GAMProcessor._gamprocessor_perp_norm_options[super(GAMProcessor, self).__getoneof__(
			GAMProcessor._gamprocessor_perp_norm_options_names,
			default_value = 'Orthonormalize all',
			show_text = 'PolyGLM Processor: How do you want to treat the features? (default: Orthonormalize all)'
		)]

		smoothing_functions = []
		print('')
		for cor in corrector_names:
			smoother_type = super(GAMProcessor, self).__getint__(
				default_value = 1,
				try_ntimes = 3,
				show_text = 'GAM Processor: Please, enter the smoothing function of the feature (corrector) \'' + str(cor)
							+ '\' (0: Polynomial Smoother, 1: Splines Smoother): ')
			smoothing_functions.append(smoother_type)

			if smoother_type == 0:
				smoothing_functions.append(1)
				smoothing_functions.append(super(GAMProcessor, self).__getint__(
					default_value = 1,
					try_ntimes = 3,
					show_text = 'GAM Processor: You have selected Polynomial smoother. Please, enter the degree of the polynomial '
								'(or leave blank to set to 3) '
				))
			elif smoother_type == 1:
				smoothing_functions.append(2)
				smoothing_functions.append(super(GAMProcessor, self).__getint__(
					default_value = 1,
					try_ntimes = 3,
					show_text = 'GAM Processor: You have selected Splines smoother. Please, enter the degree of the splines '
								'(or leave blank to set to 3) '
				))
				smoothing_functions.append(super(GAMProcessor, self).__getint__(
					default_value = 1,
					try_ntimes = 3,
					show_text = 'GAM Processor: You have selected Splines smoother. Please, enter the smoothing factor of the spline'
								'(or leave it blank to set it to default: len(observations) '
				))
		for reg in regressor_names:
			smoother_type = super(GAMProcessor, self).__getint__(
				default_value = 1,
				try_ntimes = 3,
				show_text = 'GAM Processor: Please, enter the smoothing function of the feature (regressor) \'' + str(reg)
							+ '\' (0: Polynomial Smoother, 1: Splines Smoother): ')
			smoothing_functions.append(smoother_type)

			if smoother_type == 0:
				smoothing_functions.append(1)
				smoothing_functions.append(super(GAMProcessor, self).__getint__(
					default_value = 1,
					try_ntimes = 3,
					show_text = 'GAM Processor: You have selected Polynomial smoother. Please, enter the degree of the polynomial '
								'(or leave blank to set to 3) '
				))
			elif smoother_type == 1:
				smoothing_functions.append(2)
				smoothing_functions.append(super(GAMProcessor, self).__getint__(
					default_value = 1,
					try_ntimes = 3,
					show_text = 'GAM Processor: You have selected Splines smoother. Please, enter the degree of the splines '
								'(or leave blank to set to 3) '
				))
				smoothing_functions.append(super(GAMProcessor, self).__getint__(
					default_value = 1,
					try_ntimes = 3,
					show_text = 'GAM Processor: You have selected Splines smoother. Please, enter the smoothing factor of the splines'
								'(or leave it blank to set it to default: len(observations) '
				))


		return (perp_norm_option,smoothing_functions)

	def __curve__(self, fitter, regressors, regression_parameters):
		gam = GAM()
		GAMProcessor._gamprocessor_perp_norm_options_list[self._gamprocessor_perp_norm_option](gam)
		return gam.predict(regressors=regressors, regression_parameters = regression_parameters)


GAMProcessor._gamprocessor_perp_norm_options = {GAMProcessor._gamprocessor_perp_norm_options_names[i] : i for i in xrange(len(GAMProcessor._gamprocessor_perp_norm_options_names))}

