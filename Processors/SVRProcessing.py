"""
Processor for Support Vector Regression fitters
"""
from Processors.Processing import Processor
from numpy import zeros

from Fitters.SVR import PolySVR


class PolySVRProcessor(Processor):
    """
    Processor for Polynomic Support Vector Regression
    """
    _psvrprocessor_perp_norm_options_names = [
		'Orthonormalize all',
		'Orthogonalize all',
		'Normalize all',
		'Orthonormalize predictor',
		'Orthogonalize predictor',
		'Normalize predictor',
		'Orthonormalize correctors',
		'Orthogonalize correctors',
		'Normalize correctors',
		'Use correctors and predictor as they are'
	]

    _psvrprocessor_perp_norm_options_list = [
        PolySVR.orthonormalize_all,
        PolySVR.orthogonalize_all,
        PolySVR.normalize_all,
        PolySVR.orthonormalize_predictors,
        PolySVR.orthogonalize_predictors,
        PolySVR.normalize_predictors,
        PolySVR.orthonormalize_correctors,
        PolySVR.orthogonalize_correctors,
        PolySVR.normalize_correctors,
        lambda *args, **kwargs: zeros((0, 0))
    ]

    _psvrprocessor_intercept_options_names = [
		'Do not include the intercept term',
		'As a corrector',
		'As a predictor'
	]

    _psvrprocessor_intercept_options_list = [
        PolySVR.NoIntercept,
        PolySVR.CorrectionIntercept,
        PolySVR.PredictionIntercept
    ]

    def __fitter__(self, user_defined_parameters):
        """

        Parameters
        ----------
        user_defined_parameters

        Returns
        -------

        """
        self._psvrprocessor_intercept = user_defined_parameters[0]
        self._psvrprocessor_perp_norm_option = user_defined_parameters[1]
        self._psvrprocessor_C = user_defined_parameters[2]
        self._psvrprocessor_epsilon = user_defined_parameters[3]
        self._psvrprocessor_degrees = user_defined_parameters[4:]

        # Orthonormalize/Orthogonalize/Do nothing options
        treat_data = PolySVRProcessor._psvrprocessor_perp_norm_options_list[self._psvrprocessor_perp_norm_option]
        # Intercept option
        intercept = PolySVRProcessor._psvrprocessor_intercept_options_list[self._psvrprocessor_intercept]

        # Construct data matrix from correctors and predictor
        num_regs = self.predictors.shape[1]
        num_correc = self.correctors.shape[1]
        features = zeros((self.predictors.shape[0], num_regs + num_correc))
        features[:, :num_regs] = self.predictors
        features[:, num_regs:] = self.correctors

        # Instantiate a PolySVR
        psvr = PolySVR(features=features, predictors=range(num_regs), degrees=self._psvrprocessor_degrees, intercept=intercept)
        treat_data(psvr)
        return psvr

    def __user_defined_parameters__(self, fitter):
        """

        Parameters
        ----------
        fitter

        Returns
        -------

        """
        user_params = (self._psvrprocessor_intercept, self._psvrprocessor_perp_norm_option)
        user_params += (self._psvrprocessor_C, self._psvrprocessor_epsilon)
        user_params += tuple(self._psvrprocessor_degrees)
        return user_params

    def __read_user_defined_parameters__(self, regressor_names, corrector_names):
        """

        Parameters
        ----------
        regressor_names
        corrector_names

        Returns
        -------

        """
        # Intercept term
        intercept = PolySVRProcessor._psvrprocessor_intercept_options[super(PolySVRProcessor, self).__getoneof__(
			PolySVRProcessor._psvrprocessor_intercept_options_names,
			default_value = PolySVRProcessor._psvrprocessor_intercept_options_names[2],
			show_text = 'PolySVR Processor: How do you want to include the intercept term? (default: ' + PolySVRProcessor._psvrprocessor_intercept_options_names[2] + ')'
		)]

        # Treat data option
        perp_norm_option = PolySVRProcessor._psvrprocessor_perp_norm_options[super(PolySVRProcessor, self).__getoneof__(
			PolySVRProcessor._psvrprocessor_perp_norm_options_names,
			default_value = 'Orthonormalize all',
			show_text = 'PolySVR Processor: How do you want to treat the features? (default: Orthonormalize all)'
		)]

        # C regularization parameter
        C = super(PolySVRProcessor, self).__getfloat__(
            default_value = 100.0,
            try_ntimes= 3,
            lower_limit=0.0,
            show_text='PolySVR Processor: Please, enter the regularization parameter C (default: 100.0)'
        )

        # epsilon regularization parameter
        epsilon = super(PolySVRProcessor, self).__getfloat__(
            default_value = 0.01,
            try_ntimes= 3,
            lower_limit=0.0,
            show_text='PolySVR Processor: Please, enter the epsilon-tube within which no penalty is associated in the training loss function (default: 0.01)'
        )

        # Polynomial degrees
        degrees = []
        for reg in regressor_names:
            degrees.append(super(PolySVRProcessor, self).__getint__(
                default_value = 1,
                lower_limit = 1,
                try_ntimes = 3,
                show_text = 'PolySVR Processor: Please, enter the degree of the feature (predictor) \'' + str(reg) + '\' (or leave blank to set to 1): '
            ))
        for cor in corrector_names:
            degrees.append(super(PolySVRProcessor, self).__getint__(
                default_value = 1,
                try_ntimes = 3,
                show_text = 'PolySVR Processor: Please, enter the degree of the feature (corrector) \'' + str(cor) + '\' (or leave blank to set to 1): '
            ))

        return (intercept, perp_norm_option, C, epsilon) + tuple(degrees)

    def __curve__(self, fitter, predictor, prediction_parameters):
        """
        Returns the curve values when predicting the y values associated to the x values (regressor)
        using the regression_parameters
        Parameters
        ----------
        fitter
        regressor
        regression_parameters

        Returns
        -------

        """
        psvr = PolySVR(predictor, degrees = self._psvrprocessor_degrees[:1], intercept = self._psvrprocessor_intercept)
        PolySVRProcessor._psvrprocessor_perp_norm_options_list[self._psvrprocessor_perp_norm_option](psvr)
        return psvr.predict(prediction_parameters = prediction_parameters)

    def process(self, x1 = 0, x2 = None, y1 = 0, y2 = None, z1 = 0, z2 = None,
                mem_usage = None, evaluation_kwargs = {}, *args, **kwargs):
        """
        Processes the given voxel coordinates (x1:x2, y1:y2, z1:z2)
        Parameters
        ----------
        self
        x1
        x2
        y1
        y2
        z1
        z2
        mem_usage
        evaluation_kwargs
        args
        kwargs

        Returns
        -------

        """
        # Call parent function process with additional parameters obtained through __read_user_defined_parameters__
        return super(PolySVRProcessor, self).process(x1, x2, y1, y2, z1, z2, mem_usage, evaluation_kwargs, C=self._psvrprocessor_C, epsilon=self._psvrprocessor_epsilon, *args, **kwargs)

PolySVRProcessor._psvrprocessor_perp_norm_options = {PolySVRProcessor._psvrprocessor_perp_norm_options_names[i] : i for i in xrange(len(PolySVRProcessor._psvrprocessor_perp_norm_options_names))}
PolySVRProcessor._psvrprocessor_intercept_options = {PolySVRProcessor._psvrprocessor_intercept_options_names[i] : i for i in xrange(len(PolySVRProcessor._psvrprocessor_intercept_options_names))}


