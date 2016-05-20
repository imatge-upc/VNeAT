"""
Processor for Support Vector Regression fitters
"""
from Processors.Processing import Processor
from numpy import zeros

from Fitters.SVR import PolySVR, GaussianSVR


class PolySVRProcessor(Processor):
    """
    Processor for Polynomic Support Vector Regression
    """
    _psvrprocessor_perp_norm_options_names = [
        'Normalize all',
        'Normalize predictor',
        'Normalize correctors',
        'Use correctors and predictor as they are'
    ]

    _psvrprocessor_perp_norm_options_list = [
        PolySVR.normalize_all,
        PolySVR.normalize_predictors,
        PolySVR.normalize_correctors,
        lambda *args, **kwargs: zeros((0, 0))
    ]

    _psvrprocessor_perp_norm_options = {
        'Normalize all': 0,
        'Normalize predictor': 1,
        'Normalize correctors': 2,
        'Use correctors and predictor as they are': 3
    }

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

    _psvrprocessor_intercept_options = {
        'Do not include the intercept term': 0,
        'As a corrector': 1,
        'As a predictor': 2
    }

    def __fitter__(self, user_defined_parameters):
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
        user_params = (self._psvrprocessor_intercept, self._psvrprocessor_perp_norm_option)
        user_params += (self._psvrprocessor_C, self._psvrprocessor_epsilon)
        user_params += tuple(self._psvrprocessor_degrees)
        return user_params

    def __read_user_defined_parameters__(self, predictor_names, corrector_names):
        # Intercept term
        intercept = PolySVRProcessor._psvrprocessor_intercept_options[super(PolySVRProcessor, self).__getoneof__(
            PolySVRProcessor._psvrprocessor_intercept_options_names,
            default_value = PolySVRProcessor._psvrprocessor_intercept_options_names[2],
            show_text = 'PolySVR Processor: How do you want to include the intercept term? (default: ' + PolySVRProcessor._psvrprocessor_intercept_options_names[2] + ')'
        )]

        # Treat data option
        perp_norm_option = PolySVRProcessor._psvrprocessor_perp_norm_options[super(PolySVRProcessor, self).__getoneof__(
            PolySVRProcessor._psvrprocessor_perp_norm_options_names,
            default_value = 'Use correctors and predictor as they are',
            show_text = 'PolySVR Processor: How do you want to treat the features? (default: Use correctors and predictor as they are)'
        )]

        # C regularization parameter
        C = super(PolySVRProcessor, self).__getfloat__(
            default_value=3.162,
            try_ntimes=3,
            lower_limit=0.0,
            show_text='PolySVR Processor: Please, enter the regularization parameter C (default: 3.162)'
        )

        # epsilon regularization parameter
        epsilon = super(PolySVRProcessor, self).__getfloat__(
            default_value=0.16,
            try_ntimes=3,
            lower_limit=0.0,
            show_text='PolySVR Processor: Please, enter the epsilon-tube within which no penalty is associated in the training loss function (default: 0.16)'
        )

        # Polynomial degrees
        degrees = []
        for reg in predictor_names:
            degrees.append(super(PolySVRProcessor, self).__getint__(
                default_value=1,
                lower_limit=1,
                try_ntimes=3,
                show_text='PolySVR Processor: Please, enter the degree of the feature (predictor) \'' + str(reg) + '\' (or leave blank to set to 1): '
            ))
        for cor in corrector_names:
            degrees.append(super(PolySVRProcessor, self).__getint__(
                default_value=1,
                try_ntimes=3,
                show_text='PolySVR Processor: Please, enter the degree of the feature (corrector) \'' + str(cor) + '\' (or leave blank to set to 1): '
            ))

        return (intercept, perp_norm_option, C, epsilon) + tuple(degrees)

    def __curve__(self, fitter, predictor, prediction_parameters):
        # Create a new PolySVR fitter to return the curve prediction
        psvr = PolySVR(predictor, degrees=self._psvrprocessor_degrees[:1], intercept=self._psvrprocessor_intercept)
        PolySVRProcessor._psvrprocessor_perp_norm_options_list[self._psvrprocessor_perp_norm_option](psvr)
        return psvr.predict(prediction_parameters=prediction_parameters)

    def process(self, x1=0, x2=None, y1=0, y2=None, z1=0, z2=None,
                mem_usage=None, evaluation_kwargs={}, *args, **kwargs):
        # Call parent function process with additional parameters obtained through __read_user_defined_parameters__
        return super(PolySVRProcessor, self).process(
            x1, x2, y1, y2, z1, z2,
            mem_usage, evaluation_kwargs,
            C=self._psvrprocessor_C, epsilon=self._psvrprocessor_epsilon,
            *args, **kwargs
        )


class GaussianSVRProcessor(Processor):
    """
    Processor for Support Vector Regression with Gaussian kernel
    """
    _gsvrprocessor_perp_norm_options_names = [
        'Normalize all',
        'Normalize predictor',
        'Normalize correctors',
        'Use correctors and predictor as they are'
    ]

    _gsvrprocessor_perp_norm_options_list = [
        GaussianSVR.normalize_all,
        GaussianSVR.normalize_predictors,
        GaussianSVR.normalize_correctors,
        lambda *args, **kwargs: zeros((0, 0))
    ]

    _gsvrprocessor_perp_norm_options = {
        'Normalize all': 0,
        'Normalize predictor': 1,
        'Normalize correctors': 2,
        'Use correctors and predictor as they are': 3
    }

    _gsvrprocessor_intercept_options_names = [
        'Do not include the intercept term',
        'As a corrector',
        'As a predictor'
    ]

    _gsvrprocessor_intercept_options_list = [
        PolySVR.NoIntercept,
        PolySVR.CorrectionIntercept,
        PolySVR.PredictionIntercept
    ]

    _gsvrprocessor_intercept_options = {
        'Do not include the intercept term': 0,
        'As a corrector': 1,
        'As a predictor': 2
    }

    def __fitter__(self, user_defined_parameters):
        self._gsvrprocessor_intercept = user_defined_parameters[0]
        self._gsvrprocessor_perp_norm_option = user_defined_parameters[1]
        self._gsvrprocessor_C = user_defined_parameters[2]
        self._gsvrprocessor_epsilon = user_defined_parameters[3]
        self._gsvrprocessor_gamma = user_defined_parameters[4]

        # Orthonormalize/Orthogonalize/Do nothing options
        treat_data = GaussianSVRProcessor._gsvrprocessor_perp_norm_options_list[self._gsvrprocessor_perp_norm_option]
        # Intercept option
        intercept = GaussianSVRProcessor._gsvrprocessor_intercept_options_list[self._gsvrprocessor_intercept]

        # Instantiate a Gaussian SVR
        gsvr = GaussianSVR(self.predictors, self.correctors, intercept,
                           C=self._gsvrprocessor_C, epsilon=self._gsvrprocessor_epsilon,
                           gamma=self._gsvrprocessor_gamma)
        treat_data(gsvr)
        return gsvr

    def __user_defined_parameters__(self, fitter):
        user_params = (self._gsvrprocessor_intercept, self._gsvrprocessor_perp_norm_option)
        user_params += (self._gsvrprocessor_C, self._gsvrprocessor_epsilon, self._gsvrprocessor_gamma)
        return user_params

    def __read_user_defined_parameters__(self, predictor_names, corrector_names):
        # Intercept term
        intercept = GaussianSVRProcessor._gsvrprocessor_intercept_options[super(GaussianSVRProcessor, self).__getoneof__(
            GaussianSVRProcessor._gsvrprocessor_intercept_options_names,
            default_value = GaussianSVRProcessor._gsvrprocessor_intercept_options_names[2],
            show_text = 'GaussianSVR Processor: How do you want to include the intercept term? (default: ' + GaussianSVRProcessor._gsvrprocessor_intercept_options_names[2] + ')'
        )]

        # Treat data option
        perp_norm_option = GaussianSVRProcessor._gsvrprocessor_perp_norm_options[super(GaussianSVRProcessor, self).__getoneof__(
            GaussianSVRProcessor._gsvrprocessor_perp_norm_options_names,
            default_value = 'Use correctors and predictor as they are',
            show_text = 'GaussianSVR Processor: How do you want to treat the features? (default: Use correctors and predictor as they are)'
        )]

        # C regularization parameter
        C = super(GaussianSVRProcessor, self).__getfloat__(
            default_value=3.162,
            try_ntimes=3,
            lower_limit=0.0,
            show_text='GaussianSVR Processor: Please, enter the regularization parameter C (default: 3.162)'
        )

        # epsilon regularization parameter
        epsilon = super(GaussianSVRProcessor, self).__getfloat__(
            default_value=0.08916,
            try_ntimes= 3,
            lower_limit=0.0,
            show_text='GaussianSVR Processor: Please, enter the epsilon-tube within which no penalty is associated in the training loss function (default: 0.08916)'
        )

        # gamma for Gaussian kernel
        gamma = super(GaussianSVRProcessor, self).__getfloat__(
            default_value=0.25,
            try_ntimes=3,
            lower_limit=0.0,
            show_text='GaussianSVR Processor: Please, enter the gamma for the gaussian kernel (default: 0.25)'
        )

        return intercept, perp_norm_option, C, epsilon, gamma

    def process(self, x1=0, x2=None, y1=0, y2=None, z1=0, z2=None,
                mem_usage=None, evaluation_kwargs={}, *args, **kwargs):
        # Call parent function process with additional parameters 
        return super(GaussianSVRProcessor, self).process(
            x1, x2, y1, y2, z1, z2, mem_usage, evaluation_kwargs,
            C=self._gsvrprocessor_C,
            epsilon=self._gsvrprocessor_epsilon,
            gamma=self._gsvrprocessor_gamma,
            *args, **kwargs
        )


