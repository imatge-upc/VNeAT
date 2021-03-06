"""
Processor for Support Vector Regression fitters
"""
from numpy import zeros

from vneat.Fitters.SVR import PolySVR, GaussianSVR
from vneat.Processors.Processing import Processor


class PolySVRProcessor(Processor):
    """
    Processor for Polynomic Support Vector Regression
    """
    _psvrprocessor_perp_norm_options_names = [
        'Normalize all',
        'Normalize predictors',
        'Normalize correctors',
        'Use correctors and/or predictors as they are'
    ]

    _psvrprocessor_perp_norm_options_list = [
        PolySVR.normalize_all,
        PolySVR.normalize_predictors,
        PolySVR.normalize_correctors,
        lambda *args, **kwargs: zeros((0, 0))
    ]

    _psvrprocessor_perp_norm_options = {
        'Normalize all': 0,
        'Normalize predictors': 1,
        'Normalize correctors': 2,
        'Use correctors and/or predictor as they are': 3
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
        psvr = PolySVR(features=features, predictors=range(num_regs), degrees=self._psvrprocessor_degrees,
                       intercept=intercept, C=self._psvrprocessor_C, epsilon=self._psvrprocessor_epsilon)
        treat_data(psvr)
        return psvr

    def __user_defined_parameters__(self, fitter):
        user_params = (self._psvrprocessor_intercept, self._psvrprocessor_perp_norm_option)
        user_params += (self._psvrprocessor_C, self._psvrprocessor_epsilon)
        user_params += tuple(self._psvrprocessor_degrees)
        return user_params

    def __read_user_defined_parameters__(self, predictor_names, corrector_names, perp_norm_option_global=False,
                                         *args,**kwargs):
        # Intercept term
        # If there are no predictor names, show only options NoIntercept and CorrectionIntercept,
        # and if there are no corrector names, show only NoIntercept and PredictionIntercept. Otherwise,
        # show all options
        if len(predictor_names) == 0:
            default_value = PolySVRProcessor._psvrprocessor_intercept_options_names[1]
            options_names = PolySVRProcessor._psvrprocessor_intercept_options_names[:2]
        elif len(corrector_names) == 0:
            default_value = PolySVRProcessor._psvrprocessor_intercept_options_names[2]
            options_names = PolySVRProcessor._psvrprocessor_intercept_options_names[::2]
        else:
            default_value = PolySVRProcessor._psvrprocessor_intercept_options_names[1]
            options_names = PolySVRProcessor._psvrprocessor_intercept_options_names

        intercept = PolySVRProcessor._psvrprocessor_intercept_options[super(PolySVRProcessor, self).__getoneof__(
            options_names,
            default_value=default_value,
            show_text='PolySVR Processor: How do you want to include the intercept term? (default: {})'.format(
                default_value
            )
        )]

        # Treat data option: if there is not a global variable, treat it independently for each fitter.
        if perp_norm_option_global:
            if len(predictor_names) == 0:
                default_value = PolySVRProcessor._psvrprocessor_perp_norm_options_names[3]
                options_names = PolySVRProcessor._psvrprocessor_perp_norm_options_names[2:4]
            elif len(corrector_names) == 0:
                default_value = PolySVRProcessor._psvrprocessor_perp_norm_options_names[3]
                options_names = PolySVRProcessor._psvrprocessor_perp_norm_options_names[1:2] + \
                                PolySVRProcessor._psvrprocessor_perp_norm_options_names[3:4]
            else:
                default_value = PolySVRProcessor._psvrprocessor_perp_norm_options_names[3]
                options_names = PolySVRProcessor._psvrprocessor_perp_norm_options_names

            perp_norm_option = PolySVRProcessor._psvrprocessor_perp_norm_options[
                super(PolySVRProcessor, self).__getoneof__(
                    options_names,
                    default_value=default_value,
                    show_text='PolySVR Processor: How do you want to treat the features? (default: ' +
                              default_value + ')'
                )]

        else:
            perp_norm_option = 3

        # C regularization parameter
        C = super(PolySVRProcessor, self).__getfloat__(
            default_value=1,
            try_ntimes=3,
            lower_limit=0.0,
            show_text='PolySVR Processor: Please, enter the regularization parameter C (default: 1)'
        )

        # epsilon regularization parameter
        epsilon = super(PolySVRProcessor, self).__getfloat__(
            default_value=0.1,
            try_ntimes=3,
            lower_limit=0.0,
            show_text='PolySVR Processor: Please, enter the epsilon-tube within which no penalty is '
                      'associated in the training loss function (default: 0.1)'
        )

        # Polynomial degrees
        degrees = []
        for reg in predictor_names:
            degrees.append(super(PolySVRProcessor, self).__getint__(
                default_value=1,
                lower_limit=1,
                try_ntimes=3,
                show_text='PolySVR Processor: Please, enter the degree of the feature (predictor) \'' + str(
                    reg) + '\' (or leave blank to set to 1): '
            ))
        for cor in corrector_names:
            degrees.append(super(PolySVRProcessor, self).__getint__(
                default_value=1,
                try_ntimes=3,
                show_text='PolySVR Processor: Please, enter the degree of the feature (corrector) \'' + str(
                    cor) + '\' (or leave blank to set to 1): '
            ))

        return (intercept, perp_norm_option, C, epsilon) + tuple(degrees)

    def __curve__(self, fitter, predictor, prediction_parameters, *args, **kwargs):
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

    def get_name(self):
        return 'PolySVR'


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
        treat_data = GaussianSVRProcessor._gsvrprocessor_perp_norm_options_list[
            self._gsvrprocessor_perp_norm_option]
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

    def __read_user_defined_parameters__(self, predictor_names, corrector_names, perp_norm_option_global=False,
                                         *args, **kwargs):
        # Intercept term
        # If there are no predictor names, show only options NoIntercept and CorrectionIntercept,
        # and if there are no corrector names, show only NoIntercept and PredictionIntercept. Otherwise,
        # show all options
        if len(predictor_names) == 0:
            default_value = GaussianSVRProcessor._gsvrprocessor_intercept_options_names[1]
            options_names = GaussianSVRProcessor._gsvrprocessor_intercept_options_names[:2]
        elif len(corrector_names) == 0:
            default_value = GaussianSVRProcessor._gsvrprocessor_intercept_options_names[2]
            options_names = GaussianSVRProcessor._gsvrprocessor_intercept_options_names[::2]
        else:
            default_value = GaussianSVRProcessor._gsvrprocessor_intercept_options_names[1]
            options_names = GaussianSVRProcessor._gsvrprocessor_intercept_options_names
        intercept = GaussianSVRProcessor._gsvrprocessor_intercept_options[
            super(GaussianSVRProcessor, self).__getoneof__(
                options_names,
                default_value=default_value,
                show_text='GaussianSVR Processor: How do you want to include the intercept term? (default: {})'.format(
                    default_value
                )
            )]

        # Treat data option: if there is not a global variable, treat it independently for each fitter.
        if perp_norm_option_global:
            if len(predictor_names) == 0:
                default_value = GaussianSVRProcessor._gsvrprocessor_perp_norm_options_names[3]
                options_names = GaussianSVRProcessor._gsvrprocessor_perp_norm_options_names[2:4]
            elif len(corrector_names) == 0:
                default_value = GaussianSVRProcessor._gsvrprocessor_perp_norm_options_names[3]
                options_names = GaussianSVRProcessor._gsvrprocessor_perp_norm_options_names[1:2] + \
                                GaussianSVRProcessor._gsvrprocessor_perp_norm_options_names[3:4]
            else:
                default_value = GaussianSVRProcessor._gsvrprocessor_perp_norm_options_names[3]
                options_names = GaussianSVRProcessor._gsvrprocessor_perp_norm_options_names

            perp_norm_option = GaussianSVRProcessor._gsvrprocessor_perp_norm_options[
                super(GaussianSVRProcessor, self).__getoneof__(
                    options_names,
                    default_value=default_value,
                    show_text='GaussianSVR Processor: How do you want to treat the features? (default: ' +
                              default_value + ')'
                )]

        else:
            perp_norm_option = 3

        # C regularization parameter
        C = super(GaussianSVRProcessor, self).__getfloat__(
            default_value=1,
            try_ntimes=3,
            lower_limit=0.0,
            show_text='GaussianSVR Processor: Please, enter the regularization parameter C (default: 1)'
        )

        # epsilon regularization parameter
        epsilon = super(GaussianSVRProcessor, self).__getfloat__(
            default_value=0.1,
            try_ntimes=3,
            lower_limit=0.0,
            show_text='GaussianSVR Processor: Please, enter the epsilon-tube within which no penalty '
                      'is associated in the training loss function (default: 0.1)'
        )

        # gamma for Gaussian kernel
        gamma = super(GaussianSVRProcessor, self).__getfloat__(
            default_value=0.1,
            try_ntimes=3,
            lower_limit=0.0,
            show_text='GaussianSVR Processor: Please, enter the gamma for the gaussian kernel (default: 0.1)'
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

    def get_name(self):
        return 'GaussianSVR'
