"""
Processor for Support Vector Regression fitters
"""
from Processing import Processor
from SVR import PolySVR
from numpy import zeros

class PolySVRProcessor(Processor):
    """
    Processor for Polynomic Support Vector Regression
    """
    _psvrprocessor_perp_norm_options = {
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

    _psvrprocessor_perp_norm_options_list = [
        PolySVR.orthonormalize_all,
        PolySVR.orthogonalize_all,
        PolySVR.normalize_all,
        PolySVR.orthonormalize_regressors,
        PolySVR.orthogonalize_regressors,
        PolySVR.normalize_regressors,
        PolySVR.orthonormalize_correctors,
        PolySVR.orthogonalize_correctors,
        PolySVR.normalize_correctors,
        lambda *args, **kwargs: zeros((0, 0))
    ]

    def __fitter__(self, user_defined_parameters):
        """

        Parameters
        ----------
        user_defined_parameters

        Returns
        -------

        """
        self._psvrprocessor_homogeneous = user_defined_parameters[0]
        self._psvrprocessor_perp_norm_option = user_defined_parameters[1]
        self._psvrprocessor_C = user_defined_parameters[2]
        self._psvrprocessor_epsilon = user_defined_parameters[3]
        self._psvrprocessor_degrees = user_defined_parameters[4:]

        # Orthonormalize/Orthogonalize/Do nothing options
        treat_data = PolySVRProcessor._psvrprocessor_perp_norm_options_list[self._psvrprocessor_perp_norm_option]

        # Construct data matrix from correctors and regressors
        num_regs = self.regressors.shape[1]
        num_correc = self.correctors.shape[1]
        features = zeros((self.regressors.shape[0], num_regs + num_correc))
        features[:, :num_regs] = self.regressors
        features[:, num_regs:] = self.correctors

        # Instantiate a PolySVR
        psvr = PolySVR(features=features, regressors=range(num_regs), degrees=self._psvrprocessor_degrees, homogeneous=self._psvrprocessor_homogeneous)
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
        user_params = (self._psvrprocessor_homogeneous, self._psvrprocessor_perp_norm_option)
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
        # Homogeneous term
        if super(PolySVRProcessor, self).__getyesorno__(default_value = True, show_text = 'PolySVR Processor: Do you want to include the homogeneous term? (Y/N, default Y): '):
            homogeneous = 1
        else:
            homogeneous = 0

        # Treat data option
        perp_norm_option = PolySVRProcessor._psvrprocessor_perp_norm_options[super(PolySVRProcessor, self).__getoneof__(
            PolySVRProcessor._psvrprocessor_perp_norm_options,
            default_value = 'Orthonormalize all',
            show_text = 'PolySVR Processor: How do you want to treat the features? (default: Orthonormalize all)'
        )]

        # C regularization parameter
        C = super(PolySVRProcessor, self).__getfloat__(
            default_value = 1000.0,
            try_ntimes= 3,
            lower_limit=0.0,
            show_text='PolySVR Processor: Please, enter the regularization parameter C (default: 1000.0)'
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

        return (homogeneous, perp_norm_option, C, epsilon) + tuple(degrees)

    def process(self, x1 = 0, x2 = None, y1 = 0, y2 = None, z1 = 0, z2 = None, mem_usage = None, evaluation_kwargs = {}, *args, **kwargs):
        """

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




