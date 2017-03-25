import numpy as np

from vneat.Fitters.GAM import GAM, InterceptSmoother, PolynomialSmoother, SplinesSmoother, SmootherSet
from vneat.Processors.Processing import Processor


class GAMProcessor(Processor):
    _gamprocessor_perp_norm_options_names = [
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

    _gamprocessor_perp_norm_options_list = [
        GAM.orthonormalize_all,
        GAM.orthogonalize_all,
        GAM.normalize_all,
        GAM.orthonormalize_predictors,
        GAM.orthogonalize_predictors,
        GAM.normalize_predictors,
        GAM.orthonormalize_correctors,
        GAM.orthogonalize_correctors,
        GAM.normalize_correctors,
        lambda *args, **kwargs: None
    ]

    TYPE_SMOOTHER = [InterceptSmoother, PolynomialSmoother, SplinesSmoother]

    def __fitter__(self, user_defined_parameters):
        '''Initializes the GAM fitter to be used to process the data.
        '''

        self._gamprocessor_perp_norm_option = user_defined_parameters[0]
        self._gamprocessor_smoother_parameters = user_defined_parameters[1]

        sm_index = 0
        corrector_smoothers = SmootherSet()
        predictor_smoothers = SmootherSet()
        for corr in self.correctors.T:
            smoother_function = GAMProcessor.TYPE_SMOOTHER[int(self._gamprocessor_smoother_parameters[sm_index])](corr)
            sm_index += 1
            n_param = self._gamprocessor_smoother_parameters[sm_index]
            sm_index += 1
            smoother_function.set_parameters(
                np.array(self._gamprocessor_smoother_parameters[sm_index:sm_index + n_param])[:, None])
            sm_index += n_param
            corrector_smoothers.extend(smoother_function)
        for reg in self.predictors.T:
            smoother_function = GAMProcessor.TYPE_SMOOTHER[int(self._gamprocessor_smoother_parameters[sm_index])](reg)
            sm_index += 1
            n_param = self._gamprocessor_smoother_parameters[sm_index]
            sm_index += 1
            smoother_function.set_parameters(
                np.array(self._gamprocessor_smoother_parameters[sm_index:sm_index + n_param])[:, None])
            sm_index += n_param
            predictor_smoothers.extend(smoother_function)

        treat_data = GAMProcessor._gamprocessor_perp_norm_options_list[self._gamprocessor_perp_norm_option]

        gam = GAM(corrector_smoothers=corrector_smoothers, predictor_smoothers=predictor_smoothers)

        treat_data(gam)

        return gam

    def __user_defined_parameters__(self, fitter):
        return (self._gamprocessor_perp_norm_option, self._gamprocessor_smoother_parameters)

    def __read_user_defined_parameters__(self, predictor_names, corrector_names):

        perp_norm_option = GAMProcessor._gamprocessor_perp_norm_options[super(GAMProcessor, self).__getoneof__(
            GAMProcessor._gamprocessor_perp_norm_options_names,
            default_value='Orthonormalize all',
            show_text='GAM Processor: How do you want to treat the features? (default: Orthonormalize all)'
        )]

        print('')
        smoothing_functions = []
        for cor in corrector_names:
            smoother_type = super(GAMProcessor, self).__getint__(
                default_value=1,
                try_ntimes=3,
                show_text='GAM Processor: Please, enter the smoothing function of the feature (corrector) \'' + str(cor)
                          + '\' (1: Polynomial Smoother, 2: Splines Smoother): ')

            smoothing_functions += [smoother_type]

            if smoother_type == GAMProcessor.TYPE_SMOOTHER.index(PolynomialSmoother):
                n_params = 1
                polynomial_degree = super(GAMProcessor, self).__getint__(
                    default_value=3,
                    try_ntimes=3,
                    show_text='GAM Processor: You have selected Polynomial smoother. Please, enter the degree of the polynomial '
                              '(or leave blank to set to 3): '
                )

                # Update smoothing functions list
                smoothing_functions += [n_params, polynomial_degree]

            elif smoother_type == GAMProcessor.TYPE_SMOOTHER.index(SplinesSmoother):
                n_params = 3
                specification_option = super(GAMProcessor, self).__getint__(
                    default_value=0,
                    try_ntimes=3,
                    show_text='GAM Processor: You have selected Splines smoother. Please, choose if you want to specify degrees of free'
                              'dom (0, by default) or smoothing factor (1): '
                )
                spline_degrees = super(GAMProcessor, self).__getint__(
                    default_value=3,
                    try_ntimes=3,
                    show_text='GAM Processor: You have selected Splines smoother. Please, enter the degree of the splines '
                              '(or leave blank to set to 3): '
                )
                if specification_option == 0:
                    df_option = super(GAMProcessor, self).__getint__(
                        default_value=spline_degrees + 1,
                        try_ntimes=3,
                        lower_limit=spline_degrees + 1,
                        show_text='GAM Processor: Please, enter the degree of freedom of the spline'
                                  '(or leave it blank to set it to default: spline degree + 1): '
                    )

                    # Update smoothing functions list
                    smoothing_functions += [n_params, specification_option, df_option, spline_degrees]
                else:
                    smoothing_factor = super(GAMProcessor, self).__getfloat__(
                        default_value=500,
                        try_ntimes=3,
                        show_text='GAM Processor: Please, enter the smoothing factor of the spline '
                                  '(or leave it blank to set it to default: 500): '
                    )

                    # Update smoothing functions list
                    smoothing_functions += [n_params, specification_option, smoothing_factor, spline_degrees]

        for reg in predictor_names:
            smoother_type = super(GAMProcessor, self).__getint__(
                default_value=1,
                try_ntimes=3,
                show_text='GAM Processor: Please, enter the smoothing function of the feature (predictor) \'' + str(reg)
                          + '\' (1: Polynomial Smoother, 2: Splines Smoother): ')

            smoothing_functions += [smoother_type]

            if smoother_type == GAMProcessor.TYPE_SMOOTHER.index(PolynomialSmoother):
                n_params = 1
                polynomial_degree = super(GAMProcessor, self).__getint__(
                    default_value=3,
                    try_ntimes=3,
                    show_text='GAM Processor: You have selected Polynomial smoother. Please, enter the degree of the polynomial '
                              '(or leave blank to set to 3): '
                )

                # Update smoothing functions list
                smoothing_functions += [n_params, polynomial_degree]

            elif smoother_type == GAMProcessor.TYPE_SMOOTHER.index(SplinesSmoother):
                n_params = 3
                specification_option = super(GAMProcessor, self).__getint__(
                    default_value=0,
                    try_ntimes=3,
                    show_text='GAM Processor: You have selected Splines smoother. Please, choose if you want to specify degrees of free'
                              'dom (0, by default) or smoothing factor (1): '
                )
                spline_degrees = super(GAMProcessor, self).__getint__(
                    default_value=3,
                    try_ntimes=3,
                    show_text='GAM Processor: You have selected Splines smoother. Please, enter the degree of the splines '
                              '(or leave blank to set to 3): '
                )
                if specification_option == 0:
                    df_option = super(GAMProcessor, self).__getint__(
                        default_value=spline_degrees + 1,
                        try_ntimes=3,
                        lower_limit=spline_degrees + 1,
                        show_text='GAM Processor: Please, enter the degree of freedom of the spline'
                                  '(or leave it blank to set it to default: spline degree + 1): '
                    )

                    # Update smoothing functions list
                    smoothing_functions += [n_params, specification_option, df_option, spline_degrees]
                else:
                    smoothing_factor = super(GAMProcessor, self).__getfloat__(
                        default_value=500,
                        try_ntimes=3,
                        show_text='GAM Processor: Please, enter the smoothing factor of the spline '
                                  '(or leave it blank to set it to default: 500): '
                    )

                    # Update smoothing functions list
                    smoothing_functions += [n_params, specification_option, smoothing_factor, spline_degrees]


        return (perp_norm_option, smoothing_functions)

    def __curve__(self, fitter, predictors, prediction_parameters):
        gam = GAM()
        GAMProcessor._gamprocessor_perp_norm_options_list[self._gamprocessor_perp_norm_option](gam)
        return gam.predict(predictors=predictors, prediction_parameters=prediction_parameters)

    def get_name(self):
        if self._gamprocessor_smoother_parameters[0] == 1:
            smoother_name = 'Poly'
        elif self._gamprocessor_smoother_parameters[0] == 2:
            smoother_name = 'Spline'
        else:
            smoother_name = ''
        return '{}GAM'.format(smoother_name)


GAMProcessor._gamprocessor_perp_norm_options = {GAMProcessor._gamprocessor_perp_norm_options_names[i]: i for i in
                                                xrange(len(GAMProcessor._gamprocessor_perp_norm_options_names))}
