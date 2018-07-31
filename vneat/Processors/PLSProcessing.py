import numpy as np

from vneat.Fitters.PLS import PLS
from vneat.Processors.Processing import Processor


class PLSProcessor(Processor):

    _plsprocessor_perp_norm_options_names = [
        'Normalize all',
        'Normalize predictors',
        'Normalize correctors',
        'Use correctors and/or predictors as they are'

    ]


    _plsprocessor_perp_norm_options_list = [
        PLS.normalize_all,
        PLS.normalize_correctors,
        PLS.normalize_predictors,
        lambda *args, **kwargs: np.zeros((0, 0)),

    ]



    def __fitter__(self, user_defined_parameters):
        """
        Initializes the PLS fitter to be used to process the data.
        """

        self._plsprocessor_perp_norm_option = user_defined_parameters[0]
        self._plsprocessor_num_components_corr = user_defined_parameters[1]
        self._plsprocessor_num_components_pred = user_defined_parameters[2]

        treat_data = PLSProcessor._plsprocessor_perp_norm_options_list[self._plsprocessor_perp_norm_option]


        self._plsprocessor_pls = PLS(num_components_corr=self._plsprocessor_num_components_corr,
                                     num_components_pred=self._plsprocessor_num_components_pred,
                                     predictors=self.predictors, correctors=self.correctors,
                                     )

        treat_data(self._plsprocessor_pls)

        return self._plsprocessor_pls

    def __user_defined_parameters__(self, fitter):
        return (self._plsprocessor_perp_norm_option,
                self._plsprocessor_num_components_corr, self._plsprocessor_num_components_pred)

    def __read_user_defined_parameters__(self, predictor_names, corrector_names, perp_norm_option_global=False):


        if perp_norm_option_global:
            if len(predictor_names) == 0:
                default_value = PLSProcessor._plsprocessor_perp_norm_options_names[3]
                options_names = PLSProcessor._plsprocessor_perp_norm_options_names[2:4]
            elif len(corrector_names) == 0:
                default_value = PLSProcessor._plsprocessor_perp_norm_options_names[3]
                options_names = PLSProcessor._plsprocessor_perp_norm_options_names[1:2] + \
                                PLSProcessor._plsprocessor_perp_norm_options_names[3:4]
            else:
                default_value = PLSProcessor._plsprocessor_perp_norm_options_names[3]
                options_names = PLSProcessor._plsprocessor_perp_norm_options_names

            perp_norm_option = PLSProcessor._plsprocessor_perp_norm_options[
                super(PLSProcessor, self).__getoneof__(
                    options_names,
                    default_value=default_value,
                    show_text='PolySVR Processor: How do you want to treat the features? (default: ' +
                              default_value + ')'
                )]

        else:
            perp_norm_option = 3

        num_components_pred = super(PLSProcessor, self).__getint__(
            default_value=1,
            lower_limit=1,
            try_ntimes=3,
            show_text='PLS Processor: Please, enter the number of components (predictor) \'' + str(
                [reg for reg in predictor_names]) + '\' (or leave blank to set to 1): '
        )
        num_components_corr = super(PLSProcessor, self).__getint__(
            default_value=1,
            try_ntimes=3,
            show_text='PLS Processor: Please, enter the number of components (corrector) \'' + str(
                [cor for cor in corrector_names]) + '\' (or leave blank to set to 1): '
        )


        return (perp_norm_option,num_components_corr,num_components_pred)

    def __curve__(self, fitter, predictor, prediction_parameters, *args, **kwargs):
        # Initialize the pls with such predictors
        pls = PLS(predictors=np.array(predictor))

        PLSProcessor._plsprocessor_perp_norm_options_list[self._plsprocessor_perp_norm_option](pls)

        return pls.predict(prediction_parameters=prediction_parameters, curve=True, *args, **kwargs)

    def __evaluate_latent_factor(self):
        pass

    def get_name(self):
            return 'PLS'

    def __assign_bound_data__(self, observations, predictors, prediction_parameters, correctors,
                              correction_parameters, fitting_results):
        '''
        Method that add specific data of each Processor. Needs to be first bind for general data.
        '''
        # Pre-process parameters for fitter operations (predict, correct, etc.) and leave original
        # parameters for processor operations (curve)
        processed_prediction_parameters, processed_correction_parameters = self.__pre_process__(
            prediction_parameters,
            correction_parameters,
            predictors,
            correctors
        )

        # Assign data to compute PLS maps.
        fitting_results.x_scores, fitting_results.y_scores = self._processor_fitter.transform(
            predictors=self._processor_fitter.predictors,
            prediction_parameters=processed_prediction_parameters,
            observations=fitting_results.corrected_data,
            correct=True
        )

        fitting_results.x_rotations = self._processor_fitter.get_item_parameters(
            parameters=processed_prediction_parameters,
            name='x_rotations'
        )

        fitting_results.fitter = self._processor_fitter
        fitting_results.predictors = self._processor_fitter.predictors
        fitting_results.num_components = self._processor_fitter.get_item_parameters(
            parameters=processed_prediction_parameters,
            name='num_components'
        )


        bound_functions = ['x_scores', 'y_scores', 'x_rotations', 'fitter', 'predictors', 'num_components']

        # # Call parent method
        # bound_functions += super(PLSProcessor, self).__assign_bound_data__(observations, predictors,
        #                                                                    prediction_parameters, correctors,
        #                                                                    correction_parameters, fitting_results)

        return bound_functions


PLSProcessor._plsprocessor_perp_norm_options = {PLSProcessor._plsprocessor_perp_norm_options_names[i]: i for i in
                                                range(len(PLSProcessor._plsprocessor_perp_norm_options_names))}
#