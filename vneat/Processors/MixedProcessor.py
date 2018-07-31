from collections import Counter

import numpy as np
import os

from vneat.Fitters.CurveFitting import CombinedFitter, CurveFitter
from vneat.Processors.GAMProcessing import GAMProcessor
from vneat.Processors.GLMProcessing import PolyGLMProcessor
from vneat.Processors.Processing import Processor
from vneat.Processors.SVRProcessing import PolySVRProcessor, GaussianSVRProcessor
from vneat.Processors.PLSProcessing import PLSProcessor
from vneat.Utils.Documentation import Debugger


debugger = Debugger(os.environ['DEBUG'])
class MixedProcessor(Processor):
    """
    Processor that uses MixedFitter to allow you to correct and predict the data with two
    different fitters
    """

    # Available processors
    _mixedprocessor_processor_list = [
        PolyGLMProcessor,
        GAMProcessor,
        PolySVRProcessor,
        GaussianSVRProcessor,
        PLSProcessor,
    ]

    _mixedprocessor_processor_options = {
        'Poly GLM': 0,
        'GAM': 1,
        'Poly SVR': 2,
        'Gaussian SVR': 3,
        'PLS': 4
    }

    _mixedprocessor_perp_norm_options_names = [
        'Orthonormalize all',
        'Orthogonalize all',
        'Normalize all',
        'Treat correctors and predictors independently',

        'Orthonormalize predictors',
        'Orthogonalize predictors',
        'Normalize predictors',
        'Use predictors as they are',

        'Orthonormalize correctors',
        'Orthogonalize correctors',
        'Normalize correctors',
        'Use correctors as they are',
    ]


    _mixedprocessor_perp_norm_options_list = [
        lambda combined_fitter: combined_fitter.orthonormalize_all(),
        lambda combined_fitter: combined_fitter.orthogonalize_all(),
        lambda combined_fitter: combined_fitter.normalize_all(),
        lambda *args, **kwargs: np.zeros((0, 0)),

        CurveFitter.orthonormalize_predictors,
        CurveFitter.orthogonalize_predictors,
        CurveFitter.normalize_predictors,
        lambda *args, **kwargs: np.zeros((0, 0)),

        CurveFitter.orthonormalize_correctors,
        CurveFitter.orthogonalize_correctors,
        CurveFitter.normalize_correctors,
        lambda *args, **kwargs: np.zeros((0, 0))
    ]




    def __fitter__(self, user_defined_parameters):
        # Store user defined parameters
        self._separate_predictors_by_category = user_defined_parameters[0]
        self._category_predictor_option = user_defined_parameters[1]
        self._perp_norm_option = user_defined_parameters[2]
        self._corrector_option = user_defined_parameters[3]
        self._corrector_udp = user_defined_parameters[4]
        self._predictor_option = user_defined_parameters[5]
        self._predictor_udp = user_defined_parameters[6]

        # Separate predictors as required
        all_categories = [subj.category for subj in self._processor_subjects]
        c = Counter(all_categories)
        if self._separate_predictors_by_category:
            # Create NxM ndarray, with the same number of samples as the original predictors array but with
            # M columns, where M is the number of categories, and put 0 where corresponding
            M = len(c)
            N = len(self._processor_subjects)
            predictors_array = np.zeros((N, M))
            for index, category in enumerate(list(c)):
                category_index = [i for i, x in enumerate(all_categories) if x == category]
                selected_predictors = self._processor_predictors[category_index, 0]
                predictors_array[category_index, index] = selected_predictors

            # Separate the category to be treated as a predictor from the others,
            # which will be trated as correctors
            cat_predictor = [pos for pos, x in enumerate(list(c)) if self._category_predictor_option == x][0]
            cat_corrector = [pos for pos, x in enumerate(list(c)) if self._category_predictor_option != x]
            self._processor_predictors = np.atleast_2d(predictors_array[:, cat_predictor]).T

            self._processor_correctors = np.concatenate(
                (self._processor_correctors, predictors_array[:, cat_corrector]),
                axis=1
            )
            # Change predictors and correctors names
            original_predictor_name = self._processor_predictors_names[0]
            self._processor_predictors_names = [
                original_predictor_name + ' (category {})'.format(self._category_predictor_option)
            ]
            self._processor_correctors_names += [
                original_predictor_name + ' (category {})'.format(cat) for cat in cat_corrector
                ]

        # Create correction processor
        self._correction_processor = MixedProcessor._mixedprocessor_processor_list[
            self._corrector_option
        ](self._processor_subjects, [], self._processor_correctors_names, np.zeros((len(self._processor_subjects), 0)), self._processor_correctors,
          self._processor_processing_params, tuple(self._corrector_udp),type_data=self._type_data)


        # Create prediction processor
        self._prediction_processor = MixedProcessor._mixedprocessor_processor_list[
            self._predictor_option
        ](self._processor_subjects, self._processor_predictors_names, [], self._processor_predictors, np.zeros((len(self._processor_subjects), 0)),
          self._processor_processing_params, tuple(self._predictor_udp),type_data=self._type_data)

        # Get correction fitter
        correction_fitter = self._correction_processor.fitter
        prediction_fitter = self._prediction_processor.fitter

        # Create MixedFitter
        fitter = CombinedFitter(correction_fitter, prediction_fitter)()
        treat_data = MixedProcessor._mixedprocessor_perp_norm_options_list[self._perp_norm_option]
        treat_data(fitter)

        return fitter

    def __user_defined_parameters__(self, fitter):
        return self._separate_predictors_by_category, \
               self._category_predictor_option, \
               self._corrector_option, \
               self._corrector_udp, \
               self._predictor_option, \
               self._predictor_udp

    def __read_user_defined_parameters__(self, predictor_names, corrector_names, *args, **kwargs):

        # Mixed processor options
        separate_predictors_by_category = False
        category_predictor_option = 'All'
        all_categories = [subj.category for subj in self._processor_subjects]
        c = Counter(all_categories)
        if (self._category is None) and (None not in all_categories):
            # Ask user to separate predictors if there is no category specified for this processor
            separate_predictors_by_category = MixedProcessor.__getyesorno__(
                default_value=False,
                try_ntimes=3,
                show_text='\nMixedProcessor: Do you want to separate the predictor by categories? (Y/N, default N): '
            )
            if separate_predictors_by_category:
                # Ask which category should remain in predictors
                options_list = list(c)
                category_predictor_option = MixedProcessor.__getoneof__(
                    option_list=options_list,
                    default_value=0,
                    try_ntimes=3,
                    show_text='MixedProcessor: Which category do you want to have as a predictor, thus being the rest '
                              'correctors? (default value: 0)'
                )

                cat_predictor = [pos for pos, x in enumerate(list(c)) if category_predictor_option == x]
                cat_corrector = [pos for pos, x in enumerate(list(c)) if category_predictor_option != x]
                # Change predictors and correctors names
                original_predictor_name = predictor_names[0]
                predictor_names = [
                    original_predictor_name + ' (category {})'.format(cat) for cat in cat_predictor
                ]
                corrector_names += [
                    original_predictor_name + ' (category {})'.format(cat) for cat in cat_corrector
                    ]

        # Correction fitter
        keys = list(MixedProcessor._mixedprocessor_processor_options.keys())
        keys.sort()
        correct_option_name = MixedProcessor.__getoneof__(
            keys,
            default_value='Poly GLM',
            try_ntimes=3,
            show_text='MixedProcessor: Select the fitter to be used for correction '
                      '(default value: Poly GLM)'
        )
        correct_option = MixedProcessor._mixedprocessor_processor_options[correct_option_name]

        # Prediction fitter
        keys = list(MixedProcessor._mixedprocessor_processor_options.keys())
        keys.sort()
        predict_option_name = MixedProcessor.__getoneof__(
            keys,
            default_value='Poly GLM',
            try_ntimes=3,
            show_text='MixedProcessor: Select the fitter to be used for prediction '
                      '(default value: Poly GLM)'
        )
        predict_option = MixedProcessor._mixedprocessor_processor_options[predict_option_name]

        print()
        print("----------------------")
        print(" TREAT DATA ")
        print("----------------------")
        perp_norm_option_global = MixedProcessor._mixedprocessor_perp_norm_options[
            super(MixedProcessor, self).__getoneof__(
                MixedProcessor._mixedprocessor_perp_norm_options_names[:4],
                default_value=MixedProcessor._mixedprocessor_perp_norm_options_names[0],
                show_text='PolyGLM Processor: How do you want to treat the features? (default: ' +
                          MixedProcessor._mixedprocessor_perp_norm_options_names[0] + ')'
            )]


        print()
        print( "----------------------")
        print( " CORRECTOR PARAMETERS")
        print( "----------------------")
        # Create dummy array with proper dimensions to pass it as correctors to be the same size as the names
        N = len(self._processor_subjects)
        M = len(corrector_names)
        correctors = np.zeros((N, M))
        # User defined parameters for correction fitter
        correct_processor = MixedProcessor._mixedprocessor_processor_list[
            correct_option
        ](self._processor_subjects, [], corrector_names, np.zeros((0, 0)), correctors,
          self._processor_processing_params, perp_norm_option_global=(perp_norm_option_global==3))
        correct_udp = list(correct_processor.user_defined_parameters)

        print()
        print( "----------------------")
        print( " PREDICTOR PARAMETERS")
        print( "----------------------")
        # Create dummy array with proper dimensions to pass it as correctors to be the same size as the names
        M = len(predictor_names)
        predictors = np.zeros((N, M))
        # User defined parameters for correction fitter
        predict_processor = MixedProcessor._mixedprocessor_processor_list[
            predict_option
        ](self._processor_subjects, predictor_names, [], predictors, np.zeros((0, 0)),
          self._processor_processing_params, perp_norm_option_global=(perp_norm_option_global==3))
        predict_udp = list(predict_processor.user_defined_parameters)


        return separate_predictors_by_category, category_predictor_option, perp_norm_option_global, correct_option, \
               correct_udp, predict_option, predict_udp

    def __post_process__(self, prediction_parameters, correction_parameters):
        # Route post-processing routines to the corresponding processors
        prediction_results = self._prediction_processor.__post_process__(
            prediction_parameters, correction_parameters
        )
        correction_results = self._correction_processor.__post_process__(
            prediction_parameters, correction_parameters
        )

        # Return the post_processed parameters
        return Processor.Results(
            prediction_results.prediction_parameters,
            correction_results.correction_parameters
        )

    def __pre_process__(self, prediction_parameters, correction_parameters, predictors, correctors):
        # Route pre-processing routines to the corresponding processors
        pparams, dummy = self._prediction_processor.__pre_process__(
            prediction_parameters,
            correction_parameters,
            predictors,
            correctors
        )
        dummy, cparams = self._correction_processor.__pre_process__(
            prediction_parameters,
            correction_parameters,
            predictors,
            correctors
        )
        return pparams, cparams

    def __curve__(self, fitter, predictor, prediction_parameters, *args, **kwargs):
        return self._prediction_processor.__curve__(fitter, predictor, prediction_parameters)

    def __corrected_values__(self, fitter, observations, correction_parameters, *args, **kwargs):
        return self._correction_processor.__corrected_values__(fitter, observations, correction_parameters,
                                                               *args, **kwargs)

    def assign_bound_data(self, observations, predictors, prediction_parameters, correctors, correction_parameters,
                              fitting_results):
        # Restrictive bound data assignment: only if both processors are instances of the same class call their
        # specific implementation of __assign_bound_data__


        bound_functions = super(MixedProcessor, self).assign_bound_data(observations,
                                                                        predictors,
                                                                        prediction_parameters,
                                                                        correctors,
                                                                        correction_parameters,
                                                                        fitting_results
                                                                        )

        bound_functions += self._prediction_processor.__assign_bound_data__(observations,
                                                                            predictors,
                                                                            prediction_parameters,
                                                                            correctors,
                                                                            correction_parameters,
                                                                            fitting_results
                                                                            )


        return bound_functions



    def get_name(self):
        corrector_name = 'correction_{}'.format(self._correction_processor.get_name())
        predictor_name = 'prediction_{}'.format(self._prediction_processor.get_name())
        processor_name = '{}-{}'.format(corrector_name, predictor_name)
        if self._category is not None:
            processor_name += '-category_{}'.format(self._category)
        elif self._separate_predictors_by_category:
            processor_name += '-category_{}_vs_all'.format(self._category_predictor_option)
        return processor_name

    @property
    def correction_processor(self):
        return self._correction_processor

    @property
    def prediction_processor(self):
        return self._prediction_processor


MixedProcessor._mixedprocessor_perp_norm_options = {MixedProcessor._mixedprocessor_perp_norm_options_names[i]: i for i
                                                    in
                                                    range(len(MixedProcessor._mixedprocessor_perp_norm_options_names))}