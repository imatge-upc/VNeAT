import numpy as np

from src.Fitters.CurveFitting import CombinedFitter
from src.Processors.GAMProcessing import GAMProcessor
from src.Processors.GLMProcessing import GLMProcessor, PolyGLMProcessor
from src.Processors.Processing import Processor
from src.Processors.SVRProcessing import PolySVRProcessor, GaussianSVRProcessor


class MixedProcessor(Processor):
    """
    Processor that uses MixedFitter to allow you to correct and predict the data with two
    different fitters
    """

    # Available processors
    _mixedprocessor_processor_list = [
        GLMProcessor,
        PolyGLMProcessor,
        GAMProcessor,
        PolySVRProcessor,
        GaussianSVRProcessor
    ]

    _mixedprocessor_processor_options = {
        'GLM': 0,
        'Poly GLM': 1,
        'GAM': 2,
        'Poly SVR': 3,
        'Gaussian SVR': 4
    }

    def __fitter__(self, user_defined_parameters):
        # Store user defined parameters
        self._corrector_option = user_defined_parameters[0]
        self._corrector_udp = user_defined_parameters[1]
        self._predictor_option = user_defined_parameters[2]
        self._predictor_udp = user_defined_parameters[3]

        # Create correction processor
        self._correction_processor = MixedProcessor._mixedprocessor_processor_list[
            self._corrector_option
        ](self._processor_subjects, [], self._processor_correctors_names, np.zeros((0, 0)), self._processor_correctors,
          self._processor_processing_params, tuple(self._corrector_udp))

        # Create prediction processor
        self._prediction_processor = MixedProcessor._mixedprocessor_processor_list[
            self._predictor_option
        ](self._processor_subjects, self._processor_predictors_names, [], self._processor_predictors, np.zeros((0, 0)),
          self._processor_processing_params, tuple(self._predictor_udp))

        # Get correction fitter
        correction_fitter = self._correction_processor.fitter
        prediction_fitter = self._prediction_processor.fitter

        # Create MixedFitter
        fitter = CombinedFitter(correction_fitter, prediction_fitter)
        return fitter()

    def __user_defined_parameters__(self, fitter):
        return self._corrector_option, \
               self._corrector_udp, \
               self._predictor_option, \
               self._predictor_udp

    def __read_user_defined_parameters__(self, predictor_names, corrector_names):
        # Correction fitter
        keys = MixedProcessor._mixedprocessor_processor_options.keys()
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
        keys = MixedProcessor._mixedprocessor_processor_options.keys()
        keys.sort()
        predict_option_name = MixedProcessor.__getoneof__(
            keys,
            default_value='Poly GLM',
            try_ntimes=3,
            show_text='MixedProcessor: Select the fitter to be used for prediction '
                      '(default value: Poly GLM)'
        )
        predict_option = MixedProcessor._mixedprocessor_processor_options[predict_option_name]

        print "----------------------"
        print " CORRECTOR PARAMETERS"
        print "----------------------"
        # User defined parameters for correction fitter
        correct_processor = MixedProcessor._mixedprocessor_processor_list[
            correct_option
        ](self._processor_subjects, [], self._processor_correctors_names, np.zeros((0, 0)), self._processor_correctors,
          self._processor_processing_params)

        correct_udp = list(correct_processor.user_defined_parameters)

        print "----------------------"
        print " PREDICTOR PARAMETERS"
        print "----------------------"
        # User defined parameters for correction fitter
        predict_processor = MixedProcessor._mixedprocessor_processor_list[
            predict_option
        ](self._processor_subjects, self._processor_predictors_names, [], self._processor_predictors, np.zeros((0, 0)),
          self._processor_processing_params)

        predict_udp = list(predict_processor.user_defined_parameters)

        return correct_option, correct_udp, predict_option, predict_udp

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

    def __curve__(self, fitter, predictor, prediction_parameters):
        return self._prediction_processor.__curve__(fitter, predictor, prediction_parameters)

    def __corrected_values__(self, fitter, observations, correction_parameters, *args, **kwargs):
        return self._correction_processor.__corrected_values__(fitter, observations, correction_parameters,
                                                               *args, **kwargs)

    def __assign_bound_data__(self, observations, predictors, prediction_parameters, correctors, correction_parameters,
                              fitting_results):
        # Restrictive bound data assignment: only if both processors are instances of the same class call their
        # specific implementation of __assign_bound_data__
        bound_functions = []
        if self._correction_processor.__class__ == self._prediction_processor.__class__:
            bound_functions = self._prediction_processor.__assign_bound_data__(observations,
                                                                               predictors,
                                                                               prediction_parameters,
                                                                               correctors,
                                                                               correction_parameters,
                                                                               fitting_results
                                                                               )
        else:
            return super(MixedProcessor, self).__assign_bound_data__(observations,
                                                                     predictors,
                                                                     prediction_parameters,
                                                                     correctors,
                                                                     correction_parameters,
                                                                     fitting_results
                                                                     )

        return bound_functions

    def get_name(self):
        return 'correction_' + self._correction_processor.get_name() + \
               '-prediction_' + self._prediction_processor.get_name()

    @property
    def correction_processor(self):
        return self._correction_processor

    @property
    def prediction_processor(self):
        return self._prediction_processor
