from Fitters.CurveFitting import CombinedFitter
from Processors.GAMProcessing import GAMProcessor
from Processors.GLMProcessing import GLMProcessor, PolyGLMProcessor
from Processors.Processing import Processor
from Processors.SVRProcessing import PolySVRProcessor, GaussianSVRProcessor


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

    def __init__(self, subjects, predictors, correctors=[], user_defined_parameters=()):
        self._processor_predictors_attr = predictors
        self._processor_correctors_attr = correctors
        super(MixedProcessor, self).__init__(subjects, predictors, correctors, user_defined_parameters)

    def __fitter__(self, user_defined_parameters):
        # Store user defined parameters
        self._corrector_option = user_defined_parameters[0]
        self._corrector_udp = user_defined_parameters[1]
        self._predictor_option = user_defined_parameters[2]
        self._predictor_udp = user_defined_parameters[3]

        # Create correction processor
        self._correction_processor = MixedProcessor._mixedprocessor_processor_list[
            self._corrector_option
        ](self._processor_subjects, [], self._processor_correctors_attr, tuple(self._corrector_udp))

        # Create prediction processor
        self._prediction_processor = MixedProcessor._mixedprocessor_processor_list[
            self._predictor_option
        ](self._processor_subjects, self._processor_predictors_attr, [], tuple(self._predictor_udp))

        # Get correction fitter
        correction_fitter = self._correction_processor._processor_fitter
        prediction_fitter = self._prediction_processor._processor_fitter

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
        ](self._processor_subjects, [], self._processor_correctors_attr)

        correct_udp = list(correct_processor.user_defined_parameters)

        print "----------------------"
        print " PREDICTOR PARAMETERS"
        print "----------------------"
        # User defined parameters for correction fitter
        predict_processor = MixedProcessor._mixedprocessor_processor_list[
            predict_option
        ](self._processor_subjects, self._processor_predictors_attr, [])

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

    @property
    def correction_processor(self):
        return self._correction_processor

    @property
    def prediction_processor(self):
        return self._prediction_processor
