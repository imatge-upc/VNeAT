import numpy

class LatentResults(object):
    """
    Computes and binds the generic requirements for all fit evaluation functions

    Parameters
    ----------
    observations : ndarray
        Array with the original observations
    predictors : ndarray
        Array with the predictor of the model
    prediction_parameters : ndarray
        Array with computed prediction parameters
    correctors : ndarray
        Array with the correctors of the model
    correction_parameters : ndarray
        Array with the computed correction parameters
    fitting_results : Object
        Object that stores the bound data for the fit evaluation function

    Returns
    -------
    List
        List of the names of the functions that should be bound (if they are not already) to the evaluation
        function that you are using
    """

    def __init__(self,observations, predictors, prediction_parameters, correctors, correction_parameters):

        # Pre-process parameters for fitter operations (predict, correct, etc.) and leave original
        # parameters for processor operations (curve)
        processed_prediction_parameters, processed_correction_parameters = self.__pre_process__(
            prediction_parameters,
            correction_parameters,
            predictors,
            correctors
        )

        latent_results.observations = observations
        latent_results.corrected_data = self._processor_fitter.correct(
            observations=observations,
            correctors=correctors,
            correction_parameters=processed_correction_parameters
        )
        latent_results.x_rotations = self._processor_fitter.predict(
            predictors=predictors,
            prediction_parameters=processed_prediction_parameters
        )

        latent_results.x_scores, latent_results.y_scores = self._processor_fitter.transform(
            observations=observations,
            predictors=predictors,
            prediction_parameters=processed_prediction_parameters
        )
