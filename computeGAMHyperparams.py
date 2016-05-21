import CrossValidation.score_functions as score_f
import Utils.DataLoader as DataLoader
import numpy as np
from os.path import join
from Fitters.CurveFitting import AdditiveCurveFitter
from CrossValidation.GridSearch import GridSearch
from Utils.Subject import Subject
from Fitters.GAM import GAM, SmootherSet, SplinesSmoother
from user_paths import RESULTS_DIR

RESULTS_DIR = join(RESULTS_DIR, 'SGAM')

if __name__ == "__main__":

    # Get features
    predictors = DataLoader.getFeatures([Subject.ADCSFIndex])

    # Create GAM instance
    predictor_smoother = SmootherSet(SplinesSmoother(predictors))
    gam = GAM(predictor_smoothers=predictor_smoother,
              intercept=AdditiveCurveFitter.PredictionIntercept
              )

    # Create grid of hyperparams using linear and logscale
    grid_params = {
        'df': np.arange(9, 15),
    }

    # Create GridSearch instance
    gs = GridSearch(fitter=gam, results_directory=RESULTS_DIR)

    # Compute hyperparameters
    gs.fit(grid_parameters=grid_params, N=1, m=200, score=score_f.mse,
           save_all_scores=True, filename="gam_scores_hyperparams")

    # Save results
    gs.store_results("gam_opt_hyperparams", verbose=True)

    # Plot error
    gs.plot_error()
