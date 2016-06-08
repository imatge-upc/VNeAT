import numpy as np

import CrossValidation.score_functions as score_f
import Utils.DataLoader as DataLoader
from CrossValidation.GridSearch import GridSearch
from Fitters.CurveFitting import AdditiveCurveFitter
from Fitters.GAM import GAM, SmootherSet, SplinesSmoother
from Utils.Subject import Subject

if __name__ == "__main__":
    # Get features
    predictors = DataLoader.getFeatures([Subject.ADCSFIndex])

    # Creat GAM instance
    predictor_smoother = SmootherSet(SplinesSmoother(predictors))
    gam = GAM(predictor_smoothers=predictor_smoother, intercept=AdditiveCurveFitter.PredictionIntercept)

    # Create grid of hyperparams using linear and logscale
    grid_params = {
        'df': np.arange(3, 10),
    }

    # Create GridSearch instance
    gs = GridSearch(fitter=gam)

    # Compute hyperparameters
    gs.fit(grid_parameters=grid_params, N=1, m=100, score=score_f.mse,
           saveAllScores=True, filename="gam_scores_hyperparams")

    # Plot error
    gs.plot_error()

    # Save results
    gs.store_results("gam_opt_hyperparams", verbose=True)
