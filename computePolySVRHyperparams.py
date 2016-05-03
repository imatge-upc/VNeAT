import CrossValidation.score_functions as score_f
import CrossValidation.degrees_freedom as df_f

import Utils.DataLoader as DataLoader
import numpy as np
from Fitters.CurveFitting import AdditiveCurveFitter
from CrossValidation.GridSearch import GridSearch
from Utils.Subject import Subject
from Fitters.SVR import PolySVR

if __name__ == "__main__":

    # Get features
    predictors = DataLoader.getFeatures([Subject.ADCSFIndex])

    # Creat PolySVR instance
    psvr = PolySVR(predictors, [0], [3], AdditiveCurveFitter.PredictionIntercept)

    # Create grid of hyperparams using uniform random sampling
    # epsilon = 1e-3 + (0.5 - 1e-3) * np.random.rand(20)
    # C = 10 ** (4 * np.random.rand(20))

    # Create grid of hyperparams using linear and logscale
    epsilon = np.linspace(0.05, 0.1, 20)
    C = np.logspace(0.5, 1, 10)
    grid_params = {
        'epsilon': list(epsilon),
        'C': list(C)
    }

    # Create GridSearch instance
    gs = GridSearch(fitter=psvr)

    # Compute hyperparameters
    gs.fit(grid_parameters=grid_params, N=1, m=200, degrees_of_freedom=df_f.df_SVR,
           score=score_f.mse, saveAllScores=True, filename="psvr_scores_hyperparams")

    # Plot error
    gs.plot_error()

    # Save results
    gs.store_results("psvr_opt_hyperparams", verbose=True)
