import numpy as np

from nonlinear2.CrossValidation.GridSearch import GridSearch
from nonlinear2.Fitters.SVR import PolySVR
from nonlinear2.Utils.Subject import Subject
import nonlinear2.Utils.DataLoader as DataLoader
import nonlinear2.CrossValidation.score_functions as score_f

if __name__ == "__main__":

    # Get features
    predictors = DataLoader.getFeatures([Subject.ADCSFIndex])

    # Creat PolySVR instance
    psvr = PolySVR(predictors, [0], [3], False)

    # Create grid of hyperparams using uniform random sampling
    # epsilon = 1e-3 + (0.5 - 1e-3) * np.random.rand(20)
    # C = 10 ** (4 * np.random.rand(20))

    # Create grid of hyperparams using linear and logscale
    epsilon = np.linspace(0.01, 0.2, 30)
    C = np.logspace(1, 4, 15)
    grid_params = {
        'epsilon': list(epsilon),
        'C': list(C)
    }

    # Create GridSearch instance
    gs = GridSearch(fitter=psvr)

    # Compute hyperparameters
    gs.fit(grid_parameters=grid_params, N=5, m=15, score=score_f.mse,
           saveAllScores=True, filename="psvr_scores_hyperparams")

    # Save results
    gs.store_results("psvr_opt_hyperparams", verbose=True)
