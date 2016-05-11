import CrossValidation.score_functions as score_f

import Utils.DataLoader as DataLoader
import numpy as np
from Fitters.CurveFitting import AdditiveCurveFitter
from CrossValidation.GridSearch import GridSearch
from Utils.Subject import Subject
from Fitters.SVR import GaussianSVR

if __name__ == "__main__":

    # Get features
    predictors = DataLoader.getFeatures([Subject.ADCSFIndex])

    # Creat PolySVR instance
    gsvr = GaussianSVR(predictors=predictors, correctors=None,
                       intercept=AdditiveCurveFitter.PredictionIntercept,
                       gamma=0.25)

    # Create grid of hyperparams using uniform random sampling
    # epsilon = 1e-3 + (0.5 - 1e-3) * np.random.rand(20)
    # C = 10 ** (4 * np.random.rand(20))

    # Create grid of hyperparams using linear and logscale
    epsilon = np.linspace(0.01, 0.2, 25)
    # gamma = np.linspace(0.1, 1.5, 15)
    C = np.logspace(0.5, 2.5, 20)
    grid_params = {
        'epsilon': list(epsilon),
        # 'gamma': list(gamma),
        'C': list(C)
    }

    # Create GridSearch instance
    gs = GridSearch(fitter=gsvr)

    # Compute hyperparameters
    gs.fit(grid_parameters=grid_params, N=1, m=1000, score=score_f.statisticC_p,
           saveAllScores=True, filename="gsvr_C_vs_eps_Cp")

    # Plot error
    gs.plot_error()

    # Save results
    gs.store_results("gsvr_opt_C_vs_eps_Cp", verbose=True)


