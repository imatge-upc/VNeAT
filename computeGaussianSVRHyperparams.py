from os.path import join

from user_paths import RESULTS_DIR
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
                       gamma=0.5)

    # Create grid of hyperparams using uniform random sampling
    epsilon = np.sort(np.random.uniform(0.01, 0.2, 20))
    C = np.sort([10 ** i for i in np.random.uniform(0, 3, 20)])
    gamma = np.sort(np.random.uniform(0.1, 0.7, 20))

    # Create grid of hyperparams using linear and logscale
    # epsilon = np.linspace(0.01, 0.2, 25)
    # gamma = np.linspace(0.1, 1.5, 15)
    # C = np.logspace(0.5, 2.5, 20)
    grid_params = {
        'epsilon': list(epsilon),
        'gamma': list(gamma),
        'C': list(C)
    }

    # Create GridSearch instance
    gs = GridSearch(fitter=gsvr)

    # Compute hyperparameters
    gs.fit(grid_parameters=grid_params, N=1, m=500, score=score_f.statisticC_p,
           saveAllScores=False, filename=join(RESULTS_DIR, 'GSVR', "gsvr_"))

    # Plot error
    # gs.plot_error()

    # Save results
    gs.store_results("gsvr_opt", results_directory=join(RESULTS_DIR, 'GSVR'), verbose=True)


