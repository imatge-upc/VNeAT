from os.path import join
import CrossValidation.score_functions as score_f
import Utils.DataLoader as DataLoader
import numpy as np
from Fitters.CurveFitting import AdditiveCurveFitter
from CrossValidation.GridSearch import GridSearch
from Utils.Subject import Subject
from Fitters.SVR import GaussianSVR
from user_paths import RESULTS_DIR

RESULTS_DIR = join(RESULTS_DIR, 'CROSS_VALIDATION')

if __name__ == "__main__":

    # Get features
    predictors = DataLoader.getFeatures([Subject.ADCSFIndex])

    # Creat PolySVR instance
    gsvr = GaussianSVR(predictors=predictors, correctors=None,
                       intercept=AdditiveCurveFitter.PredictionIntercept,
                       gamma=0.3)

    # Create grid of hyperparams using uniform random sampling
    epsilon = np.sort(np.random.uniform(0.01, 0.2, 10))
    C = np.sort([10 ** i for i in np.random.uniform(0, 1.5, 5)])
    # gamma = np.sort(np.random.uniform(0.1, 0.7, 5))

    # Create grid of hyperparams using linear and logscale
    # epsilon = np.linspace(0.01, 0.1, 15)
    # gamma = np.linspace(0.05, 1.5, 5)
    # C = np.logspace(0.1, 1.5, 15)
    grid_params = {
        'epsilon': list(epsilon),
        # 'gamma': list(gamma),
        'C': list(C)
    }

    # Create GridSearch instance
    gs = GridSearch(fitter=gsvr, results_directory=RESULTS_DIR, n_jobs=7)

    # Compute hyperparameters
    gs.fit(grid_parameters=grid_params, N=5, m=100, score=score_f.anova_error,
           save_all_scores=True, filename='gsvr_anova_scores')

    # Save results
    gs.store_results("gsvr_anova_optimal", verbose=True)

    # Plot error
    gs.plot_error()

