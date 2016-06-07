import CrossValidation.score_functions as score_f
import Utils.DataLoader as DataLoader
import numpy as np
from os.path import join
from Fitters.CurveFitting import AdditiveCurveFitter
from CrossValidation.GridSearch import GridSearch
from Utils.Subject import Subject
from Fitters.SVR import PolySVR
from user_paths import RESULTS_DIR

RESULTS_DIR = join(RESULTS_DIR, 'CROSS_VALIDATION')

if __name__ == "__main__":

    # Get features
    predictors = DataLoader.getFeatures([Subject.ADCSFIndex])

    # Creat PolySVR instance
    psvr = PolySVR(predictors, [0], [3], AdditiveCurveFitter.PredictionIntercept)

    # Create grid of hyperparams using uniform random sampling
    epsilon = np.sort(np.random.uniform(0.05, 0.08, 15))
    C = np.sort([10 ** i for i in np.random.uniform(0.1, 0.6, 10)])

    # Create grid of hyperparams using linear and logscale
    # epsilon = np.linspace(0.01, 0.5, 15)
    # C = np.logspace(0.01, 3, 15)
    grid_params = {
        'epsilon': list(epsilon),
        'C': list(C)
    }

    # Create GridSearch instance
    gs = GridSearch(fitter=psvr, results_directory=RESULTS_DIR, n_jobs=6)

    # Compute hyperparameters
    gs.fit(grid_parameters=grid_params, N=3, m=200, score=score_f.anova_error,
           save_all_scores=True, filename="psvr_anova_fine_scores")

    # Save results
    gs.store_results("psvr_anova_fine_optimal", verbose=True)

    # Plot error
    gs.plot_error()
