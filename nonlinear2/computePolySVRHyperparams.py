from CrossValidation.GridSearch import GridSearch
from Fitters.SVR import PolySVR
from Utils.Subject import Subject
import Utils.DataLoader as DataLoader

if __name__ == "__main__":

    # Get features
    predictors = DataLoader.getFeatures([Subject.ADCSFIndex])

    # Creat PolySVR instance
    psvr = PolySVR(predictors, [0], [3], False)

    # Create grid of hyperparams
    grid_params = {
        'C': [50, 75, 100, 125, 250],
        'epsilon': [0.05, 0.1, 0.15, 0.2],
    }

    # Create GridSearch instance
    gs = GridSearch(fitter=psvr)

    # Compute hyperparameters
    gs.fit(grid_parameters=grid_params, N=50, m=50)

    # Save results
    gs.store_results("psvr_hyperparams", verbose=True)
