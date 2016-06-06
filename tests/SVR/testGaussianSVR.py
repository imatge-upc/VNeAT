import time

import matplotlib.pyplot as plt
import numpy as np
from Utils.DataLoader import getGMData, getFeatures
from Utils.Subject import Subject

from Fitters.CurveFitting import CurveFitter
from Fitters.SVR import GaussianSVR as GSVR

if __name__ == "__main__":

    # Input from user
    show_artificial = raw_input("Show artificial data fitting (Y/N, default is N): ")
    voxel = eval(raw_input("Voxel to be fitted (use the following input format: X, Y, Z): "))

    # Coordinates of the voxels to fit
    x1 = voxel[0] #44
    x2 = x1+1
    y1 = voxel[1] #82
    y2 = y1+1
    z1 = voxel[2] #39
    z2 = z1+1

    if show_artificial == 'Y':
        # Get artificial data
        print("Getting artificial data...")
        X = np.sort(6 * np.random.rand(50, 1), axis=0)
        y = 0.5 + X + 0.8*X ** 2 - X ** 3
        y[::5] += 5 * (0.5 - np.random.rand(10, 1))
        y = np.atleast_2d(y)

        # Init Polynomial SVR fitters
        print("Creating SVR fitter for artificial data...")
        artificial_svr = GSVR(predictors=X, intercept=CurveFitter.PredictionIntercept)

    # Get data from Excel and nii files
    print("Loading Aetionomy data...")
    observations = getGMData(corrected_data=True)
    aet_regressors = getFeatures([Subject.ADCSFIndex])
    real_obs = observations[:, x1:x2, y1:y2, z1:z2]
    del observations

    # LinSVR fitter
    print("Creating SVR fitter for Aetionomy data...")
    aet_svr = GSVR(predictors=aet_regressors, intercept=CurveFitter.PredictionIntercept)

    # Exploratory Grid Search
    C_vals = [1.6, 2.33]
    epsilon_vals = [0.06, 0.07]
    gamma_vals = [0.3]
    n_jobs = 1

    for C in C_vals:
        for epsilon in epsilon_vals:
            for gamma in gamma_vals:

                print("PARAMS: ")
                print("C --> " + str(C))
                print("epsilon --> " + str(epsilon))
                print "gamma --> ", gamma

                """ PART 1: ARTIFICIAL DATA """
                if show_artificial == 'Y':
                    # Fit data
                    print("Fitting artificial data...")
                    artificial_svr.fit(y, C=C, epsilon=epsilon, gamma=gamma, n_jobs=1)
                    # Plot prediction
                    print("Plotting curves...")
                    plt.scatter(X, y, c='r', label='Original data')
                    poly_predicted = artificial_svr.predict()
                    plt.plot(X, poly_predicted, c='g', label='Gaussian SVR prediction')
                    plt.xlabel('data')
                    plt.ylabel('target')
                    plt.title('Gaussian Support Vector Regression')
                    plt.legend()
                    plt.show()

                """ PART 2: AETIONOMY DATA """

                # Fit data
                print("Fitting Aetionomy data...")
                dims = real_obs.shape
                num_voxels = dims[1]*dims[2]*dims[3]
                reshaped_obs = real_obs.reshape((dims[0], num_voxels))
                start_time = time.clock()
                # Fit PolySVR to original data to correct it
                aet_svr.fit(reshaped_obs, C=C, epsilon=epsilon, gamma=gamma, n_jobs=n_jobs)
                end_time = time.clock()

                # Print execution info
                print("Using the following parameters for the SVR fitter the fitting time was " +
                      str(end_time - start_time) + " s")
                print("\tC: " + str(C))
                print("\tepsilon: " + str(epsilon))
                print("\tgamma: " + str(gamma))
                print("\t# processes: " + str(n_jobs))
                print("\t# voxels fitted: " + str(num_voxels))
                print "\t# degrees of freedom: ", aet_svr.df_prediction(reshaped_obs)

                # Generate x data for first fitted voxel [:, 0]
                reg = aet_regressors[:, 0]
                x = np.atleast_2d(np.linspace(min(reg), max(reg), 100)).T

                # Plot fitting curves
                print("Plotting curves...")
                plt.scatter(reg, reshaped_obs[:, 0], c='k', label='Original Data')
                predicted = aet_svr.predict(x)
                plt.plot(x, predicted, c='y', label='Gaussian SVR prediction')
                plt.xlabel('data')
                plt.ylabel('target')
                # Title for figure
                title = 'voxel (' + str(x1) + ', ' + str(y1) + ', ' + str(z1) + ')  '
                title = title + 'C = ' + str(C) + ' / epsilon = ' + str(epsilon) + ' / gamma = ' + str(gamma)
                plt.title(title)
                plt.legend()
                plt.show()

