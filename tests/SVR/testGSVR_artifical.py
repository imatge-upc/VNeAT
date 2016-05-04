import numpy as np
import matplotlib.pyplot as plt

from Fitters.SVR import CurveFitter, GaussianSVR as GSVR

if __name__ == "__main__":

    # Get artificial data
    print("Getting artificial data...")
    X = np.sort(5 * (np.random.rand(100, 1) - 0.5), axis=0)
    y1 = X + np.exp(X * np.sin(X))
    y2 = X ** 3 - X**2 + 0.5*X
    y1 += 1.5 * np.random.randn(100, 1)
    y2 += 2.5 * np.random.randn(100, 1)
    Y = np.zeros((y1.shape[0], y1.shape[1] + y2.shape[1]))
    Y[:, 0] = np.ravel(y1)
    Y[:, 1] = np.ravel(y2)

    # Exploratory Grid Search
    C_vals = [10, 50]
    epsilon_vals = [0.1, 0.25]
    gamma_vals = [0.1, 0.25]
    n_jobs = 1

    for C in C_vals:
        for epsilon in epsilon_vals:
            for gamma in gamma_vals:

                print "PARAMS: "
                print "C --> ", C
                print "epsilon --> ", epsilon
                print "gamma --> ", gamma

                """ PART 1: ARTIFICIAL DATA """
                # Init Polynomial SVR fitters
                print("Creating SVR fitter for artificial data...")
                fitter = GSVR(predictors=X, intercept=CurveFitter.PredictionIntercept)
                # Fit data
                print("Fitting artificial data...")
                fitter.fit(Y, C=C, epsilon=epsilon, gamma=gamma)
                # Correct
                corr_Y = fitter.correct(Y)
                # Predict
                predicted = fitter.predict()
                # Plot prediction
                print("Plotting curves for first variable Y1...")
                plt.scatter(X, Y[:, 0], c='r', label='Original data')
                plt.scatter(X, corr_Y[:, 0], c='g', label='Corrected data')
                plt.plot(X, predicted[:, 0], c='b', label='Poly SVR prediction')
                plt.xlabel('data')
                plt.ylabel('target')
                plt.title('Support Vector Regression with Gaussian kernel')
                plt.legend()
                plt.show()
                print("Plotting curves ifor second variable Y2...")
                plt.scatter(X, Y[:, 1], c='r', label='Original data')
                plt.scatter(X, corr_Y[:, 1], c='g', label='Corrected data')
                plt.plot(X, predicted[:, 1], c='b', label='Poly SVR prediction')
                plt.xlabel('data')
                plt.ylabel('target')
                plt.title('Polynomial Support Vector Regression')
                plt.legend()
                plt.show()