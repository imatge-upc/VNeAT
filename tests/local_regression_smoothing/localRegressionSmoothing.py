import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import time

import Utils.DataLoader as DataLoader
from Utils.Subject import Subject

"""
LOCAL REGRESSION SMOOTHING (LOWESS/LOESS)
http://es.mathworks.com/help/curvefit/smoothing-data.html#bq_6ys3-3
"""


def regression_weights(data, x, span):
    """
    Returns a list of weights, one for each data point, with respect to x (the point
    to "predict"), following the formula found in the link above
    """
    partial = np.abs((x - data) / span) ** 3
    w = (1 - partial) ** 3
    ind = np.logical_or(data > x + span, data < x - span)
    w[ind] = 0
    return w


def compute_span(data, x, span_percentage):
    span_percentage = float(span_percentage)
    ordered_data = np.sort(data)
    min_value = ordered_data.min()
    max_value = ordered_data.max()
    span_distance = (max_value - min_value) * (span_percentage / 100)
    ind = np.logical_and(ordered_data <= x + span_distance, ordered_data >= x - span_distance)
    valid_data = ordered_data[ind]
    max_positive_dist = valid_data.max() - x
    max_negative_dist = x - valid_data.min()
    return max_positive_dist if max_positive_dist > max_negative_dist else max_negative_dist


def smooth(X, y, span_percentage, option='LOWESS'):
    lowess_option = 'LOWESS'
    loess_option = 'LOESS'
    if option != lowess_option and option != loess_option:
        raise Exception("You must provide one of the following options for the local regression "
                        "smoothing: "
                        "     'LOWESS': first-order regression"
                        "     'LOESS: second-order regression'")

    predictions = np.zeros(X.shape[0])
    counter = 0
    for predictor_value in X:
        span = compute_span(X, predictor_value, span_percentage)
        weights = regression_weights(X, predictor_value, span)
        # Weighted Least Squares
        if option == loess_option:
            training_data = np.concatenate((X, X ** 2), axis=1)
        else:
            training_data = X
        wls = sm.WLS(y, training_data, weights)
        res_wls = wls.fit()
        params = res_wls.params
        if option == loess_option:
            prediction = np.asarray([predictor_value, predictor_value ** 2]).T.dot(params)
        else:
            prediction = predictor_value * params
        predictions[counter] = prediction
        counter += 1

    return np.ravel(predictions)


""" MAIN SCRIPT """

if __name__ == "__main__":

    # Input from user
    show_artificial = raw_input("Show artificial data fitting (Y/N, default is N): ")
    voxel = eval(raw_input("Voxel to be fitted (use the following input format: X, Y, Z): "))

    # Coordinates of the voxels to fit
    x1 = voxel[0]
    x2 = x1+1
    y1 = voxel[1]
    y2 = y1+1
    z1 = voxel[2]
    z2 = z1+1

    if show_artificial == 'Y':
        # Get artificial data
        print("Getting artificial data...")
        X = np.sort(5 * (np.random.rand(100, 1) - 0.5), axis=0)
        y = X + np.exp(X * np.sin(X))
        y[::5] += 5 * (0.5 - np.random.rand(20, 1))
        y = np.atleast_2d(y)


    # Get data from Excel and nii files
    print("Loading Aetionomy data...")
    observations = DataLoader.getGMData(corrected_data=True)
    aet_regressors = DataLoader.getFeatures([Subject.ADCSFIndex])
    real_obs = np.ravel(observations[:, x1:x2, y1:y2, z1:z2])
    del observations

    # Order data
    reg, obs = [np.asarray(list(t)) for t in zip(*sorted(zip(aet_regressors, real_obs)))]

    # Options
    smoothing_methods = ['LOWESS', 'LOESS']
    span_percentages = [5, 10, 20, 40]

    for smoothing_method in smoothing_methods:
        for span_percentage in span_percentages:

            """ PART 1: ARTIFICIAL DATA """
            if show_artificial == 'Y':
                # Fit data
                print("Smoothing artificial data...")
                smoothed = smooth(X, y, span_percentage, option=smoothing_method)
                # Plot prediction
                print("Plotting curves...")
                plt.scatter(X, y, c='r', label='Original data')
                plt.plot(X, smoothed, c='g', label='Gaussian SVR prediction')
                plt.xlabel('data')
                plt.ylabel('target')
                plt.title('LOWESS Smoothing')
                plt.legend()
                plt.show()

            """ PART 2: AETIONOMY DATA """
            # Fit data
            print("Fitting Aetionomy data...")
            start_time = time.clock()
            # Scatter smoothing
            smoothed = smooth(reg, obs, span_percentage, option=smoothing_method)
            end_time = time.clock()

            # Print execution info
            print("Smoothing took " +
                  str(end_time - start_time) + " s")
            print "Smoothing method --> ", smoothing_method
            print "Span percentage --> ", span_percentage, "% "
            print

            # Plot fitting curves
            print("Plotting curves...")
            plt.scatter(reg, obs, c='k', label='Original Data')
            plt.plot(reg, smoothed, c='y', lw=2, marker="D", label='Local Regression Smoothing')
            plt.xlabel('data')
            plt.ylabel('target')
            # Title for figure
            title = 'voxel (' + str(x1) + ', ' + str(y1) + ', ' + str(z1) + ')  '
            plt.title(title)
            plt.legend()
            plt.show()