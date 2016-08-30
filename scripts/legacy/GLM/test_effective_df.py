import numpy as np
from nonlinear2.Utils.Subject import Subject

if __name__ == "__main__":
    # Voxel
    voxel = (82, 74, 39)

    # Load Aetionomy data
    X = DataLoader.getFeatures([Subject.Sex, Subject.Age, Subject.ADCSFIndex])
    y = DataLoader.getGMData(corrected_data=False)[:, voxel[0], voxel[1], voxel[2]]

    y = np.atleast_2d(y).T

    # Normal equation
    HAT = X.dot(np.linalg.inv(X.T.dot(X)).dot(X.T))
    y_hat = HAT.dot(y)

    # Effective degrees of freedom (using Hat definition)
    df = np.trace(HAT)
    print df

    # Effective degrees of freedom (using covariance definition)
    N = X.shape[0]
    y_mean = np.mean(y, axis=0)
    y_hat_mean = np.mean(y_hat, axis=0)
    error = y - y_hat
    error_mean = np.mean(error, axis=0)
    error_var = (error - error_mean).T.dot(error - error_mean) / (N - 1)
    cov = np.trace((y - y_mean).T.dot(y_hat - y_hat_mean))
    df = cov / error_var
    print df
