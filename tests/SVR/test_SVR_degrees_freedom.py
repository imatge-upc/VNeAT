from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np

""" FUNCTIONS """

def compute_pseudoresiduals(y_true, y_predicted, X, support_vector_index, dual_coeff, kernel):
    # Prepare dual coefficients
    dual_coeff = dual_coeff.ravel()
    coeff = np.zeros(y_true.shape)
    coeff[support_vector_index] = dual_coeff

    # Compute kernel matrix diagonal --> K(x_i, x_i)
    if kernel=='linear':
        kernel_diag = np.diag(X.dot(X.T))
    elif kernel=='rbf':
        kernel_diag = np.ones(coeff.shape)
    else:
        raise Exception("A kernel name (linear or rbf) must be provided")

    return kernel_diag, y_true - y_predicted + coeff * kernel_diag

def effective_df(X, y, fitter):
    kernel_diag, pseudoresiduals = compute_pseudoresiduals(
        y,
        fitter.predict(X),
        X,
        fitter.support_,
        fitter.dual_coef_,
        fitter.kernel
    )

    C = fitter.C
    epsilon = fitter.epsilon

    df = 0
    min_value = epsilon * np.ones(pseudoresiduals.shape)
    max_value = min_value + C * kernel_diag
    comp_min = min_value <= np.abs(pseudoresiduals)
    comp_max = np.abs(pseudoresiduals) <= max_value

    return np.sum(np.logical_and(comp_min, comp_max))


""" MAIN SCRIPT """

if __name__ == "__main__":

    # Get artificial data
    print("Getting artificial data...")
    X1 = np.sort(8 * (np.random.rand(100, 1) - 0.5), axis=0)
    X2 = np.sort(2 * np.random.rand(100, 1), axis=0)
    X = np.concatenate((X1, X2), axis=1)
    y = X1 + np.exp(X1 * np.sin(X1)) + 2 * X2
    y += 0.5 * np.random.randn(100, 1)
    y = y.ravel()
    # y = np.atleast_2d(y)

    # Exploratory Grid Search
    C_vals = [1, 10, 100, 1000]
    epsilon_vals = [0.01, 0.1, 0.25, 0.5]
    n_jobs = 1

    for C in C_vals:
        for epsilon in epsilon_vals:

            print("PARAMS: ")
            print("C --> " + str(C))
            print("epsilon --> " + str(epsilon))

            """ PART 1: ARTIFICIAL DATA """
            # Init Polynomial SVR fitters
            print("Creating SVR fitter for artificial data...")
            fitter = SVR(kernel='rbf', gamma=0.5, C=C, epsilon=epsilon)
            # Fit data
            print("Fitting artificial data...")
            fitter.fit(X, y)
            # Predict
            predicted = fitter.predict(X)
            # Compute degrees of freedom
            df = effective_df(X, y, fitter)
            print "Degrees of freedom: " + str(df)
            # Plot prediction
            print("Plotting curves in dimension 1...")
            plt.scatter(X[:, 0], y, c='r', label='Original data in dimension 1')
            plt.plot(X[:, 0], predicted, c='b', label='Poly SVR prediction')
            plt.xlabel('data')
            plt.ylabel('target')
            plt.title('Polynomial Support Vector Regression')
            plt.legend()
            plt.show()
            print("Plotting curves in dimension 2...")
            plt.scatter(X[:, 1], y, c='r', label='Original data in dimension 1')
            plt.plot(X[:, 1], predicted, c='b', label='Poly SVR prediction')
            plt.xlabel('data')
            plt.ylabel('target')
            plt.title('Polynomial Support Vector Regression')
            plt.legend()
            plt.show()




