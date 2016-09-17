"""
Script to validate the degrees of freedom implementation based on the paper
'On the Representer Theorem and Equivalent Degrees of Freedom of SVR', Francesco Dinuzzo et al.
(http://www.jmlr.org/papers/volume8/dinuzzo07a/dinuzzo07a.pdf)

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.svm import SVR


def b3_spline(x):
    """
    Cubic B-Spline Irwin-Hall distribution
    https://en.wikipedia.org/wiki/Spline_(mathematics)#Examples

    Parameters
    ----------
    x : numpy.array
        Input vector

    Returns
    -------
    numpy.array
        Output vector
    """
    y = np.zeros(x.shape)
    x0_range = np.logical_and(x >= -2, x < -1)
    x1_range = np.logical_and(x >= -1, x <= 1)
    x2_range = np.logical_and(x > 1, x <= 2)
    x0 = x[x0_range]
    x1 = x[x1_range]
    x2 = x[x2_range]
    y[x0_range] = 0.25 * np.power(x0 + 2, 3)
    y[x1_range] = 0.25 * (3 * np.power(np.abs(x1), 3) - 6 * np.power(x1, 2) + 4)
    y[x2_range] = 0.25 * np.power(2 - x2, 3)
    return y


def degrees_of_freedom(y_true, y_predicted, kernel_matrix, dual_coefficients, C, epsilon):
    # Create kernel diagonal
    kernel_diagonal = np.diag(kernel_matrix)

    # Compute pseudoresiduals (refer to F. Dinuzzo et al.
    # On the Representer Theorem and Equivalent Degrees of Freedom of SVR
    # [http://www.jmlr.org/papers/volume8/dinuzzo07a/dinuzzo07a.pdf]
    pseudoresiduals = y_true - y_predicted + dual_coefficients * kernel_diagonal

    # Compute effective degrees of freedom from pseudoresiduals
    min_value = epsilon * np.ones(pseudoresiduals.shape)
    max_value = min_value + C * kernel_diagonal
    comp_min = min_value <= np.abs(pseudoresiduals)
    comp_max = np.abs(pseudoresiduals) <= max_value
    df = np.sum(np.logical_and(comp_min, comp_max), axis=0)
    return df


def compute_kernel(x_data):
    # Kernel matrix computation (assume x data is a column vector)
    N = x_data.shape[0]
    X_horizontal = x_data.dot(np.ones((1, N)))
    X_vertical = np.ones((N, 1)).dot(x_data.T)
    X_diff = X_horizontal - X_vertical
    return b3_spline(X_diff)


if __name__ == "__main__":

    ''' CONSTANTS '''
    l = 64  # number of samples per dataset
    N = 100  # number of datasets
    grid_shape = (30, 30)  # shape of the surface grid (C, epsilon)
    noise_var = 0.09  # noise variance

    ''' GENERATE DATA '''
    x = np.array([(i - 1.0) / (l - 1) for i in range(1, l + 1)])
    x = np.atleast_2d(x).T
    f_0 = np.exp(np.sin(8 * x))
    noise = np.random.randn(l, N) * np.sqrt(noise_var)

    ''' COMPUTE CUSTOM KERNEL (CUBIC B-SPLINE) '''
    kernel = compute_kernel(x)

    ''' CALCULATE Cp ERROR SURFACE '''
    list_C = np.logspace(1, 3, grid_shape[0])
    list_epsilon = np.linspace(0.05, 0.5, grid_shape[1])
    cp_surf_shape = (grid_shape[0], grid_shape[1], N)
    Cp_surface = np.zeros(cp_surf_shape)

    for n in range(N):

        print
        print "----------------------------------------"
        print "\tIteration ", n + 1, " of ", N
        print "----------------------------------------"
        for i, C in enumerate(list_C):
            for j, epsilon in enumerate(list_epsilon):
                # Observations
                current_noise = np.atleast_2d(noise[:, n]).T
                y = np.ravel(f_0 + current_noise)

                # Create fitter and fit data
                svr = SVR(kernel='precomputed', tol=1e-6, C=C, epsilon=epsilon, cache_size=1024)
                svr.fit(kernel, y)

                # Get dual coefficients and process them
                dual_coeff = np.zeros(x.shape[0])
                dual_coeff[svr.support_] = np.ravel(svr.dual_coef_)

                # Get predicted values
                y_predicted = svr.predict(kernel)

                # Compute degrees of freedom
                df = degrees_of_freedom(y, y_predicted, kernel, dual_coeff, C, epsilon)

                # Compute Cp statistic and store it in mesh
                err = (1.0 / l) * (np.linalg.norm(y - y_predicted) ** 2)
                # Use noise variance, but in a real world example, noise variance would be estimated
                cp_val = err + (2.0 / l) * df * noise_var
                Cp_surface[i, j, n] = cp_val

    """ FIND BEST PARAMS """
    # Get mean Cp surface across all datasets
    Z = np.mean(Cp_surface, axis=2)
    C_ind, epsilon_ind = np.unravel_index(Z.argmin(), Z.shape)
    cp_value = Z[C_ind, epsilon_ind]
    print
    print "Index of best params: ", (C_ind, epsilon_ind)
    print "C (optimal) --> ", list_C[C_ind]
    print "epsilon (optimal) --> ", list_epsilon[epsilon_ind]
    print "Cp value (optimal) --> ", cp_value
    print

    """ PRINT Cp STATISTIC SURFACE """
    # Get the meshgrid
    X, Y = np.meshgrid(np.log10(list_C), list_epsilon)
    # Plot surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
