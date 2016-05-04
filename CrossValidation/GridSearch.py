import csv
import itertools as it
from os.path import join

import numpy as np
import matplotlib.pyplot as plot
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from user_paths import RESULTS_DIR
import Utils.DataLoader as DataLoader
import score_functions


class GridSearch(object):
    """ Finds hyperparameters for a fitter using an exhaustive search over all the possible
     values the parameters can take, and assessing their results with a variety of
     score functions """

    def __init__(self, fitter):
        """

        Parameters
        ----------
        fitter : Fitters.CurveFitter
            Fitter instance whose hyperparams you want to find
        """
        # Init
        self._fitter = fitter
        self._total_error = 1000000000                      # total error initialization
        self._param_names = []                              # names of parameters
        self._param_values = []                             # values of parameters
        self._total_computations = 1                        # total computations per iteration
        self._optimal_params = {}                           # optimal parameters
        self._N = 1                                         # number of iterations
        self._m = 1                                         # number of randomly selected voxels
        self._num_params = 0                                # number of parameters to use for
                                                            # plotting the error
        self._errors_vector = []                            # vector into which the errors
                                                            # are stored

        # Init gm threshold
        self._gm_threshold = 0.01                           # grey matter threshold

        # Get data from Excel and nii files
        self._observations = DataLoader.getGMData(corrected_data=True)

        # Dimensions
        self._num_subjects = self._observations.shape[0]
        self._X_dim, self._Y_dim, self._Z_dim = self._observations.shape[1:]



    def fit(self, grid_parameters, N, m, degrees_of_freedom, score=score_functions.mse,
            saveAllScores=False, filename="xvalidation_scores"):
        """
        Fits the data for all combinations of the params (cartesian product) and returns the optimal
        value of all N iterations
        Parameters
        ----------
        grid_parameters : dict{param_name: param_values}
            Dictionary with keys being the name of the hyperparam to be found
            and values being a list of possible values the parameter can take
            (e.g {'C': [10, 100], 'epsilon': [0.5, 0.1, 0.01]} for Linear SVR)
        N : int
            Number of iterations
        m : int
            Number of randomly selected voxels (without repetition) for each iteration
        degrees_of_freedom : function
            function to calculate the degrees of freedom. Must follow the protype specified
            at degrees_of_freedom.py
        score : Optional[function]
            Score function used to decide the best selection of parameters.
            Default is MSE.
        saveAllScores : Optional[boolean]
            Whether to save all scores for all possible combinations of parameters of each iteration.
            Default is False
        filename : Optional[String]
            Name of the file where all the scores will be saved.
            Default is "xvalidation_scores".
        Returns
        -------
        dictionary{param_name: optimal_param_value}
            Dictionary with the names of the parameters and their found optimal values
        """
        # Assign variables
        self._N = N
        self._m = m

        # Map parameters into two lists
        for key,value in grid_parameters.iteritems():
            self._param_names.append(key)
            self._param_values.append(value)
            self._total_computations *= len(value)

        # Check number of parameters for error plotting
        if len(self._param_names) <= 0:
            raise Exception("There are no parameters to optimize")
        else:
            self._num_params = len(self._param_names)

        # Save all scores variable (if required)
        if saveAllScores:
            errors = [['Iteration'] + self._param_names + ['Error']]

        # Pre-assign errors for N iterations
        self._errors_vector = [[] for _ in range(N)]

        # Iterate N times
        for iteration in range(N):

            print
            print
            print "-------------------------"
            print "Iteration ", iteration+1, "of ", N
            print "-------------------------"

            # Select m voxels randomly
            m_counter = 0
            selected_voxels = []
            selected_voxels_var = []
            while m_counter < m:
                # Random voxel (uniform distribution,
                # with prior knowledge of voxels without interest,
                # thus limiting its span)
                x = np.random.randint(10, 110)
                y = np.random.randint(20, 130)
                z = np.random.randint(30, 100)
                voxel = (x, y, z)
                obs = self._observations[:, x, y, z]
                # Voxel variance
                mean = np.sum(obs) / self._num_subjects
                var = np.sum( (obs - mean) ** 2) / (self._num_subjects - 1)
                # Filter out voxels with low data variance
                if (var > self._gm_threshold) and (voxel not in selected_voxels):
                    # Select this voxel
                    selected_voxels.append(voxel)
                    selected_voxels_var.append(var)
                    m_counter += 1

            # Get gm from selected voxels
            current_observations = np.asarray([ self._observations[:, voxel[0], voxel[1], voxel[2]] \
                               for voxel in selected_voxels ]).T

            # Initialize progress for this iteration
            progress_counter = 0
            # Cartesian products between all the possible parameters
            for params in it.product(*self._param_values):
                # Create temporary dictionary to pass it to fitter as optional params
                tmp_params = {self._param_names[i]: params[i] \
                              for i in range(len(params))}
                # Show progress
                print "\r",
                print (float(progress_counter) / self._total_computations)*100, "%",
                # Fit data
                self._fitter.fit(current_observations, **tmp_params)
                # Predict data
                predicted = self._fitter.predict()
                # Degrees of freedom
                df = degrees_of_freedom(current_observations, self._fitter)
                # Score function
                score_value = score(current_observations, predicted, df)
                # Convert variances list to numpy array
                vars = np.array(selected_voxels_var)
                # Calculate total error (Score value weighted by the gm data variance)
                tmp_error = np.sum(score_value / vars)

                # Update error and optimal params
                if tmp_error < self._total_error:
                    self._total_error = tmp_error
                    self._optimal_params = tmp_params

                # Store errors if it is the last iteration for posterior error plotting
                self._errors_vector[iteration].append(tmp_error)

                # Save score if required
                if saveAllScores:
                    l_params = map(lambda x: round(x, 2), list(params))
                    errors.append([iteration+1] + l_params + [round(tmp_error, 2)])

                # Update progress
                progress_counter += 1

        # Save scores to file if required
        if saveAllScores:
            with open(join(RESULTS_DIR, filename + '.csv'), 'wb') as f:
                writer = csv.writer(f, delimiter=";")
                for row in errors:
                    writer.writerow(row)

        # Return found optimal parameters
        return self._optimal_params

    def store_results(self, filename, results_directory=RESULTS_DIR, verbose=False):
        """
        Store optimal parameters found in fit method
        Parameters
        ----------
        filename : str
            Name of the file where the optimal parameters will be stored.
            Extension is not necessary, as it will be stored as .txt
        results_directory : Optional[string]
            Path into the results directory where the file will be stored.
            Default is your RESULTS_DIR in nonlinear2.user_paths
        verbose : Optional[boolean]
            Whether to print the results to stdout after saving them.
            Default is False.
        """
        # Create string to store
        string =   "m=" + str(self._m) + " randomly selected voxels, " + \
                "N=" + str(self._N) + " iterations, " + \
                "and total_error=" + str(self._total_error) + ":\n\n"
        for key, value in self._optimal_params.iteritems():
            string += str(key) + " --> " + str(value) + "\n"

        # Store results
        with open(join(results_directory, filename + ".txt"), 'wb') as f:
            f.write(string)

        # Print final results if verbose
        if verbose:
            print
            print
            print string
            print

    def plot_error(self):
        """
        Plots the error with respect to the first parameter (2D plot) if there is only one,
        or the first two parameters (3D surface plot) if there are two or more parameters
        """
        # Check if there is any error vector
        if len(self._errors_vector) == 0:
            raise Exception("There is no errors vector for this instace of GridSearch. "
                            "Please use the 'fit' method before using the 'plot_error' one")

        # Get the mean of the N iterations
        self._errors_vector = np.mean(self._errors_vector, axis=0)

        # 2D plot case
        if self._num_params == 1:
            x = self._param_values[0]
            y = self._errors_vector
            if len(x) != len(y):
                raise Exception("Something strange happened here! :(")
            plot.plot(x, y)
            plot.xlabel(self._param_names[0])
            plot.ylabel("error")
            plot.show()
        elif self._num_params == 2:
            # Get the meshgrid
            X = self._param_values[0]
            Y = self._param_values[1]
            X, Y = np.meshgrid(X, Y)
            # Arrange the error matrix to match the X, Y dimensions
            Z = np.array(self._errors_vector).reshape(
                (len(self._param_values[1]), len(self._param_values[0]))
            )
            # Plot surface
            fig = plot.figure()
            ax = fig.gca(projection='3d')
            surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
            fig.colorbar(surf, shrink=0.5, aspect=5)

            plot.show()
        else:
            print
            print "Cannot draw the error curve or surface as there are more than 2 parameters"