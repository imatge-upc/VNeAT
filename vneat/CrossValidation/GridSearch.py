from __future__ import print_function

import csv
import itertools as it
from os.path import join

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plot
import numpy as np
from matplotlib import cm

import score_functions


class GridSearch(object):
    """
    Finds hyperparameters for a fitter using an exhaustive search over all the possible
    values the parameters can take, and assessing their results with a variety of
    score functions
    """

    def __init__(self, processor, results_directory, voxel_offset=20, n_jobs=4):
        """
        Parameters
        ----------
        processor : Processors.Processing
            Processor instance with the fitter whose hyperparameters you want to find
        voxel_offset : int
            Number of voxels that will not be taken into account in all directions, both at the
            beginning and at the end. That is, for a voxel offset of v, only the following voxels
            will be taken into account: (v:x_dim-v, v:y_dim-v, v:z_dim-v)
        results_directory : String
            Path into the results directory where the file will be stored
        """
        # Init
        self._processor = processor
        self._fitter = processor.fitter
        self._results_dir = results_directory
        self._n_jobs = n_jobs  # number of parallel jobs used to fit
        self._total_error = 1000000000  # total error initialization
        self._param_names = []  # names of parameters
        self._param_values = []  # values of parameters
        self._total_computations = 1  # total computations per iteration
        self._optimal_params = {}  # optimal parameters
        self._N = 1  # number of iterations
        self._m = 1  # number of randomly selected voxels
        self._num_params = 0  # number of parameters to use for plotting the error
        self._errors_vector = []  # vector into which the errors are stored

        # Init gm threshold
        self._gm_threshold = 0.01  # grey matter threshold

        # Image boundaries (in voxels)
        self._img_shape = self._processor.image_shape
        start_boundary = ()
        end_boundary = ()
        for dim in self._img_shape:
            start_boundary += (voxel_offset,)
            end_boundary += (dim - voxel_offset,)

        # Observations
        print()
        print('Initializing GridSearch and loading observations...')
        self._obs = self._processor.gm_values(
            x1=start_boundary[0],
            x2=end_boundary[0],
            y1=start_boundary[1],
            y2=end_boundary[1],
            z1=start_boundary[2],
            z2=end_boundary[2],
        )

    def fit(self, grid_parameters, N, m, score=score_functions.mse, filename="xvalidation_scores"):
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
        score : [Optional]function
            Score function used to decide the best selection of parameters.
            Default is MSE.
        filename : [Optional]String
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
        for key, value in grid_parameters.iteritems():
            self._param_names.append(key)
            self._param_values.append(value)
            self._total_computations *= len(value)

        # Check number of parameters for error plotting
        if len(self._param_names) <= 0:
            raise Exception("There are no parameters to optimize")
        else:
            self._num_params = len(self._param_names)

        # Save all scores variable (if required)
        errors = [['Iteration'] + self._param_names + ['Error']]

        # Pre-assign errors for N iterations
        self._errors_vector = [[] for _ in range(N)]

        # Iterate N times
        for iteration in range(N):

            print()
            print()
            print("-------------------------")
            print("Iteration {} of {}".format(iteration + 1, N))
            print("-------------------------")

            # Select m voxels randomly
            print()
            print('Searching valid voxels...')
            m_counter = 0
            selected_voxels = []
            selected_voxels_var = []
            while m_counter < m:
                # Random voxel (uniform distribution,
                # with prior knowledge of voxels without interest,
                # thus limiting its span)
                x = np.random.randint(0, self._obs.shape[1])
                y = np.random.randint(0, self._obs.shape[2])
                z = np.random.randint(0, self._obs.shape[3])
                voxel = (x, y, z)
                # Voxel variance
                current_obs = self._obs[:, x, y, z]
                var = np.var(current_obs)
                # Filter out voxels with low data variance
                if (var > self._gm_threshold) and (voxel not in selected_voxels):
                    # Select this voxel
                    selected_voxels.append(voxel)
                    selected_voxels_var.append(var)
                    m_counter += 1
                    print("\r{} valid voxels of {}".format(m_counter, m), end="")
            print()

            # Get gm from selected voxels
            current_observations = np.asarray(
                [self._obs[:, voxel[0], voxel[1], voxel[2]] for voxel in selected_voxels]
            ).T

            # Convert variances list to numpy array
            variances = np.array(selected_voxels_var, dtype=np.float32)

            # Initialize progress for this iteration
            progress_counter = 0
            # Cartesian products between all the possible parameters
            for params in it.product(*self._param_values):
                # Create temporary dictionary to pass it to fitter as optional params
                tmp_params = {self._param_names[i]: params[i] for i in range(len(params))}
                # Show progress
                current_progress = (float(progress_counter) / self._total_computations) * 100
                current_progress_percentage = '{} %'.format(current_progress)
                print("\rProgress: ", end="")
                print(current_progress_percentage, end="")
                # Fit data
                self._fitter.fit(current_observations, n_jobs=self._n_jobs, **tmp_params)
                # Predict data
                predicted = self._fitter.predict()
                # Degrees of freedom
                df = self._fitter.df_prediction(current_observations)
                # Score function
                score_value = score(current_observations, predicted, df)
                # Calculate total error (Score value weighted by the gm data variance)
                tmp_error = np.sum(score_value / variances)

                # Update error and optimal params
                if tmp_error < self._total_error:
                    self._total_error = tmp_error
                    self._optimal_params = tmp_params

                # Store errors if it is the last iteration for posterior error plotting
                self._errors_vector[iteration].append(tmp_error)

                # Save scores
                l_params = map(lambda x: round(x, 2), list(params))
                errors.append(
                    ['#{} / {}'.format(iteration + 1, current_progress_percentage)] +
                    l_params +
                    [round(tmp_error, 2)]
                )
                # Update progress
                progress_counter += 1

            # Show final progress
            current_progress = (float(progress_counter) / self._total_computations) * 100
            current_progress_percentage = '{} %'.format(current_progress)
            print("\rProgress: ", end="")
            print(current_progress_percentage)

        # Save scores to file if required
        with open(join(self._results_dir, filename + '.csv'), 'wb') as f:
            writer = csv.writer(f, delimiter=";")
            for row in errors:
                writer.writerow(row)

        # Return found optimal parameters
        return self._optimal_params

    def store_results(self, filename):
        """
        Store optimal parameters found in fit method

        Parameters
        ----------
        filename : str
            Name of the file where the optimal parameters will be stored.
            Extension is not necessary, as it will be stored as .txt
        verbose : Optional[boolean]
            Whether to print the results to stdout after saving them.
            Default is False.
        """
        # Create string to store
        string = [
            "m = {} randomly selected voxels".format(self._m),
            "N = {} iterations".format(self._N),
            "Total error = {}".format(self._total_error),
            "",
            " PARAMETERS",
            " ---------- ",
            ""
        ]
        for key, value in self._optimal_params.iteritems():
            string.append('{} --> {}'.format(key, value))

        # Store results
        with open(join(self._results_dir, filename + ".txt"), 'wb') as f:
            f.write("\n".join(string))

        print()
        print()
        print("\n".join(string))
        print()

    def plot_error(self, plot_name):
        """
        Plots the error with respect to the first parameter (2D plot) if there is only one,
        or the first two parameters (3D surface plot) if there are two parameters

        Parameters
        ------
        plot_name : String
            Name that will be given to the PNG image stored with the plot
        """
        # Check if there is any error vector
        if len(self._errors_vector) == 0:
            raise Exception("There is no errors vector for this instace of GridSearch. "
                            "Please use the 'fit' method before using the 'plot_error' one")

        # Path to store the plot
        plot_path = join(self._results_dir, plot_name)

        # Convert it to numpy array
        self._errors_vector = np.asarray(self._errors_vector)

        # Filter nans
        without_nans = self._errors_vector[np.isfinite(self._errors_vector)]
        if without_nans.size == 0:
            print('There are no numeric errors. Try to change the search span of the hyperparameters '
                  'so that the fitting is more accurate and the error function, therefore, returns '
                  'numeric values.')
            return
        self._errors_vector[~np.isfinite(self._errors_vector)] = without_nans.max()

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
            plot.savefig(plot_path)
            plot.close()
        elif self._num_params == 2:
            # Get the meshgrid
            X = self._param_values[0]
            Y = self._param_values[1]
            X, Y = np.meshgrid(X, Y)
            # Arrange the error matrix to match the X, Y dimensions
            Z = np.array(self._errors_vector).reshape(
                (len(self._param_values[0]), len(self._param_values[1]))
            ).T
            # Plot surface
            fig = plot.figure()
            ax = fig.gca(projection='3d')
            surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)
            fig.colorbar(surf, shrink=0.5, aspect=5)

            plot.savefig(plot_path)
            plot.close()
        else:
            print()
            print("Cannot draw the error curve or surface as there are more than 2 parameters")
