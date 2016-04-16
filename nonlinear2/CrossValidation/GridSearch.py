from os.path import join
import numpy as np
import itertools as it

from nonlinear2.user_paths import RESULTS_DIR
import nonlinear2.Utils.DataLoader as dataloader
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

        # Init gm threshold
        self._gm_threshold = 0.01                           # grey matter threshold

        # Get data from Excel and nii files
        self._observations = dataloader.getGMData(corrected_data=True)

        # Dimensions
        self._num_subjects = self._observations.shape[0]
        self._X_dim, self._Y_dim, self._Z_dim = self._observations.shape[1:]



    def fit(self, grid_parameters, N, m, score=score_functions.mse):
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
            number of iterations
        m : int
            number of randomly selected voxels (without repetition) for each iteration
        score : Optional[function]
            Score function used to decide the best selection of parameters.
            Default is MSE.
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

        # Iterate N times
        for i in range(N):

            print
            print
            print "-------------------------"
            print "Iteration ", i+1, "of ", N
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
            # Cartesian products between all the possible parameters (C and epsilon)
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
                # Score function
                score_value = score(current_observations, predicted, self._num_subjects)
                # Convert variances list to numpy array
                vars = np.array(selected_voxels_var)
                # Calculate total error (Score value weighted by the gm data variance)
                tmp_error = np.sum(score_value / vars)

                # Update error and optimal params
                if tmp_error < self._total_error:
                    self._total_error = tmp_error
                    self._optimal_params = tmp_params

                # Update progress
                progress_counter += 1

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