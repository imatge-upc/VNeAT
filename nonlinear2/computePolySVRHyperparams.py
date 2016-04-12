import numpy as np
import itertools as it
from os.path import join
from nonlinear2.Utils.Subject import Subject
from nonlinear2.Utils.DataLoader import getGMData, getFeatures
from nonlinear2.Fitters.SVR import PolySVR as PSVR
from nonlinear2.user_paths import RESULTS_DIR

if __name__ == "__main__":

    """ LOAD DATA """

    # Get data from Excel and nii files
    print "Getting data from NIFTI files..."
    regressor = getFeatures([Subject.ADCSFIndex])
    observations = getGMData(corrected_data=True)

    # Dimensions
    num_subjects = observations.shape[0]
    X_dim, Y_dim, Z_dim = observations.shape[1:]

    """ PARAMS """
    N = 1                                       # number of iterations
    m = 100                                     # number of voxels to select randomly in each iteration
    gm_threshold = 0.1                          # grey matter threshold used to filter voxels with no gm
    params = {
        'C': np.logspace(0, 3.5, 10),
        'epsilon': np.linspace(1e-5, 1e-2, 10)
    }

    """ INITIALIZATION """
    optim_params = { 'C': -1, 'epsilon': -1 }   # optimal params initialization
    total_error = 1000000000                    # total error initialization
    fitter = PSVR(regressor, 0, [3], False)     # PolySVR fitter with 3rd order polynomic
    total_computations = len(params['C']) * len(params['epsilon'])

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
            x = np.random.randint(0, X_dim)
            y = np.random.randint(0, Y_dim)
            z = np.random.randint(0, Z_dim)
            voxel = (x, y, z)
            obs = observations[:, x, y, z]
            # Filter out voxels where all gm values are below the threshold
            abs_gm = np.abs(obs)
            filtered_obs = abs_gm > gm_threshold
            if (np.any(filtered_obs)) \
            and (voxel not in selected_voxels):
                # Select this voxel
                selected_voxels.append(voxel)
                # Calculate its variance
                mean = np.sum(obs) / num_subjects
                var = np.sum( (obs - mean) ** 2) / (num_subjects - 1)
                selected_voxels_var.append(var)
                m_counter += 1

        # Get gm from selected voxels
        current_observations = np.asarray([ observations[:, voxel[0], voxel[1], voxel[2]] \
                           for voxel in selected_voxels ]).T

        # Initialize progress for this iteration
        progress_counter = 0
        # Cartesian products between all the possible parameters (C and epsilon)
        for C, epsilon in it.product(params['C'], params['epsilon']):
            # Show progress
            print "\r",
            print (float(progress_counter) / total_computations)*100, "%",
            # Fit data with C and epsilon params
            fitter.fit(current_observations, C=C, epsilon=epsilon)
            # Predict data
            predicted = fitter.predict()
            # Compute MSE for each voxel
            sum_SE = np.sum(np.square(predicted - current_observations), axis=0)
            MSE = sum_SE / num_subjects
            # Convert variances list to numpy array
            vars = np.array(selected_voxels_var)
            # Calculate total error (MSE weighted by the gm data variance)
            tmp_error = np.sum(MSE / vars)

            # Update error and optimal params
            if tmp_error < total_error:
                total_error = tmp_error
                optim_params['C'] = C
                optim_params['epsilon'] = epsilon

            # Update progress
            progress_counter += 1

    # Print final results
    print
    print
    print "Total error: ", total_error
    print "Optimal parameters: "
    print "    C --> ", optim_params['C']
    print "    epsilon --> ", optim_params['epsilon']
    print

    # Store final results
    str = "Optimal parameters for PolySVR found for m=" + str(m) + \
          " randomly selected voxels, for N=" + str(N) + " iterations, " + \
          "and total_error=" + str(total_error) + ":\n" + \
          "    C --> " + str(optim_params['C']) + "\n" + \
          "    epsilon --> " + str(optim_params['epsilon']) + "\n"
    with open(join(RESULTS_DIR, "psvr_optimal_hyperparams.txt"), 'wb') as f:
        f.write(str)
