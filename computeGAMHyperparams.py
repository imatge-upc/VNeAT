from os.path import join

import numpy as np
from Utils.DataLoader import getGMData, getFeatures
from Utils.Subject import Subject
from user_paths import RESULTS_DIR

from Fitters.GAM import GAM, SplinesSmoother

if __name__ == "__main__":

    """ LOAD DATA """

    # Get data from Excel and nii files
    print "Getting data from NIFTI files..."
    predictor = getFeatures([Subject.ADCSFIndex])
    predictor_smoother = SplinesSmoother(predictor,order=3)
    observations = getGMData(corrected_data=True)

    # Dimensions
    num_subjects = observations.shape[0]
    X_dim, Y_dim, Z_dim = observations.shape[1:]

    """ PARAMS """
    N = 1                                       # number of iterations
    m = 100                                     # number of voxels to select randomly in each iteration
    gm_threshold = 0.1                          # grey matter threshold used to filter voxels with no gm
    params = {
        'smoothing_factor': np.linspace(110,135,50),
    }

    """ INITIALIZATION """
    optim_params = { 'smoothing_factor': -1 }    # optimal params initialization
    total_error = 1000000000                    # total error initialization
    fitter = GAM(predictor_smoothers=predictor_smoother)     # PolySVR fitter with 3rd order polynomic
    total_computations = len(params['smoothing_factor'])

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
        for s in params['smoothing_factor']:
            # Show progress
            print "\r",
            print (float(progress_counter) / total_computations)*100, "%",
            # Fit data with C and epsilon params
            predictor_smoother = SplinesSmoother(predictor,order=3,smoothing_factor=s)
            fitter = GAM(predictor_smoothers=predictor_smoother)
            fitter.fit(current_observations)
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
                optim_params['smoothing_factor'] = s

            # Update progress
            progress_counter += 1

    # Print final results
    print
    print
    print "Total error: ", total_error
    print "Optimal parameters: "
    print "    smoothing_factor --> ", optim_params['smoothing_factor']
    print

    # Store final results
    str = "Optimal parameters for PolySVR found for m=" + str(m) + \
          " randomly selected voxels, for N=" + str(N) + " iterations, " + \
          "and total_error=" + str(total_error) + ":\n" + \
          "    smoothing_factor --> " + str(optim_params['smoothing_factor']) + "\n"
    with open(join(RESULTS_DIR, "gam_poly_d3_optimal_hyperparams.txt"), 'wb') as f:
        f.write(str)
