from __future__ import print_function

from argparse import ArgumentParser
from glob import glob
from os import path

import nibabel as nib

from src import helper_functions
from src.FitScores.FitEvaluation_v2 import aic, fstat, ftest, mse, prss, r2, vnprss

if __name__ == '__main__':

    """ CONSTANTS """
    METHOD_CHOICES = {
        'mse': mse,
        'r2': r2,
        'fstat': fstat,
        'ftest': ftest,
        'aic': aic,
        'prss': prss,
        'vnprss': vnprss
    }

    Nifti = nib.Nifti1Image

    """ CLI ARGUMENTS """
    arguments_parser = ArgumentParser(description='Computes statistical maps for the fitting results '
                                                  ' computed by compute_fitting.py. By default '
                                                  ' uses all computed parameters inside the results'
                                                  ' folder specified in the configuration file.')

    arguments_parser.add_argument('configuration_file', help="Path to the YAML configuration file"
                                                             " used to load the data for this study.")

    arguments_parser.add_argument('--method', default='ftest',
                                  choices=METHOD_CHOICES, help='Method to evaluate the fitting score'
                                                               ' per voxel and create a statistical'
                                                               ' map out of these fitting scores.')

    arguments_parser.add_argument('--dirs', nargs='+', help='Specify one or several directories within the '
                                                            'results directory specified in the '
                                                            ' configuration file from which the '
                                                            ' parameters should be loaded.')

    # Optional arguments for the fitting methods
    arguments_parser.add_argument('--cluster_size', default=100, type=int,
                                  help='Value of the minimum cluster size (in voxels) that should survive'
                                       ' after thresholding.')

    arguments_parser.add_argument('--p_thresholds', default=[0.01, 0.005, 0.001],
                                  nargs='+', type=float,
                                  help='One or more values representing the maximum acceptable p-value,'
                                       ' so that all voxels with greater p-value are put to the default'
                                       ' value')

    arguments_parser.add_argument('--gamma', default=3e-4, type=float,
                                  help='Value for PRSS and vnPRSS methods that weights the roughness penalty.'
                                       ' Increasing this value means that roughness is more penalized.')

    arguments_parser.add_argument('--percentile_filter', default=5e-3, type=float,
                                  help='Value of the percentile used to determine the upper threshold'
                                       ' for PRSS and vnPRSS methods.')

    arguments_parser.add_argument('--gm_threshold', default=0.1, type=float,
                                  help='Mean grey-matter lower threshold.')

    arguments_parser.add_argument('--labels', default=True,
                                  help='Produce a map that has one label per cluster.')

    arguments = arguments_parser.parse_args()
    config_file = arguments.configuration_file
    method = arguments.method
    method_func = METHOD_CHOICES[method]
    dirs = arguments.dirs
    cluster_size = arguments.cluster_size
    p_thresholds = arguments.p_thresholds
    gamma = arguments.gamma
    percentile_filter = arguments.percentile_filter
    gm_threshold = arguments.gm_threshold
    labels = arguments.labels

    """ LOAD DATA USING DATALOADER """
    subjects, predictors_names, correctors_names, predictors, correctors, processing_parameters, \
    affine_matrix, output_dir = helper_functions.load_data_from_config_file(config_file)

    """ LOAD RESULTS DATA """
    if dirs is None:
        # Find prediction parameters inside results folder
        pathname = path.join(output_dir, '**', '*prediction_parameters.nii.gz')
        for p in glob(pathname):
            n, category, pred_p, corr_p, proc = helper_functions.get_results_from_path(
                p, subjects, predictors_names, correctors_names, predictors, correctors, processing_parameters
            )
            print()
            print('Computing fitting results for {}'.format(n), end="")
            if category is not None:
                print(', category {}'.format(category))
            print("{} progress: ".format(method))
            try:
                results = helper_functions.compute_fitting_scores(
                    proc, method, method_func, pred_p, corr_p, cluster_size, p_thresholds, gamma,
                    percentile_filter, gm_threshold, labels
                )
            except RuntimeError:
                print('{} is not supported by this fitter. Try another fit evaluation method.'.format(method))
                continue
            print('Storing fitting results')
            folder_name = path.split(p)[0]
            for name, data in results:
                full_file_path = path.join(folder_name, name)
                niiImg = Nifti(data, affine_matrix)
                nib.save(niiImg, full_file_path)
    else:
        for directory in dirs:
            full_path = path.join(output_dir, directory)
            pathname = glob(path.join(full_path, '*prediction_parameters.nii.gz'))
            # If there is no coincidence, ignore this directory
            if len(pathname) == 0:
                print('{} does not exist or contain any result.'.format(full_path))
                continue
            n, category, pred_p, corr_p, proc = helper_functions.get_results_from_path(
                pathname[0], subjects, predictors_names, correctors_names, predictors, correctors,
                processing_parameters
            )
            print()
            print('Computing fitting results for {}'.format(n), end="")
            if category is not None:
                print(', category {}'.format(category))
            print("{} progress: ".format(method), )
            try:
                results = helper_functions.compute_fitting_scores(
                    proc, method, method_func, pred_p, corr_p, cluster_size, p_thresholds, gamma,
                    percentile_filter, gm_threshold, labels
                )
            except RuntimeError:
                print('{} is not supported by this fitter. Try another fit evaluation method.'.
                      format(method)
                      )
                continue
            print('Storing fitting results')
            folder_name = path.split(pathname[0])[0]
            for name, data in results:
                full_file_path = path.join(folder_name, name)
                niiImg = Nifti(data, affine_matrix)
                nib.save(niiImg, full_file_path)
    print()
    print('Done.')
