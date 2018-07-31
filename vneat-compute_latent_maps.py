#!/usr/bin/python
from __future__ import print_function

from argparse import ArgumentParser
from glob import glob
from os import path

import nibabel as nib

from vneat import helper_functions
from vneat.FitScores.FitEvaluation import aic, fstat, ftest, mse, prss, r2, vnprss

if __name__ == '__main__':


    """ CLI ARGUMENTS """
    arguments_parser = ArgumentParser(description='Computes statistical maps for the fitting results '
                                                  ' computed by compute_fitting.py. By default '
                                                  ' uses all computed parameters inside the results'
                                                  ' folder specified in the configuration file.')

    arguments_parser.add_argument('configuration_file', help="Path to the YAML configuration file"
                                                             " used to load the data for this study.")



    arguments_parser.add_argument('--dirs', nargs='+', help='Specify one or several directories within the'
                                                            ' results directory specified in the '
                                                            ' configuration file from which the '
                                                            ' parameters should be loaded.')

    # Optional arguments for the fitting methods
    arguments_parser.add_argument('--cluster_size', default=100, type=int,
                                  help='Value of the minimum cluster size (in voxels) that should survive'
                                       ' after thresholding.')

    arguments_parser.add_argument('--p_thresholds', default=[0.01, 0.005, 0.001],
                                  nargs='+', type=float,
                                  help='One or more values (list) representing the maximum acceptable p-value,'
                                       ' so that all voxels with greater p-value are put to the default'
                                       ' value. (Optional: default is None, no threshold')

    arguments_parser.add_argument('--n_permutation', default=1000, type=int,
                                  help='Number of permutation within the permutation testing framework.'
                                        '(Optional: default is 1000)')

    arguments_parser.add_argument('--n_clusters', default=None, type=int,
                                  help='One or more values (list) indicating several values for clustering'
                                       ' effect-types. (Optional: default is None, indicating that no'
                                       ' clustering is performed).')

    arguments_parser.add_argument('--gm_threshold', default=0.1, type=float,
                                  help='Mean grey-matter lower threshold.')

    arguments_parser.add_argument('--labels', default=True,
                                  help='Produce a label maps for both effect strenght and effect type '
                                       'clusters (if computed).')

    arguments = arguments_parser.parse_args()
    config_file = arguments.configuration_file
    dirs = arguments.dirs
    cluster_size = arguments.cluster_size
    p_thresholds = arguments.p_thresholds
    n_permutation = arguments.n_permutation
    n_clusters = arguments.n_clusters
    gm_threshold = arguments.gm_threshold
    labels = arguments.labels


    """ LOAD DATA USING DATALOADER """
    subjects, predictors_names, correctors_names, predictors, correctors, processing_parameters, \
    affine_matrix, output_dir, results_io, type_data = helper_functions.load_data_from_config_file(config_file)

    """ LOAD RESULTS DATA """
    if dirs is None:
        # Find prediction parameters inside results folder
        pathname = path.join(output_dir, '**', '*prediction_parameters'+results_io.extension)
        for p in glob(pathname):
            n, category, pred_p, corr_p, proc = helper_functions.get_results_from_path(
                p, results_io, subjects, predictors_names, correctors_names, predictors, correctors,
                processing_parameters, type_data
            )
            print()
            print('Computing fitting results for {} '.format(n), end="")
            if category is not None:
                print('({})'.format(category), end="")
            try:
                results = helper_functions.compute_latent_effect_strength_type(
                    proc, pred_p, corr_p, cluster_size, p_thresholds, n_permutation,
                    n_clusters, gm_threshold, labels
                )
            except RuntimeError:
                continue
            print('Storing fitting results')
            folder_name = path.split(p)[0]
            for name, data in results:
                full_file_path = path.join(folder_name, name+results_io.extension)
                res_writer = results_io.writer(data, affine_matrix)
                res_writer.save(full_file_path)
    else:
        for directory in dirs:
            full_path = path.join(output_dir, directory)
            pathname = glob(path.join(full_path, '*prediction_parameters'+results_io.extension))
            # If there is no coincidence, ignore this directory
            if len(pathname) == 0:
                print('{} does not exist or contain any result.'.format(full_path))
                continue
            n, category, pred_p, corr_p, proc = helper_functions.get_results_from_path(
                pathname[0], results_io, subjects, predictors_names, correctors_names, predictors, correctors,
                processing_parameters, type_data
            )
            print()
            print('Computing fitting results for {} '.format(n), end="")
            if category is not None:
                print('({})'.format(category), end="")
            # try:
            results = helper_functions.compute_latent_effect_strength_type(
                proc, pred_p, corr_p, cluster_size, p_thresholds, n_permutation,
                n_clusters, gm_threshold, labels
            )
            # except RuntimeError:
            #     print('{} is not supported by this fitter. Try another fit evaluation method.'.
            #           format(method)
            #           )
            #     continue
            print('Storing fitting results')
            folder_name = path.split(pathname[0])[0]
            for name, data in results:
                full_file_path = path.join(folder_name, name+results_io.extension)
                res_writer = results_io.writer(data, affine_matrix)
                res_writer.save(full_file_path)
    print()
    print('Done.')
