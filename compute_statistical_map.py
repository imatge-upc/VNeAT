from __future__ import print_function

from argparse import ArgumentParser
from glob import glob
from os import path

import nibabel as nib
import numpy as np
from scipy.stats import norm

from src.FitScores.FitEvaluation_v2 import aic, fstat, ftest, mse, prss, r2, vnprss
from src.Processors.MixedProcessor import MixedProcessor
from src.Utils.DataLoader import DataLoader

""" FUNCTION DEFINITIONS """


def get_results_from_path(pred_params_path):
    # From path found by glob infer the paths to the other files
    # (correction parameters and user defined parameters)
    # and the name of the fitting method
    folder_path, prediction_params_name = path.split(pred_params_path)
    prefix = prediction_params_name.replace('prediction_parameters.nii.gz', '')
    prediction_params_path = pred_params_path
    correction_params_path = path.join(folder_path, '{}correction_parameters.nii.gz'.format(prefix))
    udp_path = path.join(folder_path, '{}user_defined_parameters.txt'.format(prefix))
    # Try to infer whether there is a curve for a category or not by folder name
    folder_name = path.basename(folder_path)
    if 'category' in folder_name:
        cat = int(folder_name.split('-')[-1].split('_')[-1])
        name = '-'.join(folder_name.split('-')[:-1])
    else:
        cat = None
        name = folder_name
    # Load niftis and txt files and keep them
    with open(udp_path, 'rb') as udp_file:
        udp = eval(udp_file.read())
    pred_parameters = nib.load(prediction_params_path).get_data()
    corr_parameters = nib.load(correction_params_path).get_data()
    # Create MixedProcessor and keep it
    processor = MixedProcessor(
        subjects,
        predictors_names,
        correctors_names,
        predictors,
        correctors,
        processing_parameters,
        user_defined_parameters=udp,
        category=cat
    )
    return name, cat, pred_parameters, corr_parameters, processor


def compute_fitting_scores(processor_instance, method_name, pparams, cparams, cluster_size,
                           p_thresholds, gamma, percentile_filter, gm_threshold, labels,
                           mem_usage, **kwargs):
    returned_results = []  # Variable where tuples of (name, fitting_score) are stored to be returned
    method_func = METHOD_CHOICES[method_name]
    if method_name == 'vnprss' or method_name == 'prss':
        fitting_scores = processor_instance.evaluate_fit(
            evaluation_function=method_func,
            correction_parameters=cparams,
            prediction_parameters=pparams,
            x1=0,
            x2=None,
            y1=0,
            y2=None,
            z1=0,
            z2=None,
            gm_threshold=gm_threshold,
            filter_nans=True,
            default_value=np.inf,
            gamma=gamma
        )
    else:
        fitting_scores = processor_instance.evaluate_fit(
            evaluation_function=method_func,
            correction_parameters=cparams,
            prediction_parameters=pparams,
            x1=0,
            x2=None,
            y1=0,
            y2=None,
            z1=0,
            z2=None,
            gm_threshold=gm_threshold,
            filter_nans=True,
            default_value=0.0
        )
    fit_scores_name = '{}_fitscores.nii.gz'.format(method_name)
    returned_results.append((fit_scores_name, fitting_scores))
    if method_name == 'ftest':
        for p_threshold in p_thresholds:
            inv_p_threshold = 1 - p_threshold
            if labels:
                print('Filtering and clustering p-values using threshold {}, and generating labels...'
                      .format(p_threshold))
                clusterized_fitting_scores, labels_map = MixedProcessor.clusterize(
                    fitting_scores,
                    default_value=0.0,
                    fit_lower_threshold=inv_p_threshold,
                    cluster_threshold=cluster_size,
                    produce_labels=True
                )
                name_labels = 'labels_{}.nii.gz'.format(inv_p_threshold)
                returned_results.append((name_labels, labels_map))
            else:
                print('Filtering and clustering p-values using threshold {}...'.format(p_threshold))
                clusterized_fitting_scores = MixedProcessor.clusterize(
                    fitting_scores,
                    default_value=0.0,
                    fit_lower_threshold=inv_p_threshold,
                    cluster_threshold=cluster_size,
                    produce_labels=False
                )
            print('Converting p-values to Z-scores...')
            lim_value = norm.ppf(inv_p_threshold)
            valid_voxels = clusterized_fitting_scores != 0.0
            clusterized_fitting_scores[valid_voxels] = norm.ppf(
                clusterized_fitting_scores[valid_voxels]
            ) - lim_value + 0.2
            name_z_scores = 'z-scores_{}.nii.gz'.format(inv_p_threshold)
            returned_results.append((name_z_scores, clusterized_fitting_scores))

    elif method_name == 'prss' or method_name == 'vnprss':
        print('Filtering and transforming scores for better visualization...')
        valid_voxels = np.isfinite(fitting_scores)
        sorted_scores = np.sort(fitting_scores[valid_voxels].reshape(-1))
        num_elems = int(np.ceil(percentile_filter * float(fitting_scores.size)))
        threshold = sorted_scores[:num_elems][-1]
        valid_voxels = fitting_scores <= threshold
        fitting_scores[~valid_voxels] = 0.0
        fitting_scores[valid_voxels] = threshold - fitting_scores[valid_voxels]  # Invert values
        filtered_scores_name = 'invfiltered_{}_gamma{}_percentile{}.nii.gz'.format(
            method_name,
            gamma,
            percentile_filter
        )
        returned_results.append((filtered_scores_name, fitting_scores))

    return returned_results

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
                                                  ' computed by compute_parameters.py. By default '
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
                                  help='Value of the minimum cluster size (in voxels) that should survive after'
                                       ' thresholding.')

    arguments_parser.add_argument('--p_thresholds', default=[0.01, 0.005, 0.001],
                                  nargs='+', type=float,
                                  help='One or more values representing the maximum acceptable p-value, so that'
                                       ' all voxels with greater p-value are put to the default value.')

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
    dirs = arguments.dirs
    cluster_size = arguments.cluster_size
    p_thresholds = arguments.p_thresholds
    gamma = arguments.gamma
    percentile_filter = arguments.percentile_filter
    gm_threshold = arguments.gm_threshold
    labels = arguments.labels

    """ LOAD DATA USING DATALOADER """
    print('Loading configuration data...')
    try:
        data_loader = DataLoader(config_file)
    except IOError as e:
        print()
        print(e.filename + ' does not exist.')
        data_loader = None
        exit(1)

    # Load all necessary data:
    try:
        subjects = data_loader.get_subjects()
        predictors_names = data_loader.get_predictors_names()
        correctors_names = data_loader.get_correctors_names()
        predictors = data_loader.get_predictors()
        correctors = data_loader.get_correctors()
        processing_parameters = data_loader.get_processing_parameters()
        affine_matrix = data_loader.get_template_affine()
        output_dir = data_loader.get_output_dir()
    except KeyError:
        print()
        print('Configuration file does not have the specified format.')
        print('See config/exampleConfig.yaml for further information about the format of configuration '
              'files')
        subjects = predictors = correctors = None
        predictors_names = correctors_names = None
        processing_parameters = affine_matrix = output_dir = None
        exit(1)

    """ LOAD RESULTS DATA """
    if dirs is None:
        # Find prediction parameters inside results folder
        pathname = path.join(output_dir, '**', '*prediction_parameters.nii.gz')
        for p in glob(pathname):
            n, category, pred_p, corr_p, proc = get_results_from_path(p)
            print()
            print('Computing fitting results for {}'.format(n), end="")
            if category is not None:
                print(', category {}'.format(category))
            print("{} progress: ".format(method))
            try:
                results = compute_fitting_scores(proc, method, pred_p, corr_p, cluster_size,
                                                 p_thresholds, gamma, percentile_filter, gm_threshold, labels,
                                                 processing_parameters['mem_usage'])
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
            n, category, pred_p, corr_p, proc = get_results_from_path(pathname[0])
            print()
            print('Computing fitting results for {}'.format(n), end="")
            if category is not None:
                print(', category {}'.format(category))
            print("{} progress: ".format(method),)
            try:
                results = compute_fitting_scores(proc, method, pred_p, corr_p, cluster_size,
                                                 p_thresholds, gamma, percentile_filter, gm_threshold, labels,
                                                 processing_parameters['mem_usage'])
            except RuntimeError:
                print('{} is not supported by this fitter. Try another fit evaluation method.'.format(method))
                continue
            print('Storing fitting results')
            folder_name = path.split(pathname[0])[0]
            for name, data in results:
                full_file_path = path.join(folder_name, name)
                niiImg = Nifti(data, affine_matrix)
                nib.save(niiImg, full_file_path)
    print()
    print('Done.')
