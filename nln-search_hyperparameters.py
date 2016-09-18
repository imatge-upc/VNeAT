from __future__ import print_function

from argparse import ArgumentParser

import os

from src import helper_functions
from src.Processors.SVRProcessing import GaussianSVRProcessor, PolySVRProcessor
from src.CrossValidation.GridSearch import GridSearch
from src.CrossValidation.score_functions import anova_error, mse, statisticC_p

if __name__ == '__main__':

    """ CONSTANTS """
    AVAILABLE_FITTERS = {
        'PolySVR': PolySVRProcessor,
        'GaussianSVR': GaussianSVRProcessor
    }

    AVAILABLE_SCORES = {
        'mse': mse,
        'anova': anova_error,
        'Cp': statisticC_p
    }

    """ PARSE ARGUMENTS FROM CLI """
    arg_parser = ArgumentParser(description='Finds the hyper parameters of the PolySVR or GaussianSVR'
                                            ' using a grid search approach and using several error'
                                            ' functions.')
    arg_parser.add_argument('configuration_file', help='Path to the YAML configuration file'
                                                       ' used to load the data for this study.')
    arg_parser.add_argument('fitter', choices=AVAILABLE_FITTERS.keys(),
                            help='The fitter for which the hyperparameters should be found.')
    arg_parser.add_argument('--error', choices=AVAILABLE_SCORES.keys(), default='anova',
                            help='Error function to be minimized in order to find the optimal '
                                 'hyperparameters.')
    arg_parser.add_argument('--iterations', '-i', type=int, default=5,
                            help='The number of iterations to perform.')
    arg_parser.add_argument('--voxels', '-v', type=int, default=100,
                            help='The number of voxels to be used to compute the error and therefore'
                                 ' find the optimal hyperparameters. In general, more voxels used may'
                                 ' imply better generalization, but also more computation time and'
                                 ' use of resources')
    arg_parser.add_argument('--voxel-offset', type=int, default=10,
                            help="Number of voxels that will not be taken into account in all directions, "
                                 "both at the beginning and at the end. That is, for a voxel offset of v, "
                                 "and volumes with dimensions (x_dim, y_dim, z_dim), "
                                 "only the following voxels will be taken into account: "
                                 "[v:x_dim-v, v:y_dim-v, v:z_dim-v]")
    arg_parser.add_argument('--categories', nargs='+', type=int,
                            help='Category or categories (as they are represented '
                                 'in the Excel file) for which the hyperparameters should be found.')
    arg_parser.add_argument('--prefix', help='Prefix used in the result files')

    arguments = arg_parser.parse_args()
    config_file = arguments.configuration_file
    error_func = AVAILABLE_SCORES[arguments.error]
    fitter_name = arguments.fitter
    N = arguments.iterations
    m = arguments.voxels
    voxel_offset = arguments.voxel_offset
    categories = arguments.categories
    prefix = arguments.prefix

    """ LOAD DATA USING DATALOADER """
    subjects, predictors_names, correctors_names, predictors, correctors, processing_parameters, \
    affine_matrix, output_dir = helper_functions.load_data_from_config_file(config_file)
    hyperparams_dict = helper_functions.load_hyperparams_from_config_file(config_file, fitter_name)

    """ PROCESSING """
    if categories is not None:
        for category in categories:
            print()
            print('////////////////')
            print('// CATEGORY {} //'.format(category))
            print('////////////////')
            print()

            # Create MixedProcessor instance
            processor = AVAILABLE_FITTERS[fitter_name](subjects,
                                                       predictors_names,
                                                       correctors_names,
                                                       predictors,
                                                       correctors,
                                                       processing_parameters,
                                                       category=category)
            """ RESULTS DIRECTORY """
            cat_str = 'category_{}'.format(category)
            output_folder_name = '{}-{}-{}-{}'.format(
                prefix,
                'hyperparameters',
                fitter_name,
                cat_str
            ) if prefix else '{}-{}-{}'.format(
                'hyperparameters',
                fitter_name,
                cat_str
            )
            output_folder = os.path.join(output_dir, output_folder_name)

            # Check if directory exists
            if not os.path.isdir(output_folder):
                # Create directory
                os.makedirs(output_folder)

            # Gridsearch
            grid_search = GridSearch(processor, output_folder, voxel_offset=voxel_offset,
                                     n_jobs=processing_parameters['n_jobs'])
            grid_search.fit(hyperparams_dict, N=N, m=m, score=error_func, filename='error_values')
            grid_search.store_results('optimal_hyperparameters')
            grid_search.plot_error('error_plot')

    else:
        processor = AVAILABLE_FITTERS[fitter_name](subjects,
                                                   predictors_names,
                                                   correctors_names,
                                                   predictors,
                                                   correctors,
                                                   processing_parameters)
        """ RESULTS DIRECTORY """
        output_folder_name = '{}-{}-{}'.format(
            prefix,
            'hyperparameters',
            fitter_name,
        ) if prefix else '{}-{}'.format(
            'hyperparameters',
            fitter_name,
        )
        output_folder = os.path.join(output_dir, output_folder_name)

        # Check if directory exists
        if not os.path.isdir(output_folder):
            # Create directory
            os.makedirs(output_folder)

        # Gridsearch
        grid_search = GridSearch(processor, output_folder, voxel_offset=voxel_offset,
                                 n_jobs=processing_parameters['n_jobs'])
        grid_search.fit(hyperparams_dict, N=N, m=m, score=error_func, filename='error_values')
        grid_search.store_results('optimal_hyperparameters')
        grid_search.plot_error('error_plot')

