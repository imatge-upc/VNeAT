#!/usr/bin/python
from __future__ import print_function

import os
from argparse import ArgumentParser

from vneat import helper_functions
from vneat.CrossValidation.GridSearch import GridSearch
from vneat.CrossValidation.score_functions import anova_error, mse, statisticC_p
from vneat.Processors.MixedProcessor import MixedProcessor

if __name__ == '__main__':

    """ CONSTANTS """
    AVAILABLE_FITTERS_NAMES = ['PolySVR', 'GaussianSVR']

    AVAILABLE_SCORES = {
        'mse': mse,
        'anova': anova_error,
        'Cp': statisticC_p
    }

    """ PARSE ARGUMENTS FROM CLI """
    arg_parser = ArgumentParser(description='Finds the hyperparameters of the PolySVR or GaussianSVR'
                                            ' using a grid search approach and using several error'
                                            ' functions.')
    arg_parser.add_argument('configuration_file', help='Path to the YAML configuration file'
                                                       ' used to load the data for this study.')
    arg_parser.add_argument('--parameters', help='Path to the txt file within the results directory '
                                                 'that contains the user defined '
                                                 'parameters to load a pre-configured '
                                                 'correction and prediction processor')
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
    parameters = arguments.parameters
    error = arguments.error
    error_func = AVAILABLE_SCORES[arguments.error]
    N = arguments.iterations
    m = arguments.voxels
    voxel_offset = arguments.voxel_offset
    categories = arguments.categories
    prefix = arguments.prefix

    """ LOAD DATA USING DATALOADER """
    subjects, predictors_names, correctors_names, predictors, correctors, processing_parameters, \
    affine_matrix, output_dir, results_io, type_data = helper_functions.load_data_from_config_file(config_file)

    if parameters:
        # Load user defined parameters
        try:
            parameters_path = os.path.normpath(os.path.join(output_dir, parameters))
            with open(parameters_path, 'rb') as f:
                udp = eval(f.read())
                print()
                print('User defined parameters have been successfully loaded.')
        except IOError as ioe:
            print()
            print('The provided parameters file, ' + ioe.filename + ', does not exist.')
            print('Standard input will be used to configure the correction and prediction processors instead.')
            print()
            udp = ()
        except SyntaxError:
            print()
            print('The provided parameters file is not properly formatted.')
            print('Standard input will be used to configure the correction and prediction processors instead.')
            print()
            udp = ()
        except:
            print()
            print('An unexpected error happened.')
            print('Standard input will be used to configure the correction and prediction processors instead.')
            print()
            udp = ()
    else:
        udp = ()

    """ INITIALIZE PROCESSOR """
    try:
        dummy_cat = categories[0]
    except TypeError:
        dummy_cat = None

    processor = MixedProcessor(
        subjects,
        predictors_names,
        correctors_names,
        predictors,
        correctors,
        processing_parameters,
        user_defined_parameters=udp,
        category=dummy_cat,
        type_data=type_data
    )

    # Check if prediction processor is available for grid searching
    prediction_fitter_name = processor.prediction_processor.get_name()
    udp = processor.user_defined_parameters
    if prediction_fitter_name not in AVAILABLE_FITTERS_NAMES:
        print('The selected prediction processor is not available for hyperparameters searching.')
        print('Use one of the following: {}'.format(', '.join(AVAILABLE_FITTERS_NAMES)))
        exit(1)

    hyperparams_dict = helper_functions.load_hyperparams_from_config_file(config_file, prediction_fitter_name)

    """ PROCESSING """
    if categories is not None:
        for category in categories:
            print()
            print('////////////////')
            print('// CATEGORY {} //'.format(category))
            print('////////////////')
            print()

            # Create MixedProcessor instance
            processor = MixedProcessor(subjects,
                                       predictors_names,
                                       correctors_names,
                                       predictors,
                                       correctors,
                                       processing_parameters,
                                       category=category,
                                       user_defined_parameters=udp,
                                       type_data=type_data)

            """ RESULTS DIRECTORY """
            cat_str = 'category_{}'.format(category)
            output_folder_name = '{}-{}-{}-{}-{}'.format(
                prefix,
                'hyperparameters',
                error,
                prediction_fitter_name,
                cat_str
            ) if prefix else '{}-{}-{}-{}'.format(
                'hyperparameters',
                error,
                prediction_fitter_name,
                cat_str
            )
            output_folder = os.path.join(output_dir, output_folder_name)

            # Check if directory exists
            if not os.path.isdir(output_folder):
                # Create directory
                os.makedirs(output_folder)

            # Save user defined parameters
            with open(os.path.join(output_folder, 'user_defined_parameters.txt'), 'wb') as f:
                f.write(str(udp).encode('utf-8'))
                f.write(b'\n')

            # Gridsearch
            grid_search = GridSearch(processor, output_folder, voxel_offset=voxel_offset,
                                     n_jobs=processing_parameters['n_jobs'])
            grid_search.fit(hyperparams_dict, N=N, m=m, score=error_func, filename='error_values')

            # Double check if directory exists
            if not os.path.isdir(output_folder):
                # Create directory
                os.makedirs(output_folder)

            grid_search.store_results('optimal_hyperparameters')
            grid_search.plot_error('error_plot')

    else:
        processor = MixedProcessor(subjects,
                                   predictors_names,
                                   correctors_names,
                                   predictors,
                                   correctors,
                                   processing_parameters,
                                   user_defined_parameters=udp,
                                   type_data=type_data)
        """ RESULTS DIRECTORY """
        output_folder_name = '{}-{}-{}-{}'.format(
            prefix,
            'hyperparameters',
            error,
            prediction_fitter_name,
        ) if prefix else '{}-{}-{}'.format(
            'hyperparameters',
            error,
            prediction_fitter_name,
        )
        output_folder = os.path.join(output_dir, output_folder_name)

        # Check if directory exists
        if not os.path.isdir(output_folder):
            # Create directory
            os.makedirs(output_folder)

        # Save user defined parameters
        with open(os.path.join(output_folder, 'user_defined_parameters.txt'), 'wb') as f:
            f.write(str(udp).encode('utf-8'))
            f.write(b'\n')

        # Gridsearch
        grid_search = GridSearch(processor, output_folder, voxel_offset=voxel_offset,
                                 n_jobs=processing_parameters['n_jobs'])
        grid_search.fit(hyperparams_dict, N=N, m=m, score=error_func, filename='error_values')

        # Double check if directory exists
        if not os.path.isdir(output_folder):
            # Create directory
            os.makedirs(output_folder)

        grid_search.store_results('optimal_hyperparameters')
        grid_search.plot_error('error_plot')
