import os
import os.path as path
from argparse import ArgumentParser

import nibabel as nib

from src import helper_functions
from src.Processors.SVRProcessing import GaussianSVRProcessor, PolySVRProcessor

if __name__ == '__main__':
    """ CONSTANTS """
    AVAILABLE_FITTERS = {
        'PolySVR': PolySVRProcessor,
        'GaussianSVR': GaussianSVRProcessor
    }

    """ PARSE ARGUMENTS FROM CLI """
    arg_parser = ArgumentParser(description='Finds the hyper parameters of the PolySVR or GaussianSVR'
                                            ' using a grid search approach and using several error'
                                            ' functions.')
    arg_parser.add_argument('configuration_file', help='Path to the YAML configuration file'
                                                       ' used to load the data for this study.')
    arg_parser.add_argument('fitter', choices=AVAILABLE_FITTERS.keys(),
                            help='The fitter for which the hyperparameters should be found.')
    arg_parser.add_argument('--iterations', '-i', type=int, default=5,
                            help='The number of iterations to perform.')
    arg_parser.add_argument('--voxels', '-v', type=int, default=1000,
                            help='The number of voxels to be used to compute the error and therefore'
                                 ' find the optimal hyperparameters. In general, more voxels used may'
                                 ' imply better generalization, but also more computation time and'
                                 ' use of resources.')
    arg_parser.add_argument('--categories', nargs='+', type=int,
                            help='Category or categories (as they are represented '
                                 'in the Excel file) for which the hyperparameters should be found.')
    arg_parser.add_argument('--prefix', help='Prefix used in the result files')

    arguments = arg_parser.parse_args()
    config_file = arguments.configuration_file
    fitter_name = arguments.fitter
    categories = arguments.categories
    prefix = arguments.prefix

    """ LOAD DATA USING DATALOADER """
    subjects, predictors_names, correctors_names, predictors, correctors, processing_parameters, \
    affine_matrix, output_dir = helper_functions.load_data_from_config_file(config_file)
    hyperparams_dict = helper_functions.load_hyperparams_from_config_file(config_file, fitter_name)

    """ PROCESSING """
    # Create MixedProcessor instance
    processor = AVAILABLE_FITTERS[fitter_name](subjects,
                                               predictors_names,
                                               correctors_names,
                                               predictors,
                                               correctors,
                                               processing_parameters)

    # Get fitter
    fitter = processor.fitter


