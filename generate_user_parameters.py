import os

from argparse import ArgumentParser
from os import path

from Utils.DataLoader import DataLoader
from Processors.MixedProcessor import MixedProcessor

if __name__ == '__main__':

    argument_parser = ArgumentParser(description='Generates user defined parameters for a '
                                                 'specific correction and prediction processor'
                                                 ' so you can use them in compute_parameters.py'
                                                 ' using the --parameters option')

    argument_parser.add_argument('configuration_file', help='Path to the YAML configuration file'
                                                            ' used to load the data for this study.')
    argument_parser.add_argument('--prefix', help='Prefix used in the result files')

    arguments = argument_parser.parse_args()
    config_file = arguments.configuration_file
    prefix = arguments.prefix

    try:
        data_loader = DataLoader(config_file)
    except IOError as e:
        print
        print e.filename + ' does not exist.'
        data_loader = None
        exit(1)

    # Load all necessary data: subjects, predictors_names, correctors_names, predictors, correctors, processing_params
    subjects = data_loader.get_subjects()
    predictors_names = data_loader.get_predictors_names()
    correctors_names = data_loader.get_correctors_names()
    predictors = data_loader.get_predictors()
    correctors = data_loader.get_correctors()
    processing_parameters = data_loader.get_processing_parameters()

    # Create MixedProcessor instance
    processor = MixedProcessor(subjects,
                               predictors_names,
                               correctors_names,
                               predictors,
                               correctors,
                               processing_parameters)
    # Processor name
    processor_name = processor.get_name()

    # User defined parameters
    udp = processor.user_defined_parameters

    print
    print 'Storing user defined parameters...'
    output_folder = path.join(data_loader.get_output_dir(), processor_name)

    # Check if directory exists
    if not path.isdir(output_folder):
        # Create directory
        os.makedirs(output_folder)

    # Filename
    udp_file = prefix + 'user_defined_parameters.txt' if prefix else 'user_defined_parameters.txt'

    with open(path.join(output_folder, udp_file), 'wb') as f:
        f.write(str(udp) + '\n')

    print 'Done'


