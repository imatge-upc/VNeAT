#!/usr/bin/python
from __future__ import print_function

import os
import os.path as path
from argparse import ArgumentParser

import nibabel as nib

from vneat import helper_functions
from vneat.Processors.MixedProcessor import MixedProcessor

if __name__ == '__main__':

    """ PARSE ARGUMENTS FROM CLI """
    arg_parser = ArgumentParser(description='Computes the fitting parameters for the data '
                                            'provided in the configuration file. This fitting parameters'
                                            ' can be computed for all subjects in the study (default behaviour)'
                                            ' or you can specify for which categories should the parameters be'
                                            ' computed ')
    arg_parser.add_argument('configuration_file', help='Path to the YAML configuration file'
                                                       ' used to load the data for this study.')
    arg_parser.add_argument('--categories', nargs='+', type=int, help='Category or categories (as they are represented '
                                                                      'in the Excel file) for which the fitting '
                                                                      'parameters should be computed')
    arg_parser.add_argument('--parameters', help='Path to the txt file within the results directory '
                                                 'that contains the user defined '
                                                 'parameters to load a pre-configured '
                                                 'correction and prediction processor')
    arg_parser.add_argument('--prefix', help='Prefix used in the result files')

    arguments = arg_parser.parse_args()
    config_file = arguments.configuration_file
    categories = arguments.categories
    parameters = arguments.parameters
    prefix = arguments.prefix

    """ LOAD DATA USING DATALOADER """
    subjects, predictors_names, correctors_names, predictors, correctors, processing_parameters, \
    affine_matrix, output_dir = helper_functions.load_data_from_config_file(config_file)

    if parameters:
        # Load user defined parameters
        try:
            parameters_path = path.normpath(path.join(output_dir, parameters))
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

    """ CREATE PROCESSOR """
    # Create MixedProcessor instance
    initial_category = categories[0] if (categories is not None and len(categories)) > 0 else None
    try:
        processor = MixedProcessor(subjects,
                                   predictors_names,
                                   correctors_names,
                                   predictors,
                                   correctors,
                                   processing_parameters,
                                   user_defined_parameters=udp,
                                   category=initial_category)
        # User defined parameters
        udp = processor.user_defined_parameters
    except ValueError:
        print()
        print("=" * 15)
        print("===  ERROR  ===")
        print("=" * 15)
        print('The processor parameters are not correctly specified. \n'
              'Check your user_defined_parameters file first '
              'if you used one, and if that does not solve the issue, contact the developers.')
        exit(1)

    if not categories:
        # Processor name
        processor_name = processor.get_name()

        # Process all subjects
        print()
        print('Processing...')
        results = processor.process()
        print('Done processing')

        correction_params = results.correction_parameters
        prediction_params = results.prediction_parameters

        """ STORE RESULTS """
        print('Storing the results...')
        output_folder_name = '{}-{}'.format(prefix, processor_name) if prefix else processor_name
        output_folder = path.join(output_dir, output_folder_name)

        # Check if directory exists
        if not path.isdir(output_folder):
            # Create directory
            os.makedirs(output_folder)

        # Filenames
        udp_file = prefix + '-user_defined_parameters.txt' if prefix else 'user_defined_parameters.txt'
        p_file = prefix + '-prediction_parameters.nii.gz' if prefix else 'prediction_parameters.nii.gz'
        c_file = prefix + '-correction_parameters.nii.gz' if prefix else 'correction_parameters.nii.gz'

        # Save user defined parameters
        with open(path.join(output_folder, udp_file), 'wb') as f:
            f.write(str(udp) + '\n')

        # Save correction and prediction parameters
        niiImage = nib.Nifti1Image
        p_image = niiImage(prediction_params, affine_matrix)
        c_image = niiImage(correction_params, affine_matrix)
        nib.save(p_image, path.join(output_folder, p_file))
        nib.save(c_image, path.join(output_folder, c_file))

        print('Done')

    else:
        # Process each category
        for category in categories:

            # Create processor for this category
            processor = MixedProcessor(subjects,
                                       predictors_names,
                                       correctors_names,
                                       predictors,
                                       correctors,
                                       processing_parameters,
                                       category=category,
                                       user_defined_parameters=udp)

            # Processor name
            processor_name = processor.get_name()

            print()
            print('Processing category', category, '...')
            results = processor.process()
            print('Done processing')
            print()

            processor_name = processor.get_name()
            correction_params = results.correction_parameters
            prediction_params = results.prediction_parameters

            """ STORE RESULTS """
            print('Storing the results...')
            output_folder_name = '{}-{}'.format(prefix, processor_name) if prefix else processor_name
            output_folder = path.join(output_dir, output_folder_name)

            # Check if directory exists
            if not path.isdir(output_folder):
                # Create directory
                os.makedirs(output_folder)

            # Filenames
            udp_file = prefix + '-user_defined_parameters.txt' if prefix else 'user_defined_parameters.txt'
            p_file = prefix + '-prediction_parameters.nii.gz' if prefix else 'prediction_parameters.nii.gz'
            c_file = prefix + '-correction_parameters.nii.gz' if prefix else 'correction_parameters.nii.gz'

            # Save user defined parameters
            with open(path.join(output_folder, udp_file), 'wb') as f:
                f.write(str(udp) + '\n')

            # Save correction and prediction parameters
            niiImage = nib.Nifti1Image
            p_image = niiImage(prediction_params, affine_matrix)
            c_image = niiImage(correction_params, affine_matrix)
            nib.save(p_image, path.join(output_folder, p_file))
            nib.save(c_image, path.join(output_folder, c_file))

            print('Done category', category)

        print()
        print('Done')
