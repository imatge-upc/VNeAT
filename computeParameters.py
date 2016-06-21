import nibabel as nib
import os
import os.path as path

from argparse import ArgumentParser
from Processors.MixedProcessor import MixedProcessor
from Utils.DataLoader import DataLoader

if __name__ == '__main__':

    """ PARSE ARGUMENTS FROM CLI """
    arg_parser = ArgumentParser(description='Computes the fitting parameters for the data '
                                            'provided in the configuration file.')
    arg_parser.add_argument('configuration_file', help='Path to the YAML configuration file'
                                                       ' for this study')
    arg_parser.add_argument('--parameters', '-p', help='Path to the txt file with user defined'
                                                       ' parameters to load a pre-configured'
                                                       ' correction and prediction processor.')
    arguments = arg_parser.parse_args()
    config_file = arguments.configuration_file

    if arguments.parameters:
        # Load user defined parameters
        try:
            with open(arguments.parameters, 'rb') as f:
                udp = eval(f.read())
                print
                print 'User defined parameters have been successfully loaded.'
        except IOError as ioe:
            print
            print 'The provided parameters file, ' + ioe.filename + ', does not exist.'
            print 'Standard input will be used to configure the correction and prediction processors' \
                  ' instead.'
            print
            udp = ()
        except SyntaxError:
            print
            print 'The provided parameters file is not properly formatted.'
            print 'Standard input will be used to configure the correction and prediction processors' \
                  ' instead.'
            print
            udp = ()
        except:
            print
            print 'An unexpected error happened.'
            print 'Standard input will be used to configure the correction and prediction processors' \
                  ' instead.'
            print
            udp = ()
    else:
        udp = ()

    """ LOAD DATA USING DATALOADER """
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

    """ PROCESSING """
    # Create MixedProcessor instance
    processor = MixedProcessor(subjects,
                               predictors_names,
                               correctors_names,
                               predictors,
                               correctors,
                               processing_parameters,
                               user_defined_parameters=udp)
    # Processor name
    processor_name = processor.get_name()

    print
    print 'Processing...'
    results = processor.process()
    print 'Done processing'
    print

    correction_params = results.correction_parameters
    prediction_params = results.prediction_parameters
    udp = processor.user_defined_parameters

    """ STORE RESULTS """
    print 'Storing the results...'
    output_folder = path.join(data_loader.get_output_dir(), processor_name)

    # Check if directory exists
    if not path.isdir(output_folder):
        # Create directory
        os.makedirs(output_folder)

    # Save user defined parameters
    with open(path.join(output_folder, 'user_defined_parameters.txt'), 'wb') as f:
        f.write(str(udp) + '\n')

    # Save correction and prediction parameters
    niiImage = nib.Nifti1Image
    affine_matrix = data_loader.get_template_affine()
    p_image = niiImage(prediction_params, affine_matrix)
    c_image = niiImage(correction_params, affine_matrix)
    nib.save(p_image, path.join(output_folder, 'prediction_parameters.nii.gz'))
    nib.save(c_image, path.join(output_folder, 'correction_parameters.nii.gz'))

    print 'Done'


