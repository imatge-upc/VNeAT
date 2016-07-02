from argparse import ArgumentParser
from glob import glob
from os import path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

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
    return name, pred_parameters, corr_parameters, processor


if __name__ == '__main__':

    """ CONSTANTS """
    AVAILABLE_COLORS = [
        '#34314c',
        '#ff7473',
        '#58C9B9',
        '#E71D36',
        '#9055A2'
    ]
    AVAILABLE_MARKERS = [
        'v',
        '^',
        'o',
        '>',
        '<'
    ]

    """ CLI ARGUMENTS """
    arguments_parser = ArgumentParser(description='Shows the curves for the fitting results computed by'
                                                  ' compute_parameters.py. By default shows all computed'
                                                  ' parameters inside the results folder specified in the'
                                                  ' configuration file')

    arguments_parser.add_argument('configuration_file', help="Path to the YAML configuration file"
                                                             " used to load the data for this study.")
    arguments_parser.add_argument('--dirs', nargs='+', help='Specify one or several directories within the '
                                                            'results directory specified in the '
                                                            ' configuration file from which the '
                                                            ' parameters should be loaded.')
    arguments_parser.add_argument('--compare', const=True, nargs='?',
                                  help='Plots the curves in the same figure so that you are able'
                                       ' to compare the different curves. The program does not'
                                       ' recognize whether the data has been corrected with the'
                                       ' same fitter or not, so you must ensure this to have '
                                       ' coherent results.')

    arguments = arguments_parser.parse_args()
    config_file = arguments.configuration_file
    dirs = arguments.dirs
    compare = arguments.compare

    """ LOAD DATA USING DATALOADER """
    print 'Loading configuration data...'
    try:
        data_loader = DataLoader(config_file)
    except IOError as e:
        print
        print e.filename + ' does not exist.'
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
        print
        print 'Configuration file does not have the specified format.'
        print 'See config/exampleConfig.yaml for further information about the format of configuration ' \
              'files'
        subjects = predictors = correctors = None
        predictors_names = correctors_names = None
        processing_parameters = affine_matrix = output_dir = None
        exit(1)

    # Lists to store the necessary data to show the curves
    names = []
    prediction_parameters = []
    correction_parameters = []
    processors = []

    """ LOAD DATA TO SHOW CURVES """
    if dirs is None:
        print 'Loading results data...'
        print
        # Find prediction parameters inside results folder
        pathname = path.join(output_dir, '**', '*prediction_parameters.nii.gz')
        for p in glob(pathname):
            n, pred_p, corr_p, proc = get_results_from_path(p)
            names.append(n)
            prediction_parameters.append(pred_p)
            correction_parameters.append(corr_p)
            processors.append(proc)

    else:
        print 'Loading results data...'
        print
        for directory in dirs:
            full_path = path.join(output_dir, directory)
            pathname = glob(path.join(full_path, '*prediction_parameters.nii.gz'))
            # If there is no coincidence, ignore this directory
            if len(pathname) == 0:
                print '{} does not exist or contain any result.'.format(full_path)
                continue
            n, pred_p, corr_p, proc = get_results_from_path(pathname[0])
            names.append(n)
            prediction_parameters.append(pred_p)
            correction_parameters.append(corr_p)
            processors.append(proc)

    if len(processors) == 0:
        print 'There are no results to be shown. Use compute_parameters.py first to generate them.'
        exit(0)

    """ ASK USER FOR VOXEL """
    while True:
        try:
            entry = raw_input('Write a tuple of mm coordinates (in MNI space) to display its curve '
                              '(or press Ctrl+D to exit): ')
        except EOFError:
            print
            print 'Program has finished.'
            print
            break
        except Exception as e:
            print '[ERROR] Unexpected error was found when reading input:'
            print e
            print
            continue
        try:
            x, y, z = map(float, eval(entry))
        except (NameError, TypeError, ValueError, EOFError):
            print '[ERROR] Input was not recognized'
            print 'To display the voxel with coordinates (x, y, z), please enter \'x, y, z\''
            print 'e.g., for voxel (57, 49, 82), type \'57, 49, 82\' (without inverted commas) as input'
            print
            continue
        except Exception as e:
            print '[ERROR] Unexpected error was found when reading input:'
            print e
            print
            continue

        print 'Processing request... please wait'

        try:
            # Transform mm coordinates -> voxel coordinates using affine
            mm_coordinates = np.array([x, y, z, 1])
            voxel_coordinates = map(int, np.round(np.linalg.inv(affine_matrix).dot(mm_coordinates)))
            # Get rounded mm coordinates in MNI space (due to 1.5 mm spacing)
            mm_coordinates_prima = affine_matrix.dot(voxel_coordinates)
            # Final voxel coordinates
            x = voxel_coordinates[0]
            y = voxel_coordinates[1]
            z = voxel_coordinates[2]

            print 'Voxel coordinates: {}, {}, {}'.format(x, y, z)

            color_counter = np.random.randint(0, len(AVAILABLE_COLORS) - 1)
            for i in range(len(processors)):
                # Get corrected grey matter data
                corrected_data = processors[i].corrected_values(
                    correction_parameters[i],
                    x1=x,
                    x2=x + 1,
                    y1=y,
                    y2=y + 1,
                    z1=z,
                    z2=z + 1
                )
                # Get curves
                axis, curve = processors[i].curve(
                    prediction_parameters[i],
                    x1=x,
                    x2=x + 1,
                    y1=y,
                    y2=y + 1,
                    z1=z,
                    z2=z + 1,
                    tpoints=100
                )
                # Plot data points
                category = processors[i].category
                label = 'Category ' + str(category) if category is not None else 'All subjects'
                plt.scatter(
                    processors[i].predictors,
                    corrected_data,
                    label=label,
                    s=50,
                    color=AVAILABLE_COLORS[color_counter],
                    marker=AVAILABLE_MARKERS[color_counter]
                )

                # Plot
                plt.plot(
                    axis,
                    curve[:, 0, 0, 0],
                    label=names[i],
                    lw=2,
                    color=AVAILABLE_COLORS[color_counter]
                )
                # Next color
                color_counter = color_counter + 1 if ((color_counter + 1) % len(AVAILABLE_COLORS)) != 0 else 0
                # TODO: Add optional lowess curve
                # Plot info
                plt.legend(fontsize='xx-large')
                plt.xlabel(predictors_names[0], fontsize='xx-large')
                plt.ylabel('Grey matter', fontsize='xx-large')
                plt_title = 'Coordinates: ' + \
                            str(mm_coordinates_prima[0]) + ', ' + \
                            str(mm_coordinates_prima[1]) + ', ' + \
                            str(mm_coordinates_prima[2]) + ' mm'
                plt.title(plt_title, size="xx-large")

                # Show in full screen mode
                backend = plt.get_backend()
                if backend == "Qt4Agg":
                    mng = plt.get_current_fig_manager()
                    mng.window.showMaximized()
                elif backend == 'TkAgg':
                    mng = plt.get_current_fig_manager()
                    mng.window.state('withdrawn')
                elif backend == 'wxAgg':
                    mng = plt.get_current_fig_manager()
                    mng.frame.Maximize(True)

                # Show current curve in tight mode if compare mode is off
                if compare is None:
                    plt.tight_layout()
                    plt.show()
                    print

            # Show all curves in tight mode if compare mode is on
            if compare is not None:
                plt.tight_layout()
                plt.show()
                print

        except Exception as e:
            print '[ERROR] Unexpected error occurred while computing and showing the results:'
            print e
            print
            continue
