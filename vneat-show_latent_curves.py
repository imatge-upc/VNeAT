#!/usr/bin/python
from __future__ import print_function

from argparse import ArgumentParser
from glob import glob
from os import path

import matplotlib.pyplot as plt
import numpy as np

from vneat import helper_functions
# plt.switch_backend('Agg')

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
    arguments_parser = ArgumentParser(description='Shows the curves for the fitting results computed by '
                                                  'compute_fitting.py. By default shows all computed '
                                                  'parameters inside the results folder specified in the '
                                                  'configuration file')

    arguments_parser.add_argument('configuration_file', help="Path to the YAML configuration file "
                                                             "used to load the data for this study.")
    arguments_parser.add_argument('--dirs', nargs='+', help='Specify one or several directories within the '
                                                            'results directory specified in the '
                                                            'configuration file from which the '
                                                            'parameters should be loaded.')
    arguments_parser.add_argument('--compare', default=True, nargs='?',
                                  help='Plots the curves in the same figure so that you are able '
                                       'to compare the different curves. The program does not '
                                       'recognize whether the data has been corrected with the '
                                       'same fitter or not, so you must ensure this to have '
                                       'coherent results.')

    arguments_parser.add_argument('--n_component', default=1, nargs='?',
                                  help='Number of component [1,...,L] that you want to plot. '
                                  )

    arguments = arguments_parser.parse_args()
    config_file = arguments.configuration_file
    dirs = arguments.dirs
    compare = arguments.compare
    n_component = arguments.n_component
    """ LOAD DATA USING DATALOADER """
    subjects, predictors_names, correctors_names, predictors, correctors, processing_parameters, \
    affine_matrix, output_dir, results_io, type_data = helper_functions.load_data_from_config_file(config_file)


    # Lists to store the necessary data to show the curves
    names = []
    prediction_parameters = []
    correction_parameters = []
    processors = []

    """ LOAD DATA TO SHOW CURVES """
    if dirs is None:
        print('Loading results data...')
        print()
        # Find prediction parameters inside results folder
        pathname = path.join(output_dir, '**', '*prediction_parameters'+results_io.extension)
        for p in glob(pathname):
            n, _, pred_p, corr_p, proc = helper_functions.get_results_from_path(
                p, subjects, predictors_names, correctors_names, predictors, correctors, processing_parameters
            )
            names.append(n)
            prediction_parameters.append(pred_p)
            correction_parameters.append(corr_p)
            processors.append(proc)

    else:
        print('Loading results data...')
        print()
        for directory in dirs:
            full_path = path.join(output_dir, directory)
            pathname = glob(path.join(full_path, '*prediction_parameters'+results_io.extension))
            # If there is no coincidence, ignore this directory
            if len(pathname) == 0:
                print('{} does not exist or contain any result.'.format(full_path))
                continue
            n, _, pred_p, corr_p, proc = helper_functions.get_results_from_path(
                pathname[0], results_io, subjects, predictors_names, correctors_names, predictors, correctors,
                processing_parameters, type_data
            )
            names.append(n)
            prediction_parameters.append(pred_p)
            correction_parameters.append(corr_p)
            processors.append(proc)

    if len(processors) == 0:
        print('There are no results to be shown. Use compute_fitting.py first to generate them.')
        exit(0)

    """ ASK USER FOR VOXEL """
    while True:
        try:
            entry = input('Write a tuple of mm coordinates (in MNI space) to display its curve '
                              '(or press Ctrl+D to exit): ')
        except EOFError:
            print()
            print('Program has finished.')
            print()
            break
        except Exception as e:
            print('[ERROR] Unexpected error was found when reading input:')
            print(e)
            print()
            continue
        try:
            x, y, z = list(map(float, eval(entry)))
        except (NameError, TypeError, ValueError, EOFError):
            print('[ERROR] Input was not recognized')
            if type_data == 'vol':
                print('To display the voxel with coordinates (x, y, z), please enter \'x, y, z\'')
                print('e.g., for voxel (57, 49, 82), type \'57, 49, 82\' (without inverted commas) as input')
            else:
                print('To display the vertex with coordinates (x,), please enter \'x\'')
                print('e.g., for voxel (57,), type \'57\' (without inverted commas) as input')
            print()
            continue
        except Exception as e:
            print('[ERROR] Unexpected error was found when reading input:')
            print(e)
            print()
            continue

        print('Processing request... please wait')

        try:

            if type_data == 'vol':
                # Transform mm coordinates -> voxel coordinates using affine
                mm_coordinates = np.array([x, y, z, 1])
                voxel_coordinates = list(map(int, np.round(np.linalg.inv(affine_matrix).dot(mm_coordinates))))
                # Get rounded mm coordinates in MNI space (due to 1.5 mm spacing)
                mm_coordinates_prima = affine_matrix.dot(voxel_coordinates)
                # Final voxel coordinates
                x = voxel_coordinates[0]
                y = voxel_coordinates[1]
                z = voxel_coordinates[2]

                print('Voxel coordinates: {}, {}, {}'.format(x, y, z))



            fig_list = []
            for i in range(predictors.shape[1]):
                fig_list.append(plt.figure().number)

            color_counter = np.random.randint(0, len(AVAILABLE_COLORS) - 1)

            print('[SHOW_LATENT_CURVES]: ' + str(len(processors)))
            for i in range(len(processors)):
                # Get corrected grey matter data
                if type_data == 'vol':
                    # corrected_data = processors[i].corrected_values(
                    #     correction_parameters[i],
                    #     x1=x,
                    #     x2=x + 1,
                    #     y1=y,
                    #     y2=y + 1,
                    #     z1=z,
                    #     z2=z + 1
                    # )
                    # corrected_data = np.reshape(corrected_data, (predictors.shape[0],))
                    corrected_data = np.zeros((predictors.shape[0],))
                    print('[SHOW_LATENT_CURVES] data correction DONE')

                    # Get curves
                    _, hidden_predictors = processors[i].curve(
                        prediction_parameters[i],
                        x1=x,
                        x2=x + 1,
                        y1=y,
                        y2=y + 1,
                        z1=z,
                        z2=z + 1,
                        tpoints=-1
                    )
                    hidden_predictors = hidden_predictors.reshape(predictors.shape)
                    print('[SHOW_LATENT_CURVES] hidden predictors DONE')

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
                    curve = curve.reshape((axis.shape[1], -1))
                    print('[SHOW_LATENT_CURVES] curve DONE')


                else:

                    corrected_data = processors[i].corrected_values(
                        correction_parameters[i],
                        x1=x,
                        x2=x + 1,
                    )
                    # corrected_data = np.zeros((predictors.shape[0],1,1,1))

                    # Get curves
                    _, hidden_predictors = processors[i].curve(
                        prediction_parameters[i],
                        x1=x,
                        x2=x + 1,
                        tpoints=-1
                    )
                    hidden_predictors = hidden_predictors.reshape(predictors.shape)

                    # Get curves
                    axis, curve = processors[i].curve(
                        prediction_parameters[i],
                        x1=x,
                        x2=x + 1,

                        tpoints=100
                    )
                    curve = curve.reshape((axis.shape[1], -1))


                # Plot data points
                for it_a, a in enumerate(axis):
                    category = processors[i].category

                    print(corrected_data.shape)
                    print(hidden_predictors.shape)
                    corrected_data_plot = corrected_data
                    for it_c in range(curve.shape[1]):
                        if it_c != it_a:
                            corrected_data_plot -= hidden_predictors[:,it_a]

                    plt.figure(fig_list[it_a])
                    label = 'Category ' + str(category) if category is not None else 'All subjects'
                    plt.scatter(
                        processors[i].predictors[:,it_a],
                        corrected_data_plot,
                        label=label,
                        s=50,
                        color=AVAILABLE_COLORS[color_counter],
                        marker=AVAILABLE_MARKERS[color_counter]
                    )

                    # Plot
                    plt.plot(
                        a,
                        curve[:, it_a],
                        label=names[i],
                        lw=2,
                        color=AVAILABLE_COLORS[color_counter]
                    )
                    # Next color
                    color_counter = color_counter + 1 if ((color_counter + 1) % len(AVAILABLE_COLORS)) != 0 else 0
                    # Plot info
                    plt.legend(fontsize='xx-large')
                    plt.xlabel(predictors_names[it_a], fontsize='xx-large')
                    plt.ylabel('Grey matter', fontsize='xx-large')
                    if type_data == 'vol':
                        plt_title = 'Coordinates: ' + \
                                    str(mm_coordinates_prima[0]) + ', ' + \
                                    str(mm_coordinates_prima[1]) + ', ' + \
                                    str(mm_coordinates_prima[2]) + ' mm'
                    else:
                        plt_title = 'Vertex: ' + str(x)

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
                if not compare:
                    plt.tight_layout()
                    plt.show()
                    print()

            # Show all curves in tight mode if compare mode is on
            if compare:
                plt.tight_layout()
                plt.show()
                print()

        except Exception as e:
            print('[ERROR] Unexpected error occurred while computing and showing the results:')
            print(e)
            print()
            continue