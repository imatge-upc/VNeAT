from __future__ import print_function

from argparse import ArgumentParser
from glob import glob
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import helper_functions

if __name__ == '__main__':

    """ CONSTANTS """
    AVAILABLE_PLOTS = [
        'univariate_density',
        'bivariate_density',
        'boxplot',
        'categorical_boxplot'
    ]

    """ CLI ARGUMENTS """
    arguments_parser = ArgumentParser(description='Shows the data distribution of the observations, the'
                                                  ' predictors, the correctors and the residuals.')

    arguments_parser.add_argument('configuration_file', help="Path to the YAML configuration file"
                                                             " used to load the data for this study.")
    arguments_parser.add_argument('plot', choices=AVAILABLE_PLOTS,
                                  help='Type of plot to be used. For the categorical_boxplot it is '
                                       'assumed that the dirs specified belong to different categories '
                                       'of the data. Otherwise, only the last data retrieved from a '
                                       'specific category will be taken into account.')
    arguments_parser.add_argument('--dirs', nargs='+', help='Specify one or several directories within the '
                                                            'results directory specified in the '
                                                            'configuration file from which the '
                                                            'parameters should be loaded.')

    arguments = arguments_parser.parse_args()
    config_file = arguments.configuration_file
    plot_name = arguments.plot
    dirs = arguments.dirs

    """ LOAD DATA USING DATALOADER """
    subjects, predictors_names, correctors_names, predictors, correctors, processing_parameters, \
    affine_matrix, output_dir = helper_functions.load_data_from_config_file(config_file)

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
        pathname = path.join(output_dir, '**', '*prediction_parameters.nii.gz')
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
            pathname = glob(path.join(full_path, '*prediction_parameters.nii.gz'))
            # If there is no coincidence, ignore this directory
            if len(pathname) == 0:
                print('{} does not exist or contain any result.'.format(full_path))
                continue
            n, _, pred_p, corr_p, proc = helper_functions.get_results_from_path(
                pathname[0], subjects, predictors_names, correctors_names, predictors, correctors,
                processing_parameters
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
            entry = raw_input('Write a tuple of mm coordinates (in MNI space) to display its curve '
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
            x, y, z = map(float, eval(entry))
        except (NameError, TypeError, ValueError, EOFError):
            print('[ERROR] Input was not recognized')
            print('To display the voxel with coordinates (x, y, z), please enter \'x, y, z\'')
            print('e.g., for voxel (57, 49, 82), type \'57, 49, 82\' (without inverted commas) as input')
            print()
            continue
        except Exception as e:
            print('[ERROR] Unexpected error was found when reading input:')
            print(e)
            print()
            continue

        print('Processing request... please wait')

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

            title = 'Voxel {}, {}, {}'.format(*mm_coordinates_prima[:-1])
            print('Voxel coordinates: {}, {}, {}'.format(x, y, z))

            # Define a selected option for categorical boxplot, so that when the user selects a
            # feature in the first loop it is remembered for the following loops
            cat_boxplot_feature = False

            # Define a dictionary to store the categorical data for categorical boxplot
            cat_data_dict = {}

            for i in range(len(processors)):
                # Only plot processor name in plot methods different than categorical boxplot, as in the
                # latter there will be no explicit iteration over processors. That is, the user
                # selects the feature at the beginning of the iteration and the result is shown at the
                # end of it, so there is no need for the user to know the current state of the iteration.
                if plot_name != "categorical_boxplot":
                    print()
                    print('Processor {} ({} of {})'.format(processors[i].get_name(), i+1, len(processors)))

                # Available x and y data
                options = {
                    'Observations': np.ravel(processors[i].gm_values(
                        x1=x,
                        x2=x + 1,
                        y1=y,
                        y2=y + 1,
                        z1=z,
                        z2=z + 1
                    )),
                    'Residuals': np.ravel(processors[i].corrected_values(
                        correction_parameters[i],
                        x1=x,
                        x2=x + 1,
                        y1=y,
                        y2=y + 1,
                        z1=z,
                        z2=z + 1
                    ))
                }
                for ind, p_name in enumerate(predictors_names):
                    options[p_name] = processors[i].predictors[:, ind]
                for ind, c_name in enumerate(correctors_names):
                    options[c_name] = processors[i].correctors[:, ind]

                """ ASK FOR INPUT """
                print()
                if plot_name == 'univariate_density' or plot_name == 'bivariate_density':
                    print('What feature do you want to have in the x-axis? ')
                elif (plot_name == 'boxplot') or (plot_name == 'categorical_boxplot' and not cat_boxplot_feature):
                    print('What is the feature that you want to plot in a boxplot? ')

                if plot_name != 'categorical_boxplot' or not cat_boxplot_feature:
                    print()
                    print('    OPTIONS     ')
                    print()
                    for key in options:
                        print(key)
                    print()
                    while True:
                        x_option_input = raw_input('Selected option: ')
                        if x_option_input in options:
                            break
                        else:
                            print('Invalid option. Try again.')

                x_data = options[x_option_input]

                if plot_name == 'bivariate_density':
                    print()
                    print('What do you want to have in the y-axis? ')
                    print('----------------')
                    print('    OPTIONS     ')
                    print('----------------')
                    for key in options:
                        print(key)
                    while True:
                        y_option_input = raw_input('Selected option: ')
                        if y_option_input in options:
                            break
                        else:
                            print('Invalid option. Try again.')
                    y_data = options[y_option_input]

                elif plot_name == 'categorical_boxplot':
                    # Put it to True in the first loop, so that subsequent loops are
                    # performed without a blocking input request.
                    cat_boxplot_feature = True

                """ STATISTICAL PLOTS """
                cat = processors[i].category
                label = 'Category {}'.format(cat) if cat is not None else 'All categories'
                if plot_name == 'univariate_density':
                    x_series = pd.Series(data=x_data, name=x_option_input)
                    sns.distplot(x_series, rug=True)
                    plt.title(' - '.join([title, label]))
                    plt.show()
                    print('__________________________________')
                    print()
                elif plot_name == 'bivariate_density':
                    xy_data = {
                        x_option_input: x_data,
                        y_option_input: y_data
                    }
                    xy_frame = pd.DataFrame(data=xy_data)
                    sns.jointplot(x=x_option_input, y=y_option_input, data=xy_frame, kind="kde")
                    plt.title(' - '.join([title, label]))
                    plt.show()
                    print('__________________________________')
                    print()
                elif plot_name == 'boxplot':
                    x_series = pd.Series(data=x_data, name=x_option_input)
                    sns.boxplot(data=x_series)
                    plt.ylabel(x_option_input)
                    plt.title(' - '.join([title, label]))
                    plt.show()
                    print('__________________________________')
                    print()
                elif plot_name == 'categorical_boxplot':
                    cat_data_dict[label] = pd.Series(data=x_data, name=x_option_input)
                else:
                    print('This plot option is not available.')
                    exit(1)

            # Only for categorical boxplot do the plot at the end of the loop
            if plot_name == 'categorical_boxplot':
                categorical_data = pd.DataFrame(data=cat_data_dict)
                sns.boxplot(data=categorical_data)
                plt.ylabel(x_option_input)
                plt.title(title)
                plt.show()

        except Exception as e:
            print('[ERROR] Unexpected error occurred while computing and showing the results:')
            print(e)
            print()
            continue
