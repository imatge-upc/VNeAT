from __future__ import print_function

from argparse import ArgumentParser
from glob import glob
from os import path

import nibabel as nib
import numpy as np

from src import helper_functions
from src.Visualization.GUIVisualizer import GUIVisualizer

if __name__ == '__main__':

    """ CLI ARGUMENTS """
    arguments_parser = ArgumentParser(description='Shows the graphical visualizer to display a statistical map '
                                                  'and the curves for the selected voxel.')

    arguments_parser.add_argument('configuration_file', help="Path to the YAML configuration file "
                                                             "used to load the data for this study.")
    arguments_parser.add_argument('map',
                                  help='Path relative to the output directory specified in the configuration file'
                                       'to the statistical map to be loaded')

    arguments_parser.add_argument('--colormap', choices=['hot', 'rainbow'], default='hot',
                                  help="Color map used to paint the statistical maps' values. By default it is 'hot',"
                                       "useful for statistical based measures (F-stat, p-values, Z-scores, etc.),"
                                       "but you can use 'rainbow' for labeled maps")

    arguments_parser.add_argument('dirs', nargs="+",
                                  help='Specify one or more directories within the results directory specified in the '
                                       'configuration file from which the fitting parameters should be loaded.')

    arguments_parser.add_argument('--n_points', default=100, type=int,
                                  help='Number of points used to plot the curves. More points means a smoother curve '
                                       'but requires more computational resources')

    arguments = arguments_parser.parse_args()
    config_file = arguments.configuration_file
    dirs = arguments.dirs
    n_points = arguments.n_points
    map_name = arguments.map
    color_map = arguments.colormap

    """ LOAD DATA USING DATALOADER """
    subjects, predictors_names, correctors_names, predictors, correctors, processing_parameters, affine_matrix, \
    output_dir = helper_functions.load_data_from_config_file(config_file)
    template = helper_functions.load_template_from_config_file(config_file)

    """ INIT VISUALIZER """
    visualizer = GUIVisualizer(template=template, affine=affine_matrix, num_points=n_points)

    """ LOAD STATISTICAL MAP """
    full_path = path.join(output_dir, map_name)
    if not path.isfile(full_path):
        print('{} does not exist or is not recognized as a file. Try again with a valid file.'.format(full_path))
        exit(1)
    map_data = nib.load(full_path).get_data()
    if map_data.shape != template.shape:
        print("The specified map and the template don't have the same dimensions. Try again with a valid statistical "
              "map")
        exit(1)
    # Mask the map
    masked_map_data = np.ma.masked_equal(map_data, 0.0)
    # Add it to the visualizer
    visualizer.add_image(masked_map_data, colormap=color_map)

    """ LOAD DATA TO SHOW CURVES """
    print('Loading results data...')
    print()
    for directory in dirs:
        full_path = path.join(output_dir, directory)
        pathname = glob(path.join(full_path, '*prediction_parameters.nii.gz'))

        # If there is no coincidence, ignore this directory
        if len(pathname) == 0:
            print('{} does not exist or contain any result.'.format(full_path))
            continue

        n, cat, pred_p, corr_p, proc = helper_functions.get_results_from_path(
            pathname[0], subjects, predictors_names, correctors_names, predictors, correctors,
            processing_parameters
        )

        plot_label = '{} / '.format(n)
        plot_label += cat if cat is not None else 'All subjects'
        visualizer.add_curve_processor(processor=proc, prediction_parameters=pred_p, correction_parameters=corr_p,
                                       label=plot_label)

    """ SHOW VISUALIZER """
    visualizer.show()
