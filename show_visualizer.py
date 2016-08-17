from __future__ import print_function

from argparse import ArgumentParser
from glob import glob
from os import path

from src import helper_functions
from src.Visualization.GUIVisualizer import GUIVisualizer

import numpy as np
import nibabel as nib

if __name__ == '__main__':

    """ CLI ARGUMENTS """
    arguments_parser = ArgumentParser(description='Shows the graphical visualizer to display a statistical map '
                                                  'and the curves for the selected voxel.')

    arguments_parser.add_argument('configuration_file', help="Path to the YAML configuration file "
                                                             "used to load the data for this study.")
    arguments_parser.add_argument('dir', help='Specify one directory within the '
                                              'results directory specified in the '
                                              'configuration file from which the '
                                              'fitting parameters and the map '
                                              'should be loaded.')
    arguments_parser.add_argument('map', help='Name of the statistical map to be loaded '
                                              '(e.g invfiltered_vnprss_gamma0.0003_percentile0.1.nii')

    arguments = arguments_parser.parse_args()
    config_file = arguments.configuration_file
    directory = arguments.dir
    map_name = arguments.map

    """ LOAD DATA USING DATALOADER """
    subjects, predictors_names, correctors_names, predictors, correctors, processing_parameters, affine_matrix, \
    output_dir = helper_functions.load_data_from_config_file(config_file)
    template = helper_functions.load_template_from_config_file(config_file)

    """ LOAD DATA TO SHOW CURVES """
    print('Loading results data...')
    print()
    full_path = path.join(output_dir, directory)
    pathname = glob(path.join(full_path, '*prediction_parameters.nii.gz'))
    # If there is no coincidence, ignore this directory
    if len(pathname) == 0:
        print('{} does not exist or contain any result.'.format(full_path))
        exit(1)
    n, cat, pred_p, corr_p, proc = helper_functions.get_results_from_path(
        pathname[0], subjects, predictors_names, correctors_names, predictors, correctors,
        processing_parameters
    )

    """ LOAD STATISTICAL MAP """
    full_path = path.join(output_dir, directory, map_name)
    if not path.isfile(full_path):
        print('{} does not exist or is not recognized as a file. Try again with a valid file.'.format(full_path))
        exit(1)
    map_data = nib.load(full_path).get_data()
    if map_data.shape != template.shape:
        print("The specified map and the template don't have the same dimensions. Try again with a valid statisitcal "
              "map")
        exit(1)
    # Mask the map
    masked_map_data = np.ma.masked_equal(map_data, 0.0)

    """ X-AXIS DATA FOR THE CURVES """
    pred_max, pred_min = proc.predictors.max(), proc.predictors.min()
    x = np.linspace(pred_min, pred_max, num=100)

    """ INIT VISUALIZER """
    visualizer = GUIVisualizer(template=template)
    visualizer.set_corrected_data_processor(processor=proc, correction_parameters=corr_p)
    visualizer.add_image(masked_map_data)
    plot_label = '{} / '.format(n)
    plot_label += 'Category {}'.format(cat) if cat is not None else 'All data'
    visualizer.add_curve_processor(processor=proc, prediction_parameters=pred_p, label=plot_label)
    visualizer.show()
