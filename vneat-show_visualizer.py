#!/usr/bin/python
from __future__ import print_function

from argparse import ArgumentParser
from glob import glob
from os import path

import nibabel as nib
import numpy as np

from vneat import helper_functions
from vneat.Utils.niftiIO import get_atlas_image_labels
from vneat.Visualization.GUIVisualizer import GUIVisualizer, GUIVisualizer_surface

if __name__ == '__main__':

    """ CLI ARGUMENTS """
    arguments_parser = ArgumentParser(description='Shows the graphical visualizer to display a statistical map '
                                                  'and the curves for the selected voxel.')

    arguments_parser.add_argument('configuration_file', help="Path to the YAML configuration file "
                                                             "used to load the data for this study.")

    arguments_parser.add_argument('dirs', nargs="+",
                                  help='Specify one or more directories within the results directory specified in the '
                                       'configuration file from which the fitting parameters should be loaded.')

    arguments_parser.add_argument('--map',
                                  help='Path relative to the output directory specified in the configuration file'
                                       'to the statistical map to be loaded')

    arguments_parser.add_argument('--atlas',
                                  help='Absolute path to the atlas file to load')

    arguments_parser.add_argument('--atlas_labels',
                                  help='Absolute path to the atlas_labels file to load. The csv file should have'
                                       'a header line containing ROIid and ROIname columns')

    arguments_parser.add_argument('--mask',
                                  help='Absolute path to the mask file to load')

    arguments_parser.add_argument('--colormap', choices=['hot', 'rainbow', 'rgb'], default='hot',
                                  help="Color map used to paint the statistical maps' values. By default it is 'hot',"
                                       "useful for statistical based measures (F-stat, p-values, Z-scores, etc.),"
                                       "but you can use 'rainbow' for labeled maps")

    arguments_parser.add_argument('--n-points', default=100, type=int,
                                  help='Number of points used to plot the curves. More points means a smoother curve '
                                       'but requires more computational resources')

    arguments = arguments_parser.parse_args()
    config_file = arguments.configuration_file
    dirs = arguments.dirs
    n_points = arguments.n_points
    map_name = arguments.map
    atlas_name = arguments.atlas
    atlas_dict_name = arguments.atlas_labels
    mask_name = arguments.mask
    color_map = arguments.colormap


    """ LOAD DATA USING DATALOADER """
    subjects, predictors_names, correctors_names, predictors, correctors, processing_parameters, affine_matrix, \
    output_dir, results_io, type_data = helper_functions.load_data_from_config_file(config_file)
    template = helper_functions.load_template_from_config_file(config_file)

    """ INIT VISUALIZER """
    if type_data == 'vol':
        visualizer = GUIVisualizer(template=template, affine=affine_matrix, num_points=n_points)
    else:
        visualizer = GUIVisualizer_surface(template=template, affine=affine_matrix, num_points=n_points)


    """ LOAD STATISTICAL MAP """
    if map_name is not None:
        full_path = path.join(output_dir, map_name)
        if not path.isfile(full_path):
            print('{} does not exist or is not recognized as a file. Try again with a valid file.'.format(full_path))
            exit(1)

        map_data = results_io.loader(full_path).get_data()
        if type_data == 'vol':
            if map_data.shape[:3] != template.shape[:3]:
                print("The specified map and the template don't have the same dimensions. Try again with a valid statistical "
                      "map")
                exit(1)
        else:
            if map_data.shape[0] != template[0].shape[0]:
                print(
                    "The specified map and the template don't have the same dimensions. Try again with a valid statistical "
                    "map")
                exit(1)

        # Mask the map
        if color_map == 'rgb':
            masked_map_data = np.ma.masked_equal(np.sum(map_data,axis=-1), 0.0).astype(int)
            mask = np.invert(masked_map_data.mask[..., np.newaxis]).astype(int)
            print(np.unique(mask))
            print(mask.shape)

            masked_map_data = np.concatenate((map_data, mask),axis=-1)
            # masked_map_data = np.concatenate((map_data, masked_map_data.mask[..., np.newaxis]),axis=-1)
        else:
            masked_map_data = np.ma.masked_equal(map_data, 0.0)
        # Add it to the visualizer
        visualizer.add_image(masked_map_data, colormap=color_map)

    if mask_name is not None:
        full_path = mask_name
        if not path.isfile(full_path):
            print('{} does not exist or is not recognized as a file. Try again with a valid file.'.format(full_path))
            exit(1)

        map_data = results_io.loader(full_path).get_data()
        if type_data == 'vol':
            if map_data.shape != template.shape:
                print(
                    "The specified map and the template don't have the same dimensions. Try again with a valid statistical "
                    "map")
                exit(1)
        else:
            if map_data.shape[0] != template[0].shape[0]:
                print(
                    "The specified map and the template don't have the same dimensions. Try again with a valid statistical "
                    "map")
                exit(1)

        # Mask the map
        masked_map_data = np.ma.masked_equal(map_data, 0.0)

        # Add it to the visualizer
        visualizer.add_mask(masked_map_data)

    if atlas_name is not None and atlas_dict_name is not None:
        atlas_image, atlas_dict_image = get_atlas_image_labels(results_io, atlas_name, atlas_dict_name)
        visualizer.add_atlas(atlas_image, atlas_dict_image)

    # new_voxel = np.asarray(list(map(lambda x: int(x / 2), template.shape)))
    # x, y, z = new_voxel
    # print(new_voxel)
    # print(atlas_image[x, y, z])
    # print(atlas_dict_image[atlas_image[x, y, z]])
    # exit()
    """ LOAD DATA TO SHOW CURVES """
    print('Loading results data...')
    print()
    for directory in dirs:
        full_path = path.join(output_dir, directory)
        pathname = glob(path.join(full_path, '*prediction_parameters' + results_io.extension))

        # If there is no coincidence, ignore this directory
        if len(pathname) == 0:
            print('{} does not exist or contain any result.'.format(full_path))
            continue

        n, cat, pred_p, corr_p, proc = helper_functions.get_results_from_path(
            pathname[0], results_io, subjects, predictors_names, correctors_names, predictors, correctors,
            processing_parameters, type_data
        )

        plot_label = '{} / '.format(n)
        plot_label += cat if cat is not None else 'All subjects'
        print('VISUALIZER:')
        print(pred_p.shape)
        visualizer.add_curve_processor(processor=proc, prediction_parameters=pred_p, correction_parameters=corr_p,
                                       label=plot_label)

    """ SHOW VISUALIZER """
    visualizer.show()
