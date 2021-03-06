#!/usr/bin/python
from __future__ import print_function

from argparse import ArgumentParser
from os import path

import nibabel as nib
import numpy as np

from vneat.Utils.DataLoader import DataLoader


if __name__ == '__main__':

    """ CONSTANTS """
    COMPARISON_CHOICES = [
        'best',
        'rgb',
        'absdiff',
        'se'
    ]


    """ CLI ARGUMENTS """
    arguments_parser = ArgumentParser(description='Compares statistical maps generated by '
                                                  'compute_statistical_maps.py using four possible techniques:'
                                                  'RGB map, best-fitting map, absolute difference or squared error. '
                                                  'You must specify the specific maps to compare and ensure that they '
                                                  'are comparable (Z-score map vs Z-score map, p-value map vs '
                                                  'p-value map, etc.)')

    arguments_parser.add_argument('configuration_file', help="Path to the YAML configuration file "
                                                             "used to load the data for this study.")

    arguments_parser.add_argument('files', nargs='+',
                                  help='Specify two or more files within the '
                                       'results directory to be compared.')

    arguments_parser.add_argument('--method', default='best',
                                  choices=COMPARISON_CHOICES,
                                  help='Method to compare the fitting score '
                                       'per voxel and create a new statistical'
                                       'map out of this comparison.')

    arguments_parser.add_argument('--name',  help='Name to be prepended to the output file.')

    arguments = arguments_parser.parse_args()
    config_file = arguments.configuration_file
    method = arguments.method
    files = arguments.files
    name = arguments.name

    # Check if there are at least two files
    if len(files) < 2:
        print('The minimum number of files to be compared is two.')
        exit(1)

    """ LOAD DATA USING DATALOADER """
    print('Loading configuration data...')
    try:
        data_loader = DataLoader(config_file)
    except IOError as e:
        print()
        print(e.filename + ' does not exist.')
        data_loader = None
        exit(1)

    # Load all necessary data:
    try:
        affine_matrix = data_loader.get_template_affine()
        output_dir = data_loader.get_output_dir()
        results_io = data_loader.get_results_io()


    except KeyError:
        print()
        print('Configuration file does not have the specified format.')
        print('See config/exampleConfig.yaml for further information about the format of configuration '
              'files')
        affine_matrix = output_dir = results_io = None
        exit(1)

    """ LOAD RESULTS DATA """
    nii_files = []
    for f in files:
        try:
            nii_files.append(results_io.loader(path.join(output_dir, f)))
        except IOError as e:
            print()
            print(e.filename + ' does not exist')

    if len(nii_files) < 2:
        print()
        print('You have to provide at least two valid files to be compared.')
        exit(1)

    """ APPLY COMPARISON """
    if method == 'best':
        print('Computing best fit map and best fit model...')
        # Load first available file and sequentially load and compare the rest of the files with
        # respect to the previous ones
        data = nii_files[0].get_data()
        valid_voxels = data > 0
        best_fit = data
        best_fit_model = np.zeros(best_fit.shape, dtype=np.int)
        best_fit_model[valid_voxels] = 1
        for i in range(1, len(nii_files)):
            data = nii_files[i].get_data()
            better_fit_ind = data > best_fit
            best_fit[better_fit_ind] = data[better_fit_ind]
            best_fit_model[better_fit_ind] = i + 1
        print('Storing comparison maps...')
        best_fit_name = ('{}-best_fit'+results_io.extension).format(name) if name else 'best_fit'+results_io.extension
        best_fit_model_name = ('{}-best_fit_model'+results_io.extension).format(name) if name else 'best_fit_model'+results_io.extension
        res_writer = results_io.writer(best_fit, affine_matrix)
        res_writer.save(path.join(output_dir, best_fit_name))
        res_writer = results_io.writer(best_fit_model, affine_matrix)
        res_writer.save(path.join(output_dir, best_fit_model_name))

    elif method == 'rgb':
        # Select first two or three files
        if len(nii_files) >= 3:
            nii_files = nii_files[:3]
            print('Computing RGB (Red-Green-Blue) map...')
        else:
            nii_files = nii_files[:2]
            print('Computing RG (Red-Green) map...')
        data = nii_files[0].get_data()
        fits = np.zeros(data.shape + (3,), dtype=np.float64)
        mask = np.zeros(data.shape + (3,), dtype=np.float64)
        valid_voxels = data > 0
        mask[valid_voxels, 0] = 1

        fits[..., 0] = data
        for i in range(1, len(nii_files)):
            data = nii_files[i].get_data()
            valid_voxels = data > 0
            mask[valid_voxels, i] = 1
            fits[..., i] = data

        print('Storing comparison maps...')
        rgb_name = ('{}-rgb'+results_io.extension).format(name) if name else 'rgb'+results_io.extension
        rgb_mask_name = ('{}-rgb_mask'+results_io.extension).format(name) if name else 'rgb_mask'+results_io.extension
        results_io.writer(fits, affine_matrix).save(path.join(output_dir, rgb_name))
        results_io.writer(mask, affine_matrix).save(path.join(output_dir, rgb_mask_name))

    elif method == 'absdiff':
        # Select first two files
        first_file = nii_files[0].get_data()
        second_file = nii_files[1].get_data()
        print('Computing absolute difference map...')
        absdiff = np.abs(first_file - second_file)
        print('Storing comparison maps...')
        absdiff_name = ('{}-absdiff'+results_io.extension).format(name) if name else 'absdiff'+results_io.extension
        results_io.writer(absdiff, affine_matrix).save(path.join(output_dir, absdiff_name))

    elif method == 'se':
        # Select first two files
        first_file = nii_files[0].get_data()
        second_file = nii_files[1].get_data()
        print('Computing squared error map...')
        semap = (first_file - second_file) ** 2
        print('Storing comparison maps...')
        se_name = ('{}-se'+results_io.extension).format(name) if name else 'se'+results_io.extension
        results_io.writer(semap, affine_matrix).save(path.join(output_dir, se_name))
    else:
        print('{} comparison method has not been implemented yet.'.format(method))
        exit(0)

    print()
    print('Done.')
