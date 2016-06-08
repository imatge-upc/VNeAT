from glob import glob
from os.path import join

import nibabel as nib
import numpy as np

from user_paths import DATA_DIR, CORRECTED_DATA_DIR

DATA_1 = join(DATA_DIR, '*.nii')
DATA_2 = join(CORRECTED_DATA_DIR, '*.nii')

print
print 'Path to Dataset 1: ', DATA_1
print 'Path to Dataset 2: ', DATA_2
print

print 'Reading the data from disk...'
dataset_1, dataset_2 = [], []
for file1, file2 in zip(glob(DATA_1), glob(DATA_2)):
    dataset_1.append(nib.load(file1).get_data())
    dataset_2.append(nib.load(file2).get_data())

dataset_1 = np.array(dataset_1)
dataset_2 = np.array(dataset_2)

print 'Computing the similarity using two criteria...'
print
all_close = np.allclose(dataset_1, dataset_2)
print "Is the data similar (np.allclose criteria)? --> ", "yes" if all_close else "no"

max_abs_diff = np.max(np.abs(dataset_1 - dataset_2))
print "Maximum absolute difference between 2 datasets: ", max_abs_diff
