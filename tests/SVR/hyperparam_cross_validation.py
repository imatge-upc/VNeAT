import numpy as np

from nonlinear2.Utils.Subject import Subject
from nonlinear2.Utils.DataLoader import getGMData, getFeatures

""" LOAD DATA """

# Get data from Excel and nii files
regressor = getFeatures([Subject.ADCSFIndex])
observations = getGMData(corrected_data=True)

""" PARAMS """
N = 10                                      # number of iterations
m = 100                                     # number of voxels to select randomly in each iteration
params = {
    'C': np.logspace(1, 3.5, 20),
    'epsilon': np.linspace(1e-5, 1e-2, 20)
}

""" INITIALIZATION """
optim_params = { 'C': -1, 'epsilon': -1 }   # optimal params initialization
total_error = 1000000000                    # total error initialization

for i in range(N):
    # Select randomly m voxels
    pass