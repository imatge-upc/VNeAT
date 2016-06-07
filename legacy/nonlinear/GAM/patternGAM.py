# To be executed as follows:
# python -W ignore -u ttest.py
import matplotlib.pyplot as plt
import numpy as np
from numpy import linspace

import database as db
from curve_fit import GAM
from tools import copy_iterable as copy, tolist  # , combinatorial

print
print 'Welcome to patterns2.py!'
print 'We are now going to analyze the data in the specified directory and perform some curve-fitting over it.'
print 'There is a lot of data to process, so please be pacient.'
print 'Thank you for using our software! :)'
print

# Set region
start = (73, 84, 41)
rang = (1, 1, 1)
x1, y1, z1 = start
x2, y2, z2 = tuple(start[i] + rang[i] for i in range(3))

# join('/','C:','Users','upcnet','FPM','data','Non-linear','Nonlinear_NBA_15')
# Get input data and initialize output data structure

input_data = db.get_data(DATA_DIR="C:\Users\upcnet\FPM\data\Non-linear\Nonlinear_NBA_15",
                         EXCEL_FILE="C:\Users\upcnet\FPM\data\Non-linear\work_DB_CSF.R1.final.xls",
                         x1=x1, y1=y1, z1=z1, x2=x2, y2=y2, z2=z2)
dims = input_data.dims[1:4]
output_data1 = [[[[] for _ in range(dims[2])] for _ in range(dims[1])] for _ in range(dims[0])]
output_data2 = tolist(copy(output_data1))

# Compute features (independent of voxel, dependent only on subjects)
features = [[1] * len(input_data.subjects), [], [], []]
for subject in input_data.subjects:
    features[1].append(subject.sex)
    features[2].append(subject.age)
    features[3].append(subject.adcsf)

# Minimum quantity of grey matter (per unit volume) so that we consider a voxel (otherwise it will be omitted)
gm_threshold = 0.2
npoints = 129

# Politting issues
t0_ini = np.array(features[1])
t1_ini = np.array(features[2])
t2_ini = np.array(features[3])
t0min = min(features[1])
t0max = max(features[1])
t0 = linspace(t0min, t0max, npoints)
t1min = min(features[2])
t1max = max(features[2])
t1 = linspace(t1min, t1max, npoints)
t2min = min(features[3])
t2max = max(features[3])
t2 = linspace(t2min, t2max, npoints)

# Printing progress purposes only
total_num_voxels = dims[0] * dims[1] * dims[2]
num_voxels_processed = 0
stnv = str(total_num_voxels)
ltnv = len(stnv)
progress = 0.0
pr_inc = 100.0 / total_num_voxels
last_int = 0

for voxel in input_data.voxels():
    # Gray matter values in ydata, voxel coordinates (relative to region) in i, j, k
    ydata = voxel.data
    i, j, k = voxel.coords
    i -= x1
    j -= y1
    k -= z1

    # Skip this voxel if there is no subject whose gray matter quantity in this coordinates is equal to or above the threshold
    if all(gm_value < gm_threshold for gm_value in ydata):
        output_data1[i][j][k] = ydata
        output_data2[i][j][k] = [0] * npoints
    else:
        # GAM!
        gam = GAM(ydata)
        gam.basisFunctions.set_polySmoother(features[1], 1)
        gam.basisFunctions.set_splinesSmoother(features[2], 1, s=100)
        gam.basisFunctions.set_splinesSmoother(features[3], 3, s=100)
        gam.backfitting_algorithm()
        output_data2[i][j][k] = gam.prediction()

    # Print progress
    num_voxels_processed += 1
    progress += pr_inc
    if int(10 * progress) != last_int:
        last_int = int(10 * progress)
        snvp = str(num_voxels_processed)
        lnvp = len(snvp)
        print ' ' * (ltnv - lnvp + 4) + snvp + ' / ' + stnv + '   (' + str(int(100 * progress) / 100.) + '%)'

    # plt.figure()
    # plt.plot(ydata, 'ko')
    # plt.plot(gam.results.alpha + gam.results.smoothers[0](t0) + gam.results.smoothers[1](t1) + gam.results.smoothers[2](t2),'r-' )

    plt.figure()
    plt.plot(features[1],
             ydata - gam.results.smoothers[1](t1_ini) - gam.results.smoothers[2](t2_ini) - gam.results.alpha, 'ko')
    plt.plot(t0, gam.results.smoothers[0](t0) + gam.results.offset[0], 'r-', label='AdditiveModel')

    plt.figure()
    plt.plot(features[2],
             ydata - gam.results.smoothers[0](t0_ini) - gam.results.smoothers[2](t2_ini) - gam.results.alpha, 'ko')
    plt.plot(t1, gam.results.smoothers[1](t1) + gam.results.offset[1], 'r-', label='AdditiveModel')

    plt.figure()
    plt.plot(features[3],
             ydata - gam.results.smoothers[0](t0_ini) - gam.results.smoothers[1](t1_ini) - gam.results.alpha, 'ko')
    plt.plot(t2, gam.results.smoothers[2](t2) + gam.results.offset[2], 'r-', label='AdditiveModel')

    plt.show()



    # db.save_output_data(nparray(output_data1), 'C:\Users\upcnet\FPM\data\Non-linear\output1.nii')
    # db.save_output_data(nparray(output_data2), 'C:\Users\upcnet\FPM\data\Non-linear\output2.nii')
