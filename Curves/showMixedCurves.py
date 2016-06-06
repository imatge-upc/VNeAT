"""
Shows curves for Mixed Processors:
    - PolyGLM corrected, PolyGLM predicted
    - PolyGLM corrected, PolyGAM predicted
    - PolyGLM corrected, PolySVR predicted
    - PolyGLM corrected, GaussianSVR predicted
"""
from os.path import join

import matplotlib.pyplot as plot
import nibabel as nib
import numpy as np

# Utils
from Utils.DataLoader import getSubjects, getMNIAffine
from Utils.Subject import Subject
from user_paths import RESULTS_DIR
from Processors.MixedProcessor import MixedProcessor

RESULTS_DIR = join(RESULTS_DIR, 'MIXED')

# Info
fitters = [
    # NAME                                  # PREFIX
    ['PolyGLM - PolyGLM',                   join('PGLM-PGLM', 'pglm_pglm_')],
    ['PolyGLM - PolyGAM',                   join('PGLM-PGAM', 'pglm_pgam_')],
    ['PolyGLM - PolySVR',                   join('PGLM-PSVR', 'pglm_psvr_')],
    ['PolyGLM - GaussianSVR',               join('PGLM-GSVR', 'pglm_gsvr_')],
]

print 'Obtaining data from Excel file...'
subjects = getSubjects(corrected_data=False)

print 'Obtaining affine matrix to map mm<-->voxels...'
affine = getMNIAffine()

print 'Loading precomputed parameters for all fitters...'
prediction_parameters = []
correction_parameters = []
user_defined_parameters = []
for fitter in fitters:
    prediction_parameters.append(nib.load(join(RESULTS_DIR, fitter[1] + 'pparams.nii')).get_data())
    correction_parameters.append(nib.load(join(RESULTS_DIR, fitter[1] + 'cparams.nii')).get_data())
    with open(join(RESULTS_DIR, fitter[1] + 'userdefparams.txt'), 'rb') as f:
        user_defined_parameters.append(eval(f.read()))

print 'Initializing processors'
processors = []
for user_params in user_defined_parameters:
    processors.append(
        MixedProcessor(
            subjects,
            predictors=[Subject.ADCSFIndex],
            correctors=[Subject.Age, Subject.Sex],
            user_defined_parameters=user_params
        )
    )

diagnostics = map(lambda subject: subject.get([Subject.Diagnostic])[0], subjects)
diag = [[], [], [], []]
for i in xrange(len(diagnostics)):
    diag[diagnostics[i]].append(i)

adcsf = processors[0].predictors.T[0]

print
print 'Program initialized correctly.'
print
print '--------------------------------------'
print

while True:
    try:
        entry = raw_input('Write a tuple of mm coordinates (in MNI space) to display its curve '
                          '(or press Ctrl+D to exit): ')
    except EOFError:
        print
        print 'Thank you for using our service.'
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
        voxel_coordinates = map(int, np.round(np.linalg.inv(affine).dot(mm_coordinates)))
        # Get rounded mm coordinates in MNI space (due to 1.5 mm spacing)
        mm_coordinates_prima = affine.dot(voxel_coordinates)
        # Final voxel coordinates
        x = voxel_coordinates[0]
        y = voxel_coordinates[1]
        z = voxel_coordinates[2]

        print 'This is voxel', x, y, z

        # Get curves for all processors
        for i in range(len(processors)):
            # Get (corrected) grey matter data
            corrected_data = processors[i].corrected_values(
                correction_parameters[i],
                x1=x, x2=x + 1, y1=y, y2=y + 1, z1=z, z2=z + 1)
            axis, curve = processors[i].curve(
                prediction_parameters[i],
                x1=x, x2=x + 1, y1=y, y2=y + 1, z1=z, z2=z + 1, tpoints=50)
            color = '#aa4433'
            plot.plot(axis, curve[:, 0, 0, 0],
                      lw=2, label=fitters[i][0], color=color, marker='d')

            color = ['co', 'bo', 'mo', 'ko']
            for i in xrange(len(diag)):
                l = diag[i]
                plot.plot(adcsf[l], corrected_data[l, 0, 0, 0], color[i], lw=4, label=Subject.Diagnostics[i])
            # Plot info
            plot.legend(fontsize='xx-large')
            plot.xlabel('ADCSF', fontsize='xx-large')
            plot.ylabel('Grey matter', fontsize='xx-large')
            plt_title = 'Coordinates: ' + \
                        str(mm_coordinates_prima[0]) + ', ' + \
                        str(mm_coordinates_prima[1]) + ', ' + \
                        str(mm_coordinates_prima[2]) + ' mm'
            plot.title(plt_title, size="xx-large")
            plot.show()

        print
    except Exception as e:
        print '[ERROR] Unexpected error occurred while computing and showing the results:'
        print e
        print
        continue

"""
INTERESTING COORDINATES:

    - Right Precuneus: 2, -54, 26 (voxel: 59, 48, 65)

    - Left Hippocampus: -16, -8, -14 (voxel: 71, 79, 39)

    - Right ParaHippocampal: 24, -28, -12 (voxel: 44, 65, 40)

"""


