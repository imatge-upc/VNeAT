""" Shows curves for all fitters created:
        - Poly GLM
        - Poly GAM
        - Poly SVR
"""

import nibabel as nib
import numpy as np
from os.path import join
import matplotlib.pyplot as plot

# Utils
from nonlinear2.Utils.DataLoader import getSubjects, getMNIAffine
from nonlinear2.Utils.Subject import Subject

# Processors
from nonlinear2.Processors.GLMProcessing import PolyGLMProcessor as PGLMP
from nonlinear2.Processors.GAMProcessing import GAMProcessor as GAMP
from nonlinear2.Processors.SVRProcessing import PolySVRProcessor as PSVRP

# Info
fitters = [
#     NAME       PROCESSOR  PATH                                        COLOR      MARKER
    ['Poly GLM', PGLMP,     join('..', 'results', 'PGLM', 'pglm_'),     '#8A5EB8', 'd'   ],
    ['Poly GAM', GAMP,      join('..', 'results', 'PGAM', 'gam_poly_'), '#FFFB69', 'x'   ],
    ['Poly SVR', PSVRP,     join('..', 'results', 'PSVR', 'psvr_'),     '#B22918', '+'   ]
]

print 'Obtaining data from Excel file...'
subjects = getSubjects(corrected_data=True)

print 'Obtaining affine matrix to map mm<-->voxels...'
affine = getMNIAffine()

print 'Loading precomputed parameters for all fitters...'
prediction_parameters = []
user_defined_parameters = []
for fitter in fitters:
    prediction_parameters.append(nib.load(fitter[2] + 'pparams.nii').get_data())
    with open(fitter[2] + 'userdefparams.txt', 'rb') as f:
        user_defined_parameters.append(eval(f.read()))


print 'Initializing processors'
processors = []
for fitter, user_params in zip(fitters, user_defined_parameters):
    processors.append(
        fitter[1](
            subjects,
            predictors = [Subject.ADCSFIndex],
            user_defined_parameters = user_params
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

        # Get (corrected) grey matter data
        corrected_data = processors[0].gm_values(
            x1 = x, x2 = x+1, y1 = y, y2 = y+1, z1 = z, z2 = z+1)

        # Get curves for all processors
        for i in range(len(processors)):
            axis, curve = processors[i].curve(
                prediction_parameters[i],
                x1 = x, x2 = x+1, y1 = y, y2 = y+1, z1 = z, z2 = z+1, tpoints = 50)
            random_color = np.random.rand(3,1)
            plot.plot(axis, curve[:, 0, 0, 0],
                      lw=2, label=fitters[i][0], color=fitters[i][3], marker=fitters[i][4])

        color = ['co', 'bo', 'mo', 'ko']
        for i in xrange(len(diag)):
            l = diag[i]
            plot.plot(adcsf[l], corrected_data[l, 0, 0, 0], color[i], lw=4, label = Subject.Diagnostics[i])
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




