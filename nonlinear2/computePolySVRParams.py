from os.path import join

import nibabel as nib
import numpy as np

from nonlinear2.Utils.DataLoader import getSubjects
from nonlinear2.Utils.Subject import Subject
from nonlinear2.Processors.SVRProcessing import PolySVRProcessor as PSVR
from nonlinear2.user_paths import RESULTS_DIR

C = 5.092
epsilon = 0.0096

if __name__ == "__main__":

    print 'Obtaining data from Excel file...'
    subjects = getSubjects(corrected_data=True)

    print 'Initializing PolySVR Processor with C = ', str(C), 'epsilon = ', str(epsilon) + ' ...'
    user_defined_parameters = (1, 0, C, epsilon, 3)
    psvr = PSVR(subjects, predictors = [Subject.ADCSFIndex], user_defined_parameters=user_defined_parameters)

    print 'Processing data...'
    results = psvr.process(x1=80, x2=81, y1=80, y2=81, z1=50, z2=51, n_jobs=8, mem_usage=50)

    print 'Saving results to files...'

    affine = np.array(
            [[ -1.50000000e+00,   0.00000000e+00,   0.00000000e+00,   9.00000000e+01],
             [  1.99278252e-16,   1.50000000e+00,   2.17210575e-16,  -1.26000000e+02],
             [ -1.36305018e-16,  -1.38272305e-16,   1.50000000e+00,  -7.20000000e+01],
             [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]]
    )

    niiFile = nib.Nifti1Image

    filename = 'psvr_C' + str(C) + '_eps' + str(epsilon) + '_'
    nib.save(niiFile(results.correction_parameters, affine), join(RESULTS_DIR, filename + 'cparams.nii'))
    nib.save(niiFile(results.prediction_parameters, affine), join(RESULTS_DIR, filename + 'pparams.nii'))

    with open(join(RESULTS_DIR, filename + 'userdefparams.txt'), 'wb') as f:
        f.write(str(psvr.user_defined_parameters) + '\n')

    print 'Done.'

