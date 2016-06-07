import time
from os.path import join

import nibabel as nib

import Utils.DataLoader as DataLoader
from Processors.SVRProcessing import PolySVRProcessor as PSVR
from Utils.Subject import Subject
from user_paths import RESULTS_DIR

if __name__ == "__main__":
    # Define filename prefix
    filename_prefix = join(RESULTS_DIR, 'PSVR')

    print 'Obtaining data from Excel file...'
    subjects = DataLoader.getSubjects(corrected_data=True)  # Used for prediction
    # subjects = DataLoader.getSubjects(corrected_data=False) # Used for correction and prediction

    print 'Initializing PolySVR Processor...'
    udp = (2, 3, 1, 0.1, 3)  # Used for prediction
    # udp = (1, 0, 3.0, 0.08, 3, 2, 1)            # Used for correction and prediction
    psvr = PSVR(subjects,
                predictors=[Subject.ADCSFIndex],
                user_defined_parameters=udp)

    print 'Processing data...'
    time_start = time.clock()
    results = psvr.process(n_jobs=6, mem_usage=512)
    time_end = time.clock()
    C = psvr.user_defined_parameters[2]
    epsilon = psvr.user_defined_parameters[3]
    print 'Processing done in ', time_end - time_start, " seconds"

    print 'Saving results to files...'
    affine = DataLoader.getMNIAffine()
    niiFile = nib.Nifti1Image
    filename = 'psvr_C' + str(C) + '_eps' + str(epsilon) + '_'
    nib.save(niiFile(results.correction_parameters, affine), join(filename_prefix, filename + 'cparams.nii'))
    nib.save(niiFile(results.prediction_parameters, affine), join(filename_prefix, filename + 'pparams.nii'))

    with open(join(filename_prefix, filename + 'userdefparams.txt'), 'wb') as f:
        f.write(str(psvr.user_defined_parameters) + '\n')

    print 'Done'
