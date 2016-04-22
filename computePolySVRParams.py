from os.path import join

import nibabel as nib
import Utils.DataLoader as DataLoader
from Processors.SVRProcessing import PolySVRProcessor as PSVR
from Utils.Subject import Subject
from user_paths import RESULTS_DIR

C = 10
epsilon = 0.03

if __name__ == "__main__":

    print 'Obtaining data from Excel file...'
    subjects = DataLoader.getSubjects(corrected_data=True)

    print 'Initializing PolySVR Processor with C = ', str(C), 'epsilon = ', str(epsilon) + ' ...'
    user_defined_parameters = (0, 9, C, epsilon, 3)
    psvr = PSVR(subjects, predictors = [Subject.ADCSFIndex], user_defined_parameters=user_defined_parameters)

    print 'Processing data...'
    results = psvr.process(n_jobs=8, mem_usage=64)

    print 'Saving results to files...'
    affine = DataLoader.getMNIAffine()
    niiFile = nib.Nifti1Image
    filename = 'psvr_C' + str(C) + '_eps' + str(epsilon) + '_'
    nib.save(niiFile(results.correction_parameters, affine), join(RESULTS_DIR, filename + 'cparams.nii'))
    nib.save(niiFile(results.prediction_parameters, affine), join(RESULTS_DIR, filename + 'pparams.nii'))

    with open(join(RESULTS_DIR, filename + 'userdefparams.txt'), 'wb') as f:
        f.write(str(psvr.user_defined_parameters) + '\n')

    print 'Done.'

