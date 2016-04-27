from os.path import join

import nibabel as nib
import Utils.DataLoader as DataLoader
from Processors.SVRProcessing import PolySVRProcessor as PSVR
from Utils.Subject import Subject
from user_paths import RESULTS_DIR

if __name__ == "__main__":

    print 'Obtaining data from Excel file...'
    subjects = DataLoader.getSubjects(corrected_data=True)

    print 'Initializing PolySVR Processor...'
    psvr = PSVR(subjects, predictors = [Subject.ADCSFIndex])

    print 'Processing data...'
    results = psvr.process(n_jobs=4, mem_usage=64)
    C = psvr.user_defined_parameters[2]
    epsilon = psvr.user_defined_parameters[3]

    print 'Saving results to files...'
    affine = DataLoader.getMNIAffine()
    niiFile = nib.Nifti1Image
    filename = 'psvr_C' + str(C) + '_eps' + str(epsilon) + '_'
    nib.save(niiFile(results.correction_parameters, affine), join(RESULTS_DIR, filename + 'cparams.nii'))
    nib.save(niiFile(results.prediction_parameters, affine), join(RESULTS_DIR, filename + 'pparams.nii'))

    with open(join(RESULTS_DIR, filename + 'userdefparams.txt'), 'wb') as f:
        f.write(str(psvr.user_defined_parameters) + '\n')

    print 'Done.'

