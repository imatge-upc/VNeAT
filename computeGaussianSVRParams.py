from os.path import join

import nibabel as nib
import Utils.DataLoader as DataLoader
from Processors.SVRProcessing import GaussianSVRProcessor as GSVRP
from Utils.Subject import Subject
from user_paths import RESULTS_DIR
RESULTS_DIR = join(RESULTS_DIR, 'GSVR')

if __name__ == "__main__":

    print 'Obtaining data from Excel file...'
    subjects = DataLoader.getSubjects(corrected_data=True)

    print 'Initializing GaussianSVR Processor...'
    # udp = (2, 3, 50.0, 0.2, 0.25)
    udp = (2, 3, 3.16227766017, 0.0891666666667, 0.25)
    gsvrp = GSVRP(subjects, predictors = [Subject.ADCSFIndex], user_defined_parameters=udp)

    print 'Processing data...'
    # x1=20, x2=21, y1=20, y2=21, z1=20, z2=21,
    results = gsvrp.process(n_jobs=4, mem_usage=512)
    C = gsvrp.user_defined_parameters[2]
    epsilon = gsvrp.user_defined_parameters[3]
    gamma = gsvrp.user_defined_parameters[4]

    print 'Saving results to files...'
    affine = DataLoader.getMNIAffine()
    niiFile = nib.Nifti1Image
    filename = 'gsvr_C' + str(C) + '_eps' + str(epsilon) + '_gamma' + str(gamma) +  '_'
    nib.save(niiFile(results.correction_parameters, affine), join(RESULTS_DIR, filename + 'cparams.nii'))
    nib.save(niiFile(results.prediction_parameters, affine), join(RESULTS_DIR, filename + 'pparams.nii'))

    with open(join(RESULTS_DIR, filename + 'userdefparams.txt'), 'wb') as f:
        f.write(str(gsvrp.user_defined_parameters) + '\n')

    print 'Done.'

