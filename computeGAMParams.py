import nibabel as nib
import Utils.DataLoader as DataLoader
import time
from os.path import join
from Processors.GAMProcessing import GAMProcessor as GAMP
from Utils.Subject import Subject
from user_paths import RESULTS_DIR
RESULTS_DIR = join(RESULTS_DIR, 'SGAM')

niiFile = nib.Nifti1Image
affine = DataLoader.getMNIAffine()

print 'Obtaining data from Excel file...'
subjects = DataLoader.getSubjects(corrected_data=True)

user_defined_parameters = [
    (9, [2, 2, 97, 3])
]
#
# user_defined_parameters = [
#     (9, [1, 1, 3])
# ]
filename_prefix = [
    'gam_splines_'
]

for udp, fn in zip(user_defined_parameters, filename_prefix):

    print 'Initializing GAM Polynomial Processor...'
    gamp = GAMP(subjects, predictors=[Subject.ADCSFIndex], user_defined_parameters=udp)

    print 'Processing data...'
    time_start = time.clock()
    results = gamp.process(mem_usage=128)
    time_end = time.clock()
    print 'Processing done in ', time_end - time_start, ' seconds'

    print 'Saving results to files...'
    nib.save(niiFile(results.correction_parameters, affine), join(RESULTS_DIR, fn + 'cparams.nii'))
    nib.save(niiFile(results.prediction_parameters, affine), join(RESULTS_DIR, fn + 'pparams.nii'))

    with open(join(RESULTS_DIR, fn + 'userdefparams.txt'), 'wb') as f:
        f.write(str(gamp.user_defined_parameters) + '\n')

    print 'Done.'
