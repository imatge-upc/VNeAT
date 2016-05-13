from os.path import join

import nibabel as nib
from Processors.GAMProcessing import GAMProcessor as GAMP
from Utils.Subject import Subject
import Utils.DataLoader as DataLoader
from user_paths import RESULTS_DIR

niiFile = nib.Nifti1Image
affine = DataLoader.getMNIAffine()

print 'Obtaining data from Excel file...'
subjects = DataLoader.getSubjects(corrected_data=True)

# user_defined_parameters = [
#     (9, [1, 1, 3]),
# ]

user_defined_parameters = [
    (9,[2,2,1,1])
]
filename_prefix = [
    'gam_splines_'
]

for udp, fn in zip(user_defined_parameters, filename_prefix):

    print 'Initializing GAM Polynomial Processor...'
    gamp = GAMP(subjects, predictors=[Subject.ADCSFIndex],user_defined_parameters=udp)

    print 'Processing data...'
    results = gamp.process(mem_usage=256)

    print 'Saving results to files...'

    nib.save(niiFile(results.correction_parameters, affine), join(RESULTS_DIR, fn + 'cparams.nii'))
    nib.save(niiFile(results.prediction_parameters, affine), join(RESULTS_DIR, fn + 'pparams.nii'))

    with open(join(RESULTS_DIR, fn + 'userdefparams.txt'), 'wb') as f:
        f.write(str(gamp.user_defined_parameters) + '\n')

    print 'Done.'
