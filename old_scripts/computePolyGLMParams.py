from os.path import join

import nibabel as nib

from Processors.GLMProcessing import PolyGLMProcessor as PGLMP
from Utils.DataLoader import getSubjects, getMNIAffine
from Utils.Subject import Subject
from user_paths import RESULTS_DIR

filename_prefix = join(RESULTS_DIR, 'PGLM', 'pglm_curve_')

niiFile = nib.Nifti1Image

print 'Getting data from Excel file...'
subjects = getSubjects(corrected_data=True)

affine = getMNIAffine()

print 'Initializing PolyGLM Processor...'
pglmp = PGLMP(subjects, predictors=[Subject.ADCSFIndex])

print 'Processing data...'
results = pglmp.process(mem_usage=512)  # x1 = 80, x2 = 81, y1 = 49, y2 = 50, z1 = 82, z2 = 83)

print 'Saving results to files...'

# nib.save(niiFile(results.correction_parameters, affine), filename_prefix + 'cparams.nii')
nib.save(niiFile(results.prediction_parameters, affine), filename_prefix + 'pparams.nii')

with open(filename_prefix + 'userdefparams.txt', 'wb') as f:
    f.write(str(pglmp.user_defined_parameters) + '\n')

print 'Done.'
