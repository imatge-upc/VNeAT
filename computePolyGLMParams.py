from os.path import join

import nibabel as nib
import numpy as np
from Processors.GLMProcessing import PolyGLMProcessor as PGLMP
from Utils.Subject import Subject

from Utils.DataLoader import getSubjects

filename_prefix = join('results', 'PGLM', 'pglm_')

niiFile = nib.Nifti1Image

print 'Getting data from Excel file...'
subjects = getSubjects(corrected_data=True)

print 'Initializing PolyGLM Processor...'
pglmp = PGLMP(subjects, predictors = [Subject.ADCSFIndex])

print 'Processing data...'
results = pglmp.process(mem_usage=256)# x1 = 80, x2 = 81, y1 = 49, y2 = 50, z1 = 82, z2 = 83)

print 'Saving results to files...'

affine = np.array(
		[[ -1.50000000e+00,   0.00000000e+00,   0.00000000e+00,   9.00000000e+01],
		 [  1.99278252e-16,   1.50000000e+00,   2.17210575e-16,  -1.26000000e+02],
		 [ -1.36305018e-16,  -1.38272305e-16,   1.50000000e+00,  -7.20000000e+01],
		 [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]]
)

nib.save(niiFile(results.correction_parameters, affine), filename_prefix + 'cparams.nii')
nib.save(niiFile(results.prediction_parameters, affine), filename_prefix + 'pparams.nii')

with open(filename_prefix + 'userdefparams.txt', 'wb') as f:
	f.write(str(pglmp.user_defined_parameters) + '\n')

print 'Done.'

