from os import listdir
from os.path import join, isfile, basename

import nibabel as nib
import numpy as np
from Utils.Subject import Subject

from nonlinear2.Processors.GLMProcessing import GLMProcessor as GLMP
from nonlinear2.Utils.ExcelIO import ExcelSheet as Excel

filename_prefix = join('results', 'GLM', 'glm_')





niiFile = nib.Nifti1Image

print 'Obtaining data from Excel file...'
from user_paths import DATA_DIR, EXCEL_FILE

filenames = filter(isfile, map(lambda elem: join(DATA_DIR, elem), listdir(DATA_DIR)))
filenames_by_id = {basename(fn).split('_')[0][8:] : fn for fn in filenames}

exc = Excel(EXCEL_FILE)

subjects = []
for r in exc.get_rows( fieldstype = {
				'id':(lambda s: str(s).strip().split('_')[0]),
				'diag':(lambda s: int(s) - 1),
				'age':int,
				'sex':(lambda s: 2*int(s) - 1),
				'apoe4_bin':(lambda s: 2*int(s) - 1),
				'escolaridad':int,
				'ad_csf_index_ttau':float
			 } ):
	subjects.append(
		Subject(
			r['id'],
			filenames_by_id[r['id']],
			r.get('diag', None),
			r.get('age', None),
			r.get('sex', None),
			r.get('apoe4_bin', None),
			r.get('escolaridad', None),
			r.get('ad_csf_index_ttau', None)
		)
	)

print 'Initializing GLM Processor...'
glmp = GLMP(subjects, predictors = [Subject.ADCSFIndex], correctors = [Subject.Age, Subject.Sex])

print 'Processing data...'
results = glmp.process(mem_usage = 512)# x1 = 80, x2 = 81, y1 = 49, y2 = 50, z1 = 82, z2 = 83)

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
	f.write(str(glmp.user_defined_parameters) + '\n')

print 'Done.'

