from ExcelIO import ExcelSheet as Excel
from GLMProcessing import PolyGLMProcessor as PGLMP
from Subject import Subject
from os.path import join, isfile, basename
from os import listdir

import nibabel as nib
from numpy import array as nparray


print 'Obtaining data from Excel file'
DATA_DIR = join('/', 'Users', 'Asier', 'Documents', 'TFG', 'Alan T', 'Nonlinear_NBA_15')
EXCEL_FILE = join('/', 'Users', 'Asier', 'Documents', 'TFG', 'Alan T', 'work_DB_CSF.R1.final.xls')

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

print 'Initializing PolyGLM Processor'
pglmp = PGLMP(subjects, regressors = [Subject.ADCSFIndex])

print 'Processing data'
results = pglmp.process()

print 'Saving results to files'

affine = nparray(
		[[ -1.50000000e+00,   0.00000000e+00,   0.00000000e+00,   9.00000000e+01],
		 [  1.99278252e-16,   1.50000000e+00,   2.17210575e-16,  -1.26000000e+02],
		 [ -1.36305018e-16,  -1.38272305e-16,   1.50000000e+00,  -7.20000000e+01],
		 [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]]
)

niiFile = nib.Nifti1Image

nib.save(niiFile(results.correction_parameters, affine), 'fpmalfa_cparams.nii')
nib.save(niiFile(results.regression_parameters, affine), 'fpmalfa_rparams.nii')
nib.save(niiFile(results.fitting_scores, affine), 'fpmalfa_fitscores.nii')
with open('fpmalfa_userdefparams.txt', 'wb') as f:
	f.write(str(pglmp.user_defined_parameters) + '\n')
