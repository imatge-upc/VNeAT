from ExcelIO import ExcelSheet as Excel
from SVRProcessing import PolySVRProcessor as PSVR
from Subject import Subject
from os.path import join, isfile, basename
from os import listdir

import nibabel as nib
from numpy import array as array


if __name__ == "__main__":

	print 'Obtaining data from Excel file...'


	DATA_DIR = join("C:\\", "Users", "santi", "Documents", "Santi", "Universitat", "TFG", "Data", "nonlinear_data", "Nonlinear_NBA_15")
	EXCEL_FILE = join("C:\\", "Users", "santi", "Documents", "Santi", "Universitat", "TFG", "Data", "nonlinear_data", "work_DB_CSF.R1.final.xls")
	RESULTS_DIR = join("results", "PSVR")
	"""
	DATA_DIR = join("/", "imatge", "spuch", "data-neuroimatge", "Nonlinear_NBA_15")
	EXCEL_FILE = join("/", "imatge", "spuch", "data-neuroimatge", "work_DB_CSF.R1.final.xls")
	RESULTS_DIR = join("/", "imatge", "spuch", "work", "neuro", "PolySVR")
	"""

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

	print 'Initializing PolySVR Processor...'
	psvr = PSVR(subjects, regressors = [Subject.ADCSFIndex], correctors = [Subject.Age, Subject.Sex])

	print 'Processing data...'
	results = psvr.process(n_jobs=8)

	print 'Saving results to files...'

	affine = array(
			[[ -1.50000000e+00,   0.00000000e+00,   0.00000000e+00,   9.00000000e+01],
			 [  1.99278252e-16,   1.50000000e+00,   2.17210575e-16,  -1.26000000e+02],
			 [ -1.36305018e-16,  -1.38272305e-16,   1.50000000e+00,  -7.20000000e+01],
			 [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]]
	)

	niiFile = nib.Nifti1Image

	nib.save(niiFile(results.correction_parameters, affine), join(RESULTS_DIR, 'fpmalfa_psvr_cparams.nii'))
	nib.save(niiFile(results.regression_parameters, affine), join(RESULTS_DIR, 'fpmalfa_psvr_rparams.nii'))
	nib.save(niiFile(results.fitting_scores, affine), join(RESULTS_DIR, 'fpmalfa_psvr_fitscores.nii'))

	with open(join(RESULTS_DIR, 'fpmalfa_psvr_userdefparams.txt'), 'wb') as f:
		f.write(str(psvr.user_defined_parameters) + '\n')

	print 'Obtaining, filtering and saving z-scores and labels to display them...'
	for fit_threshold in [0.99, 0.995, 0.999]:
		print '    Fitting-threshold set to', fit_threshold, '; Computing z-scores and labels...'
		z_scores, labels = psvr.fit_score(results.fitting_scores, fit_threshold = fit_threshold, produce_labels = True)

		print '    Saving z-scores and labels to file...'
		nib.save(niiFile(z_scores, affine), join(RESULTS_DIR, 'fpmalfa_psvr_zscores_' + str(fit_threshold) + '.nii'))
		nib.save(niiFile(labels, affine), join(RESULTS_DIR, 'fpmalfa_psvr_labels_' + str(fit_threshold) + '.nii'))


	print 'Done.'
