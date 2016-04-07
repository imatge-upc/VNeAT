
show_all = False

from os import listdir
from os.path import join, isfile, basename

import nibabel as nib
from GAMProcessing import GAMProcessor as GAMP
from GLMProcessing import PolyGLMProcessor as PGLMP
from matplotlib.pyplot import subplot, plot, legend, show, title
from nonlinear2.Subject import Subject
from numpy import zeros

from nonlinear2.Processors.SVRProcessing import PolySVRProcessor as PSVR
from nonlinear2.Utils.ExcelIO import ExcelSheet as Excel
from nonlinear2.user_paths import RESULTS_DIR

print 'Obtaining data from Excel file'

from nonlinear2.user_paths import DATA_DIR, EXCEL_FILE

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

print 'Loading precomputed parameters for PolyGLM'
pglm_correction_parameters = nib.load(join(RESULTS_DIR, 'PGLM', 'fpmalfa_pglm_cparams.nii')).get_data()
pglm_regression_parameters = nib.load(join(RESULTS_DIR, 'PGLM', 'fpmalfa_pglm_rparams.nii')).get_data()

with open(join(RESULTS_DIR, 'PGLM', 'fpmalfa_pglm_userdefparams.txt'), 'rb') as f:
	user_defined_parameters = eval(f.read())

print 'Initializing PolyGLM Processor'
pglmp = PGLMP(subjects, regressors = [Subject.ADCSFIndex], user_defined_parameters = user_defined_parameters, correctors = [Subject.Age, Subject.Sex])

print 'Loading precomputed parameters for GAM'
gam_correction_parameters = nib.load(join(RESULTS_DIR, 'GAM', 'fpmalfa_gam_poly3_cparams.nii')).get_data()
gam_regression_parameters = nib.load(join(RESULTS_DIR, 'GAM', 'fpmalfa_gam_poly3_rparams.nii')).get_data()

with open(join(RESULTS_DIR, 'GAM', 'fpmalfa_gam_poly3_userdefparams.txt'), 'rb') as f:
	user_defined_parameters = eval(f.read())

print 'Initializing GAM Processor'
gamp = GAMP(subjects, regressors = [Subject.ADCSFIndex], user_defined_parameters = user_defined_parameters, correctors = [Subject.Age, Subject.Sex])


print 'Loading precomputed parameters for PolySVR'
psvr_correction_parameters = nib.load(join(RESULTS_DIR, 'PSVR', 'fpmalfa_psvr_cparams.nii')).get_data()
psvr_regression_parameters = nib.load(join(RESULTS_DIR, 'PSVR', 'fpmalfa_psvr_rparams.nii')).get_data()

with open(join(RESULTS_DIR, 'PSVR', 'fpmalfa_psvr_userdefparams.txt'), 'rb') as f:
	user_defined_parameters = eval(f.read())

print 'Initializing PolyGLM Processor'
psvr = PSVR(subjects, regressors = [Subject.ADCSFIndex], user_defined_parameters = user_defined_parameters, correctors = [Subject.Age, Subject.Sex])


diagnostics = map(lambda subject: subject.get([Subject.Diagnostic])[0], pglmp.subjects)
diag = [[], [], [], []]
for i in xrange(len(diagnostics)):
	diag[diagnostics[i]].append(i)

adcsf = pglmp.regressors.T[0]

print
print 'Program initialized correctly.'
print
print '--------------------------------------'
print

while True:
	try:
		entry = raw_input('Write a tuple of voxel coordinates to display its curve (or press Ctrl+D to exit): ')
	except EOFError:
		print
		print 'Thank you for using our service.'
		print
		break
	except Exception as e:
		print '[ERROR] Unexpected error was found when reading input:'
		print e
		print
		continue
	try:
		x, y, z = map(int, eval(entry))
	except (NameError, TypeError, ValueError, EOFError):
		print '[ERROR] Input was not recognized'
		print 'To display the voxel with coordinates (x, y, z), please enter \'x, y, z\''
		print 'e.g., for voxel (57, 49, 82), type \'57, 49, 82\' (without inverted commas) as input'
		print
		continue
	except Exception as e:
		print '[ERROR] Unexpected error was found when reading input:'
		print e
		print
		continue

	print 'Processing request... please wait'

	try:
		# PolyGLM Curve
		corrected_data = pglmp.corrected_values(pglm_correction_parameters, x1 = x, x2 = x+1, y1 = y, y2 = y+1, z1 = z, z2 = z+1)

		if show_all:
			lin_rparams = zeros(pglm_regression_parameters.shape)
			lin_rparams[0] = pglm_regression_parameters[0]
			_, lin_curve = pglmp.curve(lin_rparams, x1 = x, x2 = x+1, y1 = y, y2 = y+1, z1 = z, z2 = z+1, tpoints = 50)
			
			nonlin_rparams = zeros(pglm_regression_parameters.shape)
			nonlin_rparams[1:] = pglm_regression_parameters[1:]
			_, nonlin_curve = pglmp.curve(nonlin_rparams, x1 = x, x2 = x+1, y1 = y, y2 = y+1, z1 = z, z2 = z+1, tpoints = 50)

		axis, curve = pglmp.curve(pglm_regression_parameters, x1 = x, x2 = x+1, y1 = y, y2 = y+1, z1 = z, z2 = z+1, tpoints = 50)

		subplot(3, 1, 1)
		title('GLM')

		if show_all:
			plot(axis, lin_curve[:, 0, 0, 0], 'y', label = 'Fitted linear curve')
			plot(axis, nonlin_curve[:, 0, 0, 0], 'g', label = 'Fitted nonlinear curve')
		
		plot(axis, curve[:, 0, 0, 0], 'r', label = 'Fitted total curve')

		color = ['co', 'bo', 'mo', 'ko']
		for i in xrange(len(diag)):
			l = diag[i]
			plot(adcsf[l], corrected_data[l, 0, 0, 0], color[i], label = Subject.Diagnostics[i])
		legend()

		# GAM Curve
		corrected_data = gamp.corrected_values(gam_correction_parameters, x1 = x, x2 = x+1, y1 = y, y2 = y+1, z1 = z, z2 = z+1)

		axis, curve = gamp.curve(gam_regression_parameters, x1 = x, x2 = x+1, y1 = y, y2 = y+1, z1 = z, z2 = z+1, tpoints = 50)

		subplot(3, 1, 2)
		title('GAM')

		plot(axis, curve[:, 0, 0, 0], 'r', label = 'Fitted total curve')

		color = ['co', 'bo', 'mo', 'ko']
		for i in xrange(len(diag)):
			l = diag[i]
			plot(adcsf[l], corrected_data[l, 0, 0, 0], color[i], label = Subject.Diagnostics[i])
		legend()

		# Poly SVR Curve
		corrected_data = psvr.corrected_values(psvr_correction_parameters, x1 = x, x2 = x+1, y1 = y, y2 = y+1, z1 = z, z2 = z+1)

		axis, curve = psvr.curve(psvr_regression_parameters, x1 = x, x2 = x+1, y1 = y, y2 = y+1, z1 = z, z2 = z+1, tpoints = 50)

		subplot(3, 1, 3)
		title('Poly SVR')

		plot(axis, curve[:, 0, 0, 0], 'r', label = 'Fitted total curve')

		color = ['co', 'bo', 'mo', 'ko']
		for i in xrange(len(diag)):
			l = diag[i]
			plot(adcsf[l], corrected_data[l, 0, 0, 0], color[i], label = Subject.Diagnostics[i])
		legend()

		show()
		print
	except Exception as e:
		print '[ERROR] Unexpected error occured while computing and showing the results:'
		print e
		print
		continue




