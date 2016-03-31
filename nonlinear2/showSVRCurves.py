from ExcelIO import ExcelSheet as Excel
from SVRProcessing import PolySVRProcessor as PSVR
from Subject import Subject
from os.path import join, isfile, basename
from os import listdir

import nibabel as nib
import numpy as np
from matplotlib.pyplot import plot, legend, show



filename_prefix = join('results', 'PSVR', 'psvr_')
# show_all = True


print 'Obtaining data from Excel file'

from user_paths import EXCEL_FILE, CORRECTED_DIR
filenames = filter(isfile, map(lambda elem: join(CORRECTED_DIR, elem), listdir(CORRECTED_DIR)))
filenames_by_id = {basename(fn).split('_')[1][:-4] : fn for fn in filenames}

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

print 'Loading precomputed parameters for GLM'
psvr_prediction_parameters = nib.load(filename_prefix + 'pparams.nii').get_data()

with open(filename_prefix + 'userdefparams.txt', 'rb') as f:
	user_defined_parameters = eval(f.read())

print 'Initializing GLM Processor'
psvrp = PSVR(subjects, regressors = [Subject.ADCSFIndex], user_defined_parameters = user_defined_parameters)

diagnostics = map(lambda subject: subject.get([Subject.Diagnostic])[0], psvrp.subjects)
diag = [[], [], [], []]
for i in xrange(len(diagnostics)):
	diag[diagnostics[i]].append(i)

adcsf = psvrp.regressors.T[0]

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
        # TODO No hardcoded
		corrected_data = psvrp.corrected_values(np.zeros((len(subjects), 121, 145, 121)), x1 = x, x2 = x+1, y1 = y, y2 = y+1, z1 = z, z2 = z+1)

#		if show_all:
#			axis = np.linspace(adcsf.min(), adcsf.max(), 50)
#			if user_defined_parameters[1] < 6:
#				Kx2 = glm_prediction_parameters.shape[0]
#				pparams = glm_prediction_parameters[(Kx2/2):]
#			else:
#				pparams = glm_prediction_parameters
#
#			lin_pparam = pparams[0, x, y, z]
#			lin_curve = lin_pparam*axis
#
#			nonlin_pparams = pparams[1:, x, y, z]
#			nonlin_curve = np.array([axis**(i+1) for i in xrange(1, pparams.shape[0])]).T.dot(nonlin_pparams)

		axis, curve = psvrp.curve(psvr_prediction_parameters, x1 = x, x2 = x+1, y1 = y, y2 = y+1, z1 = z, z2 = z+1, tpoints = 50)

#		if show_all:
#			plot(axis, lin_curve, 'y', label = 'Fitted linear curve')
#			plot(axis, nonlin_curve, 'g', label = 'Fitted nonlinear curve')

		plot(axis, curve[:, 0, 0, 0], 'r', label = 'Fitted total curve')

		color = ['co', 'bo', 'mo', 'ko']
		for i in xrange(len(diag)):
			l = diag[i]
			plot(adcsf[l], corrected_data[l, 0, 0, 0], color[i], label = Subject.Diagnostics[i])
		legend()

		show()
		print
	except Exception as e:
		print '[ERROR] Unexpected error occurred while computing and showing the results:'
		print e
		print
		continue




