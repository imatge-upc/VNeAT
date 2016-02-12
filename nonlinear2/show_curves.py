
show_all = False



from ExcelIO import ExcelSheet as Excel
from GLMProcessing import PolyGLMProcessor as PGLMP
from Subject import Subject
from os.path import join, isfile, basename
from os import listdir

import nibabel as nib
from numpy import array as nparray, zeros
from matplotlib.pyplot import plot, legend, show


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

print 'Loading precomputed parameters'
correction_parameters = nib.load(join('results', 'fpmalfa_cparams.nii')).get_data()
regression_parameters = nib.load(join('results', 'fpmalfa_rparams.nii')).get_data()

with open(join('results', 'fpmalfa_userdefparams.txt'), 'rb') as f:
	user_defined_parameters = eval(f.read())

print 'Initializing PolyGLM Processor'
pglmp = PGLMP(subjects, regressors = [Subject.ADCSFIndex], correctors = [Subject.Age, Subject.Sex], user_defined_parameters = user_defined_parameters)

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

		corrected_data = pglmp.corrected_values(correction_parameters, x1 = x, x2 = x+1, y1 = y, y2 = y+1, z1 = z, z2 = z+1)

		if show_all:
			lin_rparams = zeros(regression_parameters.shape)
			lin_rparams[0] = regression_parameters[0]
			_, lin_curve = pglmp.curve(lin_rparams, x1 = x, x2 = x+1, y1 = y, y2 = y+1, z1 = z, z2 = z+1, tpoints = 50)
			
			nonlin_rparams = zeros(regression_parameters.shape)
			nonlin_rparams[1:] = regression_parameters[1:]
			_, nonlin_curve = pglmp.curve(nonlin_rparams, x1 = x, x2 = x+1, y1 = y, y2 = y+1, z1 = z, z2 = z+1, tpoints = 50)

		axis, curve = pglmp.curve(regression_parameters, x1 = x, x2 = x+1, y1 = y, y2 = y+1, z1 = z, z2 = z+1, tpoints = 50)

		if show_all:
			plot(axis, lin_curve[:, 0, 0, 0], 'y', label = 'Fitted linear curve')
			plot(axis, nonlin_curve[:, 0, 0, 0], 'g', label = 'Fitted nonlinear curve')
		
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




