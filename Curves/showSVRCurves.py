from os.path import join

import nibabel as nib
from matplotlib.pyplot import plot, legend, show
#from Processors.SVRProcessing import GaussianSVRProcessor as PSVR
from Processors.SVRProcessing import PolySVRProcessor as PSVRP, GaussianSVRProcessor as GSVRP
from Utils.Subject import Subject

from Utils.DataLoader import getSubjects

# PolySVR prefix
filename_prefix = join('results', 'PSVR', 'psvr_C3.16227766017_eps0.16_')

print 'Obtaining data from Excel file'
subjects = getSubjects(corrected_data=True)

print 'Loading precomputed parameters for Polynomial SVR'
psvr_prediction_parameters = nib.load(filename_prefix + 'pparams.nii').get_data()

with open(filename_prefix + 'userdefparams.txt', 'rb') as f:
	user_defined_parameters = eval(f.read())

print 'Initializing Polynomial SVR Processor'
psvrp = PSVRP(subjects, predictors = [Subject.ADCSFIndex], user_defined_parameters = user_defined_parameters)

# GausSVR prefix
filename_prefix = join('results', 'GSVR', 'gsvr_C3.16227766017_eps0.0891666666667_gamma0.25_')

print 'Loading precomputed parameters for Gaussian SVR'
gsvr_prediction_parameters = nib.load(filename_prefix + 'pparams.nii').get_data()

with open(filename_prefix + 'userdefparams.txt', 'rb') as f:
	user_defined_parameters = eval(f.read())

print 'Initializing Gaussian SVR Processor'
gsvrp = GSVRP(subjects, predictors = [Subject.ADCSFIndex], user_defined_parameters = user_defined_parameters)



diagnostics = map(lambda subject: subject.get([Subject.Diagnostic])[0], psvrp.subjects)
diag = [[], [], [], []]
for i in xrange(len(diagnostics)):
	diag[diagnostics[i]].append(i)

adcsf = psvrp.predictors.T[0]

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
		# PolySVR Curve
		corrected_data = psvrp.gm_values(x1 = x, x2 = x+1, y1 = y, y2 = y+1, z1 = z, z2 = z+1)
		axis, pcurve = psvrp.curve(psvr_prediction_parameters,
                                   x1 = x, x2 = x+1, y1 = y, y2 = y+1, z1 = z, z2 = z+1, tpoints = 50)

		_, gcurve = gsvrp.curve(gsvr_prediction_parameters,
                                x1 = x, x2 = x+1, y1 = y, y2 = y+1, z1 = z, z2 = z+1, tpoints = 50)

		plot(axis, pcurve[:, 0, 0, 0], 'b', label = 'Fitted Polynomic SVR curve')
		plot(axis, gcurve[:, 0, 0, 0], 'r', label = 'Fitted Gaussian SVR curve')

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




