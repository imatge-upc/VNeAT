from os.path import join

import nibabel as nib
from matplotlib.pyplot import plot, legend, show
from Processors.GLMProcessing import GLMProcessor as GLMP
from Utils.Subject import Subject

import database as db



# PolyGLM prefix
filename_prefix = join('results', 'GLM', 'glm_linear_')

print 'Obtaining data from Excel file'
subjects = db.get_data()

print 'Loading precomputed parameters for GLM'
glm_correction_parameters = nib.load(filename_prefix + 'cparams.nii').get_data()
glm_prediction_parameters = nib.load(filename_prefix + 'pparams.nii').get_data()

with open(filename_prefix + 'userdefparams.txt', 'rb') as f:
	user_defined_parameters = eval(f.read())

print 'Initializing GLM Processor'
glmp = GLMP(subjects, predictors = [Subject.ADCSFIndex], user_defined_parameters = user_defined_parameters, correctors = [Subject.Age, Subject.Sex])

diagnostics = map(lambda subject: subject.get([Subject.Diagnostic])[0], glmp.subjects)
diag = [[], [], [], []]
for i in xrange(len(diagnostics)):
	diag[diagnostics[i]].append(i)

adcsf = glmp.predictors.T[0]

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
		corrected_data = glmp.corrected_values(correction_parameters = glm_correction_parameters, x1 = x, x2 = x+1, y1 = y, y2 = y+1, z1 = z, z2 = z+1)
		
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

		axis, curve = glmp.curve(glm_prediction_parameters,
                                 x1 = x, x2 = x+1, y1 = y, y2 = y+1, z1 = z, z2 = z+1, tpoints = 50)

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




