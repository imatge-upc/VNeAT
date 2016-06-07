
polynomial_degree = 3
accepted_maximum_error = 104

# x, y, z = 71, 79, 39
# mm_coordinates = -16, -8, -14, 1

# x, y, z = 44, 66, 40
# mm_coordinates = 24, -28, -12, 1

x, y, z = 59, 48, 66
mm_coordinates = 2, -54, 26, 1


from os.path import join

import nibabel as nib
from Processors.GAMProcessing import GAMProcessor as GAMP
from Utils.Subject import Subject
import Utils.DataLoader as DataLoader
from user_paths import RESULTS_DIR
RESULTS_DIR = join(RESULTS_DIR, 'SGAM')

niiFile = nib.Nifti1Image
affine = DataLoader.getMNIAffine()

print 'Obtaining data from Excel file...'
subjects = DataLoader.getSubjects(corrected_data=True)

udp = (9,[2,3,0,15,3])
gamp = GAMP(subjects, predictors=[Subject.ADCSFIndex],user_defined_parameters=udp)

results = gamp.process(x1 = x, x2 = x+1, y1 = y, y2 = y+1, z1 = z, z2 = z+1)

prediction_parameters = results.prediction_parameters

from Utils.DataLoader import getSubjects, getMNIAffine
affine = getMNIAffine()
adcsf = gamp.predictors.T[0]
diagnostics = map(lambda subject: subject.get([Subject.Diagnostic])[0], subjects)
diag = [[], [], [], []]
for i in xrange(len(diagnostics)):
	diag[diagnostics[i]].append(i)

corrected_data = gamp.gm_values(x1=x, x2=x+1, y1=y, y2=y+1, z1=z, z2=z+1)
from matplotlib import pyplot as plot
color = ['co', 'bo', 'mo', 'ko']
for i in xrange(len(diag)):
	l = diag[i]
	plot.plot(adcsf[l], corrected_data[l, 0, 0, 0], color[i], lw=4, label=Subject.Diagnostics[i])

axis, curve = gamp.curve(prediction_parameters, tpoints=-1)
plot.plot(axis, curve[:, 0, 0, 0], lw=2, color='g', marker='d')

plot.show()
a=1
#	
#	""" Shows curves for all fitters created:
#			- Poly GLM
#			- Poly GAM
#			- Poly SVR
#	"""
#	from os.path import join
#	
#	import matplotlib.pyplot as plot
#	import nibabel as nib
#	import numpy as np
#	
#	# Utils
#	from Utils.DataLoader import getSubjects, getMNIAffine
#	from Utils.Subject import Subject
#	
#	# Processors
#	from Processors.GLMProcessing import PolyGLMProcessor as PGLMP
#	from Processors.GAMProcessing import GAMProcessor as GAMP
#	from Processors.SVRProcessing import PolySVRProcessor as PSVRP, GaussianSVRProcessor as GSVRP
#	
#	# Info
#	fitters = [
#	#     NAME              PROCESSOR   PATH                                                                            COLOR       MARKER
#		['GLM',             PGLMP,      join('results', 'PGLM', 'pglm_'),                                               'm',        'd'   ],
#		['Polynomial GAM',  GAMP,       join('results', 'PGAM', 'gam_poly_'),                                           'y',        'd'   ],
#	#    ['Splines GAM',     GAMP,       join('results', 'SGAM', 'gam_splines_'),                                        'g',        'd'   ],
#		['Polynomial SVR',  PSVRP,      join('results', 'PSVR', 'psvr_C3.16227766017_eps0.16_'),                        'b',        'd'   ],
#		['Gaussian SVR',    GSVRP,      join('results', 'GSVR', 'gsvr_C3.16227766017_eps0.0891666666667_gamma0.25_'),   'r',        'd'   ]
#	]
#	
#	print 'Obtaining data from Excel file...'
#	subjects = getSubjects(corrected_data=True)
#	
#	print 'Obtaining affine matrix to map mm<-->voxels...'
#	affine = getMNIAffine()
#	
#	print 'Loading precomputed parameters for all fitters...'
#	prediction_parameters = []
#	user_defined_parameters = []
#	for fitter in fitters:
#		prediction_parameters.append(nib.load(fitter[2] + 'pparams.nii').get_data())
#		with open(fitter[2] + 'userdefparams.txt', 'rb') as f:
#			user_defined_parameters.append(eval(f.read()))
#	
#	
#	print 'Initializing processors'
#	processors = []
#	for fitter, user_params in zip(fitters, user_defined_parameters):
#		processors.append(
#			fitter[1](
#				subjects,
#				predictors = [Subject.ADCSFIndex],
#				user_defined_parameters = user_params
#			)
#		)
#	
#	diagnostics = map(lambda subject: subject.get([Subject.Diagnostic])[0], subjects)
#	diag = [[], [], [], []]
#	for i in xrange(len(diagnostics)):
#		diag[diagnostics[i]].append(i)
#	
#	adcsf = processors[0].predictors.T[0]
#	
#	
#	print 'Processing request... please wait'
#	
#	try:
#		# Transform mm coordinates -> voxel coordinates using affine
#		# mm_coordinates = np.array([x, y, z, 1])
#		voxel_coordinates = map(int, np.round(np.linalg.inv(affine).dot(mm_coordinates)))
#		# Get rounded mm coordinates in MNI space (due to 1.5 mm spacing)
#		mm_coordinates_prima = affine.dot(voxel_coordinates)
#		# Final voxel coordinates
#		# x = voxel_coordinates[0]
#		# y = voxel_coordinates[1]
#		# z = voxel_coordinates[2]
#		print 'This is voxel', x, y, z
#		# Get (corrected) grey matter data
#		corrected_data = processors[0].gm_values(
#			x1=x, x2=x+1, y1=y, y2=y+1, z1=z, z2=z+1)
#		# Get curves for all processors
#		for i in range(len(processors)):
#			axis, curve = processors[i].curve(
#				prediction_parameters[i],
#				x1=x, x2=x+1, y1=y, y2=y+1, z1=z, z2=z+1, tpoints=50)
#			random_color = np.random.rand(3,1)
#			plot.plot(axis, curve[:, 0, 0, 0],
#					  lw=2, label=fitters[i][0], color=fitters[i][3], marker=fitters[i][4])
#		color = ['co', 'bo', 'mo', 'ko']
#		for i in xrange(len(diag)):
#			l = diag[i]
#			plot.plot(adcsf[l], corrected_data[l, 0, 0, 0], color[i], lw=4, label=Subject.Diagnostics[i])
#		# Plot info
#		# plot.legend(fontsize='xx-large')
#		plot.xlabel('ADCSF', fontsize='xx-large')
#		plot.ylabel('Grey matter', fontsize='xx-large')
#		plt_title = 'Coordinates: ' + \
#					str(mm_coordinates_prima[0]) + ', ' + \
#					str(mm_coordinates_prima[1]) + ', ' + \
#					str(mm_coordinates_prima[2]) + ' mm'
#		plot.title(plt_title, size="xx-large")
#		plot.show()
#		print
#	except Exception as e:
#		print '[ERROR] Unexpected error occurred while computing and showing the results:'
#		print e
#		print
#	
#	"""
#	INTERESTING COORDINATES:
#	
#		- Right Precuneus: 2, -54, 26
#	
#		- Left Hippocampus: -16, -8, -14
#	
#		- Right ParaHippocampal: 24, -28, -12
#	
#	"""