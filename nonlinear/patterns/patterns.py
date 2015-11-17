# To be executed as follows:
# python -W ignore -u ttest.py

import database as db
from curve_fit import GLM
from numpy import array as nparray, linspace
from tools import polynomial, copy_iterable as copy, tolist #, combinatorial
from matplotlib.pyplot import plot, show

print
print 'Welcome to ttest.py!'
print 'We are now going to analyze the data in the specified directory and perform t-tests over it.'
print 'There is a lot of data to process, so please be pacient.'
print 'Thank you for using our software! :)'
print


# Set region
# x1, x2 = 0, None
# y1, y2 = 0, None
# z1, z2 = 0, None
x1, x2 = 47, 68
y1, y2 = 39, 60
z1, z2 = 72, 93

# Get input data and initialize output data structure
input_data = db.get_data(x1 = x1, y1 = y1, z1 = z1, x2 = x2, y2 = y2, z2 = z2)
dims = input_data.dims[:3]
output_data1 = [[[[] for _ in range(dims[2])] for _ in range(dims[1])] for _ in range(dims[0])]
output_data2 = tolist(copy(output_data1))

# Compute features (independent of voxel, dependent only on subjects)
features = [[1]*len(input_data.subjects), [], [], []]
for subject in input_data.subjects:
	features[1].append(subject.sex)
	features[2].append(subject.age)
	features[3].append(subject.adcsf)


# Homogeneous term and sex are set only linearly (does not make sense to power them,
# since ones(l)**k = sex**(2*k) = ones(l), and sex**(2*k + 1) = sex for all k
xdata = features[:2]

# Polynomials up to 3rd degree of age and adcsf (positions 2 and 3 of features)
degree = 3
for feature in nparray(features[2:]):
	for p in polynomial(degree, [feature]):
		xdata.append(p)
xdata = nparray(xdata)

# Polynomyals up to 3 of extended AD-CSF index axis to compute output
adcsf = features[3]
ladcsf = min(adcsf)
radcsf = max(adcsf)
npoints = 100
adcsf_axis = linspace(ladcsf, radcsf, npoints)
adcsf_polys = nparray(tolist(polynomial(degree, [adcsf_axis])))


for voxel in input_data.voxels():
	# Gray matter values in ydata, voxel coordinates (relative to region) in i, j, k
	ydata = voxel.data
	i, j, k = voxel.coords
	i -= x1
	j -= y1
	k -= z1

	# Correct GM values with sex, age (up to 3rd order polynomial), and homogeneous term
	glm1 = GLM(xdata[:5], ydata)
	glm1.orthogonalize()
	glm1.optimize()
	ydata2 = ydata - glm1.pred_function(glm1.xdata, *glm1.opt_params)

	# Store corrected values in output_data1
	output_data1[i][j][k] = ydata2

	# Try to predict corrected GM values with AD-CSF polynomial terms
	glm2 = GLM(xdata[5:], ydata2)
	glm2.optimize()

	# Store final function in output_data2
	output_data2[i][j][k] = glm2.pred_function(adcsf_polys, *glm2.opt_params)

	#TODO: redefine what results are exactly (for now, just raw function points, maybe use parameters + function so that it is reproducible?)
	#TODO: perform and store tests over results

	#	-------------  DEAD CODE: Keep scrolling  -------------
	#
	#	degree = 3
	#	xdata = tolist(copy(features[:1]))
	#	xdata += list(polynomial(degree, nparray(tolist(copy(features[1:3])))))
	#
	#
	#	adcsfindices = []
	#	i = 0
	#	for l in combinatorial(lambda x, y: x + y, map(lambda x: [x], range(degree)), degree):
	#		if all(x == 2 for x in l):
	#			adcsfindices.append(i)
	#		i += 1
	#
	#	xdata2 = map(lambda index: xdata[index], adcsfindices)
	#
	#	adcsfindices.reverse()
	#	for index in adcsfindices:
	#		del xdata[index]
	#
	#	xdata = nparray(xdata)
	#	glm1 = GLM(xdata, ydata)
	#	glm1.orthogonalize()
	#	glm1.optimize()
	#
	#	xdata2 = list(nparray(tolist(copy(features[3:]))))
	#	xdata2 = nparray(xdata2)
	#	ydata2 = ydata - glm1.pred_function(glm1.xdata, *glm1.opt_params)
	#	glm2 = GLM(xdata2, ydata2)
	#	glm2.optimize()
	#
	#	-------------------------------------------------------

x, y, z = 57, 49, 82
plot(adcsf_axis, output_data2[x-x1][y-y1][z-z1], 'r', adcsf, output_data1[x-x1][y-y1][z-z1], 'bo')
show()

db.save_output_data(nparray(output_data1), '/Users/Asier/Documents/TFG/python/output1.nii')
db.save_output_data(nparray(output_data2), '/Users/Asier/Documents/TFG/python/output2.nii')







