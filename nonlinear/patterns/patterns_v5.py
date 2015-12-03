# To be executed as follows:
# python -W ignore -u ttest.py

import database as db
from numpy import array as nparray, zeros
from sklearn.linear_model import LinearRegression as GLM
from tools import polynomial


print
print 'Welcome to patterns.py!'
print 'We are now going to analyze the data in the specified directory and perform some curve-fitting over it.'
print 'There is a lot of data to process, so please be pacient.'
print 'Thank you for using our software! :)'
print


# Set region
x1, x2 = 0, None
y1, y2 = 0, None
z1, z2 = 0, None
# rang = (20, 20, 20)
# start = (47, 39, 72)
# x2, y2, z2 = tuple(start[i] + rang[i] for i in range(3))
# x1, y1, z1 = start

# Get input data and initialize output data structure
input_data = db.get_data(x1 = x1, y1 = y1, z1 = z1, x2 = x2, y2 = y2, z2 = z2)
dims = input_data.dims
output_data = zeros((8,) + dims[1:4], dtype = float)

# Compute features (independent of voxel, dependent only on subjects)
sex = []
age = []
adcsf = []
for subject in input_data.subjects:
	sex.append(subject.sex)
	age.append(subject.age)
	adcsf.append(subject.adcsf)


# Sex is set only linearly (does not make sense to power it, since sex**(2*k) = ones(l),
# and sex**(2*k + 1) = sex for all k)
xdata1 = [sex]

# Polynomials up to 3rd degree of age and adcsf
degree = 3
for p in polynomial(degree, [age]):
	xdata1.append(p)

xdata1 = nparray(xdata1, dtype = float).T


xdata2 = []
for p in polynomial(degree, [adcsf]):
	xdata2.append(p)

xdata2 = nparray(xdata2, dtype = float).T

# Initialize GLMs outside the loop
glm = GLM(fit_intercept = True, normalize = True, copy_X = True, n_jobs = -1)
glm2 = GLM(fit_intercept = False, normalize = False, copy_X = True, n_jobs = -1)

# Minimum quantity of gray matter (per unit volume) so that we consider a voxel (otherwise it will be omitted)
#	gm_threshold = 0.2

# Progress printing purposes only
progress = 0.0
total_num_voxels = dims[1]*dims[2]*dims[3]
prog_inc = 10000./total_num_voxels

for chunk in input_data.chunks():
	# Chunk coordinates (relative to region being analyzed) in x, y, z
	x, y, z = chunk.coords
	x -= x1
	y -= y1
	z -= z1

	# Original dimensions of the chunk
	dw, dx, dy, dz = chunk.data.shape

	# Compute the voxels whose mean value of gray matter quantity is above (or equal to) the threshold
	#	valid_voxels = (sum(chunk.data) >= gm_threshold*dw).astype(int)

	# Gray matter values (in matrix form, nullified if they are below threshold) in ydata
	# ydata[i*dy*dz + j*dz + k] contains the nullified GM values associated to voxel (i, j, k) in the region
	#	ydata = nparray([chunk.data[:,i,j,k]*valid_voxels[i,j,k] for i in range(dx) for j in range(dy) for k in range(dz)], dtype = float).T

	ydata = chunk.data.reshape((dw, dx*dy*dz))

	# Correct GM values with orthogonalized sex, age (up to 3rd order polynomial),
	# and the homogeneous term, and try to predict corrected GM values with AD-CSF
	# polynomial terms (all in one optimization call)
	_ = glm.fit(xdata1, ydata)
	output_data[0, x:x+dx, y:y+dy, z:z+dz] = glm.intercept_.reshape((dx, dy, dz))
	output_data[1:5, x:x+dx, y:y+dy, z:z+dz] = glm.coef_.T.reshape((4, dx, dy, dz))
	
	_ = glm2.fit(xdata2, ydata - glm.predict(xdata1))
	output_data[5:, x:x+dx, y:y+dy, z:z+dz] = glm2.coef_.T.reshape((3, dx, dy, dz))
	
	# Print progress
	progress += prog_inc*dx*dy*dz
	print '\r  Computing parameters:  ' + str(int(progress)/100.) + '% completed',

print

db.save_output_data(output_data, '/Users/Asier/Documents/TFG/python/output_v5.nii')


#	from numpy import linspace
#	from tools import tolist
#	# Polynomyals up to 3 of extended AD-CSF index axis to compute output
#	ladcsf = min(adcsf)
#	radcsf = max(adcsf)
#	npoints = 100
#	adcsf_axis = linspace(ladcsf, radcsf, npoints)
#	adcsf_polys = nparray(tolist(polynomial(degree, [adcsf_axis])))
#	
#	x, y, z = 57, 49, 82
#	
#	vgen = db.get_data(x1 = x, x2 = x+1, y1 = y, y2 = y+1, z1 = z, z2 = z+1)
#	vgen = vgen.voxels()
#	v = vgen.next()
#	
#	glm.intercept_ = output_data[0, x-x1, y-y1, z-z1]
#	glm.coef_ = output_data[1:5, x-x1, y-y1, z-z1]
#	corrected_data = v.data - glm.predict(xdata1)
#	
#	coefs = output_data[5:, x-x1, y-y1, z-z1]
#	curve = sum(adcsf_polys[i]*coefs[i] for i in range(len(coefs)))
#	
#	from matplotlib.pyplot import show, plot
#	
#	plot(adcsf_axis, curve, 'r', adcsf, corrected_data, 'bo')
#	show()






