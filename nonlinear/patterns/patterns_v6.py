# To be executed as follows:
# python -W ignore -u ttest.py

import database as db
from numpy import array as nparray, zeros
from curve_fit_v2 import GLM
from scipy.stats import f as f_stat
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

# Get input data and initialize output data structures
input_data = db.get_data(x1 = x1, y1 = y1, z1 = z1, x2 = x2, y2 = y2, z2 = z2)
dims = input_data.dims
output_data = zeros((7,) + dims[1:4], dtype = float)
ftest_results = zeros(dims[1:4] + (3,), dtype = float) # RG(B) for each voxel
p_value_results = zeros(dims[1:4] + (2,), dtype = float) # nonlinear vs. full in 1, linear vs. full in 0

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
for p in polynomial(2, [age]):
	xdata1.append(p)

xdata1 = nparray(xdata1, dtype = float).T


xdata2 = []
for p in polynomial(3, [adcsf]):
	xdata2.append(p)

xdata3 = xdata2[1:] + xdata2[:1]

xdata2 = nparray(xdata2, dtype = float).T
xdata3 = nparray(xdata3, dtype = float).T

# Initialize GLMs outside the loop
# Correction GLM
glm1 = GLM(xdata1, xdata1[:, 0], homogeneous = True)
glm1.orthonormalize()

# Curve GLM
glm2 = GLM(xdata2, xdata2[:, 0], homogeneous = False)

# Nonlinear test GLM
glm3 = GLM(xdata2, xdata2[:, 0], homogeneous = False)
glm3.orthonormalize()

# Linear test GLM
glm4 = GLM(xdata3, xdata3[:, 0], homogeneous = False)
glm4.orthonormalize()

# Minimum quantity of gray matter (per unit volume) so that we consider a voxel (otherwise it will be omitted)
gm_threshold = 0.2

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
	#	glm1.ydata = nparray([chunk.data[:,i,j,k]*valid_voxels[i,j,k] for i in range(dx) for j in range(dy) for k in range(dz)], dtype = float).T
	glm1.ydata = chunk.data.reshape((dw, dx*dy*dz))
	# Correct GM values with orthogonalized sex, age (up to 2nd order polynomial),
	# and the homogeneous term, and try to predict corrected GM values with AD-CSF
	# polynomial terms (all in one optimization call)
	glm1.optimize()
	output_data[:4, x:x+dx, y:y+dy, z:z+dz] = glm1.opt_params.reshape((4, dx, dy, dz))
	glm2.ydata = glm1.ydata - GLM.predict(glm1.xdata, glm1.opt_params)
	glm2.optimize()
	output_data[4:, x:x+dx, y:y+dy, z:z+dz] = glm2.opt_params.reshape((3, dx, dy, dz))
	#####################################
	## Nonlinear f-scores and p-values ##
	#####################################
	glm3.ydata = glm2.ydata
	glm3.optimize()
	# Compute error with restricted model (only linear component)
	e = glm3.ydata - GLM.predict(glm3.xdata[:, :1], glm3.opt_params[:1])
	rss1 = sum(e**2)
	p1 = 1 # only linear component
	# Compute error with unrestricted model (add nonlinear components)
	e = glm3.ydata - GLM.predict(glm3.xdata, glm3.opt_params)
	rss2 = sum(e**2)
	p2 = glm3.xdata.shape[1] # number of regressors (in this case, 3)
	# Degrees of freedom
	n = glm3.xdata.shape[0] # number of samples
	df1 = p2 - p1
	df2 = n - p2 + 1
	# Compute f-scores and p-values
	var1 = (rss1 - rss2)/df1
	var2 = rss2/df2
	f_score = var1/var2
	p_value = f_stat.cdf(f_score, df1, df2)
	# Store results
	ftest_results[x:x+dx, y:y+dy, z:z+dz, 1] = f_score.reshape((dx, dy, dz)) # Green
	p_value_results[x:x+dx, y:y+dy, z:z+dz, 1] = p_value.reshape((dx, dy, dz))
	##################################
	## Linear f-scores and p-values ##
	##################################
	glm4.ydata = glm2.ydata
	glm4.optimize()
	# Compute error with restricted model (only nonlinear components)
	e = glm4.ydata - GLM.predict(glm4.xdata[:, :2], glm4.opt_params[:2])
	rss1 = sum(e**2)
	p1 = glm4.xdata.shape[1] - 1 # all components except for the linear one
	# Compute error with unrestricted model (add linear component)
	e = glm4.ydata - GLM.predict(glm4.xdata, glm4.opt_params)
	rss2 = sum(e**2)
	p2 = glm4.xdata.shape[1] # number of regressors (in this case, 3)
	# Degrees of freedom
	n = glm4.xdata.shape[0] # number of samples
	df1 = p2 - p1
	df2 = n - p2 + 1
	# Compute f-scores and p-values
	var1 = (rss1 - rss2)/df1
	var2 = rss2/df2
	f_score = var1/var2
	p_value = f_stat.cdf(f_score, df1, df2)
	# Store results
	ftest_results[x:x+dx, y:y+dy, z:z+dz, 0] = f_score.reshape((dx, dy, dz)) # Red
	p_value_results[x:x+dx, y:y+dy, z:z+dz, 0] = p_value.reshape((dx, dy, dz))
	# Print progress
	progress += prog_inc*dx*dy*dz
	print '\r  Computing parameters:  ' + str(int(progress)/100.) + '% completed',

print

db.save_output_data(output_data, '/Users/Asier/Documents/TFG/python/output_v6_2.nii')
db.save_output_data(ftest_results, '/Users/Asier/Documents/TFG/python/ftest_v6_2.nii')
db.save_output_data(p_value_results, '/Users/Asier/Documents/TFG/python/pvalue_v6_2.nii')

from numpy import linspace
from tools import tolist
# Polynomyals up to 3 of extended AD-CSF index axis to compute output
ladcsf = min(adcsf)
radcsf = max(adcsf)
npoints = 100
adcsf_axis = linspace(ladcsf, radcsf, npoints)
adcsf_polys = nparray(tolist(polynomial(3, [adcsf_axis])))

x, y, z = 57, 49, 82

vgen = db.get_data(x1 = x, x2 = x+1, y1 = y, y2 = y+1, z1 = z, z2 = z+1)
vgen = vgen.voxels()
v = vgen.next()

params = output_data[:4, x-x1, y-y1, z-z1]
corrected_data = v.data - GLM.predict(glm1.xdata, params)

params = output_data[4:, x-x1, y-y1, z-z1]
curve = GLM.predict(adcsf_polys.T, params)

from matplotlib.pyplot import show, plot

plot(adcsf_axis, curve, 'r', adcsf, corrected_data, 'bo')
show()






