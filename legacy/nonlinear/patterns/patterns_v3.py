# To be executed as follows:
# python -W ignore -u ttest.py

import database as db
from curve_fit import GLM
from tools import polynomial, copy_iterable as copy, tolist

print
print 'Welcome to patterns.py!'
print 'We are now going to analyze the data in the specified directory and perform some curve-fitting over it.'
print 'There is a lot of data to process, so please be pacient.'
print 'Thank you for using our software! :)'
print


# Set region
# x1, x2 = 0, None
# y1, y2 = 0, None
# z1, z2 = 0, None
rang = (20, 20, 20)
start = (47, 39, 72)
x2, y2, z2 = tuple(start[i] + rang[i] for i in range(3))
x1, y1, z1 = start

# Get input data and initialize output data structure
input_data = db.get_data(x1 = x1, y1 = y1, z1 = z1, x2 = x2, y2 = y2, z2 = z2)
dims = input_data.dims
output_data1 = [[[[] for _ in range(dims[3])] for _ in range(dims[2])] for _ in range(dims[1])]
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
for feature in features[2:]:
	for p in polynomial(degree, [feature]):
		xdata.append(p)

# Orthogonalize xdata parameters outside the loop (so that we don't have to repeat this over and over again)
glm = GLM(xdata[:5], xdata[0])
glm.orthogonalize()

glm2 = GLM(xdata[5:], xdata[5])

# Minimum quantity of gray matter (per unit volume) so that we consider a voxel (otherwise it will be omitted)
gm_threshold = 0.2

# Progress printing purposes only
dims = input_data.dims
prog_inc = 10000./(dims[1]*dims[2]*dims[3])
progress = 0

for chunk in input_data.chunks():
	# Chunk coordinates (relative to region being analyzed) in x, y, z
	x, y, z = chunk.coords
	x -= x1
	y -= y1
	z -= z1

	# Original dimensions of the chunk
	dw, dx, dy, dz = chunk.data.shape

	# Compute the voxels whose mean value of gray matter quantity is above (or equal to) the threshold
	valid_voxels = sum(chunk.data) >= (gm_threshold*dw)
	for i in range(dx):
		for j in range(dy):
			for k in range(dz):
				if valid_voxels[i][j][k]:
					glm.ydata = chunk.data[:, i, j, k]
					_ = glm.optimize()
					prediction = glm.predict(glm.opt_params)

					glm2.ydata = glm.ydata - prediction
					_ = glm2.optimize()

					# Store corrected values in output_data2
					output_data2[x+i][y+j][z+k] = glm2.ydata

					# Store final coefficients for the adcsf index polynomials in output_data1
					output_data1[x+i][y+j][z+k] = glm2.opt_params
					
				else:
					output_data1[x+i][y+j][z+k] = [0]*(glm.xdata.shape[0]-5)
					output_data2[x+i][y+j][z+k] = chunk.data[:, i, j, k]	
				progress += prog_inc

		# Print progress
		print '    ' + str(int(progress)/100.) + '% completed\r',
	# print '    ' + str(int(progress)/100.) + '%'
	
print

#	db.save_output_data(nparray(output_data1), '/Users/Asier/Documents/TFG/python/output1.nii')
#	db.save_output_data(nparray(output_data2), '/Users/Asier/Documents/TFG/python/output2.nii')
from numpy import array as nparray, linspace
from matplotlib.pyplot import plot, show
adcsf = features[3]
ladcsf = min(adcsf)
radcsf = max(adcsf)
npoints = 100
adcsf_axis = linspace(ladcsf, radcsf, npoints)
adcsf_polys = nparray(tolist(polynomial(degree, [adcsf_axis])))

x, y, z = 57, 49, 82
glm2 = GLM(adcsf_polys, adcsf_polys[0])
plot(adcsf_axis, glm2.predict(output_data1[x-x1][y-y1][z-z1]), 'r')
plot(adcsf, output_data2[x-x1][y-y1][z-z1], 'bo')
show()






