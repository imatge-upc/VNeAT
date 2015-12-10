print
print 'Welcome to filter_pvalues.py!'
print 'We are now going to filter the p-values and f-scores obtained by patterns_v6.py'
print 'according to the minimum mean amount of gray matter required for each voxel and'
print 'the maximum p-value accepted to reject the null hypothesis.'
print 'There is a lot of data to process, so please be pacient.'
print 'Thank you for using our software! :)'
print

print 'Setting relevant parameters for the program...',

import database as db
from graphlib import NiftiGraph as NGraph
from numpy import isfinite, zeros
from scipy.stats import norm

# Set region
x1, y1, z1 = [0]*3
x2, y2, z2 = [None]*3

# Set thresholds for p-value and gray matter quantity (per unit volume)
pv_threshold = 0.001
gm_threshold = 0.2

# Set threshold for minimum number of nodes present in a cluster for this
# to be considered and shown in the results
num_nodes_cluster = 100

print 'Done.'

print 'Reading data and filtering voxels by p-value and mean GM volume...'

# Initialize input structure
input_data = db.get_data(x1 = x1, y1 = y1, z1 = z1, x2 = x2, y2 = y2, z2 = z2)
dims = input_data.dims[1:4]

# Read f-statistics and p-values
pvalues = db.open_output_file('/Users/Asier/Documents/TFG/python/pvalue_v6_2.nii').get_data()
fscores = db.open_output_file('/Users/Asier/Documents/TFG/python/ftest_v6_2.nii').get_data()


# Compute the minimum p-value between linear and non-linear terms for each voxel
min_pvalues = (pvalues[:,:,:,0] < pvalues[:,:,:,1]).astype(int)
min_pvalues = pvalues[:,:,:,0]*min_pvalues + pvalues[:,:,:,1]*(1-min_pvalues)

# Initialize valid voxels 3D matrix
valid_voxels = isfinite(min_pvalues)

# Progress printing purposes only
progress = 0.0
total_num_voxels = dims[0]*dims[1]*dims[2]
prog_inc = 10000./total_num_voxels

for chunk in input_data.chunks():
	x, y, z = chunk.coords
	x -= x1
	y -= y1
	z -= z1

	dw, dx, dy, dz = chunk.data.shape
	valid_voxels[x:x+dx, y:y+dy, z:z+dz] &= sum(chunk.data) >= (gm_threshold*dw)

	progress += prog_inc*dx*dy*dz
	print '\r  Computing valid voxels:  ' + str(int(progress)/100.) + '% completed  ',

print 'Done.'

# Nullify non-valid voxels
print 'Adjusting p-values according to their validity...',
for x in range(dims[0]):
	for y in range(dims[1]):
		for z in range(dims[2]):
			if not valid_voxels[x, y, z]:
				min_pvalues[x, y, z] = 1.0
print 'Done'

print 'Filtering for clusters of size >= ' + str(num_nodes_cluster) + '...',

lim_value = norm.ppf(1 - pv_threshold)

g = NGraph(min_pvalues, pv_threshold)
for scc in g.sccs():
	if len(scc) < num_nodes_cluster:
		for x, y, z in scc:
			fscores[x, y, z] = zeros(fscores.shape[3:])
			min_pvalues[x, y, z] = 0.0
	else:
		for x, y, z in scc:
			min_pvalues[x, y, z] = norm.ppf(1 - min_pvalues[x, y, z]) - lim_value + 0.2

print 'Done.'

print 'Storing results to file...',

db.save_output_data(min_pvalues, '/Users/Asier/Documents/TFG/python/pvalue_v6_2_filtered.nii')
db.save_output_data(fscores, '/Users/Asier/Documents/TFG/python/ftest_v6_2_filtered.nii')

print 'Done.'

print
print 'Program finished without errors. Thank you for your patience!'
