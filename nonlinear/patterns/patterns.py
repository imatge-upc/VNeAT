# To be executed as follows:
# python -W ignore -u ttest.py

import database as db
from curve_fit import GLM
from numpy import array as nparray

print
print 'Welcome to ttest.py!'
print 'We are now going to analyze the data in the specified directory and perform t-tests over it.'
print 'There is a lot of data to process, so please be pacient.'
print 'Thank you for using our software! :)'
print

OUTPUT_DIR = 'ttests'
OUTPUT_FILENAME = 'ttest_results'

# Set region
x1 = 0
x2 = None
y1 = 0
y2 = None
z1 = 0
z2 = None

for supervoxel in db.get_data(x1 = x1, y1 = y1, z1 = z1, x2 = x2, y2 = y2, z2 = z2):
	ydata = nparray(map(lambda voxel: voxel.gmvalue, supervoxel))
	xdata = []
	#TODO: append features to xdata

	xdata = nparray(xdata)
	glm = GLM(xdata, ydata)
	glm.orthogonalize()
	glm.optimize()

	#TODO: store results for each supervoxel (also define what results are exactly)
	#TODO: perform and store tests over results