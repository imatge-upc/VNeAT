# To be executed as follows:
# python -W ignore -u ttest.py

import database as db
from curve_fit import GLM
from numpy import array as nparray, linspace
from tools import polynomial
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
x1, x2 = 58, 59
y1, y2 = 49, 50
z1, z2 = 82, 83

for supervoxel in db.get_data(x1 = x1, y1 = y1, z1 = z1, x2 = x2, y2 = y2, z2 = z2):
	ydata = nparray(map(lambda voxel: voxel.gmvalue, supervoxel))
	features = [[1]*len(supervoxel), [], [], []]
	for voxel in supervoxel:
		features[1].append(voxel.subject.sex)
		features[2].append(voxel.subject.age)
		features[3].append(voxel.subject.adcsf)

	xdata = features[:2]
	degree = 3
	for feature in nparray(features[2:]):
		for p in polynomial(degree, [feature]):
			xdata.append(p)
	xdata = nparray(xdata)

	glm1 = GLM(xdata[:5], ydata)
	glm1.orthogonalize()
	glm1.optimize()

	y = ydata - glm1.pred_function(glm1.xdata, *glm1.opt_params)
	glm2 = GLM(xdata[5:], y)
	glm2.optimize()

	adcsf = features[3]
	ladcsf = min(adcsf)
	radcsf = max(adcsf)
	npoints = 100
	adcsf_axis = linspace(ladcsf, radcsf, npoints)
	plot(adcsf_axis, glm2.pred_function(nparray([adcsf_axis, adcsf_axis**2, adcsf_axis**3]), *glm2.opt_params), 'r', adcsf, y, 'bo')
	show()

	#TODO: store results for each supervoxel (also define what results are exactly)
	#TODO: perform and store tests over results






