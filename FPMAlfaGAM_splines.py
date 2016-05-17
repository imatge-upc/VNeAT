from Utils.ExcelIO import ExcelSheet as Excel
from Processors.GAMProcessing import GAMProcessor as GAMP
from Utils.Subject import Subject
from os.path import join, isfile, basename
import Utils.DataLoader as DataLoader
import nibabel as nib
from numpy import array as nparray


print 'Obtaining data from Excel file...'

subjects = DataLoader.getSubjects(corrected_data=True)
print 'Initializing GAM Processor...'

s_target=105
user_defined_parameters = [
	(9,[2,2,s_target,3]),
]

filenames = [

	'gam_splines_d3_s1',

]
x1 = 85#  71#  103#
x2 = x1 + 1
y1 = 101#  79#    45#
y2 = y1 + 1
z1 = 45  #39# 81#
z2 = z1 + 1

for udp,filename in zip(user_defined_parameters,filenames):
	gamp = GAMP(subjects, predictors = [Subject.ADCSFIndex],user_defined_parameters=udp)

	print 'Processing data...'
	results = gamp.process(x1=x1,x2=x2,y1=y1,y2=y2,z1=z1,z2=z2)



from matplotlib import pyplot as plt
import numpy as np

adcsf = np.sort(gamp.predictors.T[0])
index_adcsf = np.argsort(gamp.predictors.T[0])

from Fitters.GAM import SplinesSmoother
spl_smoother = SplinesSmoother(adcsf,order=3,smoothing_factor=s_target)
df_target = spl_smoother.df_model(gamp.gm_values(x1=x1,x2=x2,y1=y1,y2=y2,z1=z1,z2=z2)[index_adcsf, 0, 0, 0])


plt.plot(adcsf, gamp.gm_values(x1=x1,x2=x2,y1=y1,y2=y2,z1=z1,z2=z2)[index_adcsf, 0, 0, 0], 'k.')
plt.plot(adcsf, gamp.__curve__(-1, np.sort(adcsf[:, np.newaxis]), np.squeeze(results.prediction_parameters)))
plt.title(df_target)
plt.show()

print 'Done.'

