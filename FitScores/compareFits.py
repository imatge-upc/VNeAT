import nibabel as nib
import numpy as np

from os.path import join
from scipy.stats import norm

fscores = []

# Append all files to be compared (must have same dimensions)

# Files generated with GLM
GLM_DIR = join('results', 'GLM')
fscores.append(join(GLM_DIR, 'glm_linear_alone_aicscores.nii'))
fscores.append(join(GLM_DIR, 'glm_nonlinear_alone_aicscores.nii'))

# Any other files? (Maximum allowed is 3)

# Path and prefix for the output files
filename_prefix = join('results', 'Bestfit', 'rgb', 'aic', 'linear_vs_nonlinear_')



print 'Reading data...'

fscores = map(lambda fn: nib.load(fn), fscores)
affine = fscores[0].affine
niiFile = nib.Nifti1Image
fscores = map(lambda data_file: data_file.get_data(), fscores)

print 'Building RGB maps...'

fits = np.zeros(fscores[0].shape + (3,), dtype = np.float64)

for model in xrange(len(fscores)):
	fits[:, :, :, model] = fscores[model]

print 'Saving results...'

nib.save(niiFile(fits, affine), filename_prefix + 'fit_comparison.nii')


#	print 'Obtaining, filtering and saving z-scores and labels to display them...'
#	for fit_threshold in [0.99, 0.995, 0.999]:
#		print '    Fitting-threshold set to', fit_threshold, '; Computing z-scores and labels...'
#	
#		z_scores = best_fit.copy()
#		labels = best_fit_model.copy()
#	
#		invalid_zones = z_scores < fit_threshold
#		valid_zones = ~invalid_zones
#	
#		z_scores[invalid_zones] = 0.0
#		labels[invalid_zones] = 0
#	
#		z_scores[valid_zones] = norm.ppf(z_scores[valid_zones]) - norm.ppf(fit_threshold) + 0.2
#	
#		print '    Saving z-scores and labels to file...'
#		nib.save(niiFile(z_scores, affine), filename_prefix + 'best_fit_zscores_' + str(fit_threshold) + '.nii')
#		nib.save(niiFile(labels, affine), filename_prefix + 'best_fit_labels_' + str(fit_threshold) + '.nii')







print 'Done.'
