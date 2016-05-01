import nibabel as nib
import numpy as np

from os.path import join
from scipy.stats import norm

fitting_scores = []

# Append all files to be compared (must have same dimensions)

# Files generated with GLM
GLM_DIR = join('results', 'GLM')
# fitting_scores.append(join(GLM_DIR, 'glm_all_fitscores.nii'))
fitting_scores.append(join(GLM_DIR, 'glm_linear_fitscores.nii'))
fitting_scores.append(join(GLM_DIR, 'glm_nonlinear_fitscores.nii'))
# fitting_scores.append(join(GLM_DIR, 'glm_quadratic_fitscores.nii'))
# fitting_scores.append(join(GLM_DIR, 'glm_cubic_fitscores.nii'))

# Files generated with GAM
# GAM_DIR = join('results', 'GAM')
# fitting_scores.append(join(GAM_DIR, 'gam_fitscores.nii'))

# Any other files?

# Path and prefix for the output files
filename_prefix = join('results', 'Bestfit', 'max', 'fscores', 'linear_vs_nonlinear_')



print 'Reading data...'

fitting_scores = map(lambda fn: nib.load(fn), fitting_scores)
affine = fitting_scores[0].affine
niiFile = nib.Nifti1Image
fitting_scores = map(lambda data_file: data_file.get_data(), fitting_scores)

print 'Selecting best fits and their respective models...'

best_fit = fitting_scores[0]
best_fit_model = np.ones(best_fit.shape, dtype = np.int)

for model in xrange(1, len(fitting_scores)):
	better_fit = fitting_scores[model] > best_fit
	best_fit[better_fit] = fitting_scores[model][better_fit]
	best_fit_model[better_fit] = model + 1

print 'Saving results...'

nib.save(niiFile(best_fit, affine), filename_prefix + 'best_fit.nii')
nib.save(niiFile(best_fit_model, affine), filename_prefix + 'best_fit_model.nii')


print 'Obtaining, filtering and saving z-scores and labels to display them...'
for fit_threshold in [0.99, 0.995, 0.999]:
	print '    Fitting-threshold set to', fit_threshold, '; Computing z-scores and labels...'

	z_scores = best_fit.copy()
	labels = best_fit_model.copy()

	invalid_zones = z_scores < fit_threshold
	valid_zones = ~invalid_zones

	z_scores[invalid_zones] = 0.0
	labels[invalid_zones] = 0

	z_scores[valid_zones] = norm.ppf(z_scores[valid_zones]) - norm.ppf(fit_threshold) + 0.2

	print '    Saving z-scores and labels to file...'
	nib.save(niiFile(z_scores, affine), filename_prefix + 'best_fit_zscores_' + str(fit_threshold) + '.nii')
	nib.save(niiFile(labels, affine), filename_prefix + 'best_fit_labels_' + str(fit_threshold) + '.nii')







print 'Done.'
