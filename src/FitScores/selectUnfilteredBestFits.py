from os.path import join

import nibabel as nib
import numpy as np
from scipy.stats import norm

fitting_scores = []

# Append all files to be compared (must have same dimensions)

# Files generated with GLM - AIC
GLM_DIR = join('results', 'GLM')
fitting_scores.append(join(GLM_DIR, 'glm_linear_alone_aicscores.nii'))
fitting_scores.append(join(GLM_DIR, 'glm_nonlinear_alone_aicscores.nii'))

# Any other files?

# Path and prefix for the output files
filename_prefix = join('results', 'Bestfit', 'max', 'aic', 'linear_vs_nonlinear_alone_')

print 'Reading data...'

fitting_scores = map(lambda fn: nib.load(fn), fitting_scores)
affine = fitting_scores[0].affine
niiFile = nib.Nifti1Image
fitting_scores = map(lambda data_file: data_file.get_data(), fitting_scores)

print 'Selecting best fits and their respective models...'

best_fit = fitting_scores[0]
best_fit_model = np.ones(best_fit.shape, dtype=np.int)

for model in xrange(1, len(fitting_scores)):
    better_fit = fitting_scores[
                     model] < best_fit  ####################### <<--------- LOOK OUT!!! Here we take the minimum as best!
    best_fit[better_fit] = fitting_scores[model][better_fit]
    best_fit_model[better_fit] = model + 1

print 'Saving results...'

nib.save(niiFile(best_fit, affine), filename_prefix + 'best_fit.nii')
nib.save(niiFile(best_fit_model, affine), filename_prefix + 'best_fit_model.nii')

best_fit = -best_fit
best_fit = (best_fit - best_fit.mean()) / (best_fit.std())

mask = best_fit > norm.cdf(0.99)
best_fit[~mask] = 0.0
best_fit_model[~mask] = 0.0

nib.save(niiFile(best_fit, affine), filename_prefix + 'best_fit_postprocessed.nii')
nib.save(niiFile(best_fit_model, affine), filename_prefix + 'best_fit_model_postprocessed.nii')

# best_fit_rgb = np.zeros()


print 'Done.'
