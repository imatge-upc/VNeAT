from os.path import join

import nibabel as nib
import numpy as np
from scipy.stats import norm

from user_paths import RESULTS_DIR

prss_scores = []

# Append all files to be compared (must have same dimensions)

# Files generated with GLM
GLM_DIR = join(RESULTS_DIR, 'MIXED', 'PGLM-PGLM')
prss_scores.append(join(GLM_DIR, 'pglm_pglm_inv_vnprss_0.5.nii'))

# Files generated with GAM
GAM_DIR = join(RESULTS_DIR, 'MIXED', 'PGLM-PGAM')
prss_scores.append(join(GAM_DIR, 'pglm_pgam_inv_vnprss_0.5.nii'))

# Files generated with Poly SVR
POLYSVR_DIR = join(RESULTS_DIR, 'MIXED', 'PGLM-PSVR')
prss_scores.append(join(POLYSVR_DIR, 'pglm_psvr_inv_vnprss_0.5.nii'))

# Files generated with Gaussian SVR
GAUSSIANSVR_DIR = join(RESULTS_DIR, 'MIXED', 'PGLM-GSVR')
prss_scores.append(join(GAUSSIANSVR_DIR, 'pglm_gsvr_inv_vnprss_0.5.nii'))

# Path and prefix for the output files
filename_prefix = join(RESULTS_DIR, 'BESTFIT', 'MAX_PRSS_SCORE', 'pglm_pgam_psvr_gsvr_')

print 'Reading data...'

fitting_scores = map(lambda fn: nib.load(fn), prss_scores)
affine = fitting_scores[0].affine
niiFile = nib.Nifti1Image
fitting_scores = map(lambda data_file: data_file.get_data(), fitting_scores)

print 'Selecting best fits and their respective models...'

best_fit = fitting_scores[0]
best_fit_model = np.ones(best_fit.shape, dtype=np.int)

for model in xrange(1, len(fitting_scores)):
    better_fit = fitting_scores[model] > best_fit
    best_fit[better_fit] = fitting_scores[model][better_fit]
    best_fit_model[better_fit] = model + 1

print 'Saving results...'

nib.save(niiFile(best_fit, affine), filename_prefix + 'best_fit.nii')
nib.save(niiFile(best_fit_model, affine), filename_prefix + 'best_fit_model.nii')

print 'Done.'
