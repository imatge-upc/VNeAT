import nibabel as nib
import numpy as np
from os.path import join
from scipy.stats import norm
from user_paths import RESULTS_DIR

# Append all files to be compared (must have same dimensions, maximum allowed is 3)
fitscores = [
    join(RESULTS_DIR, 'MIXED', 'PGLM-PGLM', 'pglm_pglm_zscores_0.999.nii'),
    join(RESULTS_DIR, 'MIXED', 'PGLM-PSVR', 'pglm_psvr_zscores_0.999.nii'),
    join(RESULTS_DIR, 'MIXED', 'PGLM-GSVR', 'pglm_gsvr_zscores_0.999.nii'),
    # join(RESULTS_DIR, 'MIXED', 'PGLM-PGAM', 'pglm_pgam_zscores_0.999.nii')
]

# Path and prefix for the output files
filename_prefix = join(RESULTS_DIR, 'BESTFIT', 'RGB', 'pglm_vs_psvr_vs_gsvr_')

print 'Reading data...'
fitscores = map(lambda fn: nib.load(fn), fitscores)
affine = fitscores[0].affine
niiFile = nib.Nifti1Image
fitscores = map(lambda data_file: data_file.get_data(), fitscores)

print 'Building RGB maps...'

fits = np.zeros(fitscores[0].shape + (3,), dtype=np.float64)

for model in xrange(len(fitscores)):
    fits[:, :, :, model] = fitscores[model]

print 'Saving results...'
nib.save(niiFile(fits, affine), filename_prefix + 'fit_comparison.nii')

#    print 'Obtaining, filtering and saving z-scores and labels to display them...'
#    for fit_threshold in [0.99, 0.995, 0.999]:
#        print '    Fitting-threshold set to', fit_threshold, '; Computing z-scores and labels...'
#    
#        z_scores = best_fit.copy()
#        labels = best_fit_model.copy()
#    
#        invalid_zones = z_scores < fit_threshold
#        valid_zones = ~invalid_zones
#    
#        z_scores[invalid_zones] = 0.0
#        labels[invalid_zones] = 0
#    
#        z_scores[valid_zones] = norm.ppf(z_scores[valid_zones]) - norm.ppf(fit_threshold) + 0.2
#    
#        print '    Saving z-scores and labels to file...'
#        nib.save(niiFile(z_scores, affine), filename_prefix + 'best_fit_fitscores_' + str(fit_threshold) + '.nii')
#        nib.save(niiFile(labels, affine), filename_prefix + 'best_fit_labels_' + str(fit_threshold) + '.nii')

print 'Done.'
