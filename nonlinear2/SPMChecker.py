""" IMPORTS """
from os.path import join
import nibabel as nib
from numpy import array
from scipy.stats import f, norm

""" CONSTANTS """
RESULTS_DIR = join(
    "C:\\", "Users", "santi", "Documents", "Santi",
    "Universitat", "TFG", "Results", "SPMvsGLM"
)

# FILES TO BE OPENED
PGLM = join(RESULTS_DIR, "fpmalfa_zscores.nii")
PGLM_FIT = join(RESULTS_DIR, "fpmalfa_fitscores.nii")
GAM = join(RESULTS_DIR, "fpmalfa_gam_zscores.nii")
GAM_FIT = join(RESULTS_DIR, "fpmalfa_gam_fitscores.nii")
# FILES TO BE SAVED
DIFF = join(RESULTS_DIR, "spmvspglm_diff.nii")
DIFF_GAM = join(RESULTS_DIR, "spmvsgam_diff.nii")
ABS_DIFF = join(RESULTS_DIR, "spmvspglm_abs.nii")
ABS_DIFF_GAM = join(RESULTS_DIR, "spmvsgam_abs.nii")
SQUARED_ERROR = join(RESULTS_DIR, "spmvspglm_squared.nii")
SQUARED_ERROR_GAM = join(RESULTS_DIR, "spmvsgam_squared.nii")

print("Loading data from NIFTI files...")
spm = nib.load(SPM)
spm_data = spm.get_data()
pglm = nib.load(PGLM)
pglm_data = pglm.get_data()
pglm_fit = nib.load(PGLM_FIT)
pglm_fit_data = pglm_fit.get_data()
gam = nib.load(GAM)
gam_data = gam.get_data()
gam_fit = nib.load(GAM_FIT)
gam_fit_data = gam_fit.get_data()

print("Transforming F-scores into fit scores and Z-values for the SPM...")
# fit scores (inverted p-value)
# number of regressors
dfn = 3
# number of subjects - number of regressors
dfd = 122
F_rv = f(dfn=dfn, dfd=dfd)
fit_scores = F_rv.cdf(spm_data)
# Z-scores
p_inv_threshold = 0.99
lim_value = norm.ppf(p_inv_threshold)
Z_scores = norm.ppf(fit_scores) - lim_value + 0.2


# print("Saving Z-scores for SPM...")
# nii_z_scores = nib.Nifti1Image(Z_scores, spm.affine)
# nib.save(nii_z_scores, SPM_Z_SCORES)
#
# print("Saving fitscores for SPM...")
# nii_fit_scores = nib.Nifti1Image(fit_scores, spm.affine)
# nib.save(nii_fit_scores, SPM_FIT_SCORES)
#
# print("Calculating SPM vs PGLM (Diff)...")
# diff = pglm_data - Z_scores
#
# print("Calculating SPM vs GAM (Diff)...")
# diff_gam = gam_data - Z_scores
#
# print("Calculating SPM vs PGLM (Absolute Diff)...")
# abs_diff = abs(pglm_data - Z_scores)
#
# print("Calculating SPM vs GAM (Absolute Diff)...")
# abs_diff_gam = abs(gam_data - Z_scores)
#
# print("Calculating SPM vs PGLM (Squared Error)...")
# se = (pglm_data - Z_scores) ** 2
#
# print("Calculating SPM vs GAM (Squared Error)...")
# se_gam = (gam_data - Z_scores) ** 2

print ("Saving differential maps (SPM vs PGLM)...")
niidiff = nib.Nifti1Image(diff, spm.affine)
nib.save(niidiff, DIFF)
niiabs = nib.Nifti1Image(abs_diff, spm.affine)
nib.save(niiabs, ABS_DIFF)
niise = nib.Nifti1Image(se, spm.affine)
nib.save(niise, SQUARED_ERROR)

print ("Saving differential maps (SPM vs GAM)...")
niidiff = nib.Nifti1Image(diff_gam, spm.affine)
nib.save(niidiff, DIFF_GAM)
niiabs = nib.Nifti1Image(abs_diff_gam, spm.affine)
nib.save(niiabs, ABS_DIFF_GAM)
niise = nib.Nifti1Image(se_gam, spm.affine)
nib.save(niise, SQUARED_ERROR_GAM)
