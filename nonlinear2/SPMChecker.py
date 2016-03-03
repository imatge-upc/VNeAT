# IMPORTS
from os.path import join
import nibabel as nib
from numpy import array
from scipy.stats import f,norm

# CONSTANTS
RESULTS_DIR = join("C:\\", "Users", "santi", "Documents", "Santi", "Universitat", "TFG", "Results", "SPMvsGLM")
SPM_NIFTI_MAP = join(RESULTS_DIR, "spm_F_map.nii")
PGLM_NIFTI_MAP = join(RESULTS_DIR, "fpmalfa_zscores.nii")
SPM_Z_SCORES = join(RESULTS_DIR, "spm_Z_map.nii")
DIFF = join(RESULTS_DIR, "spmvspglm_diff.nii")
ABS_DIFF = join(RESULTS_DIR, "spmvspglm_abs.nii")
SQUARED_ERROR = join(RESULTS_DIR, "spmvspglm_squared.nii")

print("Loading data from NIFTI files...")
spm = nib.load(SPM_NIFTI_MAP)
spm_data = spm.get_data()
pglm = nib.load(PGLM_NIFTI_MAP)
pglm_data = pglm.get_data()

print("Transforming F-scores into Z-scores for the SPM...")
dfn = 3
dfd = 118
p_threshold = 0.01
F_rv = f(dfn=dfn, dfd=dfd)
G_rv = norm(loc=1,scale=1)
lim_value = G_rv.ppf(1-p_threshold)
Z_scores = G_rv.ppf(F_rv.cdf(spm_data)) - lim_value + 0.2

print("Saving Z-scores for SPM...")
niizscores = nib.Nifti1Image(Z_scores, spm.affine)
nib.save(niizscores, SPM_Z_SCORES)

print("Calculating SPM vs PGLM (Diff)...")
diff = pglm_data - Z_scores

print("Calculating SPM vs PGLM (Absolute Diff)...")
abs_diff = abs(Z_scores - pglm_data)

print("Calculating SPM vs PGLM (Squared Error)...")
se = abs(Z_scores - pglm_data) ** 2

print ("Saving differential maps...")
niidiff = nib.Nifti1Image(diff, spm.affine)
nib.save(niidiff, DIFF)
niiabs = nib.Nifti1Image(abs_diff, spm.affine)
nib.save(niiabs, ABS_DIFF)
niise = nib.Nifti1Image(se, spm.affine)
nib.save(niise, SQUARED_ERROR)
