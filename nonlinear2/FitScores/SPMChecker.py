""" IMPORTS """
from os.path import join
import nibabel as nib
from scipy.stats import f, norm
from nonlinear2.user_paths import RESULTS_DIR

p_inv_thresholds = [0.99, 0.995, 0.999]

for p_inv_threshold in p_inv_thresholds:
    # FILES TO BE OPENED
    SPM = join(RESULTS_DIR, "spm_fscores_" + str(p_inv_threshold) + ".nii")
    PGLM = join(RESULTS_DIR, "fpmalfa_zscores_" + str(p_inv_threshold) + ".nii")
    GAM = join(RESULTS_DIR, "fpmalfa_gam_zscores_" + str(p_inv_threshold) + ".nii")
    # FILES TO BE SAVED
    SPM_Z_SCORES = join(RESULTS_DIR, "spm_zscores_" + str(p_inv_threshold) +".nii")
    SPM_FIT_SCORES = join(RESULTS_DIR, "spm_fitscores_map.nii")
    DIFF = join(RESULTS_DIR, "spmvspglm_diff_" + str(p_inv_threshold) + ".nii")
    DIFF_GAM = join(RESULTS_DIR, "spmvsgam_diff_" + str(p_inv_threshold) + ".nii")
    ABS_DIFF = join(RESULTS_DIR, "spmvspglm_abs_" + str(p_inv_threshold) + ".nii")
    ABS_DIFF_GAM = join(RESULTS_DIR, "spmvsgam_abs_" + str(p_inv_threshold) + ".nii")
    SQUARED_ERROR = join(RESULTS_DIR, "spmvspglm_squared_" + str(p_inv_threshold) + ".nii")
    SQUARED_ERROR_GAM = join(RESULTS_DIR, "spmvsgam_squared_" + str(p_inv_threshold) + ".nii")

    print("Loading data from NIFTI files...")
    spm = nib.load(SPM)
    spm_data = spm.get_data()
    pglm = nib.load(PGLM)
    pglm_data = pglm.get_data()
    gam = nib.load(GAM)
    gam_data = gam.get_data()

    print("Transforming F-scores into fit scores and Z-values for the SPM...")
    # fit scores (inverted p-value)
    # number of regressors
    dfn = 3
    # number of subjects - number of regressors
    dfd = 122
    F_rv = f(dfn=dfn, dfd=dfd)
    fit_scores = F_rv.cdf(spm_data)
    # Z-scores
    lim_value = norm.ppf(p_inv_threshold)
    Z_scores = norm.ppf(fit_scores) - lim_value + 0.2


    # print("Saving Z-scores for SPM...")
    nii_z_scores = nib.Nifti1Image(Z_scores, spm.affine)
    nib.save(nii_z_scores, SPM_Z_SCORES)

    print("Saving fitscores for SPM...")
    nii_fit_scores = nib.Nifti1Image(fit_scores, spm.affine)
    nib.save(nii_fit_scores, SPM_FIT_SCORES)

    print("Calculating SPM vs PGLM (Diff)...")
    diff = pglm_data - Z_scores

    print("Calculating SPM vs GAM (Diff)...")
    diff_gam = gam_data - Z_scores

    print("Calculating SPM vs PGLM (Absolute Diff)...")
    abs_diff = abs(pglm_data - Z_scores)

    print("Calculating SPM vs GAM (Absolute Diff)...")
    abs_diff_gam = abs(gam_data - Z_scores)

    print("Calculating SPM vs PGLM (Squared Error)...")
    se = (pglm_data - Z_scores) ** 2

    print("Calculating SPM vs GAM (Squared Error)...")
    se_gam = (gam_data - Z_scores) ** 2

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

