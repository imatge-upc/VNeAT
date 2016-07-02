import time
from os.path import join

import nibabel as nib
from scipy.stats import norm

import src.Utils.DataLoader as DataLoader
from src.FitScores.FitEvaluation import ftest
from src.Processors.MixedProcessor import MixedProcessor
from src.Utils.Subject import Subject
from user_paths import RESULTS_DIR

RESULTS_DIR = join(RESULTS_DIR, 'MIXED')

if __name__ == "__main__":

    """ SELECTION """
    # Pre-defined user_defined_params for MixedProcessor
    prefixes = {
        'PolyGLM-PolyGLM': join('PGLM-PGLM', 'pglm_pglm_'),
        'PolyGLM-GaussianSVR': join('PGLM-GSVR', 'pglm_gsvr_'),
        'PolyGLM-PolySVR': join('PGLM-PSVR', 'pglm_psvr_'),
        'PolyGLM-PolyGAM': join('PGLM-PGAM', 'pglm_pgam_')
    }
    # SELECT HERE YOUR PREDEFINED USER-DEFINED-PARAMS
    prefix = prefixes['PolyGLM-PolySVR']

    """ PROCESSING """
    # Get affine matrix
    niiFile = nib.Nifti1Image
    affine = DataLoader.getMNIAffine()

    print 'Obtaining data from Excel file'
    subjects = DataLoader.getSubjects(corrected_data=False)

    print 'Loading precomputed parameters for MixedProcessor'
    c_path = join(RESULTS_DIR, prefix + 'cparams.nii')
    p_path = join(RESULTS_DIR, prefix + 'pparams.nii')
    cparameters = nib.load(c_path).get_data()
    pparameters = nib.load(p_path).get_data()

    with open(join(RESULTS_DIR, prefix + 'userdefparams.txt'), 'rb') as f:
        user_defined_parameters = eval(f.read())

    print 'Initializing MixedProcessor'
    processor = MixedProcessor(
        subjects,
        predictors=[Subject.ADCSFIndex],
        correctors=[Subject.Age, Subject.Sex],
        user_defined_parameters=user_defined_parameters
    )

    print 'Computing F-scores'
    time_start = time.clock()
    fitting_scores = MixedProcessor.evaluate_fit(
        evaluation_function=ftest,
        correction_processor=processor,
        correction_parameters=cparameters,
        prediction_processor=processor,
        prediction_parameters=pparameters,
        gm_threshold=0.1,
        filter_nans=True,
        default_value=0.0,
        # x1=40,
        # x2=50,
        # y1=82,
        # y2=83,
        # z1=39,
        # z2=40,
        mem_usage=256
    )
    time_end = time.clock()
    print "Fit evaluation done in ", time_end - time_start, " seconds"

    print 'Saving inverted p-values to file'
    nib.save(niiFile(fitting_scores, affine), join(RESULTS_DIR, prefix + 'fitscores.nii'))

    print 'Obtaining, filtering and saving Z-scores and labels to display them...'
    for fit_threshold in [0.99, 0.995, 0.999]:
        print '    Fitting-threshold set to', fit_threshold, '; Computing z-scores and labels...'
        clusterized_fitting_scores, labels = MixedProcessor.clusterize(
            fitting_scores=fitting_scores,
            default_value=0.0,
            fit_lower_threshold=fit_threshold,
            cluster_threshold=100,
            produce_labels=True
        )
        # Compute Z-scores
        lim_value = norm.ppf(fit_threshold)
        valid_voxels = clusterized_fitting_scores != 0.0
        clusterized_fitting_scores[valid_voxels] = norm.ppf(clusterized_fitting_scores[valid_voxels]) - lim_value + 0.2

        print '    Saving z-scores and labels to file...'
        nib.save(niiFile(clusterized_fitting_scores, affine),
                 join(RESULTS_DIR, prefix + 'zscores_' + str(fit_threshold) + '.nii'))
        nib.save(niiFile(labels, affine), join(RESULTS_DIR, prefix + 'labels_' + str(fit_threshold) + '.nii'))

    print 'Done.'
