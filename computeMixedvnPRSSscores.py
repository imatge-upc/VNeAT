import time
from os.path import join

import nibabel as nib
import numpy as np

import Utils.DataLoader as DataLoader
from FitScores.FitEvaluation_v2 import vnprss
from Processors.MixedProcessor import MixedProcessor
from Utils.Subject import Subject
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
    prefix = prefixes['PolyGLM-PolyGAM']
    gamma = 10

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

    print 'Computing variance normalized PRSS'
    time_start = time.clock()
    fitting_scores = processor.evaluate_fit(
        evaluation_function=vnprss,
        gamma=gamma,
        correction_parameters=cparameters,
        prediction_parameters=pparameters,
        gm_threshold=0.1,
        filter_nans=True,
        default_value=np.inf,
        mem_usage=256
    )
    time_end = time.clock()
    print "Fit evaluation done in ", time_end - time_start, " seconds"

    print 'Saving VNPRSS-scores to file'
    nib.save(niiFile(fitting_scores, affine), join(RESULTS_DIR, prefix + 'vnprss_' + str(gamma) + '.nii'))
    nib.save(niiFile(-fitting_scores, affine), join(RESULTS_DIR, prefix + 'inv_vnprss_' + str(gamma) + '.nii'))

    print 'Done.'

