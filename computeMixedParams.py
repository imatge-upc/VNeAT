import time
from os.path import join

import nibabel as nib

import Utils.DataLoader as DataLoader
from Processors.MixedProcessor import MixedProcessor
from Utils.Subject import Subject
from user_paths import RESULTS_DIR

RESULTS_DIR = join(RESULTS_DIR, 'MIXED')

if __name__ == "__main__":
    """ SELECTION """

    # Pre-defined user_defined_params for MixedProcessor
    user_def_params = {
        'PolyGLM-PolyGLM': [
            (1, [1, 0, 2, 1], 1, [2, 0, 3]),  # correctors: intercept, age^2, sex; predictors: adcsf^3
            join('PGLM-PGLM', 'pglm_pglm_')
        ],
        'PolyGLM-GaussianSVR': [
            (1, [1, 0, 2, 1], 4, [2, 3, 1.6, 0.06, 0.3]),  # correctors: intercept, age^2, sex; predictors: adcsf
            join('PGLM-GSVR', 'pglm_gsvr_')
        ],
        'PolyGLM-PolySVR': [
            (1, [1, 0, 2, 1], 3, [2, 3, 1.65, 0.078, 3]),  # correctors: intercept, age^2, sex; predictors: adcsf^3
            join('PGLM-PSVR', 'pglm_psvr_')
        ],
        'PolyGLM-PolyGAM': [
            (1, [1, 0, 2, 1], 2, [9, [1, 1, 3]]),  # correctors: intercept, age^2, sex; predictors: adcsf^3
            join('PGLM-PGAM', 'pglm_pgam_')
        ],
        'None': [
            (),
            'mixedprocessor_'
        ]
    }
    # SELECT HERE YOUR PREDEFINED USER-DEFINED-PARAMS
    udp = user_def_params['PolyGLM-PolySVR']

    """ PROCESSING """

    # Get subjects
    print "Getting subjects from Excel file..."
    subjects = DataLoader.getSubjects(corrected_data=False)

    # Create MixedProcessor
    print "Creating MixedProcessor..."
    processor = MixedProcessor(
        subjects,
        predictors=[Subject.ADCSFIndex],
        correctors=[Subject.Age, Subject.Sex],
        user_defined_parameters=udp[0]
    )

    print 'Processing...'
    time_start = time.clock()
    results = processor.process(mem_usage=128, n_jobs=7, cache_size=2048)
    time_end = time.clock()
    print 'Processing done in ', time_end - time_start, " seconds"

    # User defined parameters
    print 'Storing user defined parameters...'
    user_defined_parameters = processor.user_defined_parameters
    with open(join(RESULTS_DIR, udp[1] + 'userdefparams.txt'), 'wb') as f:
        f.write(str(user_defined_parameters) + '\n')

    # Correction and prediction params
    print 'Storing correction and prediction parameters...'
    affine = DataLoader.getMNIAffine()
    niiFile = nib.Nifti1Image
    nib.save(niiFile(results.correction_parameters, affine), join(RESULTS_DIR, udp[1] + 'cparams.nii'))
    nib.save(niiFile(results.prediction_parameters, affine), join(RESULTS_DIR, udp[1] + 'pparams.nii'))
