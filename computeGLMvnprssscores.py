from os import listdir
from os.path import join, isfile, basename

import nibabel as nib
import numpy as np

from FitScores.FitEvaluation_v2 import vnprss
from Processors.GLMProcessing import GLMProcessor as GLMP
from Utils.ExcelIO import ExcelSheet as Excel
from Utils.Subject import Subject
from user_paths import DATA_DIR, EXCEL_FILE

filename_prefix = join('results', 'GLM', 'glm_all_')
gamma = 1e3

niiFile = nib.Nifti1Image
affine = np.array(
    [[-1.50000000e+00, 0.00000000e+00, 0.00000000e+00, 9.00000000e+01],
     [1.99278252e-16, 1.50000000e+00, 2.17210575e-16, -1.26000000e+02],
     [-1.36305018e-16, -1.38272305e-16, 1.50000000e+00, -7.20000000e+01],
     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
)

print 'Obtaining data from Excel file'
# DATA_DIR = join('/', 'Users', 'Asier', 'Documents', 'TFG', 'Alan T', 'Nonlinear_NBA_15')
# EXCEL_FILE = join('/', 'Users', 'Asier', 'Documents', 'TFG', 'Alan T', 'work_DB_CSF.R1.final.xls')

filenames = filter(isfile, map(lambda elem: join(DATA_DIR, elem), listdir(DATA_DIR)))
filenames_by_id = {basename(fn).split('_')[0][8:]: fn for fn in filenames}

exc = Excel(EXCEL_FILE)

subjects = []
for r in exc.get_rows(fieldstype={
    'id': (lambda s: str(s).strip().split('_')[0]),
    'diag': (lambda s: int(s) - 1),
    'age': int,
    'sex': (lambda s: 2 * int(s) - 1),
    'apoe4_bin': (lambda s: 2 * int(s) - 1),
    'escolaridad': int,
    'ad_csf_index_ttau': float
}):
    subjects.append(
        Subject(
            r['id'],
            filenames_by_id[r['id']],
            r.get('diag', None),
            r.get('age', None),
            r.get('sex', None),
            r.get('apoe4_bin', None),
            r.get('escolaridad', None),
            r.get('ad_csf_index_ttau', None)
        )
    )

print 'Loading precomputed parameters for GLM'
glm_correction_parameters = nib.load(filename_prefix + 'cparams.nii').get_data()
glm_prediction_parameters = nib.load(filename_prefix + 'pparams.nii').get_data()

with open(filename_prefix + 'userdefparams.txt', 'rb') as f:
    user_defined_parameters = eval(f.read())

print 'Initializing PolyGLM Processor'
glmp = GLMP(subjects, predictors=[Subject.ADCSFIndex], user_defined_parameters=user_defined_parameters,
            correctors=[Subject.Age, Subject.Sex])

print 'Computing VNPRSS-scores'
fitting_scores = glmp.evaluate_fit(
    evaluation_function=vnprss,
    correction_parameters=glm_correction_parameters,
    prediction_parameters=glm_prediction_parameters,
    # x1 = 0, x2 = None, y1 = 0, y2 = None, z1 = 0, z2 = None,
    # origx = 0, origy = 0, origz = 0,
    gm_threshold=0.1,
    filter_nans=True,
    default_value=np.inf,
    # mem_usage = None,
    # *args, **kwargs
    gamma=gamma
)

print 'Saving VNPRSS-scores to file'
nib.save(niiFile(fitting_scores, affine), filename_prefix + 'vnprss_' + str(gamma) + '.nii')
nib.save(niiFile(-fitting_scores, affine), filename_prefix + 'inv_vnprss_' + str(gamma) + '.nii')

print 'Done.'

# No, not done. We still have to filter these results and make them suitable for visualization.
