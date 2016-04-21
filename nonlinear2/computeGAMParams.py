from nonlinear2.Utils.ExcelIO import ExcelSheet as Excel
from nonlinear2.Processors.GAMProcessing import GAMProcessor as GAMP
from nonlinear2.Utils.Subject import Subject
from os.path import join, isfile, basename
from os import listdir

import nibabel as nib
import numpy as np

from user_paths import DATA_DIR, EXCEL_FILE, CORRECTED_DATA_DIR, RESULTS_DIR


niiFile = nib.Nifti1Image
affine = np.array(
    [[-1.50000000e+00, 0.00000000e+00, 0.00000000e+00, 9.00000000e+01],
     [1.99278252e-16, 1.50000000e+00, 2.17210575e-16, -1.26000000e+02],
     [-1.36305018e-16, -1.38272305e-16, 1.50000000e+00, -7.20000000e+01],
     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
)


print 'Obtaining data from Excel file...'

filenames = filter(isfile, map(lambda elem: join(CORRECTED_DATA_DIR, elem), listdir(CORRECTED_DATA_DIR)))
filenames_by_id = {basename(fn).split('_')[1][:-4] : fn for fn in filenames}

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

user_defined_parameters = [
    (9, [1, 1, 3]),
]

filename_prefix = [
    'gam_poly_d3_'
]

for udp, fn in zip(user_defined_parameters, filename_prefix):

    print 'Initializing GAM Polynomial Processor...'
    gamp = GAMP(subjects, predictors=[Subject.ADCSFIndex], user_defined_parameters=udp)

    print 'Processing data...'
    results = gamp.process()

    print 'Saving results to files...'

    nib.save(niiFile(results.correction_parameters, affine), join(RESULTS_DIR,'PGAM',fn + 'cparams.nii'))
    nib.save(niiFile(results.prediction_parameters, affine), join(RESULTS_DIR,'PGAM',fn+ 'pparams.nii'))

    with open(join(RESULTS_DIR,'PGAM',fn + 'userdefparams.txt'), 'wb') as f:
        f.write(str(gamp.user_defined_parameters) + '\n')

    print 'Done.'
