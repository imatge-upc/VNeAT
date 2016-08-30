from os import listdir
from os.path import join, isfile, basename

import nibabel as nib
from GAMProcessing import GAMProcessor
from numpy import array as nparray

from Subject import Subject
from Utils.ExcelIO import ExcelSheet as Excel

print 'Obtaining data from Excel file...'
DATA_DIR = join('C:\\', 'Users', 'upcnet', 'FPM', 'data_backup', 'Non-linear', 'Nonlinear_NBA_15')
EXCEL_FILE = join('C:\\', 'Users', 'upcnet', 'FPM', 'data_backup', 'Non-linear', 'work_DB_CSF.R1.final.xls')

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
print 'Initializing GAM Processor...'
gamp = GAMProcessor(subjects, regressors=[Subject.ADCSFIndex], correctors=[Subject.Age, Subject.Sex])

print 'Processing data...'
results = gamp.process(x1=73, x2=74, y1=84, y2=85, z1=41, z2=42)

print 'Formatting obtained data to display it...'
z_scores, labels = gamp.fit_score(results.fitting_scores, produce_labels=True)

print 'Saving results to files...'

affine = nparray(
    [[-1.50000000e+00, 0.00000000e+00, 0.00000000e+00, 9.00000000e+01],
     [1.99278252e-16, 1.50000000e+00, 2.17210575e-16, -1.26000000e+02],
     [-1.36305018e-16, -1.38272305e-16, 1.50000000e+00, -7.20000000e+01],
     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
)

niiFile = nib.Nifti1Image

nib.save(niiFile(results.correction_parameters, affine), 'fpmalfa_gam_cparams.nii')
nib.save(niiFile(results.regression_parameters, affine), 'fpmalfa_gam_rparams.nii')
nib.save(niiFile(results.fitting_scores, affine), 'fpmalfa_gam_fitscores.nii')
nib.save(niiFile(z_scores, affine), 'fpmalfa_gam_zscores.nii')
nib.save(niiFile(labels, affine), 'fpmalfa_gam_labels.nii')

with open('fpmalfa_gam_userdefparams.txt', 'wb') as f:
    f.write(str(gamp.user_defined_parameters) + '\n')

print 'Done.'
