from ExcelIO import ExcelSheet as Excel
from GLMProcessing import PolyGLMProcessor as PGLMP
from Subject import Subject
from os.path import join, isfile, basename
from os import listdir

import nibabel as nib
from numpy import array as nparray
from user_paths import DATA_DIR, EXCEL_FILE, CORRECTED_DIR

print 'Obtaining data from Excel file...'
filenames = filter(isfile, map(lambda elem: join(DATA_DIR, elem), listdir(DATA_DIR)))
filenames_by_id = {basename(fn).split('_')[0][8:] : fn for fn in filenames}

exc = Excel(EXCEL_FILE)

subjects = []
for r in exc.get_rows( fieldstype = {
                'id':(lambda s: str(s).strip().split('_')[0]),
                'diag':(lambda s: int(s) - 1),
                'age':int,
                'sex':(lambda s: 2*int(s) - 1),
                'apoe4_bin':(lambda s: 2*int(s) - 1),
                'escolaridad':int,
                'ad_csf_index_ttau':float
             } ):
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

print 'Initializing PolyGLM Processor...'
pglmp = PGLMP(subjects, regressors=[Subject.ADCSFIndex], correctors = [Subject.Age, Subject.Sex])

print 'Processing data...'
results = pglmp.process()

print 'Obtaining corrected values...'
corrected_values = pglmp.corrected_values(results.correction_parameters)

print 'Saving results to files...'

affine = nparray(
		[[ -1.50000000e+00,   0.00000000e+00,   0.00000000e+00,   9.00000000e+01],
		 [  1.99278252e-16,   1.50000000e+00,   2.17210575e-16,  -1.26000000e+02],
		 [ -1.36305018e-16,  -1.38272305e-16,   1.50000000e+00,  -7.20000000e+01],
		 [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]]
)

niiFile = nib.Nifti1Image

# Save corrected values per subject
for i in range(len(subjects)):
    # Get id of the subject to create the file name
    filename = 'corrected_' + subjects[i].id + '.nii'
    nib.save(niiFile(corrected_values[i, :, :, :], affine), join(CORRECTED_DIR, filename))

# Save user defined params in order to reproduce the correction
with open(join(CORRECTED_DIR, 'user_def_params.txt'), 'wb') as f:
    f.write(str(pglmp.user_defined_parameters) + '\n')

print 'Done.'

