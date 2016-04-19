from os.path import join

import nibabel as nib

import nonlinear2.Utils.DataLoader as DataLoader
from nonlinear2.Utils.Subject import Subject
from nonlinear2.Processors.GLMProcessing import PolyGLMProcessor as PGLMP
from user_paths import CORRECTED_DATA_DIR

print 'Obtaining data from Excel file...'
subjects = DataLoader.getSubjects(corrected_data=False)

print 'Initializing PolyGLM Processor...'
pglmp = PGLMP(subjects, predictors=[Subject.ADCSFIndex], correctors = [Subject.Age, Subject.Sex])

print 'Processing data...'
results = pglmp.process()

print 'Obtaining corrected values...'
corrected_values = pglmp.corrected_values(results.correction_parameters)

print 'Saving results to files...'
affine = DataLoader.getMNIAffine()

niiFile = nib.Nifti1Image

# Save corrected values per subject
for i in range(len(subjects)):
    # Get id of the subject to create the file name
    filename = 'corrected_' + subjects[i].id + '.nii'
    nib.save(niiFile(corrected_values[i, :, :, :], affine), join(CORRECTED_DATA_DIR, filename))

# Save user defined params in order to reproduce the correction
with open(join(CORRECTED_DATA_DIR, 'user_def_params.txt'), 'wb') as f:
    f.write(str(pglmp.user_defined_parameters) + '\n')

print 'Done.'

