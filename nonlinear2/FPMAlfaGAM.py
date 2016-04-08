import nibabel as nib
import numpy as np

from nonlinear2.Utils.Subject import Subject
from nonlinear2.Processors.GAMProcessing import GAMProcessor as GAMP
from nonlinear2.Utils.DataLoader import getSubjects

niiFile = nib.Nifti1Image
affine = np.array(
    [[-1.50000000e+00, 0.00000000e+00, 0.00000000e+00, 9.00000000e+01],
     [1.99278252e-16, 1.50000000e+00, 2.17210575e-16, -1.26000000e+02],
     [-1.36305018e-16, -1.38272305e-16, 1.50000000e+00, -7.20000000e+01],
     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
)

print 'Obtaining data from Excel file...'
subjects = getSubjects(corrected_data=True)


print 'Initializing GAM Splines Processor...'
user_defined_parameters = [(9, [2, 2, 10, 1]),
                           (9, [2, 2, 10, 2]),
                           (9, [2, 2, 10, 3]),
                           (9, [2, 2, 10, 4]),
                           (9, [2, 2, 10, 5])
                           ]
filenames = ['gam_splines_d1_s10',
             'gam_splines_d2_s10',
             'gam_splines_d3_s10',
             'gam_splines_d4_s10',
             'gam_splines_d5_s10',
             ]

for udf, filename in zip(user_defined_parameters, filenames):
    gamp = GAMP(subjects, predictors=[Subject.ADCSFIndex], user_defined_parameters=udf)
    print 'Processing data...'
    results = gamp.process()







print 'Initializing GAM Polynomial Processor...'
user_defined_parameters = [
    (9, [1, 1, 3]),

]

filenames = [
    'gam_poly_d3',
]

for udp, filename in zip(user_defined_parameters, filenames):
    gamp = GAMP(subjects, predictors=[Subject.ADCSFIndex], user_defined_parameters=udp)
    results = gamp.process()

