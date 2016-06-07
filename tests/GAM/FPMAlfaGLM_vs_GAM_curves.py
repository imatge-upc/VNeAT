import sys

sys.path.insert(1, 'C:\\Users\\upcnet\\Repositoris\\neuroimatge\\nonlinear2')
sys.path.insert(1, '/Users/acasamitjana/Repositories/neuroimatge/nonlinear2')
sys.stdout.flush()
from ExcelIO import ExcelSheet as Excel
from GLMProcessing import PolyGLMProcessor as PGLMP
from GAMProcessing import GAMProcessor as GAMP
from Subject import Subject
from os.path import join, isfile, basename
from os import listdir

print 'Obtaining data from Excel file...'
from user_paths import EXCEL_FILE, CORRECTED_DIR

filenames = filter(isfile, map(lambda elem: join(CORRECTED_DIR, elem), listdir(CORRECTED_DIR)))
filenames_by_id = {basename(fn).split('_')[1][:-4]: fn for fn in filenames}

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

x1 = 103  # 85  #
x2 = x1 + 1
y1 = 45  # 101  #
y2 = y1 + 1
z1 = 81  # 45  #
z2 = z1 + 1

print 'Initializing PolyGLM Processor...'
udp = (0, 9, 3)
pglmp = PGLMP(subjects, predictors=[Subject.ADCSFIndex], user_defined_parameters=udp)

print 'Processing data...'
results_glm = pglmp.process(x1=x1, x2=x2, y1=y1, y2=y2, z1=z1, z2=z2)

print 'Obtaining corrected values...'

print 'Initializing GAM Processor...'
udp = (9, [1, 1, 3])
gamp = GAMP(subjects, predictors=[Subject.ADCSFIndex], user_defined_parameters=udp)

print 'Processing data...'

results_gam = gamp.process(x1=x1, x2=x2, y1=y1, y2=y2, z1=z1, z2=z2)

print 'Saving results to files...'

from matplotlib import pyplot as plt
import numpy as np

correction_parameters_gam = np.zeros((results_gam.correction_parameters.shape[0], 200, 200, 200))
correction_parameters_gam[:, x1:x2, y1:y2, z1:z2] = results_gam.correction_parameters
corrected_data_gam = gamp.corrected_values(correction_parameters_gam, x1=x1, x2=x2, y1=y1, y2=y2, z1=z1, z2=z2)

correction_parameters_glm = np.zeros((results_glm.correction_parameters.shape[0], 200, 200, 200))
correction_parameters_glm[:, x1:x2, y1:y2, z1:z2] = results_glm.correction_parameters
corrected_data_glm = pglmp.corrected_values(correction_parameters_glm, x1=x1, x2=x2, y1=y1, y2=y2, z1=z1, z2=z2)

x = np.array(np.sort([sbj._attributes[sbj.ADCSFIndex.index] for sbj in subjects]))
plt.figure()
plt.plot(x, np.squeeze(corrected_data_glm), 'g.')
plt.plot(x, np.squeeze(corrected_data_gam), 'k.')
plt.plot(x, pglmp.__curve__(-1, x[:, np.newaxis], np.squeeze(results_glm.prediction_parameters)), 'b')
plt.plot(x, gamp.__curve__(-1, x[:, np.newaxis], np.squeeze(results_gam.prediction_parameters)), 'r')
plt.show()

a = 1
