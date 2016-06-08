print
print 'Welcome to ttest.py!'
print 'We are now going to analyze the data in the specified directory and perform t-tests over it.'
print 'There is a lot of data to process, so please be pacient.'
print 'Thank you for using our software! :)'
print

from os.path import join

from numpy import zeros
from scipy.stats import ttest_ind

import database as db
from subject import Subject

# Set paths
WORK_DIR = join('/', 'Users', 'Asier', 'Documents', 'TFG')
DATA_DIR = join('Alan T', 'Nonlinear_NBA_15')
EXCEL_FILE = join('Alan T', 'work_DB_CSF.R1.final.xls')
OUTPUT_DIR = join('python', 'ttest')
OUTPUT_FILENAME = 'ttest_results'

# Set region
x1, y1, z1 = [0] * 3
x2, y2, z2 = [None] * 3

in_data = db.get_data(x1=x1, y1=y1, z1=z1, x2=x2, y2=y2, z2=z2,
                      fields={'id': (lambda s: str(s).strip().split('_')[0]),
                              'diag': (lambda s: int(s) - 1)})

lsd = len(Subject.diagnostics)
out_data = [[zeros(in_data.dims[1:4]) for _ in range(i + 1, lsd)] for i in range(lsd - 1)]

diag = [[] for _ in range(lsd)]
for i in range(len(in_data.subjects)):
    diag[in_data.subjects[i].diag].append(i)

for chunk in in_data.chunks():
    x, y, z = chunk.coords
    dx, dy, dz = chunk.data.shape[1:4]
    data = [[chunk.data[i] for i in l] for l in diag]
    for i in range(lsd - 1):
        for j in range(lsd - i - 1):
            tt_res = ttest_ind(data[i], data[i + j + 1])
            out_data[i][j][x:(x + dx), y:(y + dy), z:(z + dz)] = tt_res.statistic

del data
del x, y, z, dx, dy, dz
del i, j
del tt_res

for i in range(len(out_data)):
    for j in range(len(out_data[i])):
        fn = OUTPUT_FILENAME + '_' + Subject.diagnostics[i] + '_' + Subject.diagnostics[i + j + 1] + '.nii'
        abs_fn = join(WORK_DIR, OUTPUT_DIR, fn)
        db.save_output_data(out_data[i][j], abs_fn)
