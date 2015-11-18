print
print 'Welcome to ttest.py!'
print 'We are now going to analyze the data in the specified directory and perform t-tests over it.'
print 'There is a lot of data to process, so please be pacient.'
print 'Thank you for using our software! :)'
print

import database as db
from os.path import join

# Set paths
WORK_DIR =  join('/', 'Users', 'Asier', 'Documents', 'TFG')
DATA_DIR =  join('Alan T', 'Nonlinear_NBA_15')
EXCEL_FILE = join('Alan T', 'work_DB_CSF.R1.final.xls')
OUTPUT_DIR = join('python', 'ttests')
OUTPUT_FILENAME = 'ttest_results'

# Set region
x1, y1, z1 = [0]*3
x2, y2, z2 = [None]*3

in_data = db.get_data(x1 = x1, y1 = y1, z1 = 1, x2 = x2, y2 = y2, z2 = z2, fields = {'id': (lambda s: str(s).strip().split('_')[0]), 'diag':int})

out_data = 