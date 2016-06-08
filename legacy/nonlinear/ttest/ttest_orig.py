# To be executed as follows:
# python -W ignore -u ttest.py


print
print 'Welcome to ttest.py!'
print 'We are now going to analyze the data in the specified directory and perform t-tests over it.'
print 'There is a lot of data to process, so please be pacient.'
print 'Thank you for using our software! :)'
print

# Set paths
WORK_DIR = '/Users/Asier/Documents/TFG/Alan T'
DATA_DIR = 'Nonlinear_NBA_15'
EXCEL_FILE = 'work_DB_CSF.R1.final.xls'
OUTPUT_DIR = 'ttests'
OUTPUT_FILENAME = 'ttest_results'

# Set region
regx_init = 0
regx_end = 1000
regy_init = 0
regy_end = 1000
regz_init = 0
regz_end = 1000

MEMORY_USE = 100  # approx. memory use in MB (depends on garbage collector, but the order of magnitude should be around this value)

# Get all files in data directory and index them by their ID

print 'Indexing files in data directory by ID...',
from os import listdir
from os.path import join, isfile

filename_by_id = {f.split('_')[0][8:]: f for f in
                  filter(lambda elem: isfile(join(WORK_DIR, DATA_DIR, elem)), listdir(join(WORK_DIR, DATA_DIR)))}
print 'Done.'

# Initialize data retrieval variables
classes = ['cont', 'prec', 'mci', 'ad']  # [controls, preclinicals, mcis, ads]
diag = [[] for _ in classes]

# Retrieve interesting data from excel file
print
print 'Retrieving data from Excel file...',
from xlrd import open_workbook as open_wb
from xlrd.sheet import ctype_text as type2text

with open_wb(join(WORK_DIR, EXCEL_FILE)) as wb:
    # Open first sheet
    ws = wb.sheet_by_index(0)
    # This could also be done as follows
    # sheet_names = wb.sheet_names()
    # ws = wb.sheet_by_name(sheet_names[0])
    #
    # Or by using the name of the sheet, i.e., DB_clinic (even if it's not the first one):
    # ws = wb.sheet_by_name('DB_clinic')

    # Get the column index for each header in the sheet (headers must be in the first row and text typed)
    h = ws.row(
        0)  # Extract first row to make it more efficient (since we're gonna make multiple sequential reads on it)
    header2col = {h[j].value.strip().lower(): j for j in range(ws.ncols) if type2text[h[j].ctype] == 'text'}
    del h

    # Separate the elements by their category
    ids_col = ws.col(header2col['id'])
    diags_col = ws.col(header2col['diag'])
    for i in range(1, len(ids_col)):
        if type2text[ids_col[i].ctype] == 'text' and type2text[diags_col[i].ctype] in ('text', 'number'):
            diag[int(diags_col[i].value) - 1].append(ids_col[i].value.strip().split('_')[0])
del ws, wb
print 'Done.'
print

# Retrieve scanner data in chunks from files for ADs and Controls and
# perform the tests for each chunk, storing the results in a NIfTI file
print 'Processing data and performing t-tests over all pairs of classes...'
print 'This may take several seconds (or even minutes), please be pacient.'
import nibabel as nib
from sys import exit

print '    Initializing data structures and variables...',
# Get output dimensions
try:
    f = nib.load(join(WORK_DIR, DATA_DIR, filename_by_id[diag[0][0]]))
    out_dims = f.shape
    del f
except IOError as e:
    print
    print 'FATAL ERROR: Could not open output file template. Aborting process...'
    exit()

# Initialize output (all zeros)
from numpy import zeros

output = []
for i in range(len(classes) - 1):
    output.append([])
    for _ in range(i + 1, len(classes)):
        output[-1].append(zeros(out_dims))

# Perform analysis chunk by chunk (chunks of 'dx' x 'dy' x 'dz') for each pair of classes
from scipy.stats import ttest_ind

nelems = MEMORY_USE * (2 ** 17) / sum(
    map(len, diag))  # (number of MBytes x 2**20 bytes/MB x 8 bits/byte)/((64 bits/elem)* #samples)

# nelems = dx*dy*dz = dx * (dy/dx * dx) * (dz/dx * dx) = (dy/dx * dz/dx) * (dx**3)
# dx = (nelems / (dy/dx * dz/dx))**(1./3)
regx_end, regy_end, regz_end = map(min, zip(out_dims, [regx_end, regy_end, regz_end]))
regx_init, regy_init, regz_init = map(lambda x: max(x, 0), [regx_init, regy_init, regz_init])

sx, sy, sz = regx_end - regx_init, regy_end - regy_init, regz_end - regz_init
dydx = sy / float(sx)
dzdx = sz / float(sx)
dx = (nelems / (dydx * dzdx)) ** (1. / 3)
dy = int(dydx * dx)
dz = int(dzdx * dx)
dx = int(dx)

from math import ceil

nchunks = int(ceil(sx / float(dx)) * ceil(sy / float(dy)) * ceil(sz / float(dz)))
chunks_processed = 0

print 'Done.'

# OK, let's go for it!
print '    Proceeding to analyze the data...'

for x in range(regx_init, regx_end, dx):
    for y in range(regy_init, regy_end, dy):
        for z in range(regz_init, regz_end, dz):
            print '        Chunk #' + str(chunks_processed + 1) + ' out of ' + str(nchunks) + ':'
            print '            Reading data...',
            data = [[] for _ in diag]
            for i in range(len(diag)):
                for elem in diag[i]:
                    try:
                        f = nib.load(join(WORK_DIR, DATA_DIR, filename_by_id[elem]))
                        data[i].append(f.get_data('unchanged')[x:(x + dx), y:(y + dy), z:(z + dz)])
                    except KeyError as e:  # Should never raise, unless keys in excel or in filenames are mistyped
                        print
                        print '    Warning: Element with id ' + elem + ' does not exist (skipped)'
                    except IOError as e:
                        print
                        print '    Warning: Error while managing file ' + filename_by_id[
                            elem] + ' (skipped in chunk [' + str(x) + ', ' + str(y) + ', ' + str(z) + '])'
                        print '      More details:', e
            print 'Done.'
            print '            Performing t-test...',
            for i in range(len(output)):
                for j in range(len(output[i])):
                    tt_res = ttest_ind(data[i], data[i + j + 1])
                    output[i][j][x:(x + dx), y:(y + dy), z:(z + dz)] = tt_res.pvalue
            print 'Done.'
            chunks_processed += 1

del data

print 'Data processed!'
print

# Store the values in NIfTI files
print 'Storing results...',
try:
    out_data = f.get_data()
    for i in range(len(output)):
        for j in range(len(output[i])):
            out_data[:, :, :] = output[i][j]
            nib.save(f,
                     join(WORK_DIR, OUTPUT_DIR, OUTPUT_FILENAME + '_' + classes[i] + '_' + classes[i + j + 1] + '.nii'))
except IOError:
    print
    print 'ERROR: Could not save data as NIfTI files.'
    print 'Proceeding to print p-values in raw format instead...',
    try:
        for i in range(len(output)):
            for j in range(len(output[i])):
                with open(join(WORK_DIR, OUTPUT_DIR,
                               'raw_' + OUTPUT_FILENAME + '_' + classes[i] + '_' + classes[i + j + 1] + '.nii'),
                          'wb') as f:
                    for x in range(regx_init, regx_end):
                        f.write('Results [' + str(x) + ', :, :] = [\n')
                        for row in output[i][j][x, :, :]:
                            f.write('    [ ')
                            for elem in row:
                                f.write(str(elem) + ' ')
                            f.write(']\n')
                        f.write('] ;\n\n')
    except:
        print
        print 'FATAL ERROR: Could not store p-values in raw format. Aborting program...'
        exit()

print 'Done.'
print

print 'Congratulations! Program executed succesfully! :D'
print
