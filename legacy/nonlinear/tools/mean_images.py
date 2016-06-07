# Set paths
WORK_DIR = '/Users/Asier/Documents/TFG/Alan T'
DATA_DIR = 'Nonlinear_NBA_15'
EXCEL_FILE = 'work_DB_CSF.R1.final.xls'
OUTPUT_FILE_AD = 'mean' + '_ad.nii'
OUTPUT_FILE_PREC = OUTPUT_FILE_AD[:-6] + 'prec' + OUTPUT_FILE_AD[-4:]
OUTPUT_FILE_MCIS = OUTPUT_FILE_AD[:-6] + 'mci' + OUTPUT_FILE_AD[-4:]
OUTPUT_FILE_CONT = OUTPUT_FILE_AD[:-6] + 'cont' + OUTPUT_FILE_AD[-4:]

# Get all files in data directory and index them by their ID
from os import listdir
from os.path import join, isfile

filename_by_id = {f.split('_')[0][8:]: f for f in
                  filter(lambda elem: isfile(join(WORK_DIR, DATA_DIR, elem)), listdir(join(WORK_DIR, DATA_DIR)))}

# Retrieve interesting data from excel file
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

    # Separate the elements by their category
    diag = [[], [], [], []]
    ids_col = ws.col(header2col['id'])
    diags_col = ws.col(header2col['diag'])
    for i in range(1, len(ids_col)):
        if type2text[ids_col[i].ctype] == 'text' and type2text[diags_col[i].ctype] in ('text', 'number'):
            diag[int(diags_col[i].value) - 1].append(ids_col[i].value.strip().split('_')[0])

# Prepare output files
import nibabel as nib
from sys import exit

# Initialize and save output (all zeros) to avoid any aliasing or pipe problems
output_files = [OUTPUT_FILE_CONT, OUTPUT_FILE_PREC, OUTPUT_FILE_MCIS, OUTPUT_FILE_AD]
outf = []

try:
    for i in range(len(output_files)):
        outfile = output_files[i]
        f = nib.load(join(WORK_DIR, DATA_DIR, filename_by_id[diag[i][0]]))
        nib.save(f, join(WORK_DIR, outfile))
        del f
        outf.append(nib.load(join(WORK_DIR, outfile)))
except IOError:
    print 'FATAL ERROR: Could not create output files. Aborting process...'
    exit()

from numpy import zeros

output_data = []
for outfile in outf:
    output_data.append(outfile.get_data())
    output_data[-1][:, :, :] = zeros(outfile.shape)

for i in range(len(diag)):
    nelems = float(len(diag[i]))
    for elem in diag[i]:
        try:
            f = nib.load(join(WORK_DIR, DATA_DIR, filename_by_id[elem]))
            output_data[i] += f.get_data('unchanged') / nelems
        except KeyError as e:
            print 'Warning: Element with id ' + elem + ' does not exist (skipped)'
        except IOError as e:
            print 'Warning: Error while managing file ' + filename_by_id[elem] + ' (skipped)'
            print '    More details:', e

# Store the values in NIfTI files

for i in range(len(outf)):
    try:
        nib.save(outf[i], join(WORK_DIR, output_files[i]))
    except IOError:
        print 'ERROR: Could not save data as NIfTI files.',
        print 'Proceeding to print mean values in raw format instead...'
        try:
            nx = outf[i].shape[0]
            with open(join(WORK_DIR, 'raw_' + output_files[i]), 'wb') as f:
                for x in range(nx):
                    f.write('Results [' + str(x) + ', :, :] = [\n')
                    for row in output_data[i][x, :, :]:
                        f.write('    [ ')
                        for elem in row:
                            f.write(str(elem) + ' ')
                        f.write(']\n')
                    f.write('] ;\n\n')
        except:
            print 'FATAL ERROR: Could not store mean values in raw format. Skipping file ' + output_files[i] + '...'
