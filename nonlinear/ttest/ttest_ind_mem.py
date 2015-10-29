# Set paths
WORK_DIR = '/Users/Asier/Documents/TFG/Alan T'
DATA_DIR = 'Nonlinear_NBA_15'
EXCEL_FILE = 'work_DB_CSF.R1.final.xls'
OUTPUT_FILE_EQ = 'ttest_results' + '_eq.nii'
OUTPUT_FILE_NEQ = OUTPUT_FILE_EQ[:-6] + 'n' + OUTPUT_FILE_EQ[-6:]

# Set region
regx_init = 0
regx_end = 1000
regy_init = 0
regy_end = 1000
regz_init = 0
regz_end = 1000

MEMORY_USE = 100 # approx. memory use in MB (depends on garbage collector, but the order of magnitude should be around this value)


# Get all files in data directory and index them by their ID
from os.path import join, isfile
from os import listdir

filename_by_id = {f.split('_')[0][8:] : f for f in filter(lambda elem: isfile(join(WORK_DIR, DATA_DIR, elem)), listdir(join(WORK_DIR, DATA_DIR)))}


# Initialize data retrieval variables
ads = []
preclinicals = []
mcis = []
controls = []


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
	h = ws.row(0) # Extract first row to make it more efficient (since we're gonna make multiple sequential reads on it)
	header2col = {h[j].value.strip().lower() : j for j in range(ws.ncols) if type2text[h[j].ctype] == 'text'}

	# Separate the elements by their category
	diag = [controls, preclinicals, mcis, ads]
	ids_col = ws.col(header2col['id'])
	diags_col = ws.col(header2col['diag'])
	for i in range(1, len(ids_col)):
		if type2text[ids_col[i].ctype] == 'text' and type2text[diags_col[i].ctype] in ('text', 'number'):
			diag[int(diags_col[i].value) - 1].append(ids_col[i].value.strip().split('_')[0])



# Retrieve scanner data in chunks from files for ADs and Controls and
# perform the tests for each chunk, storing the results in a NIfTI file
import nibabel as nib
from sys import exit

try:
	f = nib.load(join(WORK_DIR, DATA_DIR, filename_by_id[ads[0]]))
except IOError as e:
	print 'FATAL ERROR: Could not open output file template. Aborting process...'
	exit()

# Initialize and save output (all zeros) to avoid any aliasing or pipe problems
from numpy import zeros
try:
	nib.save(f, join(WORK_DIR, OUTPUT_FILE_EQ))
	nib.save(f, join(WORK_DIR, OUTPUT_FILE_NEQ))
	del f
	outf_eq = nib.load(join(WORK_DIR, OUTPUT_FILE_EQ))
	outf_neq = nib.load(join(WORK_DIR, OUTPUT_FILE_NEQ))
except IOError:
	print 'FATAL ERROR: Could not create output files. Aborting process...'
	exit()

output_eq = outf_eq.get_data()
output_eq[:, :, :] = zeros(outf_eq.shape)

output_neq = outf_neq.get_data()
output_neq[:, :, :] = zeros(outf_neq.shape)

# Perform analysis chunk by chunk (chunks of dx x dy x dz)
from scipy.stats import ttest_ind

nelems = MEMORY_USE*(2**17)/(len(ads) + len(controls)) # (number of MBytes x 2**20 bytes/MB x 8 bits/byte)/((64 bits/elem)* #samples)

# nelems = dx*dy*dz = dx * (dy/dx * dx) * (dz/dx * dx) = (dy/dx * dz/dx) * (dx**3)
# dx = (nelems / (dy/dx * dz/dx))**(1./3)
regx_end, regy_end, regz_end = map(min, zip(outf_eq.shape, [regx_end, regy_end, regz_end]))
regx_init, regy_init, regz_init = map(lambda x: max(x, 0), [regx_init, regy_init, regz_init])

sx, sy, sz = regx_end - regx_init, regy_end - regy_init, regz_end - regz_init
dydx = sy/float(sx)
dzdx = sz/float(sx)
dx = (nelems / (dydx * dzdx))**(1./3)
dy = int(dydx * dx)
dz = int(dzdx * dx)
dx = int(dx)

for x in range(regx_init, regx_end, dx):
	for y in range(regy_init, regy_end, dy):
		for z in range(regz_init, regz_end, dz):
			ad_data = []
			ct_data = []
			for data, ids in ((ad_data, ads), (ct_data, controls)):
				for elem in ids:
					try:
						f = nib.load(join(WORK_DIR, DATA_DIR, filename_by_id[elem]))
						data.append(f.get_data('unchanged')[x:(x+dx), y:(y+dy), z:(z+dz)])
					except KeyError as e: # Should never raise, unless keys in excel or in filenames are mistyped
						print 'Warning: Element with id ' + elem + ' does not exist (skipped)'
					except IOError as e:
						print 'Warning: Error while managing file ' + filename_by_id[elem] + ' (skipped in chunk [' + str(x) + ', ' + str(y) + ', ' + str(z) + '])'
						print '    More details:', e

			# Do the tests for current chunk and store them
			tt_res_eq = ttest_ind(ad_data, ct_data)
			output_eq[x:(x+dx), y:(y+dy), z:(z+dz)] = tt_res_eq.pvalue
			del tt_res_eq
			tt_res_neq = ttest_ind(ad_data, ct_data, equal_var = False)
			output_neq[x:(x+dx), y:(y+dy), z:(z+dz)] = tt_res_neq.pvalue
			del tt_res_neq
del ad_data
del ct_data


# Store the values in NIfTI files

try:
	nib.save(outf_eq, join(WORK_DIR, OUTPUT_FILE_EQ))
	nib.save(outf_neq, join(WORK_DIR, OUTPUT_FILE_NEQ))
except IOError:
	print 'ERROR: Could not save data as NIfTI files'
	print 'Proceeding to print p-values in raw format instead...'
	try:
		with open(join(WORK_DIR, 'raw_' + OUTPUT_FILE_EQ), 'wb') as f:
			for x in range(regx_init, regx_end):
				f.write('Results [' + str(x) + ', :, :] = [\n')
				for row in output_eq[x, :, :]:
					f.write('    [ ')
					for elem in row:
						f.write(str(elem) + ' ')
					f.write(']\n')
				f.write('] ;\n\n')
		with open(join(WORK_DIR, 'raw_' + OUTPUT_FILE_NEQ), 'wb') as f:
			for x in range(regx_init, regx_end):
				f.write('Results [' + str(x) + ', :, :] = [\n')
				for row in output_neq[x, :, :]:
					f.write('    [ ')
					for elem in row:
						f.write(str(elem) + ' ')
					f.write(']\n')
				f.write('] ;\n\n')
	except:
		print 'FATAL ERROR: Could not store p-values in raw format. Aborting program...'





