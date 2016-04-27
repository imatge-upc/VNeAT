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



# Retrieve scanner data from files for ADs and Controls
import nibabel as nib

ad_data = []
ct_data = []

for data, ids in ((ad_data, ads), (ct_data, controls)):
	for elem in ids:
		try:
			f = nib.load(join(WORK_DIR, DATA_DIR, filename_by_id[elem]))
			data.append(f.get_data('unchanged')[regx_init:regx_end, regy_init:regy_end, regz_init:regz_end])
		except KeyError as e:
			print 'Warning: Element with id ' + elem + ' does not exist (skipped)'
		except IOError as e:
			print 'Warning: Error while managing file ' + filename_by_id[elem] + ' (skipped)'
			print '    More details:', e


# Do the tests
from scipy.stats import ttest_ind
tt_res_eq = ttest_ind(ad_data, ct_data)
tt_res_neq = ttest_ind(ad_data, ct_data, equal_var = False)
# print 'Results assuming equal variances:'
# print tt_res_eq
# print '-'*40
# print 'Results assuming different variances (Welch\'s test):'
# print tt_res_neq


# Store the values in NIfTI files

class SaveDataError(Exception):
	def __init__(self, msg):
		self.msg = msg
	def __str__(self):
		return repr(self.msg)
from numpy import zeros

try:
	templ = f
	img = templ.get_data()
	img[:, :, :] = zeros(templ.shape)
	img[regx_init:regx_end, regy_init:regy_end, regz_init:regz_end] = tt_res_eq.pvalue
	try:
		nib.save(templ, join(WORK_DIR, OUTPUT_FILE_EQ))
	except IOError:
		raise SaveDataError('Could not save data as NIfTI files')

	img[regx_init:regx_end, regy_init:regy_end, regz_init:regz_end] = tt_res_neq.pvalue
	try:
		nib.save(templ, join(WORK_DIR, OUTPUT_FILE_NEQ))
	except IOError:
		raise SaveDataError('Could not save data as NIfTI files')

except SaveDataError as e:
	print 'ERROR:', e
	print 'Proceeding to print p-values in raw format instead...'
	try:
		with open(join(WORK_DIR, 'raw_' + OUTPUT_FILE_EQ), 'wb') as f:
			for x in range(regx_init, regx_end):
				f.write('Results [' + str(x) + ', :, :] = [\n')
				for row in tt_res_eq.pvalue[x, :, :]:
					f.write('    [ ')
					for elem in row:
						f.write(str(elem) + ' ')
					f.write(']\n')
				f.write('] ;\n\n')
		with open(join(WORK_DIR, 'raw_' + OUTPUT_FILE_NEQ), 'wb') as f:
			for x in range(regx_init, regx_end):
				f.write('Results [' + str(x) + ', :, :] = [\n')
				for row in tt_res_neq.pvalue[x, :, :]:
					f.write('    [ ')
					for elem in row:
						f.write(str(elem) + ' ')
					f.write(']\n')
				f.write('] ;\n\n')
	except:
		print 'FATAL ERROR: Could not store p-values in raw format. Aborting program...'





