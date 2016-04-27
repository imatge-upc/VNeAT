
from sys import argv, exit

if len(argv) != 2:
	print 'Usage: python ' + argv[0] + ' p-value'
	exit()

try:
	pvalue = float(argv[1])
except ValueError:
	print 'ERROR: Parameter p-value must be a floating point number'
	exit()

WORK_DIR = '/Users/Asier/Documents/TFG/Alan T/ttests'
OUTPUT_DIR = 'fsl'

from os import makedirs
from os.path import join
from errno import EEXIST
try:
	makedirs(join(WORK_DIR, OUTPUT_DIR, ''))
except OSError as e:
	if e.errno == EEXIST:
		pass
	else:
		raise

from glob import glob
from os.path import basename, splitext
import nibabel as nib
from numpy import isfinite, array

for filename in glob(join(WORK_DIR, 'ttest_results_*_*.nii')):
	f = nib.load(filename)
	data = f.get_data()
	# data[:, :, :] = [[[1 if (isfinite(elem) and elem < pvalue) else 0 for elem in row] for row in mat] for mat in data[:, :, :]]
	margin = 0.5
	data[:, :, :] = [[[margin + (1 - margin)*(1 - elem/pvalue) if (isfinite(elem) and elem < pvalue) else 0 for elem in row] for row in mat] for mat in data[:, :, :]]
	bn, ext = splitext(basename(filename))
	outfname = join(WORK_DIR, OUTPUT_DIR, bn + '_' + str(pvalue) + ext)
	nib.save(f, outfname)


