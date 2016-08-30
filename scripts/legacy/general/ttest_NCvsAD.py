from os import listdir
from os.path import join, isfile, basename

import nibabel as nib
from nonlinear2.ExcelIO import ExcelSheet as Excel
from nonlinear2.Subject import Subject
from numpy import asarray, ones, array
from scipy.stats import ttest_ind, norm

from Utils import NiftiGraph

# Constants
gm_threshold = 0.1
pv_threshold = 0.001
num_nodes_cluster = 100

# Path to Excel and NIFTI data
DATA_DIR = join("C:\\", "Users", "santi", "Documents", "Santi", "Universitat", "TFG", "Data", "nonlinear_data",
                "Nonlinear_NBA_15")
EXCEL_FILE = join("C:\\", "Users", "santi", "Documents", "Santi", "Universitat", "TFG", "Data", "nonlinear_data",
                  "work_DB_CSF.R1.final.xls")
RESULTS_FILENAME = "ttest_results_NCvsAD.nii"

# Load data from Excel
print "Loading Excel data from ", EXCEL_FILE, "...",
filenames = filter(isfile, map(lambda elem: join(DATA_DIR, elem), listdir(DATA_DIR)))
filenames_by_id = {basename(fn).split('_')[0][8:]: fn for fn in filenames}

excel = Excel(EXCEL_FILE)
print "Done."

# Create empty subjects list
subjects = []

# Fill list with subjects
for r in excel.get_rows(fieldstype={
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

# List of control and AD patients
control_patients = [x for x in subjects if x.get([Subject.Diagnostic])[0] == 0]
ad_patients = [x for x in subjects if x.get([Subject.Diagnostic])[0] == 3]

# Array of patients with their VBM images
print "Loading NIFTI files for control and AD patients... ",
control_gm = asarray(map(lambda subject: nib.load(subject.gmfile).get_data(), control_patients))
ad_gm = asarray(map(lambda subject: nib.load(subject.gmfile).get_data(), ad_patients))
del ad_patients, control_patients
print "Done."

# Compute dimensions of the data and prepare the ouput_data array
dimensions = control_gm.shape[1:]
N_voxels = dimensions[0] * dimensions[1] * dimensions[2]
output_data = ones(dimensions)
print "VBM dimensions: ", dimensions
print "Number of voxels per patient: ", N_voxels
print "Number of total voxels: ", N_voxels * control_gm.shape[0] + N_voxels * ad_gm.shape[0]

# For each voxel x,y,z compute a t-test for the difference of means between control and AD
print "Computing t-tests for all voxels...",
# 3D aray with same dimensions as "dimensions", with value 1 in voxels where grey matter mean is equal or above threshold
valid_vox = ((sum(control_gm) + sum(ad_gm)) >= gm_threshold * (len(control_gm) + len(ad_gm))).astype(int)
# 3D aray with same dimensions as "dimensions", with value 1 in voxels where grey matter mean is below threshold
nvalid_vox = 1 - valid_vox
_, p_val_array = ttest_ind(control_gm + 0.00001, ad_gm + 0.00001)  # Added 0.00001 to avoid NaN problems
output_data = p_val_array * valid_vox + nvalid_vox
print "Done."

# Clustering by means of a Strongly Connected Components algorithm
print 'Filtering for clusters of size >= ' + str(num_nodes_cluster) + '...',
lim_value = norm.ppf(1 - pv_threshold)
g = NiftiGraph(output_data, pv_threshold)
for scc in g.sccs():
    if len(scc) < num_nodes_cluster:
        for x, y, z in scc:
            output_data[x, y, z] = 0.0
    else:
        for x, y, z in scc:
            # z-score
            output_data[x, y, z] = norm.ppf(1 - output_data[x, y, z]) - lim_value + 0.2
print "Done."

# Store the results into a NIFTI file
print 'Storing results to a NIFTI file: ', RESULTS_FILENAME, " ...",
affine = array(
    [[-1.50000000e+00, 0.00000000e+00, 0.00000000e+00, 9.00000000e+01],
     [1.99278252e-16, 1.50000000e+00, 2.17210575e-16, -1.26000000e+02],
     [-1.36305018e-16, -1.38272305e-16, 1.50000000e+00, -7.20000000e+01],
     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
)
niiImage = nib.Nifti1Image(output_data, affine)
nib.save(niiImage, RESULTS_FILENAME)
print "Done."
