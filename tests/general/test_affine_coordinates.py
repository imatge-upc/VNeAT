from os.path import join

import nibabel as nib
import numpy as np

filename = join("C:\\", "Users", "santi", "Documents", "Santi", "Universitat", "TFG", "Data",
                "nonlinear_data", "MNI152_T1_15mm_template.nii")
filename_mod = join("C:\\", "Users", "santi", "Documents", "Santi", "Universitat", "TFG", "Data",
                    "nonlinear_data", "MNI152_T1_15mm_template_mod.nii")

mni_affine = nib.load(filename).affine
print "MNI affine: "
print mni_affine

custom_affine = np.array(
    [[-1.50000000e+00, 0.00000000e+00, 0.00000000e+00, 9.00000000e+01],
     [1.99278252e-16, 1.50000000e+00, 2.17210575e-16, -1.26000000e+02],
     [-1.36305018e-16, -1.38272305e-16, 1.50000000e+00, -7.20000000e+01],
     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
)
print
print "Custom affine: "
print custom_affine

prec_coord = np.array([2.0, -54.0, 26.0, 1])

c_mni = np.linalg.inv(mni_affine).dot(prec_coord)
c_custom = np.linalg.inv(custom_affine).dot(prec_coord)

print
print "Test MNI affine with precuneus coordinates (2, -54, 26 mm)..."
print "Voxel coordinates: ", c_mni[:-1]
print "Rounded voxel coordinates: ", np.round(c_mni[:-1])

print
print "Test Custom affine with precuneus coordinates (2, -54, 26 mm)..."
print "Voxel coordinates: ", c_custom[:-1]
print "Rounded voxel coordinates: ", np.round(c_custom[:-1])

print
print "Change affine matrix in template and store the modded template"
mni_template_data = nib.load(filename).get_data()
mni_template_mod = nib.Nifti1Image(mni_template_data, custom_affine)
nib.save(mni_template_mod, filename_mod)
