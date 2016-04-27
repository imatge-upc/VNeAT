import nibabel as nib

import nii

orig_img = nib.load('../Alan T/Nonlinear_NBA_15/s6m0wrp1089_ASD.nii')
img = orig_img.get_data()

img1 = nii.slice(img, x = 50)
img2 = nii.slice(img, y = 70)
img3 = nii.slice(img, z = 60)

nii.output(img1, 'imgx.mat')
nii.output(img2, 'imgy.mat')
nii.output(img3, 'imgz.mat')