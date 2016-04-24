from niftiIO import NiftiSetManager as NSF
l = []
l.append('/Users/Asier/Documents/TFG/Alan T/Nonlinear_NBA_15/s6m0wrp1089_ASD.nii')
l.append('/Users/Asier/Documents/TFG/Alan T/Nonlinear_NBA_15/s6m0wrp1100_MPZ.nii')
nsf = NSF(l)
for v in nsf.voxels(x1 = 50, y1 = 60, z1 = 50, x2 = 52, y2 = 62, z2 = 52):
	print str(v.coords) + ':', v.data

