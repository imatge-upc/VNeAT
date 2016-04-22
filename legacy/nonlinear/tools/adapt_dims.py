from nibabel import load as nibload, save as nibsave, Nifti1Image as niiFile
from math import ceil
from numpy import array as nparray

img = nibload('/Users/Asier/Documents/TFG/Alan T/template.nii.gz')
img_data = img.get_data()

sx, sy, sz = img_data.shape
sx2, sy2, sz2 = (121, 145, 121)

rx, ry, rz = [(sx-1, sy-1, sz-1)[i]/float((sx2-1, sy2-1, sz2-1)[i]) for i in range(3)]

def compute_value(data, x, y, z):
	hx, hy, hz = map(lambda n: int(ceil(n)), (x, y, z))
	lx, ly, lz = map(int, (x, y, z))
	dxl, dyl, dzl = x - lx, y - ly, z - lz
	dxh, dyh, dzh = hx - x, hy - y, hz - z
	if hx == lx:
		dxl = 1.
	if hy == ly:
		dyl = 1.
	if hz == lz:
		dzl = 1.

	v = 0
	for x, dx in ((lx, dxh), (hx, dxl)):
		for y, dy in ((ly, dyh), (hy, dyl)):
			for z, dz in ((lz, dzh), (hz, dzl)):
				v += data[x][y][z]*dx*dy*dz
	return v


data = nparray([[[compute_value(img_data, x*rx, y*ry, z*rz) for z in range(sz2)] for y in range(sy2)] for x in range(sx2)])

print 'Saving...'

nibsave(niiFile(data, img.affine), '/Users/Asier/Documents/TFG/Alan T/adapted_template.nii')
