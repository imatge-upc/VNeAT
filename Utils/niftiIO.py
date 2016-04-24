import nibabel as nib
niiFile = nib.Nifti1Image

class NiftiReader:

	def __init__(self, filename, x1 = 0, y1 = 0, z1 = 0, x2 = None, y2 = None, z2 = None):
		self.filename = filename
		f = nib.load(filename)
		self.dims = f.shape
		del f
		if x2 == None:
			x2 = self.dims[0]
		if y2 == None:
			y2 = self.dims[1]
		if z2 == None:
			z2 = self.dims[2]

		assert x1 < x2
		assert y1 < y2
		assert z1 < z2

		self.mem_usage = 0.5

		x1, y1, z1 = map(lambda a: max(a, 0), [x1, y1, z1])
		x2, y2, z2 = map(min, zip(self.dims, [x2, y2, z2]))

		self.dims = (x2 - x1, y2 - y1, z2 - z1) + self.dims[3:]
		self.coords = (x1, y1, z1)

	def chunks(self, mem_usage = None):
		if mem_usage != None:
			self.mem_usage = mem_usage

		d = 1.0
		for x in self.dims[3:]:
			d *= x
		nelems = self.mem_usage*(2**17)/d # (number of MBytes x 2**20 bytes/MB x 8 bits/byte)/(64 bits/elem)

		sx, sy, sz = self.dims[:3]

		dx = nelems/(sy*sz)
		if dx > 1:
			dx = int(dx)
			dy = sy
			dz = sz
		else:
			dx = 1
			dy = nelems/sz
			if dy > 1:
				dy = int(dy)
				dz = sz
			else:
				dy = 1
				dz = nelems

		x1, y1, z1 = self.coords
		x2, y2, z2 = (x1 + sx, y1 + sy, z1 + sz)
		for x in range(x1, x2, dx):
			for y in range(y1, y2, dy):
				for z in range(z1, z2, dz):
					f = nib.load(self.filename)
					chunk = Region((x, y, z), f.get_data('unchanged')[x:min(x2, x+dx), y:min(y2, y+dy), z:min(z2, z+dz)])
					del f
					yield chunk

	def __iter__(self):
		return self.chunks()

	def affine(self):
		f = nib.load(self.filename)
		aff = f.affine
		del f
		return aff


class NiftiWriter(niiFile):
	@staticmethod
	def open(filename):
		f = nib.load(filename)
		nw = NiftiWriter(f.get_data('unchanged'), f.affine)
		del f
		return nw

	def save(self, filename = None, *args, **kwargs):
		if filename != None:
			self._filename = filename
		nib.save(self, self._filename, *args, **kwargs)

	def chunks(self, mem_usage = None, *args, **kwargs):
		try:
			return NiftiReader(self._filename, *args, **kwargs).chunks(mem_usage)
		except AttributeError:
			return ()


class Region:

	def __init__(self, coords, data):
		self.coords = coords
		self.data = data

	def size():
		return self.data.shape[:3]


