from nibabel import load as nibload, save as nibsave, Nifti1Image as niiFile
from numpy import array as nparray


class NiftiReader:

	def __init__(self, filename, x1 = 0, y1 = 0, z1 = 0, x2 = None, y2 = None, z2 = None):
		self.filename = filename
		f = nibload(filename)
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

		# nelems = dx*dy*dz = dx * (dy/dx * dx) * (dz/dx * dx) = (dy/dx * dz/dx) * (dx**3)
		# dx = (nelems / (dy/dx * dz/dx))**(1./3)

		d = 1.0
		for x in self.dims[3:]:
			d *= x
		nelems = self.mem_usage*(2**17)/d # (number of MBytes x 2**20 bytes/MB x 8 bits/byte)/(64 bits/elem)

		sx, sy, sz = self.dims[:3]
		dydx = sy/float(sx)
		dzdx = sz/float(sx)
		dx = (nelems / (dydx * dzdx))**(1.0/3)
		dy = int(dydx * dx)
		dz = int(dzdx * dx)
		dx = int(dx)

		x1, y1, z1 = self.coords
		x2, y2, z2 = (x1 + sx, y1 + sy, z1 + sz)
		for x in range(x1, x2, dx):
			for y in range(y1, y2, dy):
				for z in range(z1, z2, dz):
					f = nibload(self.filename)
					chunk = Region((x, y, z), f.get_data('unchanged')[x:min(x2, x+dx), y:min(y2, y+dy), z:min(z2, z+dz)])
					del f
					yield chunk

	def __iter__(self):
		return self.chunks()


class NiftiInputManager(NiftiReader):

	def affine(self):
		f = nibload(self.filename)
		aff = f.affine
		del f
		return aff


class NiftiSetReader:

	def __init__(self, filenames, *args, **kwargs):
		assert len(filenames) != 0
		self.filenames = filenames
		self.nms = map(lambda fn: NiftiInputManager(fn, *args, **kwargs), self.filenames)
		self.dims = self.nms[0].dims[:3] + (len(self.filenames),) + self.nms[0].dims[3:]
		self.affine = sum(nm.affine() for nm in self.nms)/float(len(self.nms))
		self.mem_usage = 100.0

	def chunks(self, mem_usage = None):
		if mem_usage != None:
			self.mem_usage = mem_usage
		iterators = map(lambda nm: nm.chunks(float(self.mem_usage)/len(self.nms)), self.nms)
		try:
			while True:
				chunkset = map(lambda it: it.next(), iterators)
				yield Region(chunkset[0].coords, nparray(map(lambda chunk: chunk.data, chunkset)))
		except StopIteration:
			pass
		
	def __iter__(self):
		return self.voxels()

	def voxels(self, mem_usage = None):
		for chunkset in self.chunks(mem_usage):
			dims = chunkset.data.shape
			x, y, z = chunkset.coords
			for i in range(dims[1]):
				for j in range(dims[2]):
					for k in range(dims[3]):
						yield Region((x+i, y+j, z+k), chunkset.data[:, i, j, k])


class NiftiWriter(niiFile):
	@staticmethod
	def open(filename):
		f = nibload(filename)
		nw = NiftiWriter(f.get_data('unchanged'), f.affine)
		del f
		return nw

	def save(self, filename = None, *args, **kwargs):
		if filename != None:
			self.filename = filename
		nibsave(self, self.filename, *args, **kwargs)

	def chunks(self, mem_usage = None, *args, **kwargs):
		try:
			return NiftiReader(self.filename, *args, **kwargs).chunks(mem_usage)
		except AttributeError:
			return ()


class Region:

	def __init__(self, coords, data):
		self.coords = coords
		self.data = data



