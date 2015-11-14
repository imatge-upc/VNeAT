from nibabel import load as nibload, Nifti1Image as niiFile
from numpy import array as nparray


class NiftiManager:

	def __init__(self, filename):
		self.filename = filename
		f = nibload(filename)
		self.dims = f.shape
		del f

	def chunks(self, mem_usage = 0.5, x1 = 0, y1 = 0, z1 = 0, x2 = None, y2 = None, z2 = None):

		if x2 == None:
			x2 = self.dims[0]
		if y2 == None:
			y2 = self.dims[1]
		if z2 == None:
			z2 = self.dims[2]

		assert x1 < x2
		assert y1 < y2
		assert z1 < z2

		nelems = mem_usage*(2**17) # (number of MBytes x 2**20 bytes/MB x 8 bits/byte)/(64 bits/elem)

		x1, y1, z1 = map(lambda a: max(a, 0), [x1, y1, z1])
		x2, y2, z2 = map(min, zip(self.dims, [x2, y2, z2]))

		# nelems = dx*dy*dz = dx * (dy/dx * dx) * (dz/dx * dx) = (dy/dx * dz/dx) * (dx**3)
		# dx = (nelems / (dy/dx * dz/dx))**(1./3)
		sx, sy, sz = x2 - x1, y2 - y1, z2 - z1
		dydx = sy/float(sx)
		dzdx = sz/float(sx)
		dx = (nelems / (dydx * dzdx))**(1./3)
		dy = int(dydx * dx)
		dz = int(dzdx * dx)
		dx = int(dx)

		for x in range(x1, x2, dx):
			for y in range(y1, y2, dy):
				for z in range(z1, z2, dz):
					f = nibload(self.filename)
					chunk = Region((x, y, z), f.get_data('unchanged')[x:min(x2, x+dx), y:min(y2, y+dy), z:min(z2, z+dz)])
					del f
					yield chunk

	def affine(self):
		f = nibload(self.filename)
		aff = f.affine
		del f
		return aff


class NiftiSetManager:

	def __init__(self, filenames):
		assert len(filenames) != 0
		self.filenames = filenames
		self.affine = None

	def chunksets(self, mem_usage = 100.0, *args, **kwargs):
		nms = map(NiftiManager, self.filenames)
		iterators = map(lambda nm: nm.chunks(float(mem_usage)/len(nms), *args, **kwargs), nms)
		try:
			while True:
				chunkset = map(lambda it: it.next(), iterators)
				yield Region(chunkset[0].coords, nparray(map(lambda chunk: chunk.data, chunkset)))
		except StopIteration:
			pass

	def voxels(self, *args, **kwargs):
		for chunkset in self.chunksets(*args, **kwargs):
			dims = chunkset.data.shape
			x, y, z = chunkset.coords
			for i in range(dims[1]):
				for j in range(dims[2]):
					for k in range(dims[3]):
						yield Region((x+i, y+j, z+k), chunkset.data[:, i, j, k])

	def affine(self):
		if self.affine == None:
			self.affine = sum(NiftiManager(fn).affine() for fn in self.filenames)/float(len(self.filenames))
		return self.affine


class NiftiOutput(niiFile):
	pass


class Region:

	def __init__(self, coords, data):
		self.coords = coords
		self.data = data



