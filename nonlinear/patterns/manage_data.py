from nibabel import load as nibload
from numpy import array as nparray

class FileManager:

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

		assert x1 <= x2
		assert y1 <= y2
		assert z1 <= z2

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
					data = f.get_data('unchanged')[x:(x+dx), y:(y+dy), z:(z+dz)]
					del f
					yield data


class FileSetManager:

	def __init__(self, filenames):
		self.filenames = filenames

	def chunksets(self, mem_usage = 100.0, x1 = 0, y1 = 0, z1 = 0, x2 = None, y2 = None, z2 = None):
		fms = map(FileManager, self.filenames)
		iterators = map(lambda fm: fm.chunks(mem_usage, x1, y1, z1, x2, y2, z2), fms)
		try:
			while True:
				yield nparray(map(lambda it: it.next(), iterators))
		except StopIteration:
			pass

	def voxels(self, mem_usage = 100.0, x1 = 0, y1 = 0, z1 = 0, x2 = None, y2 = None, z2 = None):
		for chunkset in self.chunksets(mem_usage, x1, y1, z1, x2, y2, z2):
			dims = chunkset.shape
			for i in range(dims[1]):
				for j in range(dims[2]):
					for k in range(dims[3]):
						yield chunkset[:, i, j, k]


