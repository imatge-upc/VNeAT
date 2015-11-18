from excelIO import get_rows
from niftiIO import NiftiInputManager as NiftiIM, Region
from numpy import array as nparray
from os.path import basename

class InputData:

	def __init__(self, filenames, excel_file, *args, **kwargs):
		assert len(filenames) != 0

		filenames_by_id = {InputData.filename_to_id(fn) : fn for fn in filenames}

		self.subjects = []
		for r in get_rows(excel_file):
			self.subjects.append(Subject(r['id'],
										 NiftiIM(filenames_by_id[r['id']], *args, **kwargs),
										 r.get('diag', None),
										 r.get('age', None),
										 r.get('sex', None),
										 r.get('apoe4_bin', None),
										 r.get('escolaridad', None),
										 r.get('ad_csf_index_ttau', None)))
		self.normalize_subject_data()

		subj_dims = self.subjects[0].gmdata.dims
		self.dims = subj_dims[:3] + (len(self.subjects),) + subj_dims[3:]

		self.affine = sum(subj.gmdata.affine() for subj in self.subjects)/float(len(self.subjects))

		self.mem_usage = 100.0


	def chunks(self, mem_usage = None):
		if mem_usage != None:
			self.mem_usage = float(mem_usage)

		num_subjects = len(self.subjects)
		iterators = map(lambda subj: subj.gmdata.chunks(self.mem_usage/num_subjects), self.subjects)
		try:
			while True:
				reg = iterators[0].next()
				chunkset = [reg.data]
				for it in iterators[1:]:
					chunkset.append(it.next().data)
				yield Region(reg.coords, nparray(chunkset))
		except StopIteration:
			pass

	def voxels(self, mem_usage = None):
		for chunkset in self.chunks(mem_usage):
			dims = chunkset.data.shape
			x, y, z = chunkset.coords
			for i in range(dims[1]):
				for j in range(dims[2]):
					for k in range(dims[3]):
						yield Voxel((x+i, y+j, z+k), chunkset.data[:, i, j, k])

	def __iter__(self):
		return self.supervoxels()

	def supervoxels(self, mem_usage = None):
		return (self.wrap(voxel) for voxel in self.voxels(mem_usage))

	def wrap(self, voxel):
		return Voxel(voxel.coords, nparray([VoxelData(self.subjects[i], voxel.data[i]) for i in range(len(voxel.data))]))

	def normalize_subject_data(self):
		#TODO
		return

	@staticmethod
	def filename_to_id(filename):
		return basename(filename).split('_')[0][8:]


class VoxelData:

	def __init__(self, subject, gray_matter_value):
		self.subject = subject
		self.gmvalue = gray_matter_value

	def __repr__(self):
		return 'VoxelData( Subject: ' + repr(self.subject) + ' , gmvalue: ' + repr(self.gmvalue) + ' )'

	def __str__(self):
		s = 'VoxelData(\n'
		for x in repr(self.subject).split('\n'):
			s += '    ' + x + '\n'
		s += '    Gray Matter Value: ' + repr(self.gmvalue) + '\n'
		s += ')\n'
		return s


class Voxel(Region):
	def size():
		return (1, 1, 1)


class Subject:

	diagnostics = ['NC', 'PC', 'MCI', 'AD']
	sexes = ['Unknown', 'Male', 'Female']
	apoe4s = ['Unknown', 'Yes', 'No']

	def __init__(self, identifier, niftiInputManager, diagnostic = None, age = None, sex = None, apoe4 = None, education = None, adcsfIndex = None):
		self.id = identifier
		self.gmdata = niftiInputManager
		self.diag = diagnostic
		self.age = age
		if sex == None:
			self.sex = 0
		else:
			self.sex = 2*sex - 1
		if apoe4 == None:
			self.apoe4 = 0
		else:
			self.apoe4 = 2*apoe4 - 1
		self.ed = education
		self.adcsf = adcsfIndex

	def __hash__(self):
		return hash(self.id)

	def __repr__(self):
		return 'Subject( ' + repr(self.id) + ' )'

	def __str__(self):
		s = 'Subject ' + repr(self.id) + ':\n'
		s += '    Diagnostic: '
		if self.diag == None:
			s += 'Unknown'
		else:
			s += Subject.diagnostics[self.diag]
		s += '\n    Age: '
		if self.age == None:
			s += 'Unknown'
		else:
			s += repr(self.age)
		s += '\n    Sex: ' + Subject.sexes[self.sex]
		s += '\n    APOE4 presence: ' + Subject.apoe4s[self.apoe4]
		s += '\n    Education level: '
		if self.ed == None:
			s += 'Unkown'
		else:
			s += repr(self.ed)
		s += '\n    AD-CSF index (t-tau) value: '
		if self.adcsf == None:
			s += 'Unknown'
		else:
			s += repr(self.adcsf)
		s += '\n'
		return s


