from excelIO import get_rows
from niftiIO import NiftiSetManager as NiftiSM
from os.path import basename

class DataManager:

	def __init__(self, filenames, excel_file):
		self.subj_set = SubjectSet()
		for r in get_rows(excel_file):
			self.subj_set.add(Subject(r['id'],
									  r['diag'],
									  r['age'],
									  r['sex'],
									  r['apoe4_bin'],
									  r['escolaridad'],
									  r['ad_csf_index_ttau']))
		self.subj_set.normalize()
		self.nsm = NiftiSM(filenames)

	def voxels(self, *args, **kwargs):
		subjects = map(lambda fn: self.subj_set.get(DataManager.filename_to_id(fn)), self.nsm.filenames)
		for voxel in self.nsm.voxels(*args, **kwargs):
			yield SuperVoxel(voxel.coords, (VoxelData(subjects[i], voxel.data[i]) for i in range(len(voxel.data))))

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


class SuperVoxel(list):

	def __init__(self, coords, *args, **kwargs):
		list.__init__(self, *args, **kwargs)
		self.coords = coords

	def __repr__(self):
		return 'SuperVoxel( coords: ' + repr(self.coords) + ' , data: ' + list.__repr__(self) + ' )'

	def __str__(self):
		return repr(self)



class Subject:

	diagnostics = ['NC', 'PC', 'MCI', 'AD']
	sexes = ['Unknown', 'Male', 'Female']
	apoe4s = ['Unknown', 'Yes', 'No']

	def __init__(self, identifier, diagnostic = None, age = None, sex = None, apoe4 = None, education = None, adcsfIndex = None):
		self.id = identifier
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


class SubjectSet(dict):
	
	def add(self, subject):
		self[subject.id] = subject

	def get(self, subject_id):
		return self[subject_id]

	def normalize(self):
		return



