import niftiIO as nio
import numpy as np

class Subject(object):

	Diagnostics = ['NC', 'PC', 'MCI', 'AD']
	Sexes = ['Unknown', 'Male', 'Female']
	APOE4s = ['Unknown', 'Yes', 'No']

	class Attribute:
		def __init__(self, name, description = 'Undefined subject attribute'):
			self._description = description
			self._name = str(name)

		@property
		def description(self):
			return self._description

		def __repr__(self):
			return 'Subject.' + self._name

		def __str__(self):
			return self._name

	Diagnostic = Attribute('Diagnostic', 'A 0-based index indicating the diagnostic of the subject (see Subject.Diagnostics). None if it was not indicated.')
	Age = Attribute('Age', 'An integer that indicates the age of the subject. None if it was not indicated.')
	Sex = Attribute('Sex', 'An integer indicating the genre of the subject (-1 if Female, 1 if Male, 0 if Not Indicated).')
	APOE4 = Attribute('APOE-4', 'An integer indicating if the apoe-4 protein is present in the subject\'s organism (-1 if Not Present, 1 if Present, 0 if Not Indicated).')
	Education = Attribute('Education', 'An integer that indicates the level of academical education of the subject. None if it was not indicated.')
	ADCSFIndex = Attribute('AD-CSF Index', 'A float that represents the AD-CSF index (t-tau) value of the subject. None if it was not indicated.')

	Attributes = [Diagnostic, Age, Sex, APOE4, Education, ADCSFIndex]
	for index in xrange(len(Attributes)):
		Attributes[index].index = index

	def __init__(self, identifier, graymatter_filename, diagnostic = None, age = None, sex = None, apoe4 = None, education = None, adcsfIndex = None):
		self._id = identifier
		self._gmfile = graymatter_filename
		self._attributes = [None]*len(Subject.Attributes)

		self._attributes[Subject.Diagnostic.index] = diagnostic
		self._attributes[Subject.Age.index] = age
		self._attributes[Subject.Sex.index] = sex if not sex is None else 0
		self._attributes[Subject.APOE4.index] = apoe4 if not apoe4 is None else 0
		self._attributes[Subject.Education.index] = education
		self._attributes[Subject.ADCSFIndex.index] = adcsfIndex

	@property
	def id(self):
		return self._id

	@property
	def gmfile(self):
		return self._gmfile

	def get(self, attribute_list):
		'''Retrieves the specified attributes from the subject's data.

			Parameters:

			    - attribute_list: iterable containing the attributes that must be retrieved from the subject.
			        See Subject.Attributes to obtain a list of available attributes.

			Returns:

				- list containing the values of the attributes specified in the 'attribute_list' argument,
				    in the same order.
		'''
		return map(lambda attr: self._attributes[attr.index], attribute_list)

	def __hash__(self):
		return hash(self.id)

	def __repr__(self):
		return 'Subject( ' + repr(self.id) + ' )'

	def __str__(self):
		diag, age, sex, apoe4, ed, adcsf = self.get(Subject.Diagnostic, Subject.Age, Subject.Sex, Subject.APOE4, Subject.Education, Subject.ADCSFIndex)
		s = 'Subject ' + repr(self.id) + ':\n'
		s += '    Diagnostic: '
		if diag is None:
			s += 'Unknown'
		else:
			s += Subject.Diagnostics[diag]
		s += '\n    Age: '
		if age is None:
			s += 'Unknown'
		else:
			s += repr(age)
		s += '\n    Sex: ' + Subject.Sexes[sex]
		s += '\n    APOE4 presence: ' + Subject.APOE4s[apoe4]
		s += '\n    Education level: '
		if ed is None:
			s += 'Unkown'
		else:
			s += repr(ed)
		s += '\n    AD-CSF index (t-tau) value: '
		if adcsf is None:
			s += 'Unknown'
		else:
			s += repr(adcsf)
		s += '\n'
		return s



class chunks:
	def __init__(self, subject_list, x1 = 0, x2 = None, y1 = 0, y2 = None, z1 = 0, z2 = None, mem_usage = None):
		if mem_usage is None:
			mem_usage = 512.0
		self._gmdata_readers = map(lambda subject: nio.NiftiReader(subject.gmfile, x1 = x1, y1 = y1, z1 = z1, x2 = x2, y2 = y2, z2 = z2), subject_list)
		self._dims = self._gmdata_readers[0].dims
		self._num_subjects = np.float64(len(self._gmdata_readers))
		self._iterators = map(lambda gmdata_reader: gmdata_reader.chunks(mem_usage/self._num_subjects), self._gmdata_readers)

	@property
	def dims(self):
		return self._dims

	@property
	def num_subjects(self):
		return int(self._num_subjects)

	def __iter__(self):
		return self

	def next(self):
		reg = self._iterators[0].next() # throws StopIteration if there are not more chunks
		chunkset = [reg.data]
		chunkset += [it.next().data for it in self._iterators[1:]]
		return nio.Region(reg.coords, np.array(chunkset, dtype = np.float64))



