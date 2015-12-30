
class Subject:

	Diagnostics = ['NC', 'PC', 'MCI', 'AD']
	Sexes = ['Unknown', 'Male', 'Female']
	APOE4s = ['Unknown', 'Yes', 'No']

	Diagnostic = 'diag'
	Age = 'age'
	Sex = 'sex'
	APOE4 = 'apoe4'
	Education = 'ed'
	ADCSFIndex = 'adcsf'

	def __init__(self, identifier, graymatter_filename, diagnostic = None, age = None, sex = None, apoe4 = None, education = None, adcsfIndex = None):
		self._id = identifier
		self._gmfile = graymatter_filename
		setattr(self, Subject.Diagnostic, diagnostic)
		setattr(self, Subject.Age, age)
		setattr(self, Subject.Sex, sex if sex != None else 0)
		setattr(self, Subject.APOE4, apoe4 if apoe4 != None else 0)
		setattr(self, Subject.Education, education)
		setattr(self, Subject.ADCSFIndex, adcsfIndex)

	@property
	def id(self):
		return self._id

	@property
	def gmfile(self):
		return self._gmfile

	def get(self, attribute_list):
		return [getattr(self, attr_name) for attr_name in attribute_list]

	def __hash__(self):
		return hash(self.id)

	def __repr__(self):
		return 'Subject( ' + repr(self.id) + ' )'

	def __str__(self):
		diag, age, sex, apoe4, ed, adcsf = self.get(Subject.Diagnostic, Subject.Age, Subject.Sex, Subject.APOE4, Subject.Education, Subject.ADCSFIndex)
		s = 'Subject ' + repr(self.id) + ':\n'
		s += '    Diagnostic: '
		if diag == None:
			s += 'Unknown'
		else:
			s += Subject.Diagnostics[diag]
		s += '\n    Age: '
		if age == None:
			s += 'Unknown'
		else:
			s += repr(age)
		s += '\n    Sex: ' + Subject.Sexes[sex]
		s += '\n    APOE4 presence: ' + Subject.APOE4s[apoe4]
		s += '\n    Education level: '
		if ed == None:
			s += 'Unkown'
		else:
			s += repr(ed)
		s += '\n    AD-CSF index (t-tau) value: '
		if adcsf == None:
			s += 'Unknown'
		else:
			s += repr(adcsf)
		s += '\n'
		return s
