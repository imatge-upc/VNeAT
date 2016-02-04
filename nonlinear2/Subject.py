
class Subject:

	Diagnostics = ['NC', 'PC', 'MCI', 'AD']
	Sexes = ['Unknown', 'Male', 'Female']
	APOE4s = ['Unknown', 'Yes', 'No']

	Attributes = \
		 '''- Subject.Diagnostic: A 0-based index indicating the diagnostic of the subject (see Subject.Diagnostics).
			    None if it was not indicated.

			- Subject.Age: integer that indicates the age of the subject.
			    None if it was not indicated.

			- Subject.Sex: integer indicating the genre of the subject.
			    -1 if Female.
			     1 if Male.
			     0 if Not Indicated.

			- Subject.APOE4: integer indicating if the apoe-4 protein is present in the subject's organism.
			    -1 if Not Present.
			     1 if Present.
			     0 if Not Indicated.

			- Subject.Education: integer that indicates the level of academical education of the subject.
			    None if it was not indicated.

			- Subject.ADCSFIndex: float thar represents the AD-CSF index (t-tau) value of the subject.
			    None if it was not indicated.
		 '''
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
		setattr(self, Subject.Sex, sex if not sex is None else 0)
		setattr(self, Subject.APOE4, apoe4 if not apoe4 is None else 0)
		setattr(self, Subject.Education, education)
		setattr(self, Subject.ADCSFIndex, adcsfIndex)

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
		return [getattr(self, attr_name) for attr_name in attribute_list]

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
