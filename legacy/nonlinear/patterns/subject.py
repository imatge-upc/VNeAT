
class Subject:

	diagnostics = ['NC', 'PC', 'MCI', 'AD']
	sexes = ['Unknown', 'Male', 'Female']
	apoe4s = ['Unknown', 'Yes', 'No']

	def __init__(self, identifier, niftiInputManager, diagnostic = None, age = None, sex = None, apoe4 = None, education = None, adcsfIndex = None):
		self._id = identifier
		self._gmdata = niftiInputManager
		self.diag = diagnostic
		self.age = age
		self.sex = sex if sex != None else 0
		self.apoe4 = apoe4 if apoe4 != None else 0
		self.ed = education
		self.adcsf = adcsfIndex

	@property
	def id(self):
	    return self._id
	
	@property
	def gmdata(self):
	    return self._gmdata

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