class ProgramError(Exception):

	def __init__(self, msg = 'An unknown error occured', original_exception = None):
		Exception.__init__(self)
		self._msg = msg
		self._orig_exception = original_exception

	@property
	def msg(self):
	    return self._msg
	
	@property
	def original_exception(self):
	    return self._orig_exception
	
	def __repr__(self):
		s = 'ProgramError ( ' + self._msg
		if self._orig_exception != None:
			s += ' ; ' + repr(self._orig_exception)
		s += ' )'
		return s

	def __str__(self):
		s = self._msg
		if self._orig_exception != None:
			s += ' ( ' + repr(self._orig_exception) + ' )'
		return s


class InputOutputError(ProgramError):
	pass

class ReadError(InputOutputError):
	def __init__(self, filename, msg = None, original_exception = None):
		if msg == None:
			msg = 'Input error while reading file \'' + filename + '\''
		ProgramError.__init__(self, msg = msg, original_exception = original_exception)

class ParseError(ReadError):
	def __init__(self, filename, original_exception = None):
		msg = 'Error while parsing input of file \'' + filename + '\''
		ReadError.__init__(self, filename, msg = msg, original_exception = original_exception)

class WriteError(InputOutputError):
	def __init__(self, filename, msg = None, original_exception = None):
		if msg == None:
			msg = 'Output error while writing file \'' + filename + '\''
		ProgramError.__init__(self, msg = msg, original_exception = original_exception)
