from abc import ABCMeta, abstractmethod
from Documentation import docstring_inheritor
from niftiIO import NiftiReader, Region
from numpy import array as nparray, savez_compressed as npsave, load as npload

class Processor:
	__metaclass_ = docstring_inheritor()

	def __init__(self, type, subjects, correctors, regressors):
		self._subjects = subjects
		self._mem_usage = 100.0
		self._correctors = nparray(map(lambda subject: subject.get(correctors), subjects))
		self._regressors = nparray(map(lambda subject: subject.get(regressors), subjects))

	@property
	def subjects(self):
		return self._subjects

	@property
	def correctors(self):
		return self._correctors

	@property
	def regressors(self):
		return self._regressors

	def __chunks(self, gmdata_readers, mem_usage):
		if mem_usage != None:
			self._mem_usage = float(mem_usage)

		num_subjects = len(gmdata_readers)

		iterators = map(lambda gmdata_reader: gmdata_reader.chunks(self._mem_usage/num_subjects), gmdata_readers)
		try:
			while True:
				reg = iterators[0].next()
				chunkset = [reg.data]
				chunkset += [it.next().data for it in iterators[1:]]
				yield Region(reg.coords, nparray(chunkset))
		except StopIteration:
			pass

	def process(self, x1 = 0, x2 = None, y1 = 0, y2 = None, z1 = 0, z2 = None, mem_usage = None):
		gmdata_readers = map(lambda subject: NiftiReader(subject.gmfile, x1 = x1, y1 = y1, z1 = z1, x2 = x2, y2 = y2, z2 = z2), self._subjects)
		for chunk in self.__chunks(gmdata_readers, mem_usage):
			#TODO: Fucking finish this thing! :D

class Results:

	def __init__(self, fitter, correctors, correction_parameters, regressors, regression_parameters, *args, **kwargs):
		self._cvfitter = fitter(regressors = regressors, correctors = correctors)
		self._fitter = fitter
		self._correctors = nparray(correctors)
		self._regressors = nparray(regressors)
		self._correction_parameters = correction_parameters
		self._regression_parameters = regression_parameters
		self._args = args
		self.__kwargs = kwargs

	def curve(self):
		return self._cvfitter.predict(regression_parameters = self._regression_parameters)

	def corrected_data(self, data):
		cvfitter = self._cvfitter.correct(regression_parameters = self._regression_parameters, observations = data)

	def save(self, filename):
		#TODO: Change this
		npsave(filename, object = nparray([self]))

	@staticmethod
	def load(filename):
		#TODO: Change this
		with npload(filename) as data:
			r =  data['object'][0]
		return r

