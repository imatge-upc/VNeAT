from abc import ABCMeta, abstractmethod
from Documentation import docstring_inheritor
from niftiIO import NiftiReader, Region
from numpy import array as nparray, zeros, savez_compressed as npsave, load as npload

class Processor:
	__metaclass__ = docstring_inheritor(ABCMeta)

	class Results:
		__metaclass__ = docstring_inheritor(ABCMeta)

		@abstractmethod
		def save(self):
			raise NotImplementedError

		@abstractmethod
		def load(self, filename):
			raise NotImplementedError

		# TODO: Finish defining this class
		def curve(self, ...):
			return None # Define it, seriously, do it now ;)

	def __init__(self, subjects, regressors, correctors = []):
		self._subjects = subjects
		self._correctors = nparray(map(lambda subject: subject.get(correctors), subjects))
		self._regressors = nparray(map(lambda subject: subject.get(regressors), subjects))
		self._fitter = self._fitter_type(regressors, correctors, *args, **kwargs)
		self._fitter.orthonormalize_all()
		self.progress = 100.0

	@property
	def subjects(self):
		return self._subjects

	@property
	def correctors(self):
		return self._correctors

	@property
	def regressors(self):
		return self._regressors

	@abstractmethod
	def __fitter__(self, correctors, regressors):
		'''[Abstract method] Returns the fitter to be used to process the data.
			This method is not intended to be used outside the Processor class.

			Parameters:

			    - correctors: NxC (2-dimensional) matrix, representing the features of the subjects to be used as
			        correctors in the fitter, where N is the number of subjects and C the number of correctors.

			    - regressors: NxR (2-dimensional) matrix, representing the features of the subjects to be used as
			        regressors in the fitter, where N is the number of subjects and R the number of regressors.

			Returns:

			    - instance of a subclass of the CurveFitter class, already initialized (except for the additional
			    	parameters that are adjustable afterwards).
		'''

		raise NotImplementedError

	@abstractmethod
	def __results__(self, correctors, )

	def __chunks(self, gmdata_readers):

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

	def process(self, x1 = 0, x2 = None, y1 = 0, y2 = None, z1 = 0, z2 = None, mem_usage = None, evaluation_kwargs = {}, *args, **kwargs):
		if mem_usage != None:
			self._mem_usage = float(mem_usage)

		gmdata_readers = map(lambda subject: NiftiReader(subject.gmfile, x1 = x1, y1 = y1, z1 = z1, x2 = x2, y2 = y2, z2 = z2), self._subjects)
		dims = gmdata_readers[0].dims

		# Initialize progress
		self.progress = 0.0
		total_num_voxels = dims[0]*dims[1]*dims[2]
		prog_inc = 10000./total_num_voxels

		# Get the results of the first chunk to initialize dimensions of the solution matrices
		
		# Get frist chunk and fit the parameters
		chunks = self.__chunks(gmdata_readers)
		chunk = chunks.next()
		self._fitter.fit(chunk.data, *args, **kwargs)

		# Get the parameters and the dimensions of the solution matrices
		cparams = self._fitter.correction_parameters
		rparams = self._fitter.regression_parameters
		cpdims = cparams.shape[0] + dims
		rpdims = rparams.shape[0] + dims
		
		# Initialize solution matrices
		fitting_scores = zeros(dims)
		correction_parameters = zeros(cpdims)
		regression_parameters = zeros(rpdims)

		# Assign first chunk's solutions to solution matrices
		dx, dy, dz = cparams.shape[1:]
		correction_parameters[:, :dx, :dy, :dz] = cparams
		regression_parameters[:, :dx, :dy, :dz] = rparams
		fitting_scores[:dx, :dy, :dz] = self._fitter.evaluate_fit(chunk.data, **evaluation_kwargs)

		# Update progress
		self.progress += prog_inc*dx*dy*dz

		# Now do the same for the rest of the chunks
		for chunk in chunks:
			# Get relative (to the solution matrices) coordinates of the chunk
			x, y, z = chunk.coords
			x -= x1
			y -= y1
			z -= z1

			# Get chunk data and its dimensions
			cdata = chunk.data
			dx, dy, dz = cdata.shape

			# Fit the parameters to the data in the chunk
			self._fitter.fit(cdata, *args, **kwargs)

			# Get the optimal parameters and insert them in the solution matrices
			correction_parameters[:, x:x+dx, y:y+dy, z:z+dz] = self._fitter.correction_parameters
			regression_parameters[:, x:x+dx, y:y+dy, z:z+dz] = self._fitter.regression_parameters

			# Evaluate the fit for the voxels in this chunk and store them
			fitting_scores[:, x:x+dx, y:y+dy, z:z+dz] = self._fitter.evaluate_fit(cdata, **evaluation_kwargs)

			# Update progress
			self.progress += prog_inc*dx*dy*dz

		# TODO: Return something that can reproduce the obtained results and show them
		return Results(self._fitter, )

# TODO: Revise this (see the class Results INSIDE Processor - is that even legal? xD -)
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

