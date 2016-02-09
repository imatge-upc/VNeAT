from abc import ABCMeta, abstractmethod
from Documentation import docstring_inheritor
from niftiIO import NiftiReader, Region
from numpy import array as nparray, zeros, savez_compressed as npsave, load as npload
from sys import stdout

class Processor:
	__metaclass__ = docstring_inheritor(ABCMeta)

	class Results:

		def __init__(self, regression_parameters, correction_parameters, fitting_scores):
			self._regression_parameters = regression_parameters
			self._correction_parameters = correction_parameters
			self._fitting_scores = fitting_scores
		
		@property
		def regression_parameters(self):
			return self._regression_parameters
		
		@property
		def correction_parameters(self):
			return self._correction_parameters

		@property
		def fitting_scores(self):
			return self._fitting_scores

		def __str__(self):
			s = 'Results:'
			s += '\n    Correction parameters:' + reduce(lambda x, y: x + '\n    ' + y, repr(self._correction_parameters).split('\n'))
			s += '\n\n    Regression parameters:' + reduce(lambda x, y: x + '\n    ' + y, repr(self._regression_parameters).split('\n'))
			s += '\n\n    Fitting scores:\n' + reduce(lambda x, y: x + '\n    ' + y, repr(self._fitting_scores).split('\n'))
			return s

	def __init__(self, subjects, regressors, correctors = [], user_defined_parameters = ()):
		self._processor_subjects = subjects
		self._processor_regressors = nparray(map(lambda subject: subject.get(regressors), subjects))
		self._processor_correctors = nparray(map(lambda subject: subject.get(correctors), subjects))
		if (len(user_defined_parameters) != 0):
			self._processor_fitter = self.__fitter__(user_defined_parameters)
		else:
			self._processor_fitter = self.__fitter__(self.__read_user_defined_parameters__(regressors, correctors))

		self._processor_progress = 0.0
		self._processor_mem_usage = 500.0

	@property
	def subjects(self):
		'''List of subjects (Subject objects) of this instance.
		'''
		return self._processor_subjects

	@property
	def correctors(self):
		'''Matrix of correctors of this instance.

			NxC (2-dimensional) matrix, representing the values of the features of the subjects that are to be
			used as correctors in the fitter, where N is the number of subjects and C the number of correctors.
		'''
		return self._processor_correctors

	@property
	def regressors(self):
		'''Matrix of regressors of this instance.

			NxR (2-dimensional) matrix, representing the values of the features of the subjects that are to be
			used as regressors in the fitter, where N is the number of subjects and R the number of regressors.
		'''

		return self._processor_regressors

	@property
	def progress(self):
		'''Progress (percentage of data processed) of the last call to process. If it has not been called yet,
			this property will be 0.0, whereas if the task is already completed it will be 100.0.
		'''
		return int(self._processor_progress)/100.0
	

	@abstractmethod
	def __fitter__(self, user_defined_parameters):
		'''[Abstract method] Initializes the fitter to be used to process the data.
			This method is not intended to be used outside the Processor class.

			Parameters:

			    - user_defined_parameters: tuple of ints, containing the additional parameters (apart from correctors
			    	and regressors) necessary to succesfully initialize a new instance of a fitter (see 
			        __read_user_defined_parameters__ method).

			Returns:

			    - A fully initalized instance of a subclass of the CurveFitter class.

			[Developer notes]

			    - This method is always called in the initialization of an instance of the Processing class, which means
			        that you can consider it as the __init__ of the subclass (you can declare the variables that you
			        would otherwise initialize in the __init__ method of the subclass here, in the __fitter__ method). 
		'''

		raise NotImplementedError

	@abstractmethod
	def __user_defined_parameters__(self, fitter):
		'''[Abstract method] Gets the additional parameters obtained from the user and used by the __fitter__ method
			to initialize the fitter.
			This method is not intended to be used outside the Processor class.

			Parameters:

			    - fitter: a fully initialized instance of a subclass of the CurveFitter class.

			Returns:

			    - A tuple with the values of the additional parameters (apart from the correctors and regressors) that
			        have been used to succesfully initialize and use the fitter of this instance.

			[Developer notes]

			    - This method must output a tuple that is recognizable by the __fitter__ method when fed to it as the
			        'user_defined_parameters' argument, allowing it to initialize a new fitter equal to the one of this
			        instance (see __read_user_defined_parameters__).
		'''

		raise NotImplementedError

	@property
	def user_defined_parameters(self):
		return self.__user_defined_parameters__(self._processor_fitter)

	@abstractmethod
	def __read_user_defined_parameters__(self, regressor_names, corrector_names):
		'''[Abstract method] Read the additional parameters (apart from correctors and regressors)
			necessary to succesfully initialize a new instance of the fitter from the user.

			Parameters:

			    - regressor_names: iterable of subject attributes (e.g. Subject.ADCSFIndex) that represent the
			        names of the features to be used as regressors.

			    - corrector_names: iterable of subject attributes (e.g. Subject.Age) that represent the names of
			        the features to be used as correctors.

			Returns:
			    
			    - A tuple of numerical elements, containing the coded additional parameters (apart from correctors
			    	and regressors) necessary to succesfully initialize a new instance of a fitter; this tuple will
					be past as is to the __fitter__ method.

			[Developer notes]

			    - This method is responsible for obtaining the values of such additional parameters from the user.
			        
			    - Please, make use of the 'getter' methods implemented in this class for such purpose. This will
			        allow future subclasses to implement additional features (such as a GUI) by just overriding the
			        'getter' methods, consequently making it easier to maintain, expand and provide more features
			        together with a larger functionality for the abstract subclasses of Processing without requiring
			        any additional work from the developers that implement the concrete subclasses of the same class.

			    - When calling the 'getter' methods, make sure you use the 'super' built-in function, so that the Method
			        Resolution Order is dynamically adapted and you get to use the methods implemented in the bottom-most
			        subclass of Processing in the inheritance tree.

			    - The 'getter' methods are of the form __get***__, where *** denotes the value to be obtained from the user.
			        Here is a potentially non-exhaustive list of such methods: __getint__, __getfloat__, __getoneof__,
			        __getoneinrange__, __getyesorno__, ...
		'''

		raise NotImplementedError

	def __processor_chunks(self, gmdata_readers):

		num_subjects = len(gmdata_readers)

		iterators = map(lambda gmdata_reader: gmdata_reader.chunks(self._processor_mem_usage/num_subjects), gmdata_readers)
		try:
			while True:
				reg = iterators[0].next()
				chunkset = [reg.data]
				chunkset += [it.next().data for it in iterators[1:]]
				yield Region(reg.coords, nparray(chunkset))
		except StopIteration:
			pass

	def __processor_update_progress(self, prog_inc):
		self._processor_progress += prog_inc
		print '\r  ' + str(int(self._processor_progress)/100.0) + '%',
		if self._processor_progress == 10000.0:
			print
		stdout.flush()

	def process(self, x1 = 0, x2 = None, y1 = 0, y2 = None, z1 = 0, z2 = None, mem_usage = None, evaluation_kwargs = {}, *args, **kwargs):
		if mem_usage != None:
			self._processor_mem_usage = float(mem_usage)

		gmdata_readers = map(lambda subject: NiftiReader(subject.gmfile, x1 = x1, y1 = y1, z1 = z1, x2 = x2, y2 = y2, z2 = z2), self._processor_subjects)
		dims = gmdata_readers[0].dims

		# Initialize progress
		self._processor_progress = 0.0
		total_num_voxels = dims[0]*dims[1]*dims[2]
		prog_inc = 10000./total_num_voxels

		# Get the results of the first chunk to initialize dimensions of the solution matrices
		
		# Get frist chunk and fit the parameters
		chunks = self.__processor_chunks(gmdata_readers)
		chunk = chunks.next()

		self._processor_fitter.fit(chunk.data, *args, **kwargs)

		# Get the parameters and the dimensions of the solution matrices
		cparams = self._processor_fitter.correction_parameters
		rparams = self._processor_fitter.regression_parameters
		cpdims = (cparams.shape[0],) + dims
		rpdims = (rparams.shape[0],) + dims
		
		# Initialize solution matrices
		fitting_scores = zeros(dims)
		correction_parameters = zeros(cpdims)
		regression_parameters = zeros(rpdims)

		# Assign first chunk's solutions to solution matrices
		dx, dy, dz = cparams.shape[1:]
		correction_parameters[:, :dx, :dy, :dz] = cparams
		regression_parameters[:, :dx, :dy, :dz] = rparams
		fitting_scores[:dx, :dy, :dz] = self._processor_fitter.evaluate_fit(chunk.data, **evaluation_kwargs)

		# Update progress
		self.__processor_update_progress(prog_inc*dx*dy*dz)

		# Now do the same for the rest of the chunks
		for chunk in chunks:
			# Get relative (to the solution matrices) coordinates of the chunk
			x, y, z = chunk.coords
			x -= x1
			y -= y1
			z -= z1

			# Get chunk data and its dimensions
			cdata = chunk.data
			dx, dy, dz = cdata.shape[1:]

			# Fit the parameters to the data in the chunk
			self._processor_fitter.fit(cdata, *args, **kwargs)

			# Get the optimal parameters and insert them in the solution matrices
			correction_parameters[:, x:x+dx, y:y+dy, z:z+dz] = self._processor_fitter.correction_parameters
			regression_parameters[:, x:x+dx, y:y+dy, z:z+dz] = self._processor_fitter.regression_parameters

			# Evaluate the fit for the voxels in this chunk and store them
			fitting_scores[x:x+dx, y:y+dy, z:z+dz] = self._processor_fitter.evaluate_fit(cdata, **evaluation_kwargs)

			# Update progress
			self.__processor_update_progress(prog_inc*dx*dy*dz)

		if self.progress != 100.0:
			self.__processor_update_progress(10000.0 - self._processor_progress)
		return Processor.Results(regression_parameters, correction_parameters, fitting_scores)

	#TODO: Document properly
	def __curve__(self, fitter, regressor, regression_parameters):
		'''Computes a prediction from the regressor and the regression_parameters. If not overridden, this method
			calls the 'predict' function of the fitter passing as arguments the regressors and regression parameters
			as they are. Please, override this method if this is not the desired behavior.
			This method is not intended to be called outside the Processor class.


		'''
		return fitter.predict(regressor, regression_parameters)

	#TODO: Document properly
	def curve(self, regression_parameters, x1 = None, x2 = None, y1 = None, y2 = None, z1 = None, z2 = None, t1 = None, t2 = None, tpoints = 20):
		'''Computes tpoints predicted values in the axis of the regressor from t1 to t2 by using the results of
			a previous execution for each voxel in the region [x1:x2, y1:y2, z1:z2]. (Only valid for one regressor).


		'''
		if x1 is None:
			x1 = 0
		if x2 is None:
			x2 = regression_parameters.shape[1]
		if y1 is None:
			y1 = 0
		if y2 is None:
			y2 = regression_parameters.shape[2]
		if z1 is None:
			z1 = 0
		if z2 is None:
			z2 = regression_parameters.shape[3]
		
		if t1 is None:
			t1 = self._processor_regressors.min()
		if t2 is None:
			t2 = self._processor_regressors.max()

		rparams = regression_parameters[:, x1:x2, y1:y2, z1:z2]

		regs = zeros((tpoints, 1))
		step = float(t2 - t1)/tpoints
		t = t1
		for i in range(tpoints):
			regs[i][0] = t
			t += step

		return regs.T[0], self.__curve__(self._processor_fitter, regs, rparams)


	#TODO: Document properly
	def __corrected_values__(self, fitter, observations, correction_parameters):
		return fitter.correct(observations = observations, correction_parameters = correction_parameters)

	#TODO: Document properly
	def corrected_values(self, correction_parameters, x1 = None, x2 = None, y1 = None, y2 = None, z1 = None, z2 = None):
		if x1 is None:
			x1 = 0
		if x2 is None:
			x2 = correction_parameters.shape[1]
		if y1 is None:
			y1 = 0
		if y2 is None:
			y2 = correction_parameters.shape[2]
		if z1 is None:
			z1 = 0
		if z2 is None:
			z2 = correction_parameters.shape[3]

		gmdata_readers = map(lambda subject: NiftiReader(subject.gmfile, x1 = x1, y1 = y1, z1 = z1, x2 = x2, y2 = y2, z2 = z2), self._processor_subjects)
		dims = gmdata_readers[0].dims

		correction_parameters = correction_parameters[:, x1:x2, y1:y2, z1:z2]
		corrected_data = zeros(tuple([len(gmdata_readers)]) + dims)

		for chunk in self.__processor_chunks(gmdata_readers):
			# Get relative (to the solution matrix) coordinates of the chunk
			x, y, z = chunk.coords
			x -= x1
			y -= y1
			z -= z1

			# Get chunk data and its dimensions
			cdata = chunk.data
			dx, dy, dz = cdata.shape[1:]

			corrected_data[:, x:(x+dx), y:(y+dy), z:(z+dz)] = self.__corrected_values__(self._processor_fitter, cdata, correction_parameters[:, x:(x+dx), y:(y+dy), z:(z+dz)])

		return corrected_data

	#TODO
	def fit_score(self, fitting_scores, x1 = None, x2 = None, y1 = None, y2 = None, z1 = None, z2 = None):
		if x1 is None:
			x1 = 0
		if x2 is None:
			x2 = fitting_scores.shape[0]
		if y1 is None:
			y1 = 0
		if y2 is None:
			y2 = fitting_scores.shape[1]
		if z1 is None:
			z1 = 0
		if z2 is None:
			z2 = fitting_scores.shape[2]

		return fitting_scores[x1:x2, y1:y2, z1:z2]


	#TODO: define more of these

	@staticmethod
	def __processor_get__(obtain_input_from, apply_function, try_ntimes, default_value, show_text, show_error_text):
		if try_ntimes <= 0:
			try_ntimes = -1

		while try_ntimes != 0:
			s = obtain_input_from(show_text)
			if not s:
				print 'Default value selected.'
				return default_value
			else:
				try:
					return apply_function(s)
				except Exception as exc:
					print show_error_text(exc)

			if try_ntimes < 0:
				print 'Infinite',
			else:
				try_ntimes -= 1
				print try_ntimes,
			print 'attempts left.',

			if try_ntimes == 0:
				print 'Default value selected.'
			else:
				print 'Please, try again.'

		return default_value

	@staticmethod
	def __getint__(
		default_value = None,
		try_ntimes = 3,
		lower_limit = None,
		upper_limit = None,
		show_text = 'Please, enter an integer number (or leave blank to set by default): ',
		obtain_input_from = raw_input,
	):
		def nit(s, lower = lower_limit, upper = upper_limit):
			x = int(s)
			if (not (lower is None)) and x < lower:
				raise ValueError('The value must be greater than or equal to ' + str(lower))
			if (not (upper is None)) and x >= upper:
				raise ValueError('The value must be smaller than ' + str(upper))
			return x

		return Processor.__processor_get__(
			obtain_input_from,
			nit,
			try_ntimes,
			default_value,
			show_text,
			lambda e: 'Could not match input with integer number: ' + str(e)
		)

	@staticmethod
	def __getfloat__(
		default_value = None,
		try_ntimes = 3,
		lower_limit = None,
		upper_limit = None,
		show_text = 'Please, enter a real number (or leave blank to set by default): ',
		obtain_input_from = raw_input,
	):
		def olfat(s, lower = lower_limit, upper = upper_limit):
			x = float(s)
			if (not (lower is None)) and x < lower:
				raise ValueError('The value must be greater than or equal to ' + str(lower))
			if (not (upper is None)) and x >= upper:
				raise ValueError('The value must be smaller than ' + str(upper))
			return x

		return Processor.__processor_get__(
			obtain_input_from,
			olfat,
			try_ntimes,
			default_value,
			show_text,
			lambda e: 'Could not match input with real number: ' + str(e)
		)

	@staticmethod
	def __getoneof__(
		option_list,
		default_value = None,
		try_ntimes = 3,
		show_text = 'Please, select one of the following (enter index, or leave blank to set by default):',
		obtain_input_from = raw_input,
	):
		opt_list = list(option_list)
		lol = len(opt_list)
		lslol = len(str(lol))
		right_justify = lambda s: ' '*(lslol - len(str(s))) + str(s)

		new_show_text = show_text
		for i in range(lol):
			new_show_text += '\n  ' + right_justify(i) + ':  ' + str(opt_list[i])
		new_show_text += '\nYour choice: '

		def get_index(s, ls = lol):
			index = int(s)
			if index < 0 or index >= ls:
				raise IndexError('Index ' + s + ' is out of the accepted range [0, ' + str(ls) + '].')

		index = Processor.__processor_get__(
			obtain_input_from,
			get_index,
			try_ntimes,
			None,
			new_show_text,
			lambda e: 'Could not match input with index: ' + str(e)
		)
		if index == None:
			return default_value

		return opt_list[index]

	@staticmethod
	def __getoneinrange__(
		start,
		end,
		step = 0,
		default_value = None,
		try_ntimes = 3,
		show_text = 'Please, enter a number in the range',
		obtain_input_from = raw_input
	):
		if show_text == 'Please, enter a number in the range':
			show_text += ' [' + str(start) + ', ' + str(end) + ')'
			if step > 0:
				show_text += ' with a step of ' + str(step)
			show_text += '(or leave blank to set by default): '

		def inrange(s, start = start, end = end, step = step):
			f = float(s)
			if f >= end or f < start:
				raise ValueError('Input value is not in specified range.')
			if step > 0:
				# round number to its nearest step
				num_step = int((f - start)/step + 0.5) # round(x) = floor(x + 0.5) = int(x + 0.5)
				f = start + num_step*step
			return f

		return Processor.__processor_get__(
			obtain_input_from,
			inrange,
			try_ntimes,
			default_value,
			show_text,
			lambda e: 'Could not read input value: ' + str(e)
		)

	@staticmethod
	def __getyesorno__(
		default_value = None,
		try_ntimes = 3,
		show_text = 'Select yes (Y/y) or no (N/n), or leave blank to set by default: ',
		obtain_input_from = raw_input
	):
		def yesorno(s2):
			s = s2.strip()
			if s == 'y' or s == 'Y':
				return True
			if s == 'n' or s == 'N':
				return False
			raise ValueError('Option not recognized.')

		return Processor.__processor_get__(
			obtain_input_from,
			yesorno,
			try_ntimes,
			default_value,
			show_text,
			lambda e: 'Could not match input with any of the options: ' + str(e)
		)







