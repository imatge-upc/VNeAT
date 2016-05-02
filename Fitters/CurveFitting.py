from abc import ABCMeta

import numpy as np

from Utils.Documentation import docstring_inheritor



class abstractstatic(staticmethod):
	__slots__ = ()
	__isabstractmethod__ = True

	def __init__(self, function):
		super(abstractstatic, self).__init__(function)
		function.__isabstractmethod__ = True


class CurveFitter(object):
	'''Abstract class that implements the framework to develop curve fitting algorithms.
	'''
	__metaclass__ = docstring_inheritor(ABCMeta)
	__threshold = (1e-14 ** 2)

	NoIntercept = 0
	CorrectionIntercept = 1
	PredictionIntercept = 2


	def __init__(self, predictors=None, correctors=None, intercept=NoIntercept):
		'''[Constructor]

			Parameters:

				- predictors: NxR (2-dimensional) matrix (default None), representing the predictors,
					i.e., features to be used to try to explain/predict some observations (experimental
					data), where R is the number of predictors and N the number of elements for each
					predictor.

				- correctors: NxC (2-dimensional) matrix (default None), representing the covariates,
					i.e., features that (may) explain a part of the observational data in which we are
					not interested, where C is the number of correctors and N the number of elements
					for each corrector (the latter must be the same as that in the 'predictors' argu-
					ment).

				- intercept: one of CurveFitter.NoIntercept, CurveFitter.PredictionIntercept or
					CurveFitter.CorrectionIntercept (default CurveFitter.NoIntercept), indicating whether
					the intercept (a.k.a. homogeneous term) must be incorporated to the model or not, and
					if so, wheter it must be as a predictor or a corrector. In the last two cases, a column
					of ones will be added as the first column (feature) of the internal predictors/correctors
					matrix. Please notice that, if the matrix to which the columns of ones must precede
					does not have any elements, then this parameter will have no effect.
		'''

		if not predictors is None:
			predictors = np.array(predictors, dtype=np.float64)

			if len(predictors.shape) != 2:
				raise TypeError('Argument \'predictors\' must be a 2-dimensional matrix')

		if not correctors is None:
			correctors = np.array(correctors, dtype=np.float64)

			if len(correctors.shape) != 2:
				raise TypeError('Argument \'correctors\' must be a 2-dimensional matrix (or None)')

		if predictors is None:
			if correctors is None:
				self._crvfitter_correctors = np.zeros((0, 0))
				self._crvfitter_predictors = np.zeros((0, 0))
			else:
				if intercept == CurveFitter.PredictionIntercept:
					self._crvfitter_correctors = correctors
					self._crvfitter_predictors = np.ones((self._crvfitter_correctors.shape[0], 1))
				else:
					self._crvfitter_predictors = np.zeros((correctors.shape[0], 0))
					if intercept == CurveFitter.CorrectionIntercept:
						self._crvfitter_correctors = np.concatenate((np.ones((correctors.shape[0], 1)), correctors), axis = 1)
					else:
						self._crvfitter_correctors = correctors

		else:
			if correctors is None:
				if intercept == CurveFitter.CorrectionIntercept:
					self._crvfitter_predictors = predictors
					self._crvfitter_correctors = np.ones((self._crvfitter_predictors.shape[0], 1))
				else:
					self._crvfitter_correctors = np.zeros((predictors.shape[0], 0))
					if intercept == CurveFitter.PredictionIntercept:
						self._crvfitter_predictors = np.concatenate((np.ones((predictors.shape[0], 1)), predictors), axis = 1)
					else:
						self._crvfitter_predictors = predictors
			else:
				if correctors.shape[0] != predictors.shape[0]:
					raise ValueError('Correctors and predictors must have the same number of samples (same length in the first dimension)')

				if intercept == CurveFitter.CorrectionIntercept:
					self._crvfitter_correctors = np.concatenate((np.ones((correctors.shape[0], 1)), correctors), axis = 1)
					self._crvfitter_predictors = predictors
				elif intercept == CurveFitter.PredictionIntercept:
					self._crvfitter_predictors = np.concatenate((np.ones((predictors.shape[0], 1)), predictors), axis = 1)
					self._crvfitter_correctors = correctors
				else:
					self._crvfitter_correctors = correctors
					self._crvfitter_predictors = predictors

		C = self._crvfitter_correctors.shape[1]
		R = self._crvfitter_predictors.shape[1]
		self._crvfitter_correction_parameters = np.zeros((C, 0))
		self._crvfitter_prediction_parameters = np.zeros((R, 0))

	@property
	def correctors(self):
		'''Matrix of shape (N, C), representing the correctors of the model.
		'''
		return self._crvfitter_correctors.copy()

	@property
	def predictors(self):
		'''Matrix of shape (N, R), representing the predictors of the model.
		'''
		return self._crvfitter_predictors.copy()

	@property
	def features(self):
		'''Matrix of shape (N, C+R), representing the features (correctors and predictors) of the model.
			Note: do not modify the result (use its copy method if you need to treat this data).
		'''
		return np.array(list(self._crvfitter_correctors.T) + list(self._crvfitter_predictors.T)).T.copy()

	@property
	def correction_parameters(self):
		'''Array-like structure of shape (Kc, X1, ..., Xn), representing the correction parameters for which
			the correctors best explain the observational data passed as argument in the last call to 'fit',
			where Kc is the number of parameters for each variable in such observations, and X1, ..., Xn are
			the dimensions of the 'observations' argument in the last call to 'fit' (there are X1*...*Xn
			variables).
		'''
		return self._crvfitter_correction_parameters.copy().reshape(-1, *self._crvfitter_dims)

	@property
	def prediction_parameters(self):
		'''Array-like structure of shape (Kr, X1, ..., Xn), representing the prediction parameters for which
			the predictors best explain the observational data passed as argument in the last call to 'fit',
			where Kr is the number of parameters for each variable in such observations, and X1, ..., Xn are
			the dimensions of the 'observations' argument in the last call to 'fit' (there are X1*...*Xn
			variables).
		'''
		return self._crvfitter_prediction_parameters.copy().reshape(-1, *self._crvfitter_dims)

	def orthogonalize_correctors(self):
		'''Orthogonalizes each corrector in the structure w.r.t. all the previous np.ones. That is, for each
			column in the correctors matrix, its projection over the previous columns is computed and sub-
			tracted from it.

			Modifies:

				- Correctors: each column has been orthogonalized with respect to the previous np.ones.

			Returns:

				- Deorthogonalization matrix: A CxC (2-dimensional) upper triangular matrix that yields the
					original 'correctors' matrix when right-multiplied with the new 'correctors' matrix. That
					is, given the original 'correctors' matrix, OC, and the new, orthogonalized 'correctors'
					matrix, NC, the return value is a matrix, D, such that OC = NC x D (matrix multiplication).
		'''

		# Original 'correctors' matrix:
		# 	V = ( v_1 | v_2 | ... | v_C )

		# Gram-Schmidt:
		#	u_j = v_j - sum_{i < j} ( ( < u_i, v_j > / < u_i, u_i > ) * u_i ) # orthogonalize v_j with respect to every u_i, or equivalently, v_i, with i < j

		# New 'correctors' matrix (orthonormalized):
		#	U = ( u_1 | u_2 | ... | u_C )

		# Deorthogonalization matrix (upper triangular):
		#	D[i, j] =
		#			< u_i, v_j > / < u_i, u_i >,	if i < j
		#	 		1,								if i = j
		#	 		0,								if i > j

		C = self._crvfitter_correctors.shape[1]
		D = np.zeros((C, C))  # D[i, j] = 0, if i > j
		if (C == 0):
			return D

		threshold = self._crvfitter_correctors.shape[0] * CurveFitter.__threshold

		for i in xrange(C - 1):
			D[i, i] = 1.0  # D[i, j] = 1, if i = j

			u_i = self._crvfitter_correctors[:, i]
			norm_sq = u_i.dot(u_i)  # < u_i, u_i > = sq(||u_i||)

			if norm_sq < threshold:
				u_i[:] = 0.0  # Set whole vector to 0, since it is a linear combination of other vectors in the matrix
				# Notice that D[i, i] is set to 1, as requested (this means that the deorthogonalization will still
				# work, hopefully with a small enough precision error)
				continue

			for j in xrange(i + 1, C):  # for j > i
				v_j = self._crvfitter_correctors[:, j]

				D[i, j] = u_i.dot(v_j) / norm_sq  # D[i, j] = < u_i, v_j > / < u_i, u_i >, if i < j
				v_j -= D[
						   i, j] * u_i  # Orthogonalize v_j with respect to u_i (Gram-Schmidt, iterating over j instead of i)

		D[-1, -1] = 1.0  # D[i, j] = 1, if i = j

		return D

	def normalize_correctors(self):
		'''Normalizes the energy of each corrector (the magnitude of each feature interpreted as a vector,
			that is, the magnitude of each column of the internal correctors matrix).

			Modifies:

				- Correctors: each column has been normalized to have unit magnitude.

			Returns:

				- Denormalization matrix: A CxC (2-dimensional) diagonal matrix that yields the original
					'correctors' matrix when right-multiplied with the new 'correctors' matrix. That is,
					given the original 'correctors' matrix, OC, and the new, normalized 'correctors' matrix,
					NC, the return value is a diagonal matrix D such that OC = NC x D (matrix multiplication).
		'''

		# Original 'correctors' matrix:
		#	V = ( v_1 | v_2 | ... | v_C )

		# Normalization:
		#	u_j = v_j / ||v_j||

		# New 'correctors' matrix (normalized):
		#	U = ( u_1 | u_2 | ... | u_C )

		# Deorthogonalization matrix (diagonal):
		#	D[i, j] =
		#	 		||u_i||,	if i = j
		#	 		0,			if i != j

		C = self._crvfitter_correctors.shape[1]
		D = np.zeros((C, C))  # D[i, j] = 0, if i != j

		threshold = self._crvfitter_correctors.shape[0] * CurveFitter.__threshold

		for i in xrange(C):
			u_i = self._crvfitter_correctors[:, i]
			norm_sq = u_i.dot(u_i)
			if norm_sq >= threshold:
				D[i, i] = norm_sq ** 0.5  # D[i, j] = ||u_i||, if i = j
				u_i /= D[i, i]  # Normalization
			elif norm_sq != 0.0:
				u_i[:] = 0.0

		return D

	def orthonormalize_correctors(self):
		'''Orthogonalizes each corrector with respect to all the previous np.ones, and normalizes the results.
			This is equivalent to applying orthogonalize_correctors and normalize_correctors consecutively
			(in that same order), but slightly faster.

			Modifies:

				- Correctors: each column has been orthogonalized w.r.t. the previous np.ones, and normalized
					afterwards.

			Returns:

				- Deorthonormalization matrix: A CxC (2-dimensional) upper triangular matrix that yields the
					original 'correctors' matrix when right-multiplied with the new 'correctors' matrix. That
					is, given the original 'correctors' matrix, OC, and the new, orthonormalized 'correctors'
					matrix, NC, the return value is a matrix, D, such that OC = NC x D (matrix multiplication).
		'''

		# Original 'correctors' matrix:
		# 	V = ( v_1 | v_2 | ... | v_C )

		# Gram-Schmidt:
		#	u_j = v_j - sum_{i < j} ( < w_i, v_j > * w_i ) # orthogonalize v_j with respect to w_i, or equivalently, u_i or v_i with i < j
		#	w_j = u_j / (||u_j||) = u_j / sqrt(< u_j, u_j >) # normalize u_j

		# New 'correctors' matrix (orthonormalized):
		#	W = ( w_1 | w_2 | ... | w_C )

		# Deorthonormalization matrix (upper triangular):
		#	D[i, j] =
		#			< w_i, v_j >,		if i < j
		#	 		||u_i||,			if i = j
		#	 		0,					if i > j

		C = self._crvfitter_correctors.shape[1]
		D = np.zeros((C, C))  # D[i, j] = 0, if i > j

		threshold = self._crvfitter_correctors.shape[0] * CurveFitter.__threshold

		for i in xrange(C):
			u_i = self._crvfitter_correctors[:, i]

			norm_sq = u_i.dot(u_i)  # < u_i, u_i > = ||u_i||**2
			if norm_sq < threshold:
				u_i[:] = 0.0  # Set whole vector to 0, since it is a linear combination of other vectors in the matrix
				# Notice that D[i, i] is set to 0, which is exactly the same as ||u_i||, as requested (this means that
				# the deorthonormalization will still work, hopefully with a small enough precision error)
				continue

			D[i, i] = norm_sq ** 0.5  # D[i, j] = ||u_i||, if i = j
			u_i /= D[i, i]  # Normalize u_i, now u_i denotes w_i (step 2 of Gram-Schmidt)

			for j in xrange(i + 1, C):  # for j > i
				v_j = self._crvfitter_correctors[:, j]

				D[i, j] = u_i.dot(v_j)  # D[i, j] = < w_i, v_j >, if i < j
				v_j -= D[
						   i, j] * u_i  # Orthogonalize v_j with respect to w_i (step 1 of Gram-Schmidt, iterating over j instead of i)

		return D

	def orthogonalize_predictors(self):
		'''Orthogonalizes each predictor in the structure w.r.t. all the previous np.ones. That is, for each
			column in the predictors matrix, its projection over the previous columns is computed and sub-
			tracted from it.

			Modifies:

				- Regressors: each column has been orthogonalized with respect to the previous np.ones.

			Returns:

				- Deorthogonalization matrix: An RxR (2-dimensional) upper triangular matrix that yields the
					original 'predictors' matrix when right-multiplied with the new 'predictors' matrix. That
					is, given the original 'predictors' matrix, OR, and the new, orthogonalized 'predictors'
					matrix, NR, the return value is a matrix, D, such that OR = NR x D (matrix multiplication).
		'''

		# Original 'predictors' matrix:
		# 	V = ( v_1 | v_2 | ... | v_C )

		# Gram-Schmidt:
		#	u_j = v_j - sum_{i < j} ( ( < u_i, v_j > / < u_i, u_i > ) * u_i ) # orthogonalize v_j with respect to every u_i, or equivalently, v_i, with i < j

		# New 'predictors' matrix (orthonormalized):
		#	U = ( u_1 | u_2 | ... | u_C )

		# Deorthogonalization matrix (upper triangular):
		#	D[i, j] =
		#			< u_i, v_j > / < u_i, u_i >,	if i < j
		#	 		1,								if i = j
		#	 		0,								if i > j

		R = self._crvfitter_predictors.shape[1]
		D = np.zeros((R, R))  # D[i, j] = 0, if i > j
		if (R == 0):
			return D

		threshold = self._crvfitter_predictors.shape[0] * CurveFitter.__threshold

		for i in xrange(R - 1):
			D[i, i] = 1.0  # D[i, j] = 1, if i = j

			u_i = self._crvfitter_predictors[:, i]
			norm_sq = u_i.dot(u_i)  # < u_i, u_i > = sq(||u_i||)

			if norm_sq < threshold:
				u_i[:] = 0.0  # Set whole vector to 0, since it is a linear combination of other vectors in the matrix
				# Notice that D[i, i] is set to 1, as requested (this means that the deorthogonalization will still
				# work, hopefully with a small enough precision error)
				continue

			for j in xrange(i + 1, R):  # for j > i
				v_j = self._crvfitter_predictors[:, j]

				D[i, j] = u_i.dot(v_j) / norm_sq  # D[i, j] = < u_i, v_j > / < u_i, u_i >, if i < j
				v_j -= D[
						   i, j] * u_i  # Orthogonalize v_j with respect to u_i (Gram-Schmidt, iterating over j instead of i)

		D[-1, -1] = 1.0  # D[i, j] = 1, if i = j

		return D

	def normalize_predictors(self):
		'''Normalizes the energy of each predictor (the magnitude of each feature interpreted as a vector,
			that is, the magnitude of each column of the internal predictors matrix).

			Modifies:

				- Regressors: each column has been normalized to have unit magnitude.

			Returns:

				- Denormalization matrix: An RxR (2-dimensional) diagonal matrix that yields the original
					'predictors' matrix when right-multiplied with the new 'predictors' matrix. That is,
					given the original 'predictors' matrix, OR, and the new, normalized 'predictors' matrix,
					NR, the return value is a diagonal matrix D such that OR = NR x D (matrix multiplication).
		'''

		# Original 'predictors' matrix:
		#	V = ( v_1 | v_2 | ... | v_C )

		# Normalization:
		#	u_j = v_j / ||v_j||

		# New 'predictors' matrix (normalized):
		#	U = ( u_1 | u_2 | ... | u_C )

		# Deorthogonalization matrix (diagonal):
		#	D[i, j] =
		#	 		||u_i||,	if i = j
		#	 		0,			if i != j

		R = self._crvfitter_predictors.shape[1]
		D = np.zeros((R, R))  # D[i, j] = 0, if i != j

		threshold = self._crvfitter_predictors.shape[0] * CurveFitter.__threshold

		for i in xrange(R):
			u_i = self._crvfitter_predictors[:, i]
			norm_sq = u_i.dot(u_i)
			if norm_sq >= threshold:
				D[i, i] = norm_sq ** 0.5  # D[i, j] = ||u_i||, if i = j
				u_i /= D[i, i]  # Normalization
			elif norm_sq != 0.0:
				u_i[:] = 0.0

		return D

	def orthonormalize_predictors(self):
		'''Orthogonalizes each predictors with respect to all the previous np.ones, and normalizes the results.
			This is equivalent to applying orthonormalize_predictors and normalize_predictors consecutively
			(in that same order), but slightly faster.

			Modifies:

				- Regressors: each column has been orthogonalized w.r.t. the previous np.ones, and normalized
					afterwards.

			Returns:

				- Deorthonormalization matrix: An RxR (2-dimensional) upper triangular matrix that yields the
					original 'predictors' matrix when right-multiplied with the new 'predictors' matrix. That
					is, given the original 'predictors' matrix, OR, and the new, orthonormalized 'predictors'
					matrix, NR, the return value is a matrix, D, such that OR = NR x D (matrix multiplication).
		'''

		# Original 'predictors' matrix:
		# 	V = ( v_1 | v_2 | ... | v_C )

		# Gram-Schmidt:
		#	u_j = v_j - sum_{i < j} ( < w_i, v_j > * w_i ) # orthogonalize v_j with respect to w_i, or equivalently, u_i or v_i with i < j
		#	w_j = u_j / (||u_j||) = u_j / sqrt(< u_j, u_j >) # normalize u_j

		# New 'predictors' matrix (orthonormalized):
		#	W = ( w_1 | w_2 | ... | w_C )

		# Deorthonormalization matrix (upper triangular):
		#	D[i, j] =
		#			< w_i, v_j >,		if i < j
		#	 		||u_i||,			if i = j
		#	 		0,					if i > j

		R = self._crvfitter_predictors.shape[1]
		D = np.zeros((R, R))

		threshold = self._crvfitter_predictors.shape[0] * CurveFitter.__threshold

		for i in xrange(R):
			u_i = self._crvfitter_predictors[:, i]

			norm_sq = u_i.dot(u_i)  # < u_i, u_i > = ||u_i||**2
			if norm_sq < threshold:
				u_i[:] = 0.0  # Set whole vector to 0, since it is a linear combination of other vectors in the matrix
				# Notice that D[i, i] is set to 0, which is exactly the same as ||u_i||, as requested (this means that
				# the deorthonormalization will still work, hopefully with a small enough precision error)
				continue

			D[i, i] = norm_sq ** 0.5  # D[i, j] = ||u_i||, if i = j
			u_i /= D[i, i]  # Normalize u_i, now u_i denotes w_i (step 2 of Gram-Schmidt)

			for j in xrange(i + 1, R):  # for j > i
				v_j = self._crvfitter_predictors[:, j]

				D[i, j] = u_i.dot(v_j)  # D[i, j] = < w_i, v_j >, if i < j
				v_j -= D[
						   i, j] * u_i  # Orthogonalize v_j with respect to w_i (step 1 of Gram-Schmidt, iterating over j instead of i)

		return D

	def orthogonalize_all(self):
		'''Orthogonalizes each predictor w.r.t the others, all correctors w.r.t. the others, and all the
			predictors w.r.t. all the correctors.

			Modifies:

				- Correctors: each column has been orthogonalized with respect to the previous np.ones.
				- Regressors: each column has been orthogonalized with respect to all the columns in the correctors
					matrix and all the previous columns in the predictors matrix.

			Returns:

				- Deorthogonalization matrix: A (C+R)x(C+R) (2-dimensional) upper triangular matrix that yields the
					original 'correctors' and 'predictors' matrices when right-multiplied with the new 'correctors' and
					'predictors' matrices. More specifically, given the original 'correctors' matrix, OC, the original
					'predictors' matrix, OR, and the new, orthogonalized 'correctors' and 'predictors' matrices, NC
					and NR respectively, the return value is a matrix, D, such that (OC | OR) = (NC | NR) x D (matrix
					multiplication).
		'''

		# Original 'features' matrix:
		# 	V = (C | R) = ( v_1 | v_2 | ... | v_(C+R) )

		# Gram-Schmidt:
		#	u_j = v_j - sum_{i < j} ( ( < u_i, v_j > / < u_i, u_i > ) * u_i ) # orthogonalize v_j with respect to every u_i, or equivalently, v_i, with i < j

		# New 'features' matrix (orthonormalized):
		#	U = ( u_1 | u_2 | ... | u_(C+R) )

		# Deorthogonalization matrix (upper triangular):
		#	D[i, j] =
		#			< u_i, v_j > / < u_i, u_i >,	if i < j
		#	 		1,								if i = j
		#	 		0,								if i > j

		C = self._crvfitter_correctors.shape[1]
		R = self._crvfitter_predictors.shape[1]
		CR = C + R
		D = np.zeros((CR, CR))  # D[i, j] = 0, if i > j

		threshold = self._crvfitter_correctors.shape[0] * CurveFitter.__threshold

		for i in xrange(C):
			D[i, i] = 1.0  # D[i, j] = 1, if i = j

			u_i = self._crvfitter_correctors[:, i]
			norm_sq = u_i.dot(u_i)  # < u_i, u_i > = sq(||u_i||)

			if norm_sq < threshold:
				u_i[:] = 0.0  # Set whole vector to 0, since it is a linear combination of other vectors in the matrix
				# Notice that D[i, i] is set to 1, as requested (this means that the deorthogonalization will still
				# work, hopefully with a small enough precision error)
				continue

			for j in xrange(i + 1, C):
				v_j = self._crvfitter_correctors[:, j]

				D[i, j] = u_i.dot(v_j) / norm_sq  # D[i, j] = < u_i, v_j > / < u_i, u_i >, if i < j
				v_j -= D[i, j] * u_i

			for j in xrange(C, CR):
				v_j = self._crvfitter_predictors[:, j - C]

				D[i, j] = u_i.dot(v_j) / norm_sq  # D[i, j] = < u_i, v_j > / < u_i, u_i >, if i < j
				v_j -= D[i, j] * u_i

		D[C:, C:] = self.orthogonalize_predictors()  # R x R

		return D

	def normalize_all(self):
		'''Normalizes the energy of each corrector and each predictor (the magnitude of each feature
			interpreted as a vector, that is, the magnitude of each column of the internal correctors and
			predictors matrices).

			Modifies:

				- Correctors: each column has been normalized to have unit magnitude.
				- Regressors: each column has been normalized to have unit magnitude.

			Returns:

				- Denormalization matrix: A (C+R)x(C+R) (2-dimensional) diagonal matrix that yields the original
					'correctors' and 'predictors' matrices when right-multiplied with the new 'correctors' and
					'predictors' matrices. That is, given the original 'correctors' matrix, namely OC, the original
					'predictors' matrix, OR, and the new, normalized 'correctors' and 'predictors' matrices, NC and
					NR respectively, the return value is a diagonal matrix D such that (OC | OR) = (NC | NR) x D
					(matrix multiplication).
		'''

		# Deorthogonalization matrix (diagonal):
		#	D[i, j] =
		#	 		||u_i||,	if i = j
		#	 		0,			if i != j

		C = self._crvfitter_correctors.shape[1]
		R = self._crvfitter_predictors.shape[1]
		CR = C + R
		D = np.zeros((CR, CR))

		D[:C, :C] = self.normalize_correctors()
		D[C:, C:] = self.normalize_predictors()

		return D

	def orthonormalize_all(self):
		'''Orthogonalizes each predictor w.r.t the others, all correctors w.r.t. the others, and all the
			predictors w.r.t. all the correctors, and normalizes the results. This is equivalent to applying
			orthogonalize_all and normalize_all consecutively (in that same order), but slightly faster.

			Modifies:

				- Correctors: each column has been orthogonalized with respect to the previous np.ones and nor-
					malized afterwards.
				- Regressors: each column has been orthogonalized with respect to all the columns in the
					correctors matrix and all the previous columns in the predictors matrix, and normalized
					afterwards.

			Returns:

				- Deorthonormalization matrix: A (C+R)x(C+R) (2-dimensional) upper triangular matrix that yields
					the original 'correctors' and 'predictors' matrices when right-multiplied with the new
					'correctors and 'predictors' matrices. More specifically, given the original 'correctors'
					matrix, namely OC, the original 'predictors' matrix, OR, and the new, orthonormalized
					'correctors' and 'predictors' matrices, NC and NR respectively, the return value is a matrix,
					D, such that (OC | OR) = (NC | NR) x D (matrix multiplication).
		'''

		# Original 'features' matrix:
		# 	V = (C | R) = ( v_1 | v_2 | ... | v_(C+R) )

		# Gram-Schmidt:
		#	u_j = v_j - sum_{i < j} ( < w_i, v_j > * w_i ) # orthogonalize v_j with respect to w_i, or equivalently, u_i or v_i with i < j
		#	w_j = u_j / (||u_j||) = u_j / sqrt(< u_j, u_j >) # normalize u_j

		# New 'features' matrix (orthonormalized):
		#	W = ( w_1 | w_2 | ... | w_(C+R) )

		# Deorthonormalization matrix (upper triangular):
		#	D[i, j] =
		#			< w_i, v_j >,		if i < j
		#	 		||u_i||,			if i = j
		#	 		0,					if i > j

		C = self._crvfitter_correctors.shape[1]
		R = self._crvfitter_predictors.shape[1]
		CR = C + R
		D = np.zeros((CR, CR))

		threshold = self._crvfitter_correctors.shape[0] * CurveFitter.__threshold

		for i in xrange(C):
			u_i = self._crvfitter_correctors[:, i]

			norm_sq = u_i.dot(u_i)  # < u_i, u_i > = ||u_i||**2
			if norm_sq < threshold:
				u_i[:] = 0.0  # Set whole vector to 0, since it is a linear combination of other vectors in the matrix
				# Notice that D[i, i] is set to 0, which is exactly the same as ||u_i||, as requested (this means that
				# the deorthonormalization will still work, hopefully with a small enough precision error)
				continue

			D[i, i] = norm_sq ** 0.5  # D[i, j] = ||u_i||, if i = j
			u_i /= D[i, i]  # Normalize u_i, now u_i denotes w_i (step 2 of Gram-Schmidt)

			for j in xrange(i + 1, C):
				v_j = self._crvfitter_correctors[:, j]

				D[i, j] = u_i.dot(v_j)  # D[i, j] = < w_i, v_j >, if i < j
				v_j -= D[
						   i, j] * u_i  # Orthogonalize v_j with respect to w_i (step 1 of Gram-Schmidt, iterating over j instead of i)

			for j in xrange(C, CR):
				v_j = self._crvfitter_predictors[:, j - C]

				D[i, j] = u_i.dot(v_j)  # D[i, j] = < w_i, v_j >, if i < j
				v_j -= D[
						   i, j] * u_i  # Orthogonalize v_j with respect to w_i (step 1 of Gram-Schmidt, iterating over j instead of i)

		D[C:, C:] = self.orthonormalize_predictors()  # R x R

		return D

	@abstractstatic
	def __fit__(correctors, predictors, observations, *args, **kwargs):
		'''[Abstract method] Computes the correction and prediction parameters that best fit the observations.
			This method is not intended to be called outside the CurveFitter class.

			Parameters:

				- correctors: NxC (2-dimensional) matrix, representing the covariates, i.e., features that
					(may) explain a part of the observational data in which we are not interested, where C
					is the number of correctors and N the number of elements for each corrector.

				- predictors: NxR (2-dimensional) matrix, representing the predictors, i.e., features to be
					used to try to explain/predict the observations (experimental data), where R is the number
					of predictors and N the number of elements for each predictor (the latter is ensured to be
					the same as that in the 'correctors' argument).

				- observations: NxM (2-dimensional) matrix, representing the observational data, i.e., values
					obtained by measuring the variables of interest, whose behaviour is wanted to be explained
					by the correctors and predictors, where M is the number of variables and N the number of
					observations for each variable (the latter is ensured to be the same as those in the
					'correctors' and the 'predictors' arguments).

				- any other arguments will also be passed to the method in the subclass.

			Returns:

				- Correction parameters: (Kc)xM (2-dimensional) matrix, representing the parameters that best
					fit the correctors to the observations for each variable, where M is the number of variables
					(same as that in the 'observations' argument) and Kc is the number of correction parameters
					for each variable.

				- Regression parameters: (Kr)xM (2-dimensional) matrix, representing the parameters that best
					fit the predictors to the corrected observations for each variable, where M is the number
					of variables (same as that in the 'observations' argument) and Kr is the number of
					prediction parameters for each variable.


			[Developer notes]
				- Assertions regarding the size and type of the arguments have already been performed before
					the call to this method to ensure that the sizes of the arguments are coherent and the
					observations matrix has at least one element.

				- The 'correctors' and 'predictors' matrices may have zero elements, in which case the behaviour
					of the method is left to be decided by the subclass.

				- You may modify the 'observations' matrix if needed, but both the 'correctors' and the
					'predictors' arguments should be left unchanged.

				- The result should be returned as a tuple of 2 elements, containing the correction parameters
					in the first position and the prediction parameters in the second position.

				- Although it is defined as a static method here, this method supports a non-static implementation.
		'''
		raise NotImplementedError

	def fit(self, observations, *args, **kwargs):
		'''Computes the correction and prediction parameters that best fit the observations.

			Parameters:

				- observations: array-like structure of shape (N, X1, ..., Xn), representing the observational
					data, i.e., values obtained by measuring the variables of interest, whose behaviour is wanted
					to be explained by the correctors and predictors in the system, where M = X1*...*Xn is the
					number of variables and N the number of observations for each variable.

				- any other arguments will be passed to the __fit__ method.

			Modifies:

				- [created] Correction parameters: array-like structure of shape (Kc, X1, ..., Xn), representing
					the parameters that best fit the correctors to the observations, where X1, ..., Xn are the
					original dimensions of the 'observations' argument and Kc is the number of correction parameters
					for each variable.

				- [created] Regression parameters: array-like structure of shape (Kr, X1, ..., Xn), representing
					the parameters that best fit the predictors to the observations, where X1, ..., Xn are the
					original dimensions of the 'observations' argument and Kr is the number of prediction parameters
					for each variable.
		'''
		obs = np.array(observations, dtype=np.float64)
		dims = obs.shape
		self._crvfitter_dims = dims[1:]
		if dims[0] != self._crvfitter_predictors.shape[0]:
			raise ValueError('Observations and features (correctors and/or predictors) have incompatible sizes')

		if 0 in dims:
			raise ValueError('There are no elements in argument \'observations\'')

		obs = obs.reshape(dims[0], -1)
		self._crvfitter_correction_parameters, self._crvfitter_prediction_parameters = self.__fit__(
			self._crvfitter_correctors, self._crvfitter_predictors, obs, *args, **kwargs)

	@abstractstatic
	def __predict__(predictors, prediction_parameters, *args, **kwargs):
		'''[Abstract method] Computes a prediction using the predictors together with the prediction parameters.
			This method is not intended to be called outside the CurveFitter class.

			Parameters:

				- predictors: NxR (2-dimensional) matrix, representing the predictors, i.e., features to be used
					to try to explain/predict the observations (experimental data), where R is the number of
					predictors and N the number of elements for each predictor.

				- prediction_parameters: (Kr)xM (2-dimensional) matrix, representing the parameters that best fit
					the predictors to the corrected observations for each variable, where M is the number of
					variables and Kr is the number of prediction parameters for each variable.

				- any other arguments will also be passed to the method in the subclass.

			Returns:

				- Prediction: NxM (2-dimensional) matrix, containing N predicted values for each of the M variables.

			[Developer notes]
				- Assertions regarding the size and type of the arguments have already been performed before the
					call to this method to ensure that the sizes of the arguments are coherent and both, the
					'predictors' and the 'prediction_parameters' matrices have at least one element each.

				- Both the 'predictors' and the 'prediction_parameters' arguments should be left unchanged.

				- Although it is defined as a static method here, this method supports a non-static implementation.
		'''
		raise NotImplementedError

	def predict(self, predictors=None, prediction_parameters=None, *args, **kwargs):
		'''Computes a prediction using the predictors together with the prediction parameters.

			Parameters:

				- predictors: NxR (2-dimensional) matrix (default None), representing the predictors, i.e., features
					to be used to try to explain/predict the observations (experimental data), where R is the number
					of predictors and N the number of elements for each predictor. If set to None, the predictors of
					the instance will be used.

				- prediction_parameters: array-like structure of shape (Kr, X1, ..., Xn) (default None), representing
					the parameters to fit the predictors to the corrected observations for each variable, where M =
					X1*...*Xn is the number of variables and Kr is the number of prediction parameters for each
					variable. If set to None, the prediction parameters obtained in the last call to 'fit' will be
					used.

				- any other arguments will be passed to the __predict__ method.

			Returns:

				- Prediction: array-like structure of shape (N, X1, ..., Xn), containing N predicted values for each of
					the M = X1*...*Xn variables.
		'''
		if predictors is None:
			preds = self._crvfitter_predictors
			if 0 in preds.shape:
				raise AttributeError('There are no predictors in this instance')
		else:
			preds = np.array(predictors, dtype=np.float64)
			if len(preds.shape) != 2:
				raise TypeError('Argument \'predictors\' must be a 2-dimensional matrix')
			if 0 in preds.shape:
				raise ValueError('There are no elements in argument \'predictors\'')

		if prediction_parameters is None:
			params = self._crvfitter_prediction_parameters
			dims = (1,) + self._crvfitter_dims
		else:
			params = np.array(prediction_parameters, dtype=np.float64)
			# Keep original dimensions (to reset dimensions of prediction)
			dims = params.shape
			# Make matrix 2-dimensional
			params = params.reshape(dims[0], -1)

		if 0 in dims:
			raise ValueError('There are no elements in argument \'prediction_parameters\'')

		prediction = self.__predict__(preds, params, *args, **kwargs)

		# Restore original dimensions (except for the first axis)
		return prediction.reshape(-1, *dims[1:])

	@abstractstatic
	def __correct__(observations, correctors, correction_parameters, *args, **kwargs):
		'''[Abstract method] Computes the values of the data after accounting for the correctors by using the
			correction parameters.
			This method is not intended to be called outside the CurveFitter class.

			Parameters:

				- observations: NxM (2-dimensional) matrix, representing the observational data to be corrected,
					i.e., the values obtained by measuring the variables of interest from which we want to subtract
					the contribution of the correctors, where M is the number of variables and N the number of
					observations for each variable.

				- correctors: NxC (2-dimensional) matrix, representing the covariates, i.e., features that (may)
					explain a part of the observational data in which we are not interested, where C is the number
					of correctors and N the number of elements for each corrector.

				- correction_parameters: (Kc)xM (2-dimensional) matrix, representing the parameters that best fit
					the correctors to the observations for each variable, where M is the number of variables and Kc
					is the number of correction parameters for each variable.

				- any other arguments will also be passed to the method in the subclass.

			Returns:

				- Corrected data: NxM (2-dimensional) matrix, containing the observational data after having sub-
					tracted the contribution of the correctors by using the correction parameters.

			[Developer notes]
				- Assertions regarding the size and type of the arguments have already been performed before the
					call to this method to ensure that the sizes of the arguments are coherent and all, the
					'observations', 'correctors', and 'correction_parameters' matrices have at least one element
					each.

				- The 'correction_parameters' matrix may have zero elements.

				- You may modify the 'observations' matrix if needed. However, both the 'correctors' and the
					'correction_parameters' arguments should be left unchanged.

				- Although it is defined as a static method here, this method supports a non-static implementation.
		'''
		raise NotImplementedError

	def correct(self, observations, correctors=None, correction_parameters=None, *args, **kwargs):
		'''Computes the values of the data after accounting for the correctors by using the correction parameters.

			Parameters:

				- observations: array-like matrix of shape (N, X1, ..., Xn), representing the observational data,
					i.e., values obtained by measuring the variables of interest, whose behaviour is wanted to be
					explained by the correctors and predictors in the system, where M = X1*...*Xn is the number of
					variables and N the number of observations for each variable.

				- correctors: NxC (2-dimensional) matrix (default None), representing the covariates, i.e., features
					that (may) explain a part of the observational data in which we are not interested, where C is
					the number of correctors and N the number of elements for each corrector. If set to None, the
					internal correctors will be used.

				- correction_parameters: array-like structure of shape (Kc, X1, ..., Xn) (default None), representing
					the parameters to fit the correctors to the observations for each variable, where M = X1*...*Xn
					is the number of variables and Kc the number of correction parameters for each variable. If set
					to None, the correction parameters obtained in the last call to 'fit' will be used.

				- any other arguments will be passed to the __correct__ method.

			Returns:

				- Corrected data: array-like matrix of shape (N, X1, ..., Xn), containing the observational data
					after having subtracted the contribution of the correctors by using the correction parameters.
		'''

		## Treat observations
		obs = np.array(observations, dtype=np.float64)
		# Keep original dimensions (to reset dimensions of corrected data)
		dims = obs.shape
		# Make matrix 2-dimensional
		obs = obs.reshape(dims[0], -1)

		# Check correctness of matrix
		if 0 in dims:
			return np.zeros(dims)

		## Treat correctors
		if correctors is None:
			cors = self._crvfitter_correctors
			if 0 in cors.shape:
				return observations
		else:
			cors = np.array(correctors, dtype=np.float64)
			if len(cors.shape) != 2:
				raise TypeError('Argument \'correctors\' must be a 2-dimensional matrix')
			if 0 in cors.shape:
				raise ValueError('There are no elements in argument \'correctors\'')

		if obs.shape[0] != cors.shape[0]:
			raise ValueError('The dimensions of the observations and the correctors are incompatible')

		## Treat correction parameters
		if correction_parameters is None:
			params = self._crvfitter_correction_parameters
		else:
			params = np.array(correction_parameters, dtype=np.float64)
			params = params.reshape(params.shape[0], -1)

			if 0 in params.shape:
				raise ValueError('There are no elements in argument \'correction_parameters\'')

		if obs.shape[1] != params.shape[1]:
			raise ValueError('The dimensions of the observations and the correction parameters are incompatible')

		## Compute corrected data
		corrected_data = self.__correct__(obs, cors, params, *args, **kwargs)

		# Restore original dimensions
		return corrected_data.reshape(dims)


class AdditiveCurveFitter(CurveFitter):
	'''CurveFitter subclass for the cases in which the following model is assumed:
			Y = fp(C, CP) + fp(R, RP) + err
		where 'Y' is the set of observations, 'C' is the set of correctors, 'CP' is the set of correction parameters,
		'R' is the set of predictors, 'RP' the set of prediction parameters, 'fp' is an arbitrary function (more
		specifically, it is the prediction function), and 'err' is the residual error of the model.
	'''

	def __correct__(self, observations, correctors, correction_parameters, *args, **kwargs):
		'''The correction is performed by applying the following formula:
				Y - fp(C, CP)
			where 'Y' is the 'observations' argument, 'C' is the 'correctors' argument, 'CP' is the 'correction_parameters'
			argument and 'fp' is the '__predict__' method of the subclass. Any additional arguments are also passed
			to the 'fp' function.
			(See the '__predict__' and '__correct__' abstract methods in CurveFitter)
		'''
		return observations - self.__predict__(correctors, correction_parameters, *args, **kwargs)


#TODO: use metaclass instead of method

def MixedFitter(correction_fitter_type, prediction_fitter_type):

	class MixedFitter(CurveFitter):

		def __init__(self, predictors = None, correctors = None, intercept = CurveFitter.NoIntercept):
			self._mixedfitter_correction_fitter = correction_fitter_type(predictors = None, correctors = correctors, intercept = intercept)
			self._mixedfitter_prediction_fitter = prediction_fitter_type(predictors = predictors, correctors = None, intercept = intercept)
			self._crvfitter_correctors = self._mixedfitter_correction_fitter._crvfitter_correctors
			self._crvfitter_predictors = self._mixedfitter_prediction_fitter._crvfitter_predictors
			self._crvfitter_correction_parameters = self._mixedfitter_correction_fitter._crvfitter_correction_parameters
			self._crvfitter_prediction_parameters = self._mixedfitter_prediction_fitter._crvfitter_prediction_parameters
		
		def fit(self, observations, **kwargs):
			kwargs_correction = {}
			kwargs_prediction = {}
			correction_varnames = set(self._mixedfitter_correction_fitter.__fit__.func_code.co_varnames)
			prediction_varnames = set(self._mixedfitter_prediction_fitter.__fit__.func_code.co_varnames)
			for (arg, value) in kwargs.iteritems():
				if arg in correction_varnames:
					correction_varnames.remove(arg)
					kwargs_correction[arg] = value
				if arg in prediction_varnames:
					prediction_varnames.remove(arg)
					kwargs_prediction[arg] = value

			self._mixedfitter_correction_fitter.fit(observations = observations, **kwargs_correction)
			self._crvfitter_correction_parameters = self._mixedfitter_correction_fitter._crvfitter_correction_parameters
			obs = self.correct(observations)
			
			self._mixedfitter_prediction_fitter.fit(observations = obs, **kwargs_prediction)
			self._crvfitter_prediction_parameters = self._mixedfitter_prediction_fitter._crvfitter_prediction_parameters

		def correct(self, observations, correctors = None, correction_parameters = None, *args, **kwargs):
			return self._mixedfitter_correction_fitter.correct(observations = observations, correctors = correctors, correction_parameters = correction_parameters, *args, **kwargs)

		def predict(self, predictors = None, prediction_parameters = None, *args, **kwargs):
			return self._mixedfitter_prediction_fitter.predict(predictors = predictors, prediction_parameters = prediction_parameters, *args, **kwargs)

		def getattr(self, attr_name):
			try:
				return getattr(self._mixedfitter_prediction_fitter, attr_name)
			except AttributeError:
				return getattr(self._mixedfitter_correction_fitter, attr_name)

		def set_pred_attr(self, attr_name, attr_value):
			setattr(self._mixedfitter_prediction_fitter, attr_name, attr_value)

		def set_corr_attr(self, attr_name, attr_value):
			setattr(self._mixedfitter_correction_fitter, attr_name, attr_value)

	return MixedFitter
