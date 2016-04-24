import numpy as np

from Processors import GLMProcessor as GLMP


class Test(GLMP):
	def __init__(self):
		self._glmprocessor_perp_norm_option = 0

	@staticmethod
	def _generate_upper_triangular_matrix(K):
		M = np.random.random((K, K))
		for i in xrange(K):
			for j in xrange(i):
				M[i, j] = 0.0
		return M

	def test(self, K):
		Gamma = self._generate_upper_triangular_matrix(K)
		Identity = np.identity(K)
		GammaInv = self._glmprocessor_compute_original_parameters(Gamma, Identity)

		print 'Maximum difference between Gamma*GammaInv and Identity:', np.abs(Gamma.dot(GammaInv) - Identity).max()
		print 'Maximum difference between GammaInv*Gamma and Identity:', np.abs(GammaInv.dot(Gamma) - Identity).max()

		return Gamma, GammaInv