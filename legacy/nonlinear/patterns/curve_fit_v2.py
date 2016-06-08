from numpy import array as nparray, ones
from sklearn.linear_model import LinearRegression as LR


class GLM:
    ''' Class that implements the General Linear Method.

        This method assumes the following situation:
            - There are M (random) variables whose behaviour we want to explain
            - Each of the M variables has been measured N times, obtaining thus
              an NxM matrix of observations (i-th column contains the N observa-
              tions for i-th variable)
            - There are K regressors that might explain the behaviour of the M
              variables in an additive manner, i.e., a ponderated sum of the K
              regressors might fit each of the variables
            - Each of the K regressors has been measured at the same moments in
              which the M variables were measured, giving thus a NxK matrix where
              the i-th column represents the N observations of the i-th regressor

        In this situation, the relationship of the different elements can be ex-
        pressed as follows:

            OBS(NxM) = MODEL(NxK) * PARAMS(KxM) + eps(NxM),

        where OBS denotes the NxM matrix containing the N observations of each of
        the M variables, MODEL denotes the NxK matrix containing the N observations
        of each of the K regressors, PARAMS denotes the KxM matrix of ponderation
        coefficients (one for each variable and regressor, that is, the amplitude
        each regressor has in each variable), and eps denotes the error commited
        when making the aforementioned assumptions, i.e., a NxM matrix that contains
        the data that is left unexplained after accounting for all the regressors
        in the model.

        This class provides the tools to orthogonalize each of the regressors in
        the matrix with respect to the ones in the previous columns, and to esti-
        mate the ponderation coefficients (the PARAMS matrix) so that the energy
        of the error (the MSE) is minimized.
    '''

    def __init__(self, xdata, ydata, homogeneous=True):
        '''Constructor.

            Parameters:

                - xdata: NxK (2-dimensional) matrix, representing the model (independent data),
                    where N > 0 is the number of samples and K > 0 the number of regressors.

                - ydata: vector of length N / NxM (2-dimensional) matrix, representing the obser-
                    vations (dependent data), where M > 0 is the number of variables.

                - homogeneous: boolean (default True), indicating whether the homogeneous term
                    must be incorporated to the model or not. If so, a column of ones will be
                    added to the left of the model represented by xdata

            Raises:

                - AssertionError if any of the dimension constraints are violated

            Returns:

                - A new instance of the GLM class
        '''

        xdata = nparray(xdata, dtype=float)
        assert len(xdata.shape) == 2 and xdata.shape[1] != 0

        if homogeneous:
            dims = xdata.shape
            self._xdata = ones((dims[0], dims[1] + 1))
            self._xdata[:, 1:] = xdata
        else:
            self._xdata = xdata

        self.ydata = ydata
        self.__threshold = self._xdata.shape[0] * (1e-14 ** 2)

    def __setattr__(self, name, value):
        if name == 'ydata':
            ydata = nparray(value, dtype=float)
            assert len(ydata.shape) <= 2 and self._xdata.shape[0] == ydata.shape[0]
            if len(ydata.shape) == 2:
                assert ydata.shape[1] != 0
        self.__dict__[name] = value

    @property
    def xdata(self):
        '''Matrix of shape (N, K), representing the model matrix of the system.
        '''
        return self._xdata

    @property
    def opt_params(self):
        '''Matrix of shape (K, M), containing the optimum parameters for the current model matrix (xdata)
            and observation matrix (ydata). Only created after a call to optimize().
        '''

        return self._opt_params

    @staticmethod
    def predict(xdata, theta):
        '''Prediction function used in GLM.

            Parameters:

                - xdata: matrix of shape (N, K), representing the model matrix of the system, where N is
                    the number of samples and K the number of regressors.

                - theta: vector of length K / matrix of shape (K, M), representing the ponderation coef-
                    ficients of the system, that is, the PARAMS matrix, being K the number of regressors
                    and M the number of variables.

            Returns:

                - vector of length N / matrix of shape (N, M), result of computing the expression
                    xdata * theta (matrix multiplication)
        '''

        return xdata.dot(theta)  # same as numpy.dot(theta, self.xdata[:K])

    def orthogonalize(self):
        '''Orthogonalizes each regressor in self w.r.t. all the previous ones. That is, for each
           column in self.xdata, its projection over the previous rows is computed and subtracted
           from it.

            Modifies:

                - self.xdata: each column has been orthogonalized with respect to the previous ones.
                    Note that column 0 is never modified.

            Returns:

                - None
        '''

        for i in range(self._xdata.shape[1] - 1):
            u = self._xdata[:, i]
            norm_sq = u.dot(u)
            if norm_sq < self.__threshold:
                u[:] = 0.0
                continue
            u2 = u / norm_sq
            for j in range(i + 1, self._xdata.shape[1]):
                v = self._xdata[:, j]
                v -= v.dot(u) * u2

    def normalize(self):
        '''Normalizes the energy of each feature (the magnitude of each regressor interpreted as a
            vector, that is, the magnitude of each column of the model in the system, xdata)

            Modifies:

                - self.xdata: each column has been normalized to have unit magnitude.

            Returns:

                - None
        '''

        for i in range(self._xdata.shape[1]):
            u = self._xdata[:, i]
            norm_sq = u.dot(u)
            if norm_sq >= self.__threshold:
                u /= norm_sq ** 0.5
            elif norm_sq != 0.0:
                u[:] = 0.0

    def orthonormalize(self):
        '''Orthogonalizes each regressor with respect to all the previous ones, and normalizes the
            results. This is equivalent to applying orthogonalize and normalize consecutively (in that
            same order), but slightly faster.

            Modifies:

                - self.xdata: each column has been orthogonalized w.r.t. the previous ones, and normalized
                    afterwards.

            Returns:

                - None

        '''

        for i in range(self._xdata.shape[1]):
            u = self._xdata[:, i]
            norm_sq = u.dot(u)
            if norm_sq < self.__threshold:
                u[:] = 0.0
                continue
            u /= norm_sq ** 0.5  # Normalize u
            for j in range(i + 1, self._xdata.shape[1]):
                v = self._xdata[:, j]
                v -= v.dot(u) * u  # Orthogonalize v with respect to u

    def optimize(self, sample_weight=None, num_threads=-1):
        '''Computes optimal ponderation coefficients so that the error's energy is minimized.

            Parameters:

                - sample_weight: (optional) vector of length N, representing the ponderation of the error
                    term for each sample, where N is the number of samples (or observations).
                    This value must be consistent with the data in the GLM instace.

                - num_threads: (default -1) the number of threads to be used by the algorithm, -1 indicates
                    to use all CPUs.

            Modifies:

                - [created/modified] self.opt_params: vector of length K / matrix of shape (K, M)
                    Represents the optimal coefficients obtained by the algorithm, being M the number of variables
                    and K the number of regressors in the system, including the homogeneous term if it has been re-
                    quested (if M = 1, self.opt_params will be a vector; otherwise, it will be a matrix with the
                    specified shape).

            Returns:

                - None
        '''
        curve = LR(fit_intercept=False, normalize=False, copy_X=True, n_jobs=num_threads)
        curve.fit(self._xdata, self.ydata, sample_weight)
        self._opt_params = curve.coef_.T
