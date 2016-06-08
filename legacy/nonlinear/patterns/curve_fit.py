from numpy import array as nparray
from scipy.optimize import minimize


class GLM:
    ''' Class that implements the General Linear Method.

        This method assumes the following situation:
            - There are M (random) variables whose behaviour we want to explain
            - Each of the M variables has been measured N times, obtaining thus
              an MxN matrix of observations (i-th row contains the N observations
              for i-th variable)
            - There are K regressors that might explain the behaviour of the M
              variables in an additive manner, i.e., a ponderated sum of the K
              regressors might fit each of the variables
            - Each of the K regressors has been measured at the same moments in
              which the M variables were measured, giving thus a KxN matrix where
              the i-th row represents the N observations of the i-th regressor

        In this situation, the relationship of the different elements can be ex-
        pressed as follows:

            OBS(MxN) = PARAMS(MxK) * MODEL(KxN) + eps(MxN),

        where OBS denotes the MxN matrix containing the N observations of each of
        the M variables, MODEL denotes the KxN matrix containing the N observations
        of each of the K regressors, PARAMS denotes the MxK matrix of ponderation
        coefficients (one for each variable and regressor, that is, the amplitude
        each regressor has in each variable), and eps denotes the error commited
        when making the aforementioned assumptions, i.e., a MxN matrix that contains
        the data that is left unexplained after accounting for all the regressors
        in the model.

        This class provides the tools to orthogonalize each of the regressors in
        the matrix with respect to the ones in the previous columns, and to esti-
        mate the ponderation coefficients (the PARAMS matrix) so that the energy
        of the error (the MSE) is minimized.
    '''

    def __init__(self, xdata, ydata):
        '''Constructor.

            Parameters:

                - xdata: KxN (2-dimensional) matrix, representing the model (independent data),
                    where N > 0 is the number of samples and K > 0 the number of regressors.

                - ydata: vector of length N / MxN (2-dimensional) matrix, representing the obser-
                    vations (dependent data), where M > 0 is the number of variables.

            Modifies:

                - [created] self.xdata: matrix
                    Copy of the parameter xdata

                - [created] self.ydata: vector / matrix
                    Copy of the parameter ydata

            Raises:

                - AssertionError if any of the dimension constraints are violated

            Returns:

                - A new instance of the GLM class
        '''

        self.xdata = nparray(xdata, dtype=float)
        self.ydata = nparray(ydata, dtype=float)

        assert len(self.xdata.shape) == 2 and self.xdata.shape[1] != 0
        assert len(self.ydata.shape) <= 2
        if len(self.ydata.shape) == 1:
            assert self.xdata.shape[1] == self.ydata.shape[0]
        else:
            assert self.xdata.shape[1] == self.ydata.shape[1]

        self.__sigma = None

    def predict(self, theta):
        '''Prediction function used in GLM.

            Parameters:

                - theta: array-like with M * K elements, representing the ponderation coefficients of
                    the system, that is, the PARAMS matrix (in any shape, but ensuring that there are
                    exactly K*M elements and they are ordered BY ROWS - first row 1, then row 2, up to
                    row M -), being K the number of regressors and M the number of variables.

            Returns:

                - array-like structure with the same shape as that of self.ydata
                    result of computing the expression
                        -> reshape(theta, (M, K)) * xdata, if M > 1 (returns a matrix)
                        -> reshape(theta, (K)) * xdata, if M = 1 (returns vector)
        '''

        # result should be an array-like structure with shape (M, N) or just (N)
        # xdata is a matrix with shape (K, N)
        # theta must be of shape (M, K) or just (K)
        K = self.xdata.shape[0]
        if len(self.ydata.shape) == 2:
            M = self.ydata.shape[0]
            if theta.shape != (M, K):
                theta = theta.reshape((M, K), order='C')
        elif len(theta.shape) != 1:
            theta = theta.reshape(K)

        return theta.dot(self.xdata[:K])  # same as numpy.dot(theta, self.xdata[:K])

    def __error_energy(self, theta):
        error = (self.ydata - self.predict(theta)).reshape(-1)
        if self.__sigma != None:
            error *= self.__sigma
        return sum(error ** 2)

    def orthogonalize(self):
        '''Orthogonalizes each regressor in self w.r.t. all the previous ones. That is, for each
           row in self.xdata, its projection over the previous rows is computed and subtracted
           from it.

            Modifies:

                - self.xdata: each row has been orthogonalized with respect to the previous ones.
                    Note that row 0 is never modified.

            Returns:

                - None
        '''

        for i in range(0, self.xdata.shape[0] - 1):
            u = self.xdata[i]
            norm_sq = u.dot(u)
            u2 = u / norm_sq
            for j in range(i + 1, self.xdata.shape[0]):
                v = self.xdata[j]
                v -= v.dot(u) * u2

    def optimize(self, x0=None, sigma=None, method='BFGS', *args, **kwargs):
        '''Computes optimal ponderation coefficients so that the error's energy is minimized.

            Parameters:

                - x0 (optional): vector of length M*K, initial guess for the coefficients, where K is the
                    number of regressors being used (self.num_regressors) and M the number of variables

                - sigma: (optional) vector of length M*N, representing the ponderation of each error term,
                    where M is the number of variables and N the number of observations or samples.
                    These values must be consistent with the data in the GLM instace.

                - method: (optional, default = 'BFGS') string indicating solver to be used to minimize the
                    MSE in the system (or equivalently, to optimize the fitting). Refer to the documentation
                    of the scipy.optimize.minimize method to see the full description.

                - Any other parameters will be passed to the function scipy.optimize.minimize, please refer
                    to its documentation to get more information.

            Modifies:

                - [created/modified] self.results: OptimizeResult object
                    Contains the results of the optimization. Refer to scipy.optimize.OptimizeResult documentation
                    for more information.

                - [created/modified] self.opt_params: vector of length K / matrix of shape (M, K)
                    Represents the optimal coefficients obtained by the algorithm, being M the number of variables
                    and K the number of regressors in the system (if M = 1, self.opt_params will be a vector; other-
                    wise, it will be a matrix with the specified shape).
                    In case the optimization method finished abruptly, this attribute contains the last coefficients
                    computed by such method.

            Returns:

                - boolean, indicating whether the optimization was successful or not

                - string, containing a message that describes the exit status of the optimization method
        '''

        if x0 == None:
            if len(self.ydata.shape) == 1:
                M = 1
            else:
                M = self.ydata.shape[0]
            K = self.xdata.shape[0]
            x0 = nparray([0.0] * (M * K))
        self.__sigma = sigma
        self.results = minimize(self.__error_energy, x0, method=method, *args, **kwargs)
        self.opt_params = nparray(self.results.x)
        if M != 1:
            self.opt_params = self.opt_params.reshape((M, K))
        return self.results.success, self.results.message
