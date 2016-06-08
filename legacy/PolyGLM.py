from GLM import GLM
from Transforms import polynomial
from numpy import array as nparray


class PolyGLM:
    def __init__(self, features, homogeneous=True, degrees=None, regressor_index=None):
        self._pglm_features = nparray(features)
        if len(self._pglm_features.shape) != 2:
            raise ValueError('Argument \'features\' must be a 2-dimensional matrix')

        if degrees is None:
            self._pglm_degrees = [1] * self._pglm_features.shape[1]
        else:
            if len(degrees) != self._pglm_features.shape[1]:
                raise ValueError('Argument \'degrees\' must have a length equal to the number of features')
            self._pglm_degrees = []
            for deg in degrees:
                if not isinstance(deg, int):
                    raise ValueError('Expected integer in \'degrees\' list, got ' + str(type(deg)))
                if deg < 1:
                    raise ValueError('All degrees must be >= 1')

        self._pglm_homogeneous = homogeneous

        if regressor_index is None:
            self._pglm_regressor = len(self._pglm_degrees) - 1
        else:
            self.select_regressor(regressor_index)

        self._pglm_up_to_date = False

    def __pglm_update_GLM(self):
        correctors = []
        regressors = []
        for index in range(len(self._pglm_degrees)):
            for p in polynomial(self._pglm_degrees[index], self._pglm_features[:, index]):
                if index == self._pglm_regressor:
                    regressors.append(p)
                else:
                    correctors.append(p)

        correctors = nparray(correctors).T
        regressors = nparray(regressors).T

        self._pglm_glm = GLM(regressors, correctors, self._pglm_homogeneous)
        self._pglm_glm.orthonormalize_all()

        self._pglm_up_to_date = True

    def set_degree(self, index, degree):
        if index < 0:
            raise IndexError
        if not isinstance(degree, int):
            raise ValueError('Expected integer in \'degree\' argument, got ' + str(type(degree)))
        if degree < 1:
            raise ValueError('Degree must be >= 1')
        self._pglm_degrees[index] = degree
        self._pglm_up_to_date = False

    def select_regressor(self, index):
        if index < 0 or index >= len(self._pglm_degrees):
            raise IndexError
        self._pglm_regressor = index
        self._pglm_up_to_date = False

    def fit(self, observations, sample_weight=None, num_threads=-1):
        if (not self._pglm_up_to_date):
            self.__pglm_update_GLM()

        self._pglm_correction_parameters, self._pglm_regression_parameters = self._pglm_glm.fit(observations,
                                                                                                sample_weight,
                                                                                                num_threads)

    def correct(self, observations):
        return self._pglm_glm.correct(observations, correction_parameters=self._pglm_correction_params)

    def predict(self):
        return self._pglm_glm.predict(regression_parameters=self._pglm_regression_parameters)

    @property
    def correctors(self):
        if not self._pglm_up_to_date:
            self.__pglm_update_GLM()
        return self._pglm_glm.correctors

    @property
    def regressors(self):
        if not self._pglm_up_to_date:
            self.__pglm_update_GLM()
        return self._pglm_glm.regressors

    @property
    def features(self):
        return self._pglm_features

    @property
    def correction_parameters(self):
        return self._pglm_correction_parameters

    @property
    def regression_parameters(self):
        return self._pglm_regression_parameters

    def evaluate_fit(self, observations):
        return self._pglm_glm.evaluate_fit(observations)

    @staticmethod
    def load_from(obj):
        d = {key: value for (key, value) in obj}
        pglm = PolyGLM(d['features'], d['homogeneous'])
        pglm._pglm_degrees = d['degrees']
        pglm._regressor = d['regressor']
        pglm._pglm_correction_parameters = d['correction_params']
        pglm._pglm_regression_parameters = d['regression_params']
        pglm.__pglm_update_GLM()
        return pglm

    def to_save(self):
        try:
            obj = []
            obj.append(('features', self._pglm_features))
            obj.append(('degrees', self._pglm_degrees))
            obj.append(('regressor', self._pglm_regressor))
            obj.append(('homogeneous', self._pglm_homogeneous))
            obj.append(('correction_params', self.correction_parameters))
            obj.append(('regression_params', self.regression_parameters))
            return obj
        except AttributeError:
            return []
