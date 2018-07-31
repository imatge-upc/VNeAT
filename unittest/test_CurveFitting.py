import unittest
from vneat.Fitters.CurveFitting import CurveFitter
import numpy as np


class TestTreatData(unittest.TestCase):


    def setUp(self):

        self.predictors = 2*(np.random.randn(10,1)+0.5)
        self.correctors = np.concatenate((0.1*np.random.randn(10,1), 1.2*(-0.9+np.random.randn(10,1))),axis=1)

    def test_normalize_predictors(self):
        self.fitter = CurveFitter(self.predictors, self.correctors, intercept=CurveFitter.PredictionIntercept)
        self.fitter.normalize_predictors()

        for p in self.fitter.predictors.T:
            self.assertAlmostEqual(np.dot(p.T,p), 1, 5)

    def test_normalize_all(self):
        self.fitter = CurveFitter(self.predictors, self.correctors, intercept=CurveFitter.CorrectionIntercept)
        self.fitter.normalize_all()

        for c in self.fitter.correctors.T:
            self.assertAlmostEqual(np.dot(c.T, c), 1, 5)

        for p in self.fitter.predictors.T:
            self.assertAlmostEqual(np.dot(p.T, p), 1, 5)


    def test_orthogonalize_predictors(self):
        self.fitter = CurveFitter(self.predictors, self.correctors, intercept=CurveFitter.PredictionIntercept)
        self.fitter.orthogonalize_predictors()

        for p0,p1 in zip(self.fitter.predictors.T[:-1],self.fitter.predictors.T[1:]):
            self.assertAlmostEqual(np.dot(p0.T, p1), 0, 5)

    def test_orthogonalize_correctors(self):
        self.fitter = CurveFitter(self.predictors, self.correctors, intercept=CurveFitter.CorrectionIntercept)
        self.fitter.orthogonalize_correctors()

        for c0,c1 in zip(self.fitter.correctors.T[:-1],self.fitter.correctors.T[1:]):
            self.assertAlmostEqual(np.dot(c0.T, c1), 0, 5)


    def test_orthogonalize_all(self):
        self.fitter = CurveFitter(self.predictors, self.correctors, intercept=CurveFitter.CorrectionIntercept)
        self.fitter.orthogonalize_all()

        features = np.concatenate((self.fitter.correctors, self.fitter.predictors),axis=1)
        for f0, f1 in zip(features.T[:-1], features.T[1:]):
            self.assertAlmostEqual(np.dot(f0.T, f1), 0, 5)



if __name__ == '__main__':
    unittest.main()
