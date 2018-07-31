import unittest
from vneat.Processors.MixedProcessor import MixedProcessor
from vneat.Utils.Subject import Subject
import numpy as np


class TestOrthogonalization(unittest.TestCase):


    def setUp(self):

        self.subject_list = [Subject(i,str(i)+'.nii.gz','.nii.gz',category=np.mod(i,5)) for i in range(10)]
        self.predictors = 2*(np.random.randn(10,1)+0.5)
        self.correctors = np.concatenate((0.1*np.random.randn(10,1), 1.2*(-0.9+np.random.randn(10,1))),axis=1)




    def test_normalize_all(self):
        user_defined_parameters = (False, 'All', 2, 0, [1, 9, 1, 1], 0, [0, 9, 1])
        processor = MixedProcessor(self.subject_list,
                                   ['c'],
                                   ['a', 'b'],
                                   self.predictors,
                                   self.correctors,
                                   None,
                                   user_defined_parameters=user_defined_parameters)

        features = np.concatenate((processor.fitter.correctors, processor.fitter.predictors),axis=1)

        for f in features.T:
            self.assertAlmostEqual(np.dot(f.T, f), 1, 5)

    def test_orthogonalize_all(self):
        user_defined_parameters = (False, 'All', 1, 0, [1, 9, 1, 1], 0, [2, 9, 1])
        processor = MixedProcessor(self.subject_list,
                                   ['c'],
                                   ['a', 'b'],
                                   self.predictors,
                                   self.correctors,
                                   None,
                                   user_defined_parameters=user_defined_parameters)

        features = np.concatenate((processor.fitter.correctors, processor.fitter.predictors), axis=1)
        for it_f in range(1,features.shape[1]):
            for it_ff in range(it_f):
                self.assertAlmostEqual(np.dot(features[:,it_f].T, features[:,it_ff]), 0, 5)

    def test_orthonormalize_all(self):
        user_defined_parameters = (False, 'All', 0, 0, [0, 9, 1, 1], 0, [2, 9, 1])
        processor = MixedProcessor(self.subject_list,
                                   ['c'],
                                   ['a', 'b'],
                                   self.predictors,
                                   self.correctors,
                                   None,
                                   user_defined_parameters=user_defined_parameters)

        features = np.concatenate((processor.fitter.correctors, processor.fitter.predictors), axis=1)
        for it_f in range(1,features.shape[1]):
            for it_ff in range(it_f):
                self.assertAlmostEqual(np.dot(features[:,it_ff].T, features[:,it_ff]), 1, 5)
                self.assertAlmostEqual(np.dot(features[:,it_f].T, features[:,it_ff]), 0, 5)


    def test_orthonormalize_correctors_predictors(self):
        user_defined_parameters = (False, 'All', 3, 0, [1, 6, 1, 1], 0, [2, 3, 1])
        processor = MixedProcessor(self.subject_list,
                                   ['c'],
                                   ['a', 'b'],
                                   self.predictors,
                                   self.correctors,
                                   None,
                                   user_defined_parameters=user_defined_parameters)


        features = np.concatenate((processor.fitter.correctors, processor.fitter.predictors), axis=1)
        print(features)

        for it_c in range(1, processor.fitter.correctors.shape[1]):
            for it_cc in range(it_c):
                self.assertAlmostEqual(np.dot(processor.fitter.correctors[:, it_cc].T, processor.fitter.correctors[:, it_cc]), 1, 5)
                self.assertAlmostEqual(np.dot(processor.fitter.correctors[:, it_c].T, processor.fitter.correctors[:, it_cc]), 0, 5)

        for it_p in range(1, processor.fitter.predictors.shape[1]):
            for it_pp in range(it_p):
                self.assertAlmostEqual(
                    np.dot(processor.fitter.predictors[:, it_pp].T, processor.fitter.predictors[:, it_pp]), 1, 5)
                self.assertAlmostEqual(
                    np.dot(processor.fitter.predictors[:, it_p].T, processor.fitter.predictors[:, it_pp]), 0, 5)


if __name__ == '__main__':
    unittest.main()