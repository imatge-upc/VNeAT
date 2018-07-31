import unittest
from vneat.Processors.MixedProcessor import MixedProcessor
from vneat.Utils.Subject import Subject
import numpy as np


class TestOrthogonalization(unittest.TestCase):


    def setUp(self):

        self.subject_list = [Subject(i,str(i)+'.nii.gz','.nii.gz',category=np.mod(i,5)) for i in range(10)]
        self.predictors = np.random.randn(10,1)
        self.correctors = np.asarray([np.random.randn(10), np.random.randn(10)])




    def test_normalize_correctors(self):
        user_defined_parameters = []
        processor = MixedProcessor(self.subject_list,
                                   ['c'],
                                   ['a', 'b'],
                                   self.predictors,
                                   self.correctors,
                                   None,
                                   user_defined_parameters=user_defined_parameters)

        print(processor.user_defined_parameters)

    def test_orthogonalize_correctors(self):
        pass



if __name__ == '__main__':
    unittest.main()