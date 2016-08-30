import unittest

import numpy as np

from src.Utils.DataLoader import DataLoader


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.data_loader = DataLoader('../config/exampleConfig.yaml')

    def test_create_data_loader(self):
        ko_path = '../confi/exampleConfig.yaml'
        self.assertRaises(IOError, DataLoader, ko_path)

    def test_load_subjects(self):
        subjects = self.data_loader.get_subjects()
        self.assertEqual(len(subjects), 3)
        few_subjects = self.data_loader.get_subjects(start=0, end=1)
        self.assertEqual(len(few_subjects), 1)

    def test_load_predictor(self):
        predictor = np.array([[0.265], [0.013], [0.487]])
        loaded_predictor = self.data_loader.get_predictor()
        self.assertTrue(predictor.shape == loaded_predictor.shape,
                        msg='Loaded predictor does not have the appropiate shape')
        self.assertTrue(np.all(predictor == loaded_predictor),
                        msg='Loaded predictor does not have the appropiate values')

    def test_load_correctors(self):
        correctors = np.array([[42, 0], [32, 1], [85, 1]])
        loaded_correctors = self.data_loader.get_correctors()
        self.assertTrue(correctors.shape == loaded_correctors.shape,
                        msg='Loaded correctors do not have the appropiate shape')
        self.assertTrue(np.all(correctors == loaded_correctors),
                        msg='Loaded predictor do not have the appropiate values')

    def test_load_predictor_correctors_names(self):
        predictor_name = ['biomarker']
        corrector_names = ['age', 'sex']
        self.assertEqual(self.data_loader.get_predictor_name(), predictor_name)
        self.assertEqual(self.data_loader.get_correctors_names(), corrector_names)


if __name__ == '__main__':
    unittest.main()
