import unittest

import numpy as np

from src.Utils.DataLoader import DataLoader


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.data_loader = DataLoader('mock_data/config_category.yaml')
        self.data_loader_no_cat = DataLoader('mock_data/config_no_category.yaml')

    def test_create_data_loader(self):
        ko_path = '../confi/exampleConfig.yaml'
        self.assertRaises(IOError, DataLoader, ko_path)

    def test_load_subjects(self):
        few_subjects = self.data_loader.get_subjects(start=0, end=1)
        self.assertEqual(len(few_subjects), 1,
                         msg="The number of a subset of subjects loaded does not match the expected one")
        subjects = self.data_loader.get_subjects()
        self.assertEqual(len(subjects), 3,
                         msg="The number of total subjects loaded does not match the expected one")

    def test_load_subjects_no_category(self):
        few_subjects = self.data_loader_no_cat.get_subjects(start=0, end=1)
        self.assertEqual(len(few_subjects), 1,
                         msg="The number of a subset of subjects loaded does not match the expected one")
        subjects = self.data_loader_no_cat.get_subjects()
        self.assertEqual(len(subjects), 3,
                         msg="The number of total subjects loaded does not match the expected one")

    def test_load_predictor(self):
        predictor = np.array([[0.265], [0.013], [0.487]])
        loaded_predictor = self.data_loader.get_predictor()
        self.assertTrue(predictor.shape == loaded_predictor.shape,
                        msg='Loaded predictor does not have the appropiate shape')
        self.assertTrue(np.all(predictor == loaded_predictor),
                        msg='Loaded predictor does not have the appropiate values')
        loaded_predictor_from_cache = self.data_loader.get_predictor()
        self.assertTrue(predictor.shape == loaded_predictor_from_cache.shape,
                        msg='Loaded predictor from cache does not have the appropiate shape')
        self.assertTrue(np.all(predictor == loaded_predictor_from_cache),
                        msg='Loaded predictor from cache does not have the appropiate values')

    def test_load_correctors(self):
        correctors = np.array([[42, 0], [32, 1], [85, 1]])
        loaded_correctors = self.data_loader.get_correctors()
        self.assertTrue(correctors.shape == loaded_correctors.shape,
                        msg='Loaded correctors do not have the appropiate shape')
        self.assertTrue(np.all(correctors == loaded_correctors),
                        msg='Loaded predictor do not have the appropiate values')
        loaded_correctors_from_cache = self.data_loader.get_correctors()
        self.assertTrue(correctors.shape == loaded_correctors_from_cache.shape,
                        msg='Loaded correctors from cache do not have the appropiate shape')
        self.assertTrue(np.all(correctors == loaded_correctors_from_cache),
                        msg='Loaded predictor from cache do not have the appropiate values')

    def test_load_predictor_correctors_names(self):
        predictor_name = ['biomarker']
        corrector_names = ['age', 'sex']
        self.assertEqual(self.data_loader.get_predictor_name(), predictor_name)
        self.assertEqual(self.data_loader.get_correctors_names(), corrector_names)

    def test_load_processing_parameters(self):
        processing_parameters = {
            'n_jobs': 2,
            'mem_usage': 512,
            'cache_size': 1024
        }
        self.assertEqual(self.data_loader.get_processing_parameters(), processing_parameters,
                         msg="Processing parameters getter does not get the parameters as expected")

    def test_load_output_dir(self):
        output_dir = 'mock_data/results'
        self.assertEqual(self.data_loader.get_output_dir(), output_dir,
                         msg="Output directory getter does not behave as expected")

    def test_load_hyperparameters_config(self):
        """
        TODO: Assert that all combinations (linear, logarithmic), (deterministic, random) with both fitting methods
        PolySVR and GaussianSVR produce the expected dictionary with the expected number of keys and types of values.
        """
        pass

if __name__ == '__main__':
    unittest.main()
