import unittest

from vneat.Utils.Subject import Subject


class TestSubject(unittest.TestCase):
    def setUp(self):
        # Subject without category
        self.id = 1
        self.gm_file = './mock_gm_file'
        self.subject = Subject(self.id, self.gm_file, category=None)
        # Subject with category
        self.id_cat = 2
        self.gm_file_cat = './mock_gm_file_cat'
        self.category = 0
        self.subject_with_category = Subject(self.id_cat, self.gm_file_cat, category=self.category)

    def test_attributes(self):
        self.assertEqual(self.subject.id, self.id,
                         msg='Id getter for subject with no category')
        self.assertEqual(self.subject.gmfile, self.gm_file,
                         msg='Gm file getter for subject with no category')
        self.assertEqual(self.subject.category, None,
                         msg='Category getter for subject with no category')
        self.assertEqual(self.subject_with_category.id, self.id_cat,
                         msg='Id getter for subject with category')
        self.assertEqual(self.subject_with_category.gmfile, self.gm_file_cat,
                         msg='Gm file getter for subject with no category')
        self.assertEqual(self.subject_with_category.category, self.category,
                         msg='Category getter for subject with no category')

    def test_parameters(self):
        # Put a parameter in the subject and get it
        parameter_name, parameter_value = 'example', 'value'
        self.subject.set_parameter(parameter_name, parameter_value)
        self.assertEqual(self.subject.get_parameter(parameter_name), parameter_value,
                         msg='Single parameter setter and getter')
        # Put several parameters and get them
        parameter_names, parameter_values = ['p1', 'p2', 'p3'], [1, 2, 3]
        self.subject.set_parameters(parameter_names, parameter_values)
        self.assertEqual(self.subject.get_parameters(parameter_names), parameter_values,
                         msg='Multiple parameters setter and getter')
        # Try to put more parameter values than names
        parameter_values += [4, 5]
        self.subject.set_parameters(parameter_names, parameter_values)
        self.assertEqual(self.subject.get_parameters(parameter_names), parameter_values[:-2],
                         msg='Multiple parameters setter and getter when parameter names and values have different '
                             'lengths')
        # Test override: put 'p1' with override and 'p2' without it
        self.subject.set_parameter(parameter_names[0], parameter_values[1], override=True)
        self.subject.set_parameter(parameter_names[1], parameter_values[2], override=False)
        self.assertEqual(self.subject.get_parameter(parameter_names[0]), parameter_values[1],
                         msg='Set a parameter that already exists with override option on')
        self.assertEqual(self.subject.get_parameter(parameter_names[1]), parameter_values[1],
                         msg='Set a parameter that already exists with override option off')


if __name__ == '__main__':
    unittest.main()
