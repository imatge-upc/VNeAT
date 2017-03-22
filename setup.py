from distutils.core import setup

setup(
    name='VNeAT',
    version='0.0.0',
    packages=['vneat', 'vneat.Utils', 'vneat.Fitters', 'vneat.FitScores', 'vneat.Processors', 'vneat.Visualization',
              'vneat.CrossValidation'],
    scripts=['vneat-compare_statistical_maps.py', 'vneat-compute_fitting.py', 'vneat-compute_statistical_maps.py',
             'vneat-generate_user_parameters.py', 'vneat-search_hyperparameters.py', 'vneat-show_curves.py',
             'vneat-show_data_distribution.py', 'vneat-show_visualizer.py'],
    url='',
    license='MIT',
    author='Image Processing Group',
    author_email='santiago.puch@alu-etsetb.upc.edu',
    description=''
)
