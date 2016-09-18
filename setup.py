from distutils.core import setup

setup(
    name='Neuroimatge',
    version='1.0.0',
    packages=['src', 'src.Utils', 'src.Fitters', 'src.FitScores', 'src.Processors', 'src.Visualization',
              'src.CrossValidation'],
    scripts=['nln-compare_statistical_maps.py', 'nln-compute_fitting.py', 'nln-compute_statistical_maps.py',
             'nln-generate_user_parameters.py', 'nln-search_hyperparameters.py', 'nln-show_curves.py',
             'nln-show_data_distribution.py', 'nln-show_visualizer.py'],
    url='',
    license='MIT',
    author='Image Processing Group',
    author_email='santiago.puch@alu-etsetb.upc.edu',
    description=''
)
