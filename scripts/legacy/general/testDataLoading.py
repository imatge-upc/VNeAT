from os.path import join
from Utils.DataLoader import DataLoader

config_file = join("..", "..", "config", "adContinuum.yaml")
# config_file = join("..", "..", "config", "agingApoe.yaml")

subjectsLoader = DataLoader(config_file)

# Load subjects
subjects = subjectsLoader.get_subjects()

# Load grey matter data
# gm1 = subjectsLoader.get_gm_data()

# Load grey matter data from few subjects
gm2 = subjectsLoader.get_gm_data(start=3, end=13)

# Load affine matrix
affine = subjectsLoader.get_template_affine()

# Load predictors
predictors = subjectsLoader.get_predictor()

# Load predictors without using cache
predictors2 = subjectsLoader.get_predictor(use_cache=False)

# Load correctors
correctors = subjectsLoader.get_correctors()

# Load processing parameters
processing_params = subjectsLoader.get_processing_parameters()

print 'Done'
