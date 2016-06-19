from os.path import join
from Utils.DataLoader import SubjectsLoader

config_file = join("..", "..", "config", "adContinuum.yaml")
# config_file = join("..", "..", "config", "agingApoe.yaml")

subjectsLoader = SubjectsLoader(config_file)

# Load subjects
subjects = subjectsLoader.get_subjects()

# Load grey matter data
gm1 = subjectsLoader.get_gm_data()

# Load grey matter data from few subjects
gm2 = subjectsLoader.get_gm_data(start=3, end=13)

# Load affine matrix
affine = subjectsLoader.get_template_affine()

# Load predictors
predictors = subjectsLoader.get_predictors()

# Load predictors without using cache
predictors2 = subjectsLoader.get_predictors(use_cache=False)

# Load correctors
correctors = subjectsLoader.get_correctors()

print 'Done'