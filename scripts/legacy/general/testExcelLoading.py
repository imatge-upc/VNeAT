import yaml
from os.path import join
from Utils.ExcelIO import ExcelSheet
from Utils.Subject import Subject

# Load configuration
CONFIG_PATH = join("..", "..", "config", "adContinuum.yaml")
try:
    with open(CONFIG_PATH, 'r') as conf_file:
        conf = yaml.load(conf_file)

    # Load input parameters
    print "\nConfiguration file loaded."
    excel_file = conf['input']['excel_file']
    data_folder = conf['input']['data_folder']
    study_prefix = conf['input']['study_prefix']
    gzip_nifti = conf['input']['gzip_nifti']

    # Extension for GM files
    extension = '.nii.gz' if gzip_nifti else '.nii'

    # Load model parameters
    id_identifier = conf['model']['id_identifier']                                  # ID identifier
    category_identifier = conf['model']['category_identifier']                      # Category identifier
    fields_names = []
    fields_names = fields_names + list(conf['model']['correctors_identifiers'])     # Correctors
    fields_names = fields_names + list(conf['model']['predictors_identifiers'])     # Predictors

    # Load excel file
    xls = ExcelSheet(excel_file)

    # Prepare fields type parameter
    if category_identifier:
        # If there is a category identifier, add the id identifier and the category identifier
        fields = {
            id_identifier: str,
            category_identifier: int
        }
    else:
        # Otherwise, just add the id identifier
        fields = {
            id_identifier: str
        }
    for field in fields_names:
        fields[field] = float

    # Load the predictors and correctors for all subjects
    subjects = []
    for row in xls.get_rows(fieldstype=fields):
        # The subjects must have a non-empty ID
        if row[id_identifier] != "":
            # Create path to nifti file
            nifti_path = join(data_folder, study_prefix + row[id_identifier] + extension)
            # Category
            category = row[category_identifier] if category_identifier else 0
            # Create subject
            subj = Subject(row[id_identifier], nifti_path, category=category)
            # Add prediction and correction parameters
            for param_name in fields_names:
                subj.set_parameter(parameter_name=param_name, parameter_value=row[param_name])
            # Append subject to the subjects' list
            subjects.append(subj)

    print 'Done loading subjects.'

except Exception as e:
    print e

