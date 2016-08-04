import nibabel as nib
import numpy as np
import yaml

from os.path import join
from Subject import Subject
from src.Utils.ExcelIO import ExcelSheet
from src.Utils.niftiIO import NiftiReader


class DataLoader(object):
    """
    Loads the subjects and the configuration of a study given the path to the configuration file for this study
    """

    def __init__(self, configuration_path):
        """
        Initializes a DataLoader with the given configuration file

        Parameters
        ----------
        configuration_path : String
            Path to the YAMP configuration file with the configuration parameters expected for a study.
            See config/exampleConfig.yaml for more information about the format of configuration files.
        """
        # Load the configuration
        with open(configuration_path, 'r') as conf_file:
            conf = yaml.load(conf_file)
        self._conf = conf
        self._cached_subjects = []
        self._start = None
        self._end = None

    def get_subjects(self, start=None, end=None):
        """
        Gets all the subjects from the study given the configuration parameters of this instance of DataLoader

        Parameters
        ----------
        start : [Optional] int
            0-based index where the first subject to be loaded is located
        end : [Optional] int
            0-based index where the last subject to be loaded is located

        Returns
        -------
        list<Subject>
            List of all subjects

        Raises
        ------
        KeyError
            If the configuration file doesn't follow the format rules or if one or more identifiers used in
            the configuration file don't exist.
        """

        # Load model parameters from configuration
        excel_file = self._conf['input']['excel_file']
        data_folder = self._conf['input']['data_folder']
        study_prefix = self._conf['input']['study_prefix']
        gzip_nifti = self._conf['input']['gzip_nifti']

        # Extension for GM files
        extension = '.nii.gz' if gzip_nifti else '.nii'

        # Load model parameters
        id_identifier = self._conf['model']['id_identifier']  # ID identifier
        id_type = int if self._conf['model']['id_type'] == 'Number' else str
        category_identifier = self._conf['model']['category_identifier']  # Category identifier
        fields_names = []
        fields_names = fields_names + list(self._conf['model']['correctors_identifiers'])  # Correctors
        fields_names = fields_names + list(self._conf['model']['predictors_identifiers'])  # Predictors

        # Load excel file
        xls = ExcelSheet(excel_file)

        # Prepare fields type parameter
        if category_identifier:
            # If there is a category identifier, add the id identifier and the category identifier
            fields = {
                id_identifier: id_type,
                category_identifier: int
            }
        else:
            # Otherwise, just add the id identifier
            fields = {
                id_identifier: id_type
            }
        for field in fields_names:
            fields[field] = float

        # Load the predictors and correctors for all subjects
        subjects = []
        for row in xls.get_rows(start=start, end=end, fieldstype=fields):
            # The subjects must have a non-empty ID
            if row[id_identifier] != "":
                # Create path to nifti file
                nifti_path = join(data_folder, study_prefix + str(row[id_identifier]) + extension)
                # Category
                category = row[category_identifier] if category_identifier else None
                # Create subject
                subj = Subject(row[id_identifier], nifti_path, category=category)
                # Add prediction and correction parameters
                for param_name in fields_names:
                    subj.set_parameter(parameter_name=param_name, parameter_value=row[param_name])
                # Append subject to the subjects' list
                subjects.append(subj)

        # Cache subjects
        self._cached_subjects = subjects
        self._start = start
        self._end = end
        # Return the cached subjects
        return self._cached_subjects

    def get_template_affine(self):
        """
        Returns the affine matrix used to map between the template coordinates space and the voxel coordinates space

        Returns
        -------
        numpy.array
            The affine matrix in the template NIFTI file
        """
        template_path = self._conf['input']['template_file']
        return NiftiReader(template_path).affine()

    def get_gm_data(self, start=None, end=None, use_cache=True):
        """
        Returns the grey-matter data of all subjects between "start" and "end" indices [start, end)

        Parameters
        ----------
        start : [Optional] int
            0-based index where the first subject to be loaded is located
        end : [Optional] int
            0-based index where the last subject to be loaded is located
        use_cache : [Optional] Boolean
            Uses cached subjects (if any) when it is set to True.
            Default value: True

        Returns
        -------
        numpy.array
            4D matrix with the grey matter values of all voxels for all subjects
        """
        if use_cache and (len(self._cached_subjects) > 0) and (self._start == start) and (self._end == end):
            subjects = self._cached_subjects
        else:
            subjects = self.get_subjects(start, end)
            # Update cache
            self._cached_subjects = subjects
            self._start = start
            self._end = end

        gm_values = map(lambda subject: nib.load(subject.gmfile).get_data(), subjects)
        return np.asarray(gm_values)

    def get_predictors(self, start=None, end=None, use_cache=True):
        """
        Returns the predictors of the study for all the subjects between "start" and "end" [start, end)

        Parameters
        ----------
        start : [Optional] int
            0-based index where the first subject to be loaded is located
        end : [Optional] int
            0-based index where the last subject to be loaded is located
        use_cache : [Optional] Boolean
            Uses cached subjects (if any) when it is set to True.
            Default value: True

        Returns
        -------
        numpy.array
            2D matrix with the predictors for all subjects
        """
        # Get subjects
        if use_cache and (len(self._cached_subjects) > 0) and (self._start == start) and (self._end == end):
            subjects = self._cached_subjects
        else:
            subjects = self.get_subjects(start, end)
            # Update cache
            self._cached_subjects = subjects
            self._start = start
            self._end = end

        # Get predictors
        predictors_names = self._conf['model']['predictors_identifiers']
        predictors = map(lambda subject: subject.get_parameters(predictors_names), subjects)
        return np.asarray(predictors)

    def get_correctors(self, start=None, end=None, use_cache=True):
        """
        Returns the correctors of the study for all the subjects between "start" and "end" [start, end)

        Parameters
        ----------
        start : [Optional] int
            0-based index where the first subject to be loaded is located
        end : [Optional] int
            0-based index where the last subject to be loaded is located
        use_cache : [Optional] Boolean
            Uses cached subjects (if any) when it is set to True.
            Default value: True

        Returns
        -------
        numpy.array
            2D matrix with the correctors for all subjects
        """
        # Get subjects
        if use_cache and (len(self._cached_subjects) > 0) and (self._start == start) and (self._end == end):
            subjects = self._cached_subjects
        else:
            subjects = self.get_subjects(start, end)
            # Update cache
            self._cached_subjects = subjects
            self._start = start
            self._end = end

        # Get predictors
        correctors_names = self._conf['model']['correctors_identifiers']
        correctors = map(lambda subject: subject.get_parameters(correctors_names), subjects)
        return np.asarray(correctors)

    def get_predictors_names(self):
        """
        Returns the names of the predictors of this study

        Returns
        -------
        List<String>
            List of predictors' names
        """
        return self._conf['model']['predictors_identifiers']

    def get_correctors_names(self):
        """
        Returns the correctors' names of this study

        Returns
        -------
        List<String>
            List of correctors' names
        """
        return self._conf['model']['correctors_identifiers']

    def get_processing_parameters(self):
        """
        Returns the parameters used for processing, that is, number of jobs, chunk memory, and cache size

        Returns
        -------
        dict
            Dictionary with keys 'n_jobs', 'mem_usage' and 'cache_size' representing
            number of jobs used for fitting, amount of memory in MB per chunck, and amount of memory
            reserved for SVR fitting, respectively.
        """

        return self._conf['processing_params']

    def get_output_dir(self):
        """
        Returns the path to the output folder set in the configuration file

        Returns
        -------
        String
            Path to the output folder
        """
        return self._conf['output']['output_path']

    def get_hyperparams_finding_configuration(self, fitting_method='PolySVR'):
        """
        Returns a GridSearch ready dictionary with the possible values for the specified hyperparameters

        Returns
        -------
        Dictionary
            The keys of the dictionary are the name of the hyperparameter and the values the numpy array
            containing all the possible values amongst which the optimal will be found.
        """

        # Inner function
        def get_hyperparams(hyperparams_dict, hyperparam_name):

            identifier = '{}_values'.format(hyperparam_name)

            start_val = hyperparams_config[identifier]['start']
            end_val = hyperparams_config[identifier]['end']
            N = hyperparams_config[identifier]['N']

            if hyperparams_config[identifier]['spacing'] == 'logarithmic':
                if hyperparams_config[identifier]['method'] == 'random':
                    # Logarithmic spacing and random search
                    hyperparams_dict[hyperparam_name] = np.sort([10 ** i for i in np.random.uniform(
                        start_val, end_val, N
                    )])
                else:
                    # Logarithmic spacing and deterministic search
                    hyperparams_dict[hyperparam_name] = np.logspace(start_val, end_val, N)
            else:
                if hyperparams_config[identifier]['method'] == 'random':
                    # Linear spacing and random search
                    hyperparams_dict[hyperparam_name] = np.sort(np.random.uniform(
                        start_val, end_val, N
                    ))
                else:
                    # Logarithmic spacing and deterministic search
                    hyperparams_dict[hyperparam_name] = np.linspace(start_val, end_val, N)

        hyperparams_config = self._conf['hyperparameters_finding']
        hyperparams_dict = {}
        if hyperparams_config['epsilon']:
            get_hyperparams(hyperparams_dict, 'epsilon')
        if hyperparams_config['C']:
            get_hyperparams(hyperparams_dict, 'C')
        if fitting_method == 'GaussianSVR':
            if hyperparams_config['gamma']:
                get_hyperparams(hyperparams_dict, 'gamma')

        return hyperparams_dict

