import numpy as np
import niftiIO as nio


class Participant(object):
    """
    Class that represents a participant (or subject) in a study, and holds the information related to him/her.
    """

    def __init__(self, participant_id, gmfile, category=0):
        """
        Initializes a participant

        Parameters
        ----------
        id : string
            Unique identifier for this participant
        gmfile : string
            Full path to the NIFTI file that contains the grey matter data for this participant
        category : [Optional] int
            Numerical category assigned to this participant. Default value: 0
        """
        self._id = participant_id
        self._gmfile = str(gmfile)
        self._category = int(category)
        self._parameters = {}

    @property
    def id(self):
        return self._id

    @property
    def gmfile(self):
        return self._gmfile

    @property
    def category(self):
        return self._category

    def set_parameter(self, parameter_name, parameter_value, override=True):
        """
        Sets a parameter in this participant, identified by parameter_name

        Parameters
        ----------
        parameter_name : string
            Name that identifies this parameter
        parameter_value : int/float
            Value for this parameter
        override : Optional[boolean]
            If the parameter associated to parameter_name already exists, override it or not.
            Default value: True
        """
        if override or (parameter_name not in self._parameters):
            self._parameters[parameter_name] = parameter_value

    def set_parameters(self, parameter_names, parameter_values, override=True):
        """
        Sets several parameters in this participant, identified by parameter_names.

        Parameters
        ----------
        parameter_names : list
            Names that identify the parameters.
        parameter_values : list
            Values for these parameters. Must have the same length as parameter_names
        override : Optional[boolean]
            If the parameter associated to parameter_name already exists, override it or not.
            Default value: True
        """
        for name, value in zip(parameter_names, parameter_values):
            self.set_parameter(name, value, override)

    def get_parameter(self, parameter_name):
        """
        Returns the value associated to parameter_name. Raises a KeyError if parameter_name does not exist

        Parameters
        ----------
        parameter_name : string
            String that identifies the parameter whose value should be returned

        Returns
        -------
        int/float
            The value associated to parameter_name

        Raises
        ------
        KeyError
            If parameter_name has not been set previously in this participant
        """
        return self._parameters[parameter_name]

    def get_parameters(self, parameter_names):
        """
        Returns the values associated to parameter_names. Raises a KeyError if at least one of the
        parameter_name does not exist

        Parameters
        ----------
        parameter_names : list
            List of string that identifies the parameters whose values should be returned

        Returns
        -------
        list
            The values associated to parameter_names

        Raises
        ------
        KeyError
            If one of the parameter_names has not been set previously in this participant

        """
        return map(lambda name: self._parameters[name], parameter_names)


class Subject(object):
    Diagnostics = ['NC', 'PC', 'MCI', 'AD']
    Sexes = ['Unknown', 'Male', 'Female']
    APOE4s = ['Unknown', 'Yes', 'No']

    class Attribute:
        def __init__(self, name, description='Undefined subject attribute'):
            self._description = description
            self._name = str(name)

        @property
        def description(self):
            return self._description

        def __repr__(self):
            return 'Subject.' + self._name

        def __str__(self):
            return self._name

    Diagnostic = Attribute('Diagnostic',
                           'A 0-based index indicating the diagnostic of the subject (see Subject.Diagnostics). None if it was not indicated.')
    Age = Attribute('Age', 'An integer that indicates the age of the subject. None if it was not indicated.')
    Sex = Attribute('Sex',
                    'An integer indicating the genre of the subject (-1 if Female, 1 if Male, 0 if Not Indicated).')
    APOE4 = Attribute('APOE-4',
                      'An integer indicating if the apoe-4 protein is present in the subject\'s organism (-1 if Not Present, 1 if Present, 0 if Not Indicated).')
    Education = Attribute('Education',
                          'An integer that indicates the level of academical education of the subject. None if it was not indicated.')
    ADCSFIndex = Attribute('AD-CSF Index',
                           'A float that represents the AD-CSF index (t-tau) value of the subject. None if it was not indicated.')

    Attributes = [Diagnostic, Age, Sex, APOE4, Education, ADCSFIndex]
    for index in xrange(len(Attributes)):
        Attributes[index].index = index

    def __init__(self, identifier, graymatter_filename, diagnostic=None, age=None, sex=None, apoe4=None, education=None,
                 adcsfIndex=None):
        self._id = identifier
        self._gmfile = graymatter_filename
        self._attributes = [None] * len(Subject.Attributes)

        self._attributes[Subject.Diagnostic.index] = diagnostic
        self._attributes[Subject.Age.index] = age
        self._attributes[Subject.Sex.index] = sex if not sex is None else 0
        self._attributes[Subject.APOE4.index] = apoe4 if not apoe4 is None else 0
        self._attributes[Subject.Education.index] = education
        self._attributes[Subject.ADCSFIndex.index] = adcsfIndex

    @property
    def id(self):
        return self._id

    @property
    def gmfile(self):
        return self._gmfile

    def get(self, attribute_list):
        '''Retrieves the specified attributes from the subject's data.

            Parameters:

                - attribute_list: iterable containing the attributes that must be retrieved from the subject.
                    See Subject.Attributes to obtain a list of available attributes.

            Returns:

                - list containing the values of the attributes specified in the 'attribute_list' argument,
                    in the same order.
        '''
        return map(lambda attr: self._attributes[attr.index], attribute_list)

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return 'Subject( ' + repr(self.id) + ' )'

    def __str__(self):
        diag, age, sex, apoe4, ed, adcsf = self.get(Subject.Diagnostic, Subject.Age, Subject.Sex, Subject.APOE4,
                                                    Subject.Education, Subject.ADCSFIndex)
        s = 'Subject ' + repr(self.id) + ':\n'
        s += '    Diagnostic: '
        if diag is None:
            s += 'Unknown'
        else:
            s += Subject.Diagnostics[diag]
        s += '\n    Age: '
        if age is None:
            s += 'Unknown'
        else:
            s += repr(age)
        s += '\n    Sex: ' + Subject.Sexes[sex]
        s += '\n    APOE4 presence: ' + Subject.APOE4s[apoe4]
        s += '\n    Education level: '
        if ed is None:
            s += 'Unkown'
        else:
            s += repr(ed)
        s += '\n    AD-CSF index (t-tau) value: '
        if adcsf is None:
            s += 'Unknown'
        else:
            s += repr(adcsf)
        s += '\n'
        return s


class Chunks:
    """
    Class that lets you load the data in small chunks to have better memory performance
    """
    def __init__(self, subject_list, x1=0, x2=None, y1=0, y2=None, z1=0, z2=None, mem_usage=None):
        """
        Initialize a chunk of data

        Parameters
        ----------
        subject_list : List
            List of subjects/participants. The only requirement for the underlying class is that implements the
            "gmfile" attribute, which returns the path to the NIFTI file of this subject/participant
        x1 : [Optional] int
            First voxel in x axis
        x2 : [Optional] int
            Last voxel in x axis
        y1 : [Optional] int
            First voxel in y axis
        y2 : [Optional] int
            Last voxel in y axis
        z1 : [Optional] int
            First voxel in z axis
        z2 : [Optional] int
            Last voxel in z axis
        mem_usage : [Optional] int
            Number of MB of memory reserved to store the chunk
            Default value: 512
        """

        if mem_usage is None:
            mem_usage = 512.0
        self._gmdata_readers = map(
            lambda subject: nio.NiftiReader(subject.gmfile, x1=x1, y1=y1, z1=z1, x2=x2, y2=y2, z2=z2), subject_list)
        self._dims = self._gmdata_readers[0].dims
        self._num_subjects = np.float64(len(self._gmdata_readers))
        self._iterators = map(lambda gmdata_reader: gmdata_reader.chunks(mem_usage / self._num_subjects),
                              self._gmdata_readers)

    @property
    def dims(self):
        return self._dims

    @property
    def num_subjects(self):
        return int(self._num_subjects)

    def __iter__(self):
        return self

    def next(self):
        reg = self._iterators[0].next()  # throws StopIteration if there are not more Chunks
        chunkset = [reg.data]
        chunkset += [it.next().data for it in self._iterators[1:]]
        return nio.Region(reg.coords, np.array(chunkset, dtype=np.float64))
