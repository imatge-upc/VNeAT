import numpy as np
import niftiIO as nio


class Subject(object):
    """
    Class that represents a participant (or subject) in a study, and holds the information related to him/her.
    """

    def __init__(self, participant_id, gmfile, category=None):
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
        self._category = category
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
