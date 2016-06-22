from abc import ABCMeta, abstractmethod
from sys import stdout

import numpy as np

import Utils.Subject
from FitScores.FitEvaluation_v2 import evaluation_function as eval_func
from Utils.Documentation import docstring_inheritor
from Utils.graphlib import NiftiGraph


class Processor(object):
    """
    Class that interfaces between the data and the fitting library
    """
    __metaclass__ = docstring_inheritor(ABCMeta)

    class Results:
        def __init__(self, prediction_parameters, correction_parameters):  # , fitting_scores):
            self._prediction_parameters = prediction_parameters
            self._correction_parameters = correction_parameters

        @property
        def prediction_parameters(self):
            return self._prediction_parameters

        @property
        def correction_parameters(self):
            return self._correction_parameters

        def __str__(self):
            s = 'Results:'
            s += '\n    Correction parameters:' + reduce(lambda x, y: x + '\n    ' + y,
                                                         repr(self._correction_parameters).split('\n'))
            s += '\n\n    Prediction parameters:' + reduce(lambda x, y: x + '\n    ' + y,
                                                           repr(self._prediction_parameters).split('\n'))
            return s

    def __init__(self, subjects, predictors_names, correctors_names, predictors, correctors,
                 processing_parameters, user_defined_parameters=(), category=None):
        """
        Creates a Processor instance

        Parameters
        ----------
        subjects : List<Subject>
            List of subjects to be used in this processor
        predictors_names : List<String>
            List of the names of the features that should be used as predictors
        correctors_names : List<String>
            List of the names of the features that should be used as correctors
        predictors : numpy.array(NxP)
            Array with the values of the features that should be used as predictors, where N is the number of subjects
            and P the number of predictors
        correctors : numpy.array(NxC)
            Array with the values of the features that should be used as correctors, where N is the number of subjects
            and C the number of correctors
        processing_parameters : dict
            Dictionary with the processing parameters specified in the configuration file, that is, 'mem_usage',
            'n_jobs' and 'cache_size'
        user_defined_parameters : [Optional] tuple
            Parameters passed to the processor. If a empty tuple is passed, the parameters are requested by input.
            Default value: ()
        category : [Optional] String
            Specifies the category for which the fitting should be done. If not specified or None, the fitting is
            computed over all subjects.
            Default value: None
        """
        # Filter subjects by category if needed
        if category is not None:
            all_indices = range(len(subjects))
            category_indices = filter(lambda index: subjects[index].category == category, all_indices)
            subjects = [subjects[i] for i in category_indices]
            predictors = predictors[category_indices, :]
            correctors = correctors[category_indices, :]
        self._processor_subjects = subjects
        # Load predictors and correctors' names
        self._processor_predictors_names = predictors_names
        self._processor_correctors_names = correctors_names
        # Load predictors and correctors
        self._processor_predictors = predictors
        self._processor_correctors = correctors
        # Load processing parameters
        self._processor_processing_params = processing_parameters

        if 0 in self._processor_predictors.shape:
            self._processor_predictors = np.zeros((len(self._processor_subjects), 0))
        if 0 in self._processor_correctors.shape:
            self._processor_correctors = np.zeros((len(self._processor_subjects), 0))

        self._processor_progress = 0.0

        if len(user_defined_parameters) != 0:
            self._processor_fitter = self.__fitter__(user_defined_parameters)
        else:
            self._processor_fitter = self.__fitter__(self.__read_user_defined_parameters__(
                self._processor_predictors_names,
                self._processor_correctors_names
            ))

    @property
    def subjects(self):
        """
        List of subjects (Subject objects) of this instance.

        Returns
        -------
        List<Subject>
            List of subjects
        """
        return self._processor_subjects

    @property
    def correctors(self):
        """
        Matrix of correctors of this instance

        Returns
        -------
        numpy.array (NxC)
            Values of the features of the subjects that are to be used as correctors in the fitter, where N is the
            number of subjects and C the number of correctors
        """
        return self._processor_correctors

    @property
    def predictors(self):
        """
        Matrix of predictors of this instance.

        Returns
        -------
        numpy.array (NxR)
            Values of the features of the subjects that are to be used as predictors in the fitter, where N is the
            number of subjects and R the number of predictors
        """
        return self._processor_predictors

    @property
    def progress(self):
        """
        Progress (percentage of data processed) of the last call to process. If it has not been called yet,
        this property will be 0.0, whereas if the task is already completed it will be 100.0.

        Returns
        -------
        float
            Percentage of data processed
        """
        return int(self._processor_progress) / 100.0

    @property
    def fitter(self):
        """
        Fitter used in this processor

        Returns
        -------
        CurveFitter
            Instance of CurveFitter subclass used by this processor
        """
        return self._processor_fitter

    def get_name(self):
        """
        Name of the processor, that is, with which fitting method the processor interfaces

        Returns
        -------
        String
            The name of the processor
        """
        return 'Base'

    @abstractmethod
    def __fitter__(self, user_defined_parameters):
        """
        [Abstract method] Initializes the fitter to be used to process the data.
        This method is not intended to be used outside the Processor class.

        Parameters
        ----------
        user_defined_parameters : tuple
            tuple of ints, containing the additional parameters (apart from correctors
            and predictors) necessary to successfully initialize a new instance of a fitter (see
            __read_user_defined_parameters__ method).

        Returns
        -------
        CurveFitter
            A fully initialized instance of a CurveFitter subclass

        Notes
        -----
        This method is always called in the initialization of an instance of the Processing class, which means
        that you can consider it as the __init__ of the subclass (you can declare the variables that you
        would otherwise initialize in the __init__ method of the subclass here, in the __fitter__ method).
        """
        raise NotImplementedError

    @abstractmethod
    def __user_defined_parameters__(self, fitter):
        """
        [Abstract method] Gets the additional parameters obtained from the user and used by the __fitter__ method
        to initialize the fitter.
        This method is not intended to be used outside the Processor class.

        Parameters
        ----------
        fitter : CurveFitter
            A fully initialized instance of a subclass of the CurveFitter class

        Returns
        -------
        tuple
            Tuple with the values of the additional parameters (apart from the correctors and predictors) that
            have been used to succesfully initialize and use the fitter of this instance.

        Notes
        -----
        This method must output a tuple that is recognizable by the __fitter__ method when fed to it as the
        'user_defined_parameters' argument, allowing it to initialize a new fitter equal to the one of this
        instance (see __read_user_defined_parameters__)
        """
        raise NotImplementedError

    @property
    def user_defined_parameters(self):
        """
        User defined parameters for this processor

        Returns
        -------
        tuple
            tuple of numerical elements, containing the coded additional parameters (apart from correctors
            and predictors) necessary to successfully initialize a new instance of a fitter
        """
        return self.__user_defined_parameters__(self._processor_fitter)

    @abstractmethod
    def __read_user_defined_parameters__(self, predictor_names, corrector_names):
        """
        [Abstract method] Read the additional parameters (apart from correctors and predictors)
        necessary to succesfully initialize a new instance of the fitter from the user

        Parameters
        ----------
        predictor_names : List<String>
            List of subject attributes that represent the names of the features to be used as predictors
        corrector_names : List<String>
            Iterable of subject attributes that represent the names of the features to be used as correctors

        Returns
        -------
        tuple
            Tuple of numerical elements, containing the coded additional parameters (apart from correctors
            and predictors) necessary to succesfully initialize a new instance of a fitter

        Notes
        -----
        - This method is responsible for obtaining the values of such additional parameters from the user.

        - Please, make use of the 'getter' methods implemented in this class for such purpose. This will
        allow future subclasses to implement additional features (such as a GUI) by just overriding the
        'getter' methods, consequently making it easier to maintain, expand and provide more features
        together with a larger functionality for the abstract subclasses of Processing without requiring
        any additional work from the developers that implement the concrete subclasses of the same class.

        - When calling the 'getter' methods, make sure you use the 'super' built-in function, so that the Method
        Resolution Order is dynamically adapted and you get to use the methods implemented in the bottom-most
        subclass of Processing in the inheritance tree.

        - The 'getter' methods are of the form __get***__, where *** denotes the value to be obtained from the user.
        Here is a potentially non-exhaustive list of such methods: __getint__, __getfloat__, __getoneof__,
        __getoneinrange__, __getyesorno__, ...
        """
        raise NotImplementedError

    # TODO Document
    def __processor_update_progress(self, prog_inc):
        """

        Parameters
        ----------
        prog_inc

        Returns
        -------

        """
        self._processor_progress += prog_inc
        print '\r  ' + str(int(self._processor_progress) / 100.0) + '%',
        if self._processor_progress == 10000.0:
            print
        stdout.flush()

    # TODO Document
    def process(self, x1=0, x2=None, y1=0, y2=None, z1=0, z2=None, *args, **kwargs):
        """

        Parameters
        ----------
        x1
        x2
        y1
        y2
        z1
        z2
        args
        kwargs

        Returns
        -------

        """
        chunks = Utils.Subject.Chunks(
            self._processor_subjects,
            x1=x1, y1=y1, z1=z1, x2=x2, y2=y2, z2=z2,
            mem_usage=self._processor_processing_params['mem_usage']
        )
        dims = chunks.dims

        # Initialize progress
        self._processor_progress = 0.0
        total_num_voxels = dims[-3] * dims[-2] * dims[-1]
        prog_inc = 10000. / total_num_voxels

        # Add processing parameters to kwargs
        kwargs['n_jobs'] = self._processor_processing_params['n_jobs']
        kwargs['cache_size'] = self._processor_processing_params['cache_size']

        # Get the results of the first chunk to initialize dimensions of the solution matrices
        # Get first chunk and fit the parameters
        chunk = chunks.next()

        self._processor_fitter.fit(chunk.data, *args, **kwargs)

        # Get the parameters and the dimensions of the solution matrices
        cparams = self._processor_fitter.correction_parameters
        pparams = self._processor_fitter.prediction_parameters
        cpdims = tuple(cparams.shape[:-3] + dims)
        rpdims = tuple(pparams.shape[:-3] + dims)

        # Initialize solution matrices
        correction_parameters = np.zeros(cpdims, dtype=np.float64)
        prediction_parameters = np.zeros(rpdims, dtype=np.float64)

        # Assign first chunk's solutions to solution matrices
        dx, dy, dz = cparams.shape[-3:]
        correction_parameters[:, :dx, :dy, :dz] = cparams
        prediction_parameters[:, :dx, :dy, :dz] = pparams

        # Update progress
        self.__processor_update_progress(prog_inc * dx * dy * dz)

        # Now do the same for the rest of the Chunks
        for chunk in chunks:
            # Get relative (to the solution matrices) coordinates of the chunk
            x, y, z = chunk.coords
            x -= x1
            y -= y1
            z -= z1

            # Get chunk data and its dimensions
            cdata = chunk.data
            dx, dy, dz = cdata.shape[-3:]

            # Fit the parameters to the data in the chunk
            self._processor_fitter.fit(cdata, *args, **kwargs)

            # Get the optimal parameters and insert them in the solution matrices
            correction_parameters[:, x:x + dx, y:y + dy, z:z + dz] = self._processor_fitter.correction_parameters
            prediction_parameters[:, x:x + dx, y:y + dy, z:z + dz] = self._processor_fitter.prediction_parameters

            # Update progress
            self.__processor_update_progress(prog_inc * dx * dy * dz)

        if self.progress != 100.0:
            self.__processor_update_progress(10000.0 - self._processor_progress)

        # Call post_processing routine
        return self.__post_process__(prediction_parameters, correction_parameters)

    def __post_process__(self, prediction_parameters, correction_parameters):
        """
        [Private method] Allows the post-processing of the prediction and correction parameters
        after the process() method has finished. By default returns the same prediction and correction
        parameters that were found in process.

        Parameters
        ----------
        prediction_parameters : numpy.array
            Parameters found for the predictors
        correction_parameters : numpy.array
            Parameters found for the correctors

        Returns
        -------
        Processor.Results
            Results instance with the post-processed prediction and correction parameters
        """
        return Processor.Results(prediction_parameters, correction_parameters)

    def __pre_process__(self, prediction_parameters, correction_parameters, predictors, correctors):
        """
        [Private method] Allows the pre-processing of the prediction and correction
        parameters before other methods are called (e.g. curve(), evaluate_fit() ).

        Parameters
        ----------
        prediction_parameters : numpy.array
            Prediction parameters to be pre-processed
        correction_parameters : numpy.array
            Correction parameters to be pre-processed
        predictors : numpy.array
            Predictors used to compute the prediction parameters
        correctors : numpy.array
            Correctors used to compute the correction parameters

        Returns
        -------
        tuple(numpy.array, numpy.array)
            The pre-processed prediction_parameters and correction_parameters in this particular order
        """
        return prediction_parameters, correction_parameters

    # TODO: Document
    def __curve__(self, fitter, predictor, prediction_parameters):
        """
        Computes a prediction from the predictor and the prediction_parameters. If not overridden, this method
        calls the 'predict' function of the fitter passing as arguments the predictors and prediction parameters
        as they are. Please, override this method if this is not the desired behavior.
        This method is not intended to be called outside the Processor class

        Parameters
        ----------
        fitter
        predictor
        prediction_parameters

        Returns
        -------

        """
        return fitter.predict(predictor, prediction_parameters)

    # TODO: Document
    def curve(self, prediction_parameters, x1=0, x2=None, y1=0, y2=None, z1=0, z2=None, t1=None, t2=None, tpoints=50):
        """
        Computes tpoints predicted values in the axis of the predictor from t1 to t2 by using the results of
        a previous execution for each voxel in the relative region [x1:x2, y1:y2, z1:z2]. (Only valid for
        one predictor)

        Parameters
        ----------
        prediction_parameters
        x1
        x2
        y1
        y2
        z1
        z2
        t1
        t2
        tpoints

        Returns
        -------

        """
        if x2 is None:
            x2 = prediction_parameters.shape[-3]
        if y2 is None:
            y2 = prediction_parameters.shape[-2]
        if z2 is None:
            z2 = prediction_parameters.shape[-1]

        if t1 is None:
            t1 = self._processor_predictors.min()
        if t2 is None:
            t2 = self._processor_predictors.max()

        pparams = prediction_parameters[:, x1:x2, y1:y2, z1:z2]

        if tpoints == -1:
            preds = np.sort(np.squeeze(self._processor_predictors))[:, np.newaxis]
        else:
            preds = np.zeros((tpoints, 1), dtype=np.float64)
            step = float(t2 - t1) / (tpoints - 1)
            t = t1
            for i in xrange(tpoints):
                preds[i][0] = t
                t += step

        return preds.T[0], self.__curve__(self._processor_fitter, preds, pparams)

    # TODO: Document
    def __corrected_values__(self, fitter, observations, correction_parameters, *args, **kwargs):
        """

        Parameters
        ----------
        fitter
        observations
        correction_parameters
        args
        kwargs

        Returns
        -------

        """
        return fitter.correct(observations=observations, correction_parameters=correction_parameters)

    # TODO: Document
    def corrected_values(self, correction_parameters, x1=0, x2=None, y1=0, y2=None, z1=0, z2=None, origx=0, origy=0,
                         origz=0, *args, **kwargs):
        """
        x1, x2, y1, y2, z1 and z2 are relative coordinates to (origx, origy, origz), being the latter coordinates
        in absolute value (by default, (0, 0, 0)); that is, (origx + x, origy + y, origz + z) is the point to
        which the correction parameters in the voxel (x, y, z) of 'correction_parameters' correspond

        Parameters
        ----------
        correction_parameters
        x1
        x2
        y1
        y2
        z1
        z2
        origx
        origy
        origz
        args
        kwargs

        Returns
        -------

        """
        if x2 is None:
            x2 = correction_parameters.shape[-3]
        if y2 is None:
            y2 = correction_parameters.shape[-2]
        if z2 is None:
            z2 = correction_parameters.shape[-1]

        correction_parameters = correction_parameters[:, x1:x2, y1:y2, z1:z2]

        x1 += origx
        x2 += origx
        y1 += origy
        y2 += origy
        z1 += origz
        z2 += origz

        chunks = Utils.Subject.Chunks(self._processor_subjects, x1=x1, y1=y1, z1=z1, x2=x2, y2=y2, z2=z2,
                                      mem_usage=self._processor_processing_params['mem_usage'])
        dims = chunks.dims

        corrected_data = np.zeros(tuple([chunks.num_subjects]) + dims, dtype=np.float64)

        for chunk in chunks:
            # Get relative (to the solution matrix) coordinates of the chunk
            x, y, z = chunk.coords
            x -= x1
            y -= y1
            z -= z1

            # Get chunk data and its dimensions
            cdata = chunk.data
            dx, dy, dz = cdata.shape[-3:]

            corrected_data[:, x:(x + dx), y:(y + dy), z:(z + dz)] = self.__corrected_values__(self._processor_fitter,
                                                                                              cdata,
                                                                                              correction_parameters[:,
                                                                                              x:(x + dx), y:(y + dy),
                                                                                              z:(z + dz)], *args,
                                                                                              **kwargs)

        return corrected_data

    # TODO Document
    def gm_values(self, x1=0, x2=None, y1=0, y2=None, z1=0, z2=None):
        """


        Parameters
        ----------
        x1
        x2
        y1
        y2
        z1
        z2

        Returns
        -------

        """
        chunks = Utils.Subject.Chunks(self._processor_subjects, x1=x1, y1=y1, z1=z1, x2=x2, y2=y2, z2=z2,
                                      mem_usage=self._processor_processing_params['mem_usage'])
        dims = chunks.dims

        gm_data = np.zeros(tuple([chunks.num_subjects]) + dims, dtype=np.float64)

        for chunk in chunks:
            # Get relative (to the solution matrix) coordinates of the chunk
            x, y, z = chunk.coords
            x -= x1
            y -= y1
            z -= z1

            # Get chunk data and its dimensions
            cdata = chunk.data
            dx, dy, dz = cdata.shape[-3:]

            gm_data[:, x:(x + dx), y:(y + dy), z:(z + dz)] = cdata

        return gm_data

    # TODO: should analyze the surroundings of the indicated region even if they are not going to be displayed
    # since such values affect the values inside the region (if not considered, the clusters could potentially
    # seem smaller and thus be filtered accordingly)
    # TODO Document
    def evaluate_fit(self, evaluation_function, correction_parameters, prediction_parameters, x1=0, x2=None, y1=0,
                     y2=None, z1=0, z2=None, origx=0, origy=0, origz=0, gm_threshold=None, filter_nans=True,
                     default_value=0.0, *args, **kwargs):
        """


        Parameters
        ----------
        evaluation_function
        correction_parameters
        prediction_parameters
        x1
        x2
        y1
        y2
        z1
        z2
        origx
        origy
        origz
        gm_threshold
        filter_nans
        default_value
        args
        kwargs

        Returns
        -------

        """
        # Preprocess parameters
        orig_pparams = prediction_parameters
        prediction_parameters, correction_parameters = self.__pre_process__(
            prediction_parameters,
            correction_parameters,
            self.predictors,
            self.correctors
        )

        # Evaluate fitting from pre-processed parameters

        if correction_parameters.shape[-3] != prediction_parameters.shape[-3] or correction_parameters.shape[-2] != \
                prediction_parameters.shape[-2] or correction_parameters.shape[-1] != prediction_parameters.shape[-1]:
            raise ValueError('The dimensions of the correction parameters and the prediction parameters do not match')

        if x2 is None:
            x2 = x1 + correction_parameters.shape[-3]
        if y2 is None:
            y2 = y1 + correction_parameters.shape[-2]
        if z2 is None:
            z2 = z1 + correction_parameters.shape[-1]

        orig_pparams = orig_pparams[:, x1:x2, y1:y2, z1:z2]
        correction_parameters = correction_parameters[:, x1:x2, y1:y2, z1:z2]
        prediction_parameters = prediction_parameters[:, x1:x2, y1:y2, z1:z2]

        x1 += origx
        x2 += origx
        y1 += origy
        y2 += origy
        z1 += origz
        z2 += origz

        chunks = Utils.Subject.Chunks(self._processor_subjects, x1=x1, y1=y1, z1=z1, x2=x2, y2=y2, z2=z2,
                                      mem_usage=self._processor_processing_params['mem_usage'])
        dims = chunks.dims

        # Initialize solution matrix
        fitting_scores = np.zeros(dims, dtype=np.float64)

        if gm_threshold is not None:
            # Instead of comparing the mean to the original gm_threshold,
            # we compare the sum to such gm_threshold times the number of subjects
            gm_threshold *= chunks.num_subjects
            invalid_voxels = np.zeros(fitting_scores.shape, dtype=np.bool)

        # Initialize progress
        self._processor_progress = 0.0
        total_num_voxels = dims[-3] * dims[-2] * dims[-1]
        prog_inc = 10000. / total_num_voxels

        # Evaluate the fit for each chunk
        for chunk in chunks:
            # Get relative (to the solution matrices) coordinates of the chunk
            x, y, z = chunk.coords
            x -= x1
            y -= y1
            z -= z1

            # Get chunk data and its dimensions
            cdata = chunk.data
            dx, dy, dz = cdata.shape[-3:]

            if gm_threshold is not None:
                invalid_voxels[x:(x + dx), y:(y + dy), z:(z + dz)] = np.sum(cdata, axis=0) < gm_threshold

            # Create auxiliar structure to access chunk data inside the evaluation function
            class FittingResults(object):
                pass

            fitres = FittingResults()

            fitres.observations = cdata
            fitres.corrected_data = self._processor_fitter.correct(
                observations=cdata,
                correctors=self._processor_fitter.correctors,
                correction_parameters=correction_parameters[:, x:(x + dx), y:(y + dy), z:(z + dz)]
            )
            fitres.predicted_data = self._processor_fitter.predict(
                predictors=self._processor_fitter.predictors,
                prediction_parameters=prediction_parameters[:, x:(x + dx), y:(y + dy), z:(z + dz)]
            )
            fitres.df_correction = self._processor_fitter.df_correction(
                observations=cdata,
                correctors=self._processor_fitter.correctors,
                correction_parameters=correction_parameters[:, x:(x + dx), y:(y + dy), z:(z + dz)]
            )
            fitres.df_prediction = self._processor_fitter.df_prediction(
                observations=cdata,
                predictors=self._processor_fitter.predictors,
                prediction_parameters=prediction_parameters[:, x:(x + dx), y:(y + dy), z:(z + dz)]
            )
            axis, curve = self.curve(
                prediction_parameters=orig_pparams[:, x:(x + dx), y:(y + dy), z:(z + dz)],
                tpoints=128  # We set a high granularity to evaluate the curve more precisely
                # Another option could be to set it to a value proportional to the number of subjects
                # tpoints = 2*len(self.target.subjects)
            )
            fitres.curve = curve
            fitres.xdiff = axis[1] - axis[0]

            # Evaluate the fit for the voxels in this chunk and store them
            fitting_scores[x:x + dx, y:y + dy, z:z + dz] = evaluation_function[self].evaluate(fitres, *args, **kwargs)

            # Update progress
            self.__processor_update_progress(prog_inc * dx * dy * dz)

        if self.progress != 100.0:
            self.__processor_update_progress(10000.0 - self._processor_progress)

        # Filter non-finite elements
        if filter_nans:
            fitting_scores[~np.isfinite(fitting_scores)] = default_value

        # Filter by gray-matter threshold
        if gm_threshold is not None:
            fitting_scores[invalid_voxels] = default_value

        return fitting_scores

    # TODO Document
    @staticmethod
    def clusterize(fitting_scores, default_value=0.0, fit_lower_threshold=None, fit_upper_threshold=None,
                   cluster_threshold=None, produce_labels=False):
        """

        Parameters
        ----------
        fitting_scores
        default_value
        fit_lower_threshold
        fit_upper_threshold
        cluster_threshold
        produce_labels

        Returns
        -------

        """

        fitscores = np.ones(fitting_scores.shape, dtype=np.float64) * default_value
        if produce_labels:
            labels = np.zeros(fitting_scores.shape, dtype=np.float64)
            label = 0

        ng = NiftiGraph(fitting_scores, fit_lower_threshold, fit_upper_threshold)
        for scc in ng.sccs():
            if len(scc) >= cluster_threshold:
                # lscc = np.array(list(scc)).T
                # fitscores[lscc] = fitting_scores[lscc]
                for x, y, z in scc:
                    fitscores[x, y, z] = fitting_scores[x, y, z]

                if produce_labels:
                    label += 1
                    for x, y, z in scc:
                        labels[x, y, z] = label

        if produce_labels:
            return fitscores, labels
        else:
            return fitscores

    # TODO Document
    @staticmethod
    def __processor_get__(obtain_input_from, apply_function, try_ntimes, default_value, show_text, show_error_text):
        """


        Parameters
        ----------
        obtain_input_from
        apply_function
        try_ntimes
        default_value
        show_text
        show_error_text

        Returns
        -------

        """
        if try_ntimes <= 0:
            try_ntimes = -1

        while try_ntimes != 0:
            s = obtain_input_from(show_text)
            if not s:
                print 'Default value selected.'
                return default_value
            else:
                try:
                    return apply_function(s)
                except Exception as exc:
                    print show_error_text(exc)

            if try_ntimes < 0:
                print 'Infinite',
            else:
                try_ntimes -= 1
                print try_ntimes,
            print 'attempts left.',

            if try_ntimes == 0:
                print 'Default value selected.'
            else:
                print 'Please, try again.'

        return default_value

    # TODO Document
    @staticmethod
    def __getint__(default_value=None,
                   try_ntimes=3,
                   lower_limit=None,
                   upper_limit=None,
                   show_text='Please, enter an integer number (or leave blank to set by default): ',
                   obtain_input_from=raw_input,
                   ):
        """


        Parameters
        ----------
        default_value
        try_ntimes
        lower_limit
        upper_limit
        show_text
        obtain_input_from

        Returns
        -------

        """

        def nit(s, lower=lower_limit, upper=upper_limit):
            x = int(s)
            if (not (lower is None)) and x < lower:
                raise ValueError('The value must be greater than or equal to ' + str(lower))
            if (not (upper is None)) and x >= upper:
                raise ValueError('The value must be smaller than ' + str(upper))
            return x

        return Processor.__processor_get__(
            obtain_input_from,
            nit,
            try_ntimes,
            default_value,
            show_text,
            lambda e: 'Could not match input with integer number: ' + str(e)
        )

    # TODO Document
    @staticmethod
    def __getfloat__(default_value=None,
                     try_ntimes=3,
                     lower_limit=None,
                     upper_limit=None,
                     show_text='Please, enter a real number (or leave blank to set by default): ',
                     obtain_input_from=raw_input,
                     ):
        """


        Parameters
        ----------
        default_value
        try_ntimes
        lower_limit
        upper_limit
        show_text
        obtain_input_from

        Returns
        -------

        """

        def olfat(s, lower=lower_limit, upper=upper_limit):
            x = float(s)
            if (not (lower is None)) and x < lower:
                raise ValueError('The value must be greater than or equal to ' + str(lower))
            if (not (upper is None)) and x >= upper:
                raise ValueError('The value must be smaller than ' + str(upper))
            return x

        return Processor.__processor_get__(
            obtain_input_from,
            olfat,
            try_ntimes,
            default_value,
            show_text,
            lambda e: 'Could not match input with real number: ' + str(e)
        )

    # TODO Document
    @staticmethod
    def __getoneof__(option_list,
                     default_value=None,
                     try_ntimes=3,
                     show_text='Please, select one of the following (enter index, or leave blank to set by default):',
                     obtain_input_from=raw_input,
                     ):
        """


        Parameters
        ----------
        option_list
        default_value
        try_ntimes
        show_text
        obtain_input_from

        Returns
        -------

        """
        opt_list = list(option_list)
        lol = len(opt_list)
        lslol = len(str(lol))
        right_justify = lambda s: ' ' * (lslol - len(str(s))) + str(s)

        new_show_text = show_text
        for i in xrange(lol):
            new_show_text += '\n  ' + right_justify(i) + ':  ' + str(opt_list[i])
        new_show_text += '\nYour choice: '

        def get_index(s, ls=lol):
            index = int(s)
            if index < 0 or index >= ls:
                raise IndexError('Index ' + s + ' is out of the accepted range [0, ' + str(ls) + '].')
            return index

        index = Processor.__processor_get__(
            obtain_input_from,
            get_index,
            try_ntimes,
            None,
            new_show_text,
            lambda e: 'Could not match input with index: ' + str(e)
        )
        if index == None:
            return default_value

        return opt_list[index]

    # TODO Document
    @staticmethod
    def __getoneinrange__(start,
                          end,
                          step=0,
                          default_value=None,
                          try_ntimes=3,
                          show_text='Please, enter a number in the range',
                          obtain_input_from=raw_input
                          ):
        """


        Parameters
        ----------
        start
        end
        step
        default_value
        try_ntimes
        show_text
        obtain_input_from

        Returns
        -------

        """
        if show_text == 'Please, enter a number in the range':
            show_text += ' [' + str(start) + ', ' + str(end) + ')'
            if step > 0:
                show_text += ' with a step of ' + str(step)
            show_text += '(or leave blank to set by default): '

        def inrange(s, start=start, end=end, step=step):
            f = float(s)
            if f >= end or f < start:
                raise ValueError('Input value is not in specified range.')
            if step > 0:
                # round number to its nearest step
                num_step = int((f - start) / step + 0.5)  # round(x) = floor(x + 0.5) = int(x + 0.5)
                f = start + num_step * step
            return f

        return Processor.__processor_get__(
            obtain_input_from,
            inrange,
            try_ntimes,
            default_value,
            show_text,
            lambda e: 'Could not read input value: ' + str(e)
        )

    # TODO Document
    @staticmethod
    def __getyesorno__(default_value=None,
                       try_ntimes=3,
                       show_text='Select yes (Y/y) or no (N/n), or leave blank to set by default: ',
                       obtain_input_from=raw_input
                       ):
        """


        Parameters
        ----------
        default_value
        try_ntimes
        show_text
        obtain_input_from

        Returns
        -------

        """
        def yesorno(s2):
            s = s2.strip()
            if s == 'y' or s == 'Y':
                return True
            if s == 'n' or s == 'N':
                return False
            raise ValueError('Option not recognized.')

        return Processor.__processor_get__(
            obtain_input_from,
            yesorno,
            try_ntimes,
            default_value,
            show_text,
            lambda e: 'Could not match input with any of the options: ' + str(e)
        )


eval_func[Processor].bind(
    'curve',
    lambda self: self.fitting_results.curve
)
eval_func[Processor].bind(
    'xdiff',
    lambda self: self.fitting_results.xdiff
)
eval_func[Processor].bind(
    'corrected_data',
    lambda self: self.fitting_results.corrected_data
)
eval_func[Processor].bind(
    'predicted_data',
    lambda self: self.fitting_results.predicted_data
)
eval_func[Processor].bind(
    'df_correction',
    lambda self: self.fitting_results.df_correction
)
eval_func[Processor].bind(
    'df_prediction',
    lambda self: self.fitting_results.df_prediction
)
