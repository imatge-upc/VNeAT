from os import path

import nibabel as nib
import numpy as np
from scipy.stats import norm
from sklearn.cluster import KMeans

from vneat.Processors.MixedProcessor import MixedProcessor
from vneat.Utils.DataLoader import DataLoader


def load_data_from_config_file(config_file):
    """
    Loads all the data specified in the configuration file using src.Utils.DataLoader.DataLoader

    Parameters
    ----------
    config_file : String
        Path to the YAML configuration file that DataLoader uses to load the data

    Returns
    -------
    List
        List of the subjects of the study
    List
        List of the names of the predictors
    List
        List of the names of the correctors
    np.array
        Array with the predictors' data
    np.array
        Array with the correctors' data
    Dictionary
        Dictionary with the processing parameters
    np.array
        Array that contains the affine matrix to go from the voxel to the mm space and viceversa
    String
        Path to the output directory specified in the configuration file
    """
    print('Loading configuration data...')
    try:
        data_loader = DataLoader(config_file)
    except IOError as e:
        print()
        print(e.filename + ' does not exist.')
        data_loader = None
        exit(1)

    # Load all necessary data:
    try:
        subjects = data_loader.get_subjects()
        predictors_names = data_loader.get_predictor_name()
        correctors_names = data_loader.get_correctors_names()
        predictor = data_loader.get_predictor()
        correctors = data_loader.get_correctors()
        processing_parameters = data_loader.get_processing_parameters()
        affine_matrix = data_loader.get_template_affine()
        output_dir = data_loader.get_output_dir()
        results_io = data_loader.get_results_io()
        type_data = volume_or_surface(data_loader.get_extension())
        return subjects, predictors_names, correctors_names, predictor, correctors, processing_parameters, \
               affine_matrix, output_dir, results_io, type_data

    except KeyError:
        print()
        print('Configuration file does not have the specified format.')
        print('See config/exampleConfig.yaml for further information about the format of configuration '
              'files')
        exit(1)


def load_hyperparams_from_config_file(config_file, fitting_method):
    """
    Loads all the data specified in the configuration file using src.Utils.DataLoader.DataLoader

    Parameters
    ----------
    config_file : String
        Path to the YAML configuration file that DataLoader uses to load the data
    fitting_method : String
        String that identifies the fitting method (PolySVR or GaussianSVR)

    Returns
    -------
    Dictionary
        The keys of the dictionary are the name of the hyperparameter and the values the numpy array
        containing all the possible values amongst which the optimal will be found.
    """
    print('Loading hyperparameters data...')
    try:
        data_loader = DataLoader(config_file)
    except IOError as e:
        print()
        print(e.filename + ' does not exist.')
        data_loader = None
        exit(1)

    # Load all necessary data:
    try:
        return data_loader.get_hyperparams_finding_configuration(fitting_method=fitting_method)
    except KeyError:
        print()
        print('Configuration file does not have the specified format.')
        print('See config/exampleConfig.yaml for further information about the format of configuration '
              'files')
        exit(1)


def load_template_from_config_file(config_file):
    """
    Loads the template used to co-register all the subjects from the configuration file

    Parameters
    ----------
    config_file : String
        Path to the YAML configuration file that DataLoader uses to load the data

    Returns
    -------
    ndarray
        3D numpy array with the template image
    """
    print('Loading template data...')
    try:
        data_loader = DataLoader(config_file)
    except IOError as e:
        print()
        print(e.filename + ' does not exist.')
        data_loader = None
        exit(1)

    # Load all necessary data:
    try:
        return data_loader.get_template()
    except KeyError:
        print()
        print('Configuration file does not have the specified format.')
        print('See config/exampleConfig.yaml for further information about the format of configuration '
              'files')
        exit(1)


def get_results_from_path(pred_params_path, results_io, subjects, predictors_names, correctors_names,
                          predictors, correctors, processing_parameters, type_data):
    """
    Returns the all the fitting results previously computed by compute_fitting.py found in the
    specified path for the prediction parameters.

    Parameters
    ----------
    pred_params_path : String
        Path to the prediction parameters, where the rest of the results are also found
    results_io : niftiIO.Results class
        Path to the prediction parameters, where the rest of the results are also found
    subjects : List
        List of subjects loaded using the DataLoader class
    predictors_names : List
        List of the names of the predictors
    correctors_names : List
        List of the names of the correctors
    predictors : np.array
        Array with the predictors' data
    correctors : np.array
        Array with the correctors' data
    processing_parameters : Dictionary
        Dictionary with the necessary processing parameters, that is, 'mem_usage', 'n_jobs' and 'cache_size'
    type_data : String
        Type of data (surf or vol)

    Returns
    -------
    String
        Name of the processor that computed the found results
    int
        Category of the loaded data
    np.array
        Array with the prediction parameters
    np.array
        Array with the correction parameters
    MixedProcessor
        Instance of MixedProcessor with the same parameters as the one used to compute the fitting results
    """
    # From path found by glob infer the paths to the other files
    # (correction parameters and user defined parameters)
    # and the name of the fitting method
    folder_path, prediction_params_name = path.split(pred_params_path)
    prefix = prediction_params_name.replace('prediction_parameters' + results_io.extension, '')
    prediction_params_path = pred_params_path
    correction_params_path = path.join(folder_path, ('{}correction_parameters' + results_io.extension).format(prefix))
    udp_path = path.join(folder_path, '{}user_defined_parameters.txt'.format(prefix))

    # Try to infer whether there is a curve for a category or not by folder name
    folder_name = path.basename(folder_path)
    if 'category' in folder_name:
        cat_name = folder_name.split('-')[-1].replace('_', ' ')
        if 'all' in cat_name:
            cat = None
        else:
            cat = int(cat_name.split(' ')[-1])
        name = '-'.join(folder_name.split('-')[:-1])
    else:
        cat_name = None
        cat = None
        name = folder_name
    # Load niftis and txt files and keep them
    with open(udp_path, 'rb') as udp_file:
        udp = eval(udp_file.read())

    pred_parameters = results_io.loader(prediction_params_path).get_data()
    corr_parameters = results_io.loader(correction_params_path).get_data()

    # Create MixedProcessor and keep it
    processor = MixedProcessor(
        subjects,
        predictors_names,
        correctors_names,
        predictors,
        correctors,
        processing_parameters,
        user_defined_parameters=udp,
        category=cat,
        type_data=type_data
    )

    return name, cat_name, pred_parameters, corr_parameters, processor


def compute_fitting_scores(processor_instance, method_name, method_func, pparams, cparams, cluster_size,
                           p_thresholds, gamma, percentile_filter, gm_threshold, labels):
    """
    Computes the specific fitting scores for a given method (e.g vnPRSS)

    Parameters
    ----------
    processor_instance : MixedProcessor
        Instance of a MixedProcessor
    method_name : String
        Name of the method to compute the fitting scores (e.g f_stat)
    method_func : src.FitScores.FitEvaluation.evaluation_function
        Evaluation function that computes the fitting scores
    pparams : np.array
        Array with prediction parameters
    cparams : np.array
        Array with correction parameters
    cluster_size : int
        Integer number that specifies the size (in voxels) of the cluster
    p_thresholds : List
        List of p-value thresholds for which the fitting method (if possible) filters its results
        (e.g [0.01, 0.005, 0.001]
    gamma : float
        Weighting value for PRSS and vn-PRSS scores that indicates how much does the roughness of the curve
        penalize the score (i.e a big gamma value means that the lowest score would be for a curve that,
        even if it doesn't fit very well the data, is very smooth)
    percentile_filter : float
        Value of the percentile used to determine the upper threshold for PRSS and vnPRSS methods
    gm_threshold : float
        Mean grey-matter lower threshold
    labels : Boolean
        Whether to produce a labeled clusters map or not

    Returns
    -------
    List<Tuple>
        List of tuples with the following format: (name, fitting_score). The list will contain as many
        tuples as fitting scores an evaluation method is supposed to compute,
        That includes the filtered, clusterd and/or transformed scores that can be computed
        for each evaluation method.
    """
    returned_results = []  # Variable where tuples of (name, fitting_score) are stored to be returned
    if method_name == 'vnprss' or method_name == 'prss':
        fitting_scores = processor_instance.evaluate_fit(
            evaluation_function=method_func,
            correction_parameters=cparams,
            prediction_parameters=pparams,
            x1=0,
            x2=None,
            y1=0,
            y2=None,
            z1=0,
            z2=None,
            gm_threshold=gm_threshold,
            filter_nans=True,
            default_value=np.inf,
            gamma=gamma
        )
    else:
        fitting_scores = processor_instance.evaluate_fit(
            evaluation_function=method_func,
            correction_parameters=cparams,
            prediction_parameters=pparams,
            x1=0,
            x2=None,
            y1=0,
            y2=None,
            z1=0,
            z2=None,
            gm_threshold=gm_threshold,
            filter_nans=True,
            default_value=0.0
        )
    fit_scores_name = '{}_fitscores'.format(method_name)
    returned_results.append((fit_scores_name, fitting_scores))
    if method_name == 'ftest':
        for p_threshold in p_thresholds:
            inv_p_threshold = 1 - p_threshold
            if labels:
                print('Filtering and clustering p-values using threshold {}, and generating labels...'
                      .format(p_threshold))
                clusterized_fitting_scores, labels_map = MixedProcessor.clusterize(
                    fitting_scores,
                    default_value=0.0,
                    fit_lower_threshold=inv_p_threshold,
                    cluster_threshold=cluster_size,
                    produce_labels=True
                )
                name_labels = 'labels_{}'.format(inv_p_threshold)
                returned_results.append((name_labels, labels_map))
            else:
                print('Filtering and clustering p-values using threshold {}...'.format(p_threshold))
                clusterized_fitting_scores = MixedProcessor.clusterize(
                    fitting_scores,
                    default_value=0.0,
                    fit_lower_threshold=inv_p_threshold,
                    cluster_threshold=cluster_size,
                    produce_labels=False
                )
            print('Converting p-values to Z-scores...')
            lim_value = norm.ppf(inv_p_threshold)
            valid_voxels = clusterized_fitting_scores != 0.0
            clusterized_fitting_scores[valid_voxels] = norm.ppf(
                clusterized_fitting_scores[valid_voxels]
            ) - lim_value + 0.2
            name_z_scores = 'z-scores_{}'.format(inv_p_threshold)
            returned_results.append((name_z_scores, clusterized_fitting_scores))

    elif method_name == 'prss' or method_name == 'vnprss':
        print('Filtering and transforming scores for better visualization...')
        valid_voxels = np.isfinite(fitting_scores)
        sorted_scores = np.sort(fitting_scores[valid_voxels].reshape(-1))
        num_elems = int(np.ceil(percentile_filter * float(fitting_scores.size)))
        threshold = sorted_scores[:num_elems][-1]
        valid_voxels = fitting_scores <= threshold
        fitting_scores[~valid_voxels] = 0.0
        fitting_scores[valid_voxels] = threshold - fitting_scores[valid_voxels]  # Invert values
        filtered_scores_name = 'invfiltered_{}_gamma{}_percentile{}'.format(
            method_name,
            gamma,
            percentile_filter
        )
        returned_results.append((filtered_scores_name, fitting_scores))
    elif method_name == 'aic':
        print('Transforming and filtering AIC fit scores for better visualization...')
        masked_scores = np.ma.masked_equal(np.abs(fitting_scores), 0)
        fit_scores_max = masked_scores.max()
        fit_scores_min = masked_scores.min()
        transformed_scores = (masked_scores - fit_scores_min) / (fit_scores_max - fit_scores_min)
        mean = np.mean(transformed_scores)
        transformed_scores[transformed_scores < mean] = 0
        returned_results.append(('aic_filtered_fitscores', transformed_scores))

    return returned_results


def compute_latent_effect_strength_type(processor_instance, pparams, cparams, cluster_size,
                                        p_thresholds, n_permutation, n_clusters, gm_threshold, labels):
    """
        Computes effect strenght, p-value and effect-type of each latent factor of the analysis.

        Parameters
        ----------
        processor_instance : MixedProcessor
            Instance of a MixedProcessor
        pparams : np.array
            Array with prediction parameters
        cparams : np.array
            Array with correction parameters
        cluster_size : int
            Integer number that specifies the size (in voxels) of the cluster
        p_thresholds : List
            List of p-value thresholds for which the fitting method (if possible) filters its results
            (e.g [0.01, 0.005, 0.001].
        n_permutation : None or float
            Number of permutations to perform statistical permutation testing.
            If None, no permutation testing is performed.
        n_clusters : None, float or list of floats
            Number of clusters to compute effect type clustering. If None, no clustering is performed
        gm_threshold : float
            Mean grey-matter lower threshold
        labels : Boolean
            Whether to produce a labeled clusters map or not

        Returns
        -------
        List<Tuple>
            List of tuples with the following format: (name, fitting_score). The list will contain as many
            tuples as fitting scores an evaluation method is supposed to compute,
            That includes the filtered, clusterd and/or transformed scores that can be computed
            for each evaluation method.
        """
    returned_results = []
    effect_strength, p_value_strength, effect_type = processor_instance.evaluate_latent_space(
        correction_parameters=cparams,
        prediction_parameters=pparams,
        x1=0,
        x2=None,
        y1=0,
        y2=None,
        z1=0,
        z2=None,
        gm_threshold=gm_threshold,
        filter_nans=True,
        default_value=0,
        n_permutations=n_permutation
    )

    for it_nc in range(effect_strength.shape[0]):
        effect_strength_name = 'effect-strenght-map_{}'.format(it_nc)
        returned_results.append((effect_strength_name, effect_strength[it_nc]))

        p_value_name = 'p-value-map_{}'.format(it_nc)
        returned_results.append((p_value_name, p_value_strength[it_nc]))

        effect_type_name = 'effect-type_{}'.format(it_nc)
        returned_results.append((effect_type_name, effect_type[it_nc]))

    #Effect strenght and p-value
    for it_nc in range(effect_strength.shape[0]):

        for p_threshold in p_thresholds:
            inv_p_threshold = 1 - p_threshold
            if labels:
                print('Filtering and clustering p-values using threshold {}, and generating labels...'
                      .format(p_threshold))
                clusterized_fitting_scores, labels_map = MixedProcessor.clusterize(
                    p_value_strength[it_nc],
                    default_value=0.0,
                    fit_lower_threshold=inv_p_threshold,
                    cluster_threshold=cluster_size,
                    produce_labels=True
                )
                name_labels = 'labels_{}_{}'.format(it_nc,inv_p_threshold)
                returned_results.append((name_labels, labels_map))
            else:
                print('Filtering and clustering p-values using threshold {}...'.format(p_threshold))
                clusterized_fitting_scores = MixedProcessor.clusterize(
                    p_value_strength[it_nc],
                    default_value=0.0,
                    fit_lower_threshold=inv_p_threshold,
                    cluster_threshold=cluster_size,
                    produce_labels=False
                )
            print('Converting p-values to Z-scores...')
            lim_value = norm.ppf(inv_p_threshold)
            valid_voxels = clusterized_fitting_scores != 0.0
            clusterized_fitting_scores[valid_voxels] = norm.ppf(
                clusterized_fitting_scores[valid_voxels]
            ) - lim_value + 0.2
            z_scores_name = 'z-scores_{}_{}'.format(it_nc,inv_p_threshold)
            returned_results.append((z_scores_name, clusterized_fitting_scores))


            if n_clusters is not None:
                print('Clustering effect_type into ' + str(n_clusters) + ' clusters ...')
                image_shape = effect_strength.shape
                kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(effect_type[it_nc].reshape(-1,np.prod(image_shape)).T)
                label_map = kmeans.predict(effect_type[it_nc].reshape(-1,np.prod(image_shape)).T)

                effect_type_cluster_name = 'effect-type-map-clustered_{}_{}'.format(it_nc,inv_p_threshold)
                returned_results.append((effect_type_cluster_name, label_map.reshape((-1,)+image_shape)))

                centroid_effect_type = np.zeros((effect_type[it_nc].shape[0],) + label_map.shape)
                for it_cluster in range(n_clusters):
                    index_cluster_center = np.where(label_map == it_cluster)[0]
                    centroid_effect_type[:,index_cluster_center] = kmeans.cluster_centers_[it_cluster][:,np.newaxis]

                effect_type_centroid_name = 'effect-type-centroid-clustered_{}'.format(inv_p_threshold)
                returned_results.append((effect_type_centroid_name, centroid_effect_type.reshape((-1,)+image_shape)))


    return returned_results


def volume_or_surface(extension):

    if extension == '.nii' or extension == '.nii.gz':
        return 'vol'
    elif extension == '.mgz' or extension == '.mgh':
        return 'vol'
    else:
        return 'surf'

