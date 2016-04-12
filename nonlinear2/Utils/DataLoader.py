import numpy as np
import nibabel as nib
from os import listdir
from os.path import join, isfile, basename
from nonlinear2.Utils.ExcelIO import ExcelSheet as Excel
from nonlinear2.Utils.Subject import Subject
from nonlinear2.user_paths import DATA_DIR, CORRECTED_DATA_DIR, EXCEL_FILE, MNI_TEMPLATE

def getSubjects(corrected_data=False):
    """
    Gets a list of subjects (nonlinear2.Utils.Subject.Subject instances) from the Excel file

    Parameters
    ----------
    corrected_data Boolean indicating whether normal data or corrected data should be loaded

    Returns
    -------
    [list] list with all the Subject instances
    """
    if corrected_data:
        filenames = filter(isfile, map(lambda elem: join(CORRECTED_DATA_DIR, elem),
                                       listdir(CORRECTED_DATA_DIR)))
        filenames_by_id = {basename(fn).split('_')[1][:-4] : fn for fn in filenames}
    else:
        filenames = filter(isfile, map(lambda elem: join(DATA_DIR, elem),
                                       listdir(DATA_DIR)))
        filenames_by_id = {basename(fn).split('_')[0][8:] : fn for fn in filenames}

    exc = Excel(EXCEL_FILE)

    subjects = []
    for r in exc.get_rows( fieldstype = {
                    'id':(lambda s: str(s).strip().split('_')[0]),
                    'diag':(lambda s: int(s) - 1),
                    'age':int,
                    'sex':(lambda s: 2*int(s) - 1),
                    'apoe4_bin':(lambda s: 2*int(s) - 1),
                    'escolaridad':int,
                    'ad_csf_index_ttau':float
                 } ):
        subjects.append(
            Subject(
                r['id'],
                filenames_by_id[r['id']],
                r.get('diag', None),
                r.get('age', None),
                r.get('sex', None),
                r.get('apoe4_bin', None),
                r.get('escolaridad', None),
                r.get('ad_csf_index_ttau', None)
            )
        )
    return subjects

def getGMData(corrected_data=False):
    """
    Gets the grey matter from the Nifti files
    Parameters
    ----------
    corrected_data

    Returns
    -------
    [numpy.array] 4-dimensional array with gray matter values for all voxels and subjects
    """
    subjects = getSubjects(corrected_data)
    return np.asarray(map(lambda subject: nib.load(subject.gmfile).get_data(), subjects))

def getFeatures(features_array):
    """
    Gets a list of features from the Subjects
    Parameters
    ----------
    features_array list list of the features to be obtained from Subject
    (e.g [Subject.ADSCFIndex, Subject.Age])

    Returns
    -------
    [numpy.array] 2-dimensional array with the features
    """
    subjects = getSubjects(False)
    return np.array(map(lambda subject: subject.get(features_array), subjects), dtype = np.float64)

def getMNIAffine():
    return nib.load(MNI_TEMPLATE).affine