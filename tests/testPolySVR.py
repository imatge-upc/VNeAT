from nonlinear2.SVR import LinSVR, PolySVR
from sklearn.datasets import make_regression
from sklearn.metrics import explained_variance_score
from numpy import array, zeros, float64, asarray
import matplotlib.pyplot as plt
from os.path import join, isfile, basename
from os import listdir
from nonlinear2.ExcelIO import ExcelSheet as Excel
from nonlinear2.Subject import Subject
import nibabel as nib
import time

if __name__ == "__main__":


    """ PART 1: ARTIFICIAL DATA """

    # Get artificial data
    print("Getting artificial data...")
    X, y = make_regression(130, 2, bias=50, noise=10)
    regressors = zeros((len(X), 1))
    correctors = zeros((len(X), 1))
    regressors[:, 0] = X[:, 0]
    correctors[:, 0] = X[:, 1]

    # Init Polynomial SVR fitters
    print("Creating instance of SVR fitters...")
    poly_svr = LinSVR(regressors, correctors)  # First feature regressor and second feature corrector

    # Fit data
    print("Fitting data...")
    poly_svr.fit(y, C=100, epsilon=1e-5, n_jobs=2)

    # Plot prediction
    print("Plotting curves...")
    poly_predicted = poly_svr.predict()
    poly_corrected = poly_svr.correct(y)
    plt.scatter(regressors, poly_predicted, c='g', label='Poly SVR prediction')
    plt.scatter(regressors, poly_corrected, c='r', label='Poly SVR correction')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Polynomial Support Vector Regression')
    plt.legend()
    plt.show()

    # Print fit evaluation
    print("Evaluating fits...")
    print("Poly SVR explained variance score: ", explained_variance_score(poly_corrected, poly_predicted))



    """ PART 2: AETIONOMY DATA """

    # Get data from Excel and nii files
    print("Loading Aetionomy data...")
    DATA_DIR = join("C:\\", "Users", "santi", "Documents", "Santi", "Universitat", "TFG", "Data", "nonlinear_data", "Nonlinear_NBA_15")
    EXCEL_FILE = join("C:\\", "Users", "santi", "Documents", "Santi", "Universitat", "TFG", "Data", "nonlinear_data", "work_DB_CSF.R1.final.xls")
    RESULTS_DIR = join("C:\\", "Users", "santi", "Documents", "Santi", "Universitat", "TFG", "Results", "PSVR")

    filenames = filter(isfile, map(lambda elem: join(DATA_DIR, elem), listdir(DATA_DIR)))
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

    # Coordinates of the voxels to fit
    x1 = 10
    x2 = 30
    y1 = 10
    y2 = 20
    z1 = 10
    z2 = 20

    # Execution params
    C = 50
    epsilon = 1e-6
    n_jobs = 2

    # Get regressors, correctors and observations
    aet_regressors = array(map(lambda subject: subject.get([Subject.ADCSFIndex]), subjects), dtype = float64)
    aet_correctors = array(map(lambda subject: subject.get([Subject.Age, Subject.Sex]), subjects), dtype = float64)
    observations = asarray(map(lambda subject: nib.load(subject.gmfile).get_data(), subjects))
    real_obs = observations[:, x1:x2, y1:y2, z1:z2]
    del observations

    # LinSVR fitter
    print("Creating LinSVR fitter...")
    lin_svr = LinSVR(aet_regressors, aet_correctors)

    # Fit data
    print("Fitting Aetionomy data...")
    dims = real_obs.shape
    num_voxels = dims[1]*dims[2]*dims[3]
    reshaped_obs = real_obs.reshape((dims[0], num_voxels))
    start_time = time.clock()
    lin_svr.fit(reshaped_obs, C=C, epsilon=epsilon, n_jobs=n_jobs)
    end_time = time.clock()

    # Plot fitting curves
    print("Plotting curves...")
    reg = aet_regressors[:, 0]
    # plt.scatter(reg, reshaped_obs[:, 0], c='k', label='Original Data')
    lin_corrected = lin_svr.correct(reshaped_obs)[:, 0]
    plt.scatter(reg, lin_corrected, c='r', label='Linear SVR correction')
    lin_predicted = lin_svr.predict()[:, 0]
    plt.plot(reg, lin_predicted, c='g', label='Linear SVR prediction')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Linear SVR fitting on ')
    plt.legend()
    plt.show()

    # Print fit evaluation
    print("Evaluating fits...")
    print("Linear SVR explained variance score: ", explained_variance_score(lin_corrected, lin_predicted))

    # Print execution info
    print("Using the following parameters for the SVR fitter the fitting time was " + \
          str(end_time - start_time) + " s")
    print("\tC: " + str(C))
    print("\tepsilon: " + str(epsilon))
    print("\t# processes: " + str(n_jobs))
    print("\t# voxels fitted: " + str(num_voxels))
