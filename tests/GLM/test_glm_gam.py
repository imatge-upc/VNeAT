import sys
sys.path.insert(1,'C:\\Users\\upcnet\\Repositoris\\neuroimatge\\nonlinear2')
sys.stdout.flush()
from Utils.ExcelIO import ExcelSheet as Excel
from Processors import PolyGLMProcessor as PGLMP
from Processors import GAMProcessor
from Utils.Subject import Subject
from os.path import join, isfile, basename
from os import listdir
import nibabel as nib
from numpy import array as nparray


print 'Obtaining data from Excel file...'
DATA_DIR = join('C:\\','Users','upcnet','FPM','data_backup','Non-linear', 'Nonlinear_NBA_15')
EXCEL_FILE = join('C:\\','Users','upcnet','FPM','data_backup','Non-linear', 'work_DB_CSF.R1.final.xls')
observations = [0.2784765362739563, 0.24556149542331696, 0.3095938265323639, 0.3048464059829712, 0.26070791482925415, 0.15151332318782806, 0.3006623387336731, 0.2808901071548462, 0.25516417622566223, 0.3316618502140045, 0.30973631143569946, 0.37002524733543396, 0.3213159739971161, 0.3467503488063812, 0.4094521105289459, 0.2750070095062256, 0.27810442447662354, 0.28163591027259827, 0.2537412643432617, 0.3250562250614166, 0.34289345145225525, 0.22144272923469543, 0.3624565601348877, 0.3173443675041199, 0.2585237920284271, 0.3014221489429474, 0.41157791018486023, 0.3544761836528778, 0.30399376153945923, 0.26001420617103577, 0.27112865447998047, 0.3054848909378052, 0.26999059319496155, 0.3316003382205963, 0.3346855640411377, 0.34855127334594727, 0.28872963786125183, 0.42628028988838196, 0.4596898853778839, 0.30583295226097107, 0.28759369254112244, 0.27858930826187134, 0.3581596314907074, 0.28936967253685, 0.2677311897277832, 0.32336705923080444, 0.3066917359828949, 0.25871947407722473, 0.3901004195213318, 0.25603029131889343, 0.3349824547767639, 0.3411894142627716, 0.29385727643966675, 0.2390265315771103, 0.2856198847293854, 0.3241741955280304, 0.3451581299304962, 0.3011266589164734, 0.303628146648407, 0.281001478433609, 0.27349668741226196, 0.26146239042282104, 0.2781486511230469, 0.2565879821777344, 0.3863266408443451, 0.33011889457702637, 0.3556863069534302, 0.2609650194644928, 0.220051571726799, 0.42904773354530334, 0.35917094349861145, 0.21222135424613953, 0.3042793571949005, 0.22294996678829193, 0.35733962059020996, 0.3183954656124115, 0.278012752532959, 0.22756223380565643, 0.3604796826839447, 0.2609047293663025, 0.21358883380889893, 0.29254892468452454, 0.24075961112976074, 0.3065904974937439, 0.30007368326187134, 0.429603636264801, 0.36638107895851135, 0.2718782126903534, 0.3247428834438324, 0.4444065988063812, 0.3375723361968994, 0.2923850119113922, 0.2820183038711548, 0.3199133574962616, 0.37877288460731506, 0.3873538672924042, 0.3950746953487396, 0.33014264702796936, 0.2593349814414978, 0.3743993043899536, 0.3679767847061157, 0.33566731214523315, 0.33407989144325256, 0.41721007227897644, 0.42419081926345825, 0.38705915212631226, 0.42910948395729065, 0.3889360725879669, 0.43863430619239807, 0.29597049951553345, 0.3638444244861603, 0.5047481060028076, 0.3418075740337372, 0.3438158929347992, 0.32369306683540344, 0.36633098125457764, 0.2583903670310974, 0.34897589683532715, 0.3453844487667084, 0.3567051589488983, 0.30068838596343994, 0.24640996754169464, 0.4144071638584137, 0.39200517535209656, 0.3264894187450409, 0.3394283950328827, 0.400669127702713, 0.3835843503475189, 0.3778901696205139]
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

print 'Initializing PolyGLM Processor...'
pglmp = PGLMP(subjects, regressors = [Subject.ADCSFIndex], correctors = [Subject.Age,Subject.Sex])
gamp = GAMProcessor(subjects, regressors = [Subject.ADCSFIndex],correctors = [Subject.Age,Subject.Sex])

print 'Processing data...'
x1=54#105
x2=x1+1
y1=101#67
y2=y1+1
z1=89#53
z2=z1+1
results_glm = pglmp.process(x1=x1,x2=x2,y1=y1,y2=y2,z1=z1,z2=z2)
results_gam = gamp.process(x1=x1,x2=x2,y1=y1,y2=y2,z1=z1,z2=z2)

print 'Formatting obtained data to display it...'
z_scores_glm, labels_glm = pglmp.fit_score(results_glm.fitting_scores, produce_labels = True)
z_scores_gam, labels_gam = gamp.fit_score(results_gam.fitting_scores, produce_labels = True)

print 'Saving results to files...'

affine = nparray(
		[[ -1.50000000e+00,   0.00000000e+00,   0.00000000e+00,   9.00000000e+01],
		 [  1.99278252e-16,   1.50000000e+00,   2.17210575e-16,  -1.26000000e+02],
		 [ -1.36305018e-16,  -1.38272305e-16,   1.50000000e+00,  -7.20000000e+01],
		 [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]]
)

# from matplotlib import pyplot as plt
# import numpy as np
# plt.figure()
# corrected_data_glm = pglmp.corrected_values(results_glm.correction_parameters, x1=x1,x2=x2,y1=y1,y2=y2,z1=z1,z2=z2)
# x_glm=np.array(np.sort([sbj.adcsf for sbj in subjects]))
# x_cub_glm = np.array([np.squeeze(x_glm)**i for i in np.arange(1,4)]).T
# plt.plot(x_glm,np.squeeze(corrected_data_glm),'k.')
# plt.plot(x_glm,x_cub_glm.dot(np.squeeze(results_glm.regression_parameters)),'y')
#
#
# corrected_data_gam = gamp.corrected_values(results_gam.correction_parameters, x1=x1,x2=x2,y1=y1,y2=y2,z1=z1,z2=z2)
# x_gam=np.array(np.sort([sbj.adcsf for sbj in subjects]))
# x_cub_gam = np.array([np.squeeze(x_gam)**i for i in range(4)]).T
# plt.plot(x_gam,np.squeeze(corrected_data_gam),'r.')
# plt.plot(x_gam,x_cub_gam.dot(np.squeeze(results_gam.regression_parameters[3:])),'b')
#
# plt.show()

niiFile = nib.Nifti1Image

nib.save(niiFile(results_glm.correction_parameters, affine), 'fpmalfa_glm_cparams.nii')
nib.save(niiFile(results_glm.regression_parameters, affine), 'fpmalfa_glm_rparams.nii')
nib.save(niiFile(results_glm.fitting_scores, affine), 'fpmalfa_glm_fitscores.nii')
nib.save(niiFile(z_scores_glm, affine), 'fpmalfa_glm_zscores.nii')
nib.save(niiFile(labels_glm, affine), 'fpmalfa_glm_labels.nii')

with open('fpmalfa_glm_userdefparams.txt', 'wb') as f:
	f.write(str(pglmp.user_defined_parameters) + '\n')


nib.save(niiFile(results_gam.correction_parameters, affine), 'fpmalfa_gam_cparams.nii')
nib.save(niiFile(results_gam.regression_parameters, affine), 'fpmalfa_gam_rparams.nii')
nib.save(niiFile(results_gam.fitting_scores, affine), 'fpmalfa_gam_fitscores.nii')
nib.save(niiFile(z_scores_gam, affine), 'fpmalfa_gam_zscores.nii')
nib.save(niiFile(labels_gam, affine), 'fpmalfa_gam_labels.nii')

with open('fpmalfa_gam_userdefparams.txt', 'wb') as f:
	f.write(str(gamp.user_defined_parameters) + '\n')

print 'Done.'
