from manage_data import DataManager
from os.path import join, isfile
from os import listdir


def get_data(WORK_DIR = join('/', 'Users', 'Asier', 'Documents', 'TFG', 'Alan T'),
			 DATA_DIR = 'Nonlinear_NBA_15',
			 EXCEL_FILE = 'work_DB_CSF.R1.final.xls',
			 *args, **kwargs):
	filenames = filter(isfile, map(lambda elem: join(WORK_DIR, DATA_DIR, elem), listdir(join(WORK_DIR, DATA_DIR))))
	dm = DataManager(filenames, join(WORK_DIR, EXCEL_FILE))
	return dm.voxels(*args, **kwargs)