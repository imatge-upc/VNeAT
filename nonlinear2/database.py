from ExcelIO import ExcelSheet as Excel
from Subject import Subject

from os.path import join, isfile, basename
from os import listdir
import nibabel as nib

from user_paths import DATA_DIR, EXCEL_FILE

niiFile = nib.Nifti1Image

class Focus(object):
	pass

cur_focus = Focus()

def get_data():
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

	cur_focus.affine = nib.load(subjects[0].gmfile).affine

	return subjects

def open_output_file(filename, *args, **kwargs):
	cur_focus.output = nib.load(filename, *args, **kwargs)
	return cur_focus.output

def save_output_data(data, filename = None, *args, **kwargs):
	try:
		d = cur_focus.output.get_data()
		try:
			d[:] = data
		except ValueError:
			if filename == None:
				filename = cur_focus.output.filename
			cur_focus.output = niiFile(data, cur_focus.affine)
	except AttributeError:
		cur_focus.output = niiFile(data, cur_focus.affine)
	
	nib.save(cur_focus.output, filename, *args, **kwargs)



