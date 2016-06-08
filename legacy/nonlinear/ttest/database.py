from os import listdir
from os.path import join, isfile

from manage_input import InputData
from manage_output import OutputData


class Focus:
    pass


cur_focus = Focus()


def get_data(DATA_DIR=join('/', 'Users', 'Asier', 'Documents', 'TFG', 'Alan T', 'Nonlinear_NBA_15'),
             EXCEL_FILE=join('/', 'Users', 'Asier', 'Documents', 'TFG', 'Alan T', 'work_DB_CSF.R1.final.xls'),
             mem_usage=None, *args, **kwargs):
    filenames = filter(isfile, map(lambda elem: join(DATA_DIR, elem), listdir(DATA_DIR)))
    cur_focus.input = InputData(filenames, EXCEL_FILE, *args, **kwargs)
    return cur_focus.input


def open_output_file(filename, *args, **kwargs):
    cur_focus.output = OutputData.open(filename, *args, **kwargs)
    return cur_focus.output


def save_output_data(data, filename=None, *args, **kwargs):
    try:
        d = cur_focus.output.get_data()
        try:
            d[:] = data
        except ValueError:
            if filename == None:
                filename = cur_focus.output.filename
            cur_focus.output = OutputData(data, cur_focus.output.affine)
    except AttributeError:
        cur_focus.output = OutputData(data, cur_focus.input.affine)

    cur_focus.output.save(filename, *args, **kwargs)
