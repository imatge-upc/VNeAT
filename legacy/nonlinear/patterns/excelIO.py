from xlrd import open_workbook as open_wb
from xlrd.sheet import ctype_text as type2text

from tools import TransformedDict as tdict


class ExcelRow(tdict):
    def __keytransform__(self, key):
        return key.strip().lower()

    def __repr__(self):
        return 'ExcelRow( ' + tdict.__repr__(self) + ' )'

    def __str__(self):
        return 'ExcelRow( ' + tdict.__str__(self) + ' )'


def get_rows(filename,
             start=1, end=None,
             sheet_index=0,
             header_row=0,
             fields={
                 'id': (lambda s: str(s).strip().split('_')[0]),
                 'diag': (lambda s: int(s) - 1),
                 'age': int,
                 'sex': (lambda s: 2 * int(s) - 1),
                 'apoe4_bin': (lambda s: 2 * int(s) - 1),
                 'escolaridad': int,
                 'ad_csf_index_ttau': float
             }
             ):
    info = []
    with open_wb(filename) as wb:
        # Open first sheet
        ws = wb.sheet_by_index(sheet_index)
        # This could also be done as follows
        # sheet_names = wb.sheet_names()
        # ws = wb.sheet_by_name(sheet_names[0])
        #
        # Or by using the name of the sheet, i.e., DB_clinic (even if it's not the first one):
        # ws = wb.sheet_by_name('DB_clinic')

        # Get the column index for each header in the sheet (headers must be in the first row and text typed)
        h = ws.row(
            header_row)  # Extract first row to make it more efficient (since we're gonna make multiple sequential reads on it)
        header2col = ExcelRow()
        for j in range(ws.ncols):
            if type2text[h[j].ctype] == 'text':
                header2col[h[j].value] = j

        yes = []
        no = []
        for field in fields:
            try:
                yes.append((field, header2col[field]))
            except KeyError:
                no.append(field)

        if len(yes) == 0:
            return []

        if end == None:
            end = ws.nrows

        for i in range(start, end):
            r = ExcelRow()
            rinfo = ws.row(i)
            for field, j in yes:
                if type2text[rinfo[j].ctype] in ('text', 'number'):
                    r[field] = fields[field](rinfo[j].value)
                else:
                    r[field] = None

            if any(x != None for x in r.values()):
                for field in no:
                    r[field] = None
                info.append(r)
    return info
