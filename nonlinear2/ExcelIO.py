from Dictionaries import TransformedDict as tdict
from xlrd import open_workbook as open_wb
from xlrd.sheet import ctype_text as type2text

class ExcelSheet:
	'''Class that implements a reading method for a well-organized sheet in an Excel
		file (.xls only).
	'''

	class Row(tdict):
		'''Dictionary-like class that represents a row in an Excel sheet. Keys are
			both, case-insensitive and initial- and final-whitespace-insensitive.
		'''

		def __keytransform__(self, key):
			return key.strip().lower()

		def __repr__(self):
			return 'ExcelRow( ' + tdict.__repr__(self) + ' )'

		def __str__(self):
			return 'ExcelRow( ' + tdict.__str__(self) + ' )'


	def __init__(self, filename, sheet_index = 0, header_row = 0):
		'''Constructor.

			Parameters:

			    - filename (str): Path to the Excel file.

			    - sheet_index (int) (optional, default 0): 0-based index of the sheet
			       to be read.

			    - header_row (int) (optional, default 0): 0-based index of the row in
			       which the header can be found.

			Modifies:

			    - [created] self.filename (str, read-only property): same as 'filename'
			       parameter.

			    - [created] self.sheet (int, read-only property): same as 'sheet_index'
			       parameter.

			    - [created] self.headers (list, read-only property): headers found in
			       the row indicated by 'header_row' parameter.

			Returns:

			    - New instance of ExcelSheet.
		'''

		self._filename = filename
		self._sheet = sheet_index
		self._header_row = header_row

		with open_wb(filename) as wb:
			# Open sheet
			ws = wb.sheet_by_index(sheet_index)

			# Get row of headers
			headers = ws.row(header_row)

			# Read it and extract headers and the columns where they are
			self._header2col = ExcelSheet.Row()
			for col in xrange(ws.ncols):
				if type2text[headers[col].ctype] == 'text':
					self._header2col[headers[col].value] = col

	@property
	def filename(self):
		'''(str) Path to the Excel file from which the data is retrieved.
		'''
		return self._filename

	@property
	def sheet(self):
		'''(int) Sheet inside the Excel file from which the data is retrieved.
		'''
		return self._sheet
	
	@property
	def headers(self):
		'''(list) Valid fields for which information can be retrieved in this
			Excel sheet.
		'''
		return self._header2col.keys()
	
	def get_rows(self, start = None, end = None, fieldstype = str):
		'''Reads the specified fields for the rows in [start, end) in this Excel
			sheet.

			Parameters:

			    - start, end (int) (optional): 0-based row indices for which the data
			       must be retrieved. Any row in [start..end-1] that has a valid value
			       for at least one field will be returned. By default, all rows below
			       the headers row will be analyzed.

			    - fieldstype (type / dict) (optional, default str): If this parameter
			       is a dictionary, then it indicates the fields to be retrieved and
			       the function that must be applied to each of them before retrieving
			       them. Otherwise, all fields shall be retrieved after casting them to
			       the specified type (if possible).

			Returns:

			    - (generator) A generator of ExcelSheet.Row instances such that each
			       instance contains the information retrieved in each row. Any fields
			       that raise errors when applying the specified function/casting for
			       a certain row will be set to None in the corresponding instance. If
			       a requested field does not exist in the Excel sheet, its value will
			       be set to None in all the rows. Rows with no valid fields (all Nones)
			       are not yielded.
		'''

		# Open file
		with open_wb(self._filename) as wb:
			# Open sheet
			ws = wb.sheet_by_index(self._sheet)

			if isinstance(fieldstype, type):
				# Apply type casting to each element of Excel headers
				yes = self._header2col.items()
				no = []
				fieldstype = {field: fieldstype for field in self._header2col.iterkeys()}
			else:
				# Compute intersection with valid fields (in 'yes', invalids in 'no')
				yes = []
				no = []
				for field in fieldstype:
					try:
						yes.append((field, self._header2col[field]))
					except KeyError:
						no.append(field)

			if len(yes) == 0:
				# No valid fields, all rows will be all Nones, so we just return an empty list
				return

			# Set default start and end (if not already set)
			if start == None:
				start = self._header_row + 1
			elif start < 0:
				start == 0
			if end == None or end > ws.nrows:
				end = ws.nrows

			for i in xrange(start, end):
				r = ExcelSheet.Row()
				# Retrieve row
				rinfo = ws.row(i)
				for field, col in yes:
					# Valid elements are in 
					if type2text[rinfo[col].ctype] in ('text', 'number'):
						try:
							r[field] = fieldstype[field](rinfo[col].value)
						except:
							r[field] = None
					else:
						r[field] = None

				if any(x != None for x in r.values()):
					for field in no:
						r[field] = None
					yield r

