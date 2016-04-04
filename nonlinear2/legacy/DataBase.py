import csv

from nonlinear2.Utils.ExcelIO import ExcelSheet as Excel


class InputManager:
	'''Class that manages the input database of the system.
	'''

	class InputDataDialect(csv.Dialect):
		delimiter = ','
		doublequote = True
		lineterminator = '\r\n'
		quotechar = '"'
		quoting = csv.QUOTE_NONE
		skipinitialspace = True
		escapechar = None

	def __init__(self, input_file, data_dir):
		raise NotImplementedError

	@staticmethod
	def fromExcelToCSV(excel_file, csv_file, sheet_index = 0, header_row = 0, *args, **kwargs):
		'''Creates a new CSV file by using the information in a well-organized Excel file.
			
			Parameters:
			    
			    - excel_file (str): specifies the path of the Excel file from which the
			       data must be retrieved.
			    
			    - csv_file (str): specifies the path of the CSV file to be created with
			       the retrieved data.
			    
			    - sheet_index (int) (optional, default 0): sheet in the Excel file from
			       which the data will be retrieved (see ExcelSheet in ExcelIO module).
			    
			    - header_row (int) (optional, default 0): row in the Excel file's sheet
			       where the headers are explicited (see ExcelSheet in ExcelIO Module).
			       Such header must contain a field named 'ID' (case insensitive)
			    
			    - excel_args (tuple) (optional, default ()): unnamed arguments to be passed
			       to the 'get_rows' function in ExcelSheet class (see ExcelIO module).
			    
			    - excel_kwargs (dict) (optional, default {}): named arguments to be passed
			       to the 'get_rows' function in ExcelSheet class (see ExcelIO module).
			    
			    - any other options/parameters will be passed to the 'get_rows' function in
			       the ExcelSheet class (see ExcelIO module).
			       Note: if argument 'fieldstype' is included in the form of a dictionary as
			       an additional parameter, the field 'ID' (case insensitve) must also be
			       present in it (otherwise a KeyError will be raised).
			
			Actions:
			    
			    - A new file is created (replaced if it already existed) in the file system
			       with the path indicated by argument csv_file. This new file contains the
			       information from the Excel file in a CSV (Comma Separated Values) format,
			       suitable to be read by the InputManager class.
			
			Returns:
			    
			    - None

		'''

		src = Excel(excel_file, sheet_index, header_row)

		key = 'id'
		headers = src.headers

		if not key in src.headers:
			raise KeyError(key + ' field must be present in the header')

		with open(csv_file, 'wb') as dst:
			writer = csv.DictWriter(dst, fieldnames = headers, dialect = InputManager.InputDataDialect)

			writer.writeheader()
			writer.writerows(filter(lambda row: row[key] != None, src.get_rows(*args, **kwargs)))


