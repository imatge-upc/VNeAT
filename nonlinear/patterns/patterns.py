# To be executed as follows:
# python -W ignore -u ttest.py


print
print 'Welcome to ttest.py!'
print 'We are now going to analyze the data in the specified directory and perform t-tests over it.'
print 'There is a lot of data to process, so please be pacient.'
print 'Thank you for using our software! :)'
print

# Set paths
WORK_DIR = '/Users/Asier/Documents/TFG/Alan T'
DATA_DIR = 'Nonlinear_NBA_15'
EXCEL_FILE = 'work_DB_CSF.R1.final.xls'
OUTPUT_DIR = 'ttests'
OUTPUT_FILENAME = 'ttest_results'

# Set region
regx_init = 0
regx_end = 1000
regy_init = 0
regy_end = 1000
regz_init = 0
regz_end = 1000

MEMORY_USE = 100 # approx. memory use in MB (depends on garbage collector, but the order of magnitude should be around this value)
