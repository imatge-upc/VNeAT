from sys import argv, exit
from GLM import PolyGLM as GLM

argc = len(argv)

fitter_names = ['GLM', 'GAM', 'SVR', 'ANN']
fitters = [GLM]

def usage():
	print 'Usage: ' + argv[0] + ' <fitter>'
	print '  Parameters:'
	print '    - <fitter>: integer'
	i = 0
	while i < len(fitters):
		print i, ' --> ', fitter_names[i]
		i += 1
	while i < len(fitter_names):
		print i, ' --> ', fitter_names[i], '(not implemented)'
		i += 1
	exit()

if argc != 2:
	usage()

try:
	fitter_index = int(argv[1])
except:
	usage()
if fitter_index > 3 or fitter_index < 0:
	usage()

try:
	fitter = fitters[fitter_index]
except IndexError:
	print 'Sorry, this fitter is not implemented yet. Please, select a different fitter.'
	exit()


