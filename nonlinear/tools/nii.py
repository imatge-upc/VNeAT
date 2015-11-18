def slice(data, x = None, y = None, z = None):

	def outofrange(axis, x = x, y = y, z = z, let = ['x', 'y', 'z']):
		if eval(let[axis]) < 0 or eval(let[axis]) >= data.shape[axis]:
			print 'ERROR: Out of range index; size of ' + let[axis] + ' is ' + str(data.shape[axis]) + ' (entered index was ' + str(eval(let[axis])) + ')'
			return True
		return False

	if x == None and y == None and z == None:
		print 'Hey! Select a fucking plane!'
		return
	elif (x != None and (y != None or z != None)) or (y != None and z != None):
		print 'Ha ha, very funny. Select JUST ONE fucking plane!'
		return
	if x != None:
		if outofrange(0):
			return
		return data[x]

	if y != None:
		if outofrange(1):
			return
		return data[:, y]
	if z != None:
		if outofrange(2):
			return
		return data[:, :, z]


def output(img, filename):
	f = open(filename, 'wb')
	for row in img:
		for x in row:
			f.write(str(x) + '\t')
		f.write('\r\n')
	f.close()