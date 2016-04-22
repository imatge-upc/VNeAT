def slice(data, x = None, y = None, z = None):

	if sum(1 if elem != None else 0 for elem in [x, y, z]) != 1:
		raise ValueError("Slice function: One and only one plane must be selected")

	if x != None:
		return data[x]

	if y != None:
		return data[:, y]

	if z != None:
		return data[:, :, z]


def output(img, filename):
	f = open(filename, 'wb')
	for row in img:
		for x in row:
			f.write(str(x) + '\t')
		f.write('\r\n')
	f.close()