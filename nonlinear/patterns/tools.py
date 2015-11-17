from collections import MutableMapping
from numpy import array as nparray


class TransformedDict(MutableMapping):
	"""A dictionary that applies an arbitrary key-altering
	   function before accessing the keys"""

	def __init__(self, *args, **kwargs):
		self.store = dict()
		self.update(dict(*args, **kwargs))  # use the free update to set keys

	def __getitem__(self, key):
		return self.store[self.__keytransform__(key)]

	def __setitem__(self, key, value):
		self.store[self.__keytransform__(key)] = value

	def __delitem__(self, key):
		del self.store[self.__keytransform__(key)]

	def __iter__(self):
		return iter(self.store)

	def __len__(self):
		return len(self.store)

	def __repr__(self):
		return dict.__repr__(self.store)

	def __str__(self):
		return dict.__str__(self.store)

	def __keytransform__(self, key):
		return key


def combinatorial(func, features, n, start = 0):
	num_features = len(features) - start
	if num_features > 0 and n > 0:
		for y in combinatorial(func, features, n, start+1):
			yield y
		x = features[start]
		for d in range(1, n):
			for y in combinatorial(func, features, n - d, start+1):
				yield func(x, y)
			x = func(x, features[start])
		yield x


#	def mem(f):
#		mem = {}
#		def f2(*args):
#			try:
#				return mem[args]
#			except KeyError:
#				mem[args] = f(*args)
#				print args, ' --> ', mem[args]
#				return mem[args]
#		return f2
#
#	combinatorial = mem(combinatorial)


def polynomial(degree, features, complete_polynomy = True, constant_term = False):
	if constant_term:
		assert len(features) > 0
		yield nparray([1]*len(features[0]))

	if complete_polynomy:
		init = 1
	else:
		init = degree

	for d in range(init, degree+1):
		for term in combinatorial(lambda x, y: x*y, features, d):
			yield term


def copy_iterable(it):
	try:
		return (copy_iterable(x) for x in it)
	except TypeError:
		try:
			return it.copy()
		except AttributeError:
			return it


def tolist(it, first_call = True):
	try:
		return [tolist(x, False) for x in it]
	except TypeError:
		if first_call:
			return [it]
		else:
			return it

