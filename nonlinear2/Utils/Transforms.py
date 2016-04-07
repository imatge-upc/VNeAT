'''Module that defines useful functions to transform data
	and data-structures'''

import numpy as np

def combinatorial(func, elements, k, start = 0):
	'''Generates func(x1, ...(func(x(k-2), func(x(k-1), xk))) for each possible
		combination of 'k' elements in 'elements' (repetitions are allowed).
		Example:
		    func = lambda x, y: x + y # concatenate the strings
		    elements = ['w', 'x', 'y', 'z']
		    k = 3
		    for c in combinatorial(func, elements, k):
		        print c

		    # The possible combinations of 3 elements taken from ['w', 'x', 'y', 'z']
		    # (regardless of the order but allowing repetitions) will be printed, since
		    # we just concatenated the elements selected by the function.
	'''

	n = len(elements) - start
	if n > 0 and k > 0:
		for y in combinatorial(func, elements, k, start+1):
			yield y
		x = elements[start]
		for d in xrange(1, k):
			for y in combinatorial(func, elements, k - d, start+1):
				yield func(x, y)
			x = func(x, elements[start])
		yield x


def polynomial(degree, features, complete_polynomy = True, constant_term = False):
	if constant_term:
		assert len(features) > 0
		yield np.array([1]*len(features[0]))

	if not isinstance(features, np.ndarray):
		features = np.array(features)

	if complete_polynomy:
		init = 1
	else:
		init = degree

	for d in xrange(init, degree+1):
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

def flatten(l, ndims_to_reduce = 1):
	r = l
	for _ in xrange(ndims_to_reduce):
		try:
			r = list(chain.from_iterable(r))
		except TypeError:
			break
	return r



