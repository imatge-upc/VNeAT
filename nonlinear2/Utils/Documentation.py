
def docstring_inheritor(superclass = type):
	class DocStringInheritor(superclass):
		'''A variation on
			http://groups.google.com/group/comp.lang.python/msg/26f7b4fcb4d66c95
			by Paul McGuire
		'''

		def __new__(meta, name, bases, clsdict):
			#	print 'meta =', meta
			#	print 'name =', name
			#	print 'bases =', bases
			#	print 'clsdict:'
			#	for key, value in clsdict.iteritems():
			#		print '    ', key, '=', value
			#	print

			if not ('__doc__' in clsdict and clsdict['__doc__']):
				for mro_cls in (mro_cls_i for base in bases for mro_cls_i in base.mro()):
					doc = mro_cls.__doc__
					if doc:
						clsdict['__doc__'] = doc
						break

			for attr_name, attribute in clsdict.iteritems():
				if not attribute.__doc__:
					for mro_cls in (mro_cls_i for base in bases for mro_cls_i in base.mro() if hasattr(mro_cls_i, attr_name)):
						doc = getattr(getattr(mro_cls, attr_name), '__doc__')
						if doc:
							attribute.__doc__ = doc
							break

			return superclass.__new__(meta, name, bases, clsdict)

	return DocStringInheritor
