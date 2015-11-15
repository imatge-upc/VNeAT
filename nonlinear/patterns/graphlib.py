from abc import ABCMeta, abstractmethod

class Graph:
	__metaclass__ = ABCMeta

	@abstractmethod
	def nodes(self):
		return []

	@abstractmethod
	def neighbours(self, node):
		return []


class NiftiGraph(Graph):
	
	def __init__(self, data, limit_pvalue):
		self.data = data
		self.pv = limit_pvalue

	def nodes(self):
		dims = self.data.shape
		for i in range(dims[0]):
			for j in range(dims[1]):
				for k in range(dims[2]):
					yield (i, j, k)

	def neighbours(self, node):
		x, y, z = node
		if self.data[x, y, z] > self.pv:
			return
		for dx in (-1, 0, 1):
			for dy in (-1, 0, 1):
				for dz in (-1, 0, 1):
					if dx == 0 and dy == 0 and dz == 0:
						continue
					try:
						if self.data[x + dx, y + dy, z + dz] < self.pv:
							yield (x+dx, y+dy, z+dz)
					except IndexError:
						continue


class GenericGraph(Graph):

	def __init__(self, nodes = []):
		self.node_info = []
		self.edges = []
		for node in nodes:
			self.node_info.append(node)
			self.edges.append(set())

	def nodes(self):
		return range(len(self.node_info))

	def neighbours(self, u):
		try:
			return self.edges[u]
		except IndexError:
			return set()

	def add_edge(self, u, v, bidirectional = False):
		if v > len(self.node_info) or v < 0 or u < 0:
			return
		try:
			self.edges[u].add(v)
		except IndexError:
			return
		if bidirectional:
			self.edges[v].add(u)

	def remove_edge(self, u, v, bidirectional = False):
		if v > len(self.node_info) or v < 0 or u < 0:
			return
		try:
			self.edges[u].discard(v)
		except IndexError:
			return
		if bidirectional:
			self.edges[v].discard(u)

	def __repr__(self):
		return 'GenericGraph{ Nodes(' + repr(self.node_info) + ') ; Edges(' + repr(self.edges) + ') }'

	def __str__(self):
		s = 'GenericGraph{\n'
		s += '    Nodes( ' + repr(self.node_info) + ' )\n'
		s += '    Edges(\n'
		for i in range(len(self.edges)):
			stri = str(i)
			s += ' '*(10-len(stri)) + stri + ': ' + repr(self.edges[i]) + '\n'
		s += '    )\n'
		s += '}'
		return s


def tarjan(nodes, neighbours):

	class TarjanNode:
		def __init__(self, node):
			self.node = node
			self.onstack = False

		def __eq__(self, other):
			return self.node == other.node


	tnodes = {node : TarjanNode(node) for node in nodes}

	tarjan.index = 0
	S = []

	def strongconnect(v):
		v.index = tarjan.index
		v.low_link = tarjan.index
		tarjan.index += 1
		S.append(v)
		v.onstack = True

		for w in neighbours(v.node):
			w = tnodes[w]
			try:
				if w.index < v.low_link and w.onstack:
					v.low_link = w.index
			except AttributeError:
				for x in strongconnect(w):
					yield x
				if w.low_link < v.low_link:
					v.low_link = w.low_link

		if v.low_link == v.index:
			scc = set()
			w = None
			while w != v:
				w = S[-1]
				del S[-1]
				w.onstack = False
				scc.add(w.node)
			yield scc

	for v in tnodes.values():
		if not hasattr(v, 'index'):
			for scc in strongconnect(v):
				yield scc

